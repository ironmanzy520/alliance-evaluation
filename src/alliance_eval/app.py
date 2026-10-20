#!/usr/bin/env python3
"""Local Streamlit setup, execution progress and read-only evidence explorer.

The page uses the installed pipeline for evaluation and preset demonstrations.
Saved results provide dialogue annotations, summary evidence and descriptive
execution statistics, never aggregate alliance scores or accuracy estimates.
Credentials remain in the current session; original sources are read from their
recorded paths and checked against any recorded checksums.
"""

from __future__ import annotations

import ast
import hashlib
import html
import json
import os
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import streamlit as st
import streamlit.components.v1 as components
from alliance_eval.evaluate import DeepSeekJudge, LocalHTTPJudge, OpenAIJudge, MISTRAL_BASE_URL, MISTRAL_MODEL_ID
from alliance_eval.evaluation_models import (
    BOND_DIMENSIONS, SUMMARY_LEVEL_DIMENSIONS, TASK_DIMENSIONS, SamplePlan,
)
from alliance_eval.input_data import parse_json_value, validate_dialogue
from alliance_eval.pipeline import (
    check_runtime_version, load_selected_dialogues, run_all,
)


WORKSPACE_ROOT = Path(os.environ.get("ALLIANCE_EVAL_WORKSPACE", Path.cwd())).expanduser().resolve()
LOADED_APP_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
BUILTIN_PROVIDERS = ("deepseek", "openai", "mistral", "ollama")


def validate_model_url(base_url: str) -> None:
    """Validate a service address locally without contacting the endpoint."""

    try:
        _ = urlsplit(base_url).port
        url = httpx.URL(base_url)
    except (ValueError, httpx.InvalidURL):
        raise ValueError("Use a valid HTTP(S) base URL.") from None
    if url.scheme not in {"http", "https"} or not url.host or url.userinfo or url.query or url.fragment:
        raise ValueError("Use an HTTP(S) base URL without credentials, query or fragment.")


def load_custom_providers(workspace: Path) -> dict[str, dict[str, Any]]:
    """Read optional named connections locally; never discover or contact a service."""

    path = workspace / "providers.json"
    if not path.exists():
        return {}
    providers = parse_json_value(path.read_text(encoding="utf-8"))
    if not isinstance(providers, dict):
        raise ValueError("providers.json must contain an object of named connections.")
    defaults = {"dialect": "openai_compatible", "model_id": "", "api_key_env": "",
                "response_format": "json_schema", "parameters": {}}
    connections = {}
    for name, supplied in providers.items():
        if not name.strip() or name != name.strip() or name.casefold() in BUILTIN_PROVIDERS:
            raise ValueError("Connection names must be non-empty, trimmed and different from built-in providers.")
        if not isinstance(supplied, dict) or set(supplied) - {*defaults, "base_url"}:
            raise ValueError(f"Connection {name!r}: unknown fields; see docs/models.md.")
        config = {**defaults, **supplied}
        if any(not isinstance(config.get(field), str) for field in (
            "base_url", "dialect", "model_id", "api_key_env", "response_format",
        )) or not isinstance(config["parameters"], dict):
            raise ValueError(f"Connection {name!r}: expected string fields, base_url and a parameters object.")
        if config["dialect"] not in {"openai_compatible", "ollama"}:
            raise ValueError(f"Connection {name!r}: dialect must be openai_compatible or ollama.")
        if config["response_format"] not in {"json_schema", "json_object"} or (
            config["dialect"] == "ollama" and config["response_format"] != "json_schema"
        ):
            raise ValueError(f"Connection {name!r}: invalid response_format for this dialect.")
        if config["api_key_env"] and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", config["api_key_env"]):
            raise ValueError(f"Connection {name!r}: api_key_env must be an environment variable name, not a key.")
        try:
            validate_model_url(config["base_url"])
        except ValueError as error:
            raise ValueError(f"Connection {name!r}: {error}") from None
        reserved = {"model", "messages", "response_format", "stream", "api_key", "headers", "authorization", "base_url"}
        if reserved.intersection(key.lower() for key in config["parameters"]):
            raise ValueError(f"Connection {name!r}: parameters cannot contain authentication or request structure.")
        connections[name] = config
    return connections


def _render_runtime_status() -> bool:
    """Check local runtime identity without reading credentials or calling models."""

    try:
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != LOADED_APP_SOURCE_SHA256:
            raise RuntimeError("Loaded app differs from disk. Restart the Streamlit service.")
        check_runtime_version()
    except (OSError, ValueError, RuntimeError) as error:
        st.error(f"Run blocked — {error}")
        return False
    return True


def format_elapsed(seconds: float) -> str:
    """Format non-negative seconds as a duration with three-digit hours: hhh:mm:ss."""

    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:03d}:{minutes:02d}:{secs:02d}"


def running_timer_html() -> str:
    """Return a browser-updated timer that continues during blocking model calls."""

    return """
    <div id="elapsed" style="font: 600 24px monospace; color: #31333f;">
      000:00:00
    </div>
    <script>
      const startedAt = Date.now();
      const output = document.getElementById("elapsed");
      function updateElapsed() {
        const total = Math.floor((Date.now() - startedAt) / 1000);
        const hours = String(Math.floor(total / 3600)).padStart(3, "0");
        const minutes = String(Math.floor((total % 3600) / 60)).padStart(2, "0");
        const seconds = String(total % 60).padStart(2, "0");
        output.textContent = `${hours}:${minutes}:${seconds}`;
      }
      updateElapsed();
      setInterval(updateElapsed, 1000);
    </script>
    """


def validate_plan_input(sample_plan: Path, data_root: Path | None) -> int:
    """Validate the plan and sources locally; return the sample count without creating a run."""

    plan = SamplePlan.model_validate(parse_json_value(sample_plan.read_text(encoding="utf-8")))
    return len(load_selected_dialogues(plan, data_root or sample_plan.parent))


def _file_version(path: Path) -> tuple[str, int, int, int, int, int]:
    """Identify a local file revision without reading its contents on each rerun."""

    path = path.resolve()
    stat = path.stat()
    return (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _check_file_versions(versions: tuple[tuple, ...]) -> None:
    """Do not cache a read if its files changed while it was in progress."""

    for version in versions:
        if _file_version(Path(version[0])) != version:
            raise OSError(f"{Path(version[0]).name} changed while being read; refresh to retry.")


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_plan_input(sample_plan: Path, data_root: Path | None, versions: tuple[tuple, ...]) -> int:
    count = validate_plan_input(sample_plan, data_root)
    _check_file_versions(versions)
    return count


def _preview_plan_input(sample_plan: Path, data_root: Path | None) -> int:
    """Cache only the page precheck; run_all still reads and validates fresh inputs."""

    sample_plan = sample_plan.resolve()
    plan_version = _file_version(sample_plan)
    plan = SamplePlan.model_validate(parse_json_value(sample_plan.read_text(encoding="utf-8")))
    root = data_root.resolve() if data_root is not None else sample_plan.parent
    sources = sorted({(root / item.source_file).resolve() for item in plan.items})
    versions = (plan_version, *(_file_version(path) for path in sources))
    return _cached_plan_input(sample_plan, root, versions)


def list_sample_plans(sample_dir: Path) -> list[Path]:
    """List structurally valid plans without reading or selecting their source data."""

    plans = []
    for path in sorted(sample_dir.glob("*.json")):
        try:
            SamplePlan.model_validate(parse_json_value(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            continue
        plans.append(path)
    return plans


def _run_id_part(value: str) -> str:
    """Convert a model or sample name into a short, stable run ID component."""

    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def build_run_id(
    model_id: str,
    sample_plan: Path,
    run_root: Path,
    reasoning_effort: str | None = None,
) -> str:
    """Build a date/model/thinking/sample run name, adding a suffix on collision."""

    parts = [datetime.now().strftime("%Y%m%d"), _run_id_part(model_id)]
    if reasoning_effort and reasoning_effort != "none":
        parts.extend(["thinking", _run_id_part(reasoning_effort)])
    parts.append(_run_id_part(sample_plan.stem))
    base = "_".join(parts)
    if not (run_root / base).exists():
        return base
    revision = 2
    while (run_root / f"{base}_r{revision:02d}").exists():
        revision += 1
    return f"{base}_r{revision:02d}"


def read_jsonl(path: Path, *, warnings: list[str] | None = None) -> list[dict[str, Any]]:
    """Read saved objects, reporting unreadable lines without rewriting the file."""

    if not path.exists():
        return []
    records = []
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
    except (OSError, UnicodeError) as error:
        if warnings is None:
            raise
        warnings.append(f"{path.name}: {error}")
        return []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = parse_json_value(line)
            if not isinstance(record, dict):
                raise ValueError("expected a JSON object")
            if path.name in {"task_identification.jsonl", "activity_identification.jsonl",
                             "turn_annotations.jsonl", "summary_annotations.jsonl", "core_evidence.jsonl"}:
                record = _saved_activity_terms(record)
            _validate_view_record(record, path.name.replace("task_identification", "activity_identification"))
            records.append(record)
        except ValueError as error:
            if warnings is None:
                raise
            warnings.append(f"{path.name}:{number}: {error}")
    return records


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_saved_output(path: Path, version: tuple) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    records = read_jsonl(path, warnings=warnings)
    _check_file_versions((version,))
    return records, warnings


def _preview_saved_output(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    """Reuse parsed results and their warnings, including on every warm rerun."""

    if not path.exists():
        return []
    try:
        records, saved_warnings = _cached_saved_output(path.resolve(), _file_version(path))
    except OSError as error:
        warnings.append(f"{path.name}: {error}")
        return []
    warnings.extend(saved_warnings)
    return records


def _saved_activity_terms(value: Any, field: str = "") -> Any:
    """Adapt saved structured terms for viewing only; never change quotes or files."""

    fields = {"new_tasks": "new_activities", "task_reference": "activity_reference",
              "task_id": "activity_id", "task_label": "activity_label"}
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            name = fields.get(key, key)
            if name in result:
                raise ValueError("conflicting legacy and current activity fields")
            result[name] = _saved_activity_terms(item, name)
        return result
    if isinstance(value, list):
        return [_saved_activity_terms(item, field) for item in value]
    if isinstance(value, str):
        if field == "activity_id" and re.fullmatch(r"TASK_[0-9]+", value):
            return "ACTIVITY_" + value.removeprefix("TASK_")
        if field in {"dimension", "source_dimension", "rating_label", "core_element", "source_level"}:
            if value == "Client engagement in the work":
                return "Client engagement in the activity"
            names = {"task": "activity", "Task": "Activity", "tasks": "activities"}
            return re.sub(r"\b(?:tasks|Task|task)\b", lambda match: names[match[0]], value)
    return value


def _validate_view_record(record: dict[str, Any], filename: str) -> None:
    """Check display fields only; this is not a repeat of model/evidence validation."""

    def strings(value: Any, fields: tuple[str, ...]) -> None:
        if not isinstance(value, dict) or any(not isinstance(value.get(key), str) for key in fields):
            raise ValueError("missing or invalid display fields")

    def rows(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            raise ValueError("expected a list of saved objects")
        return value

    if filename not in {"turn_annotations.jsonl", "activity_identification.jsonl", "summary_annotations.jsonl",
                        "core_evidence.jsonl", "validation_errors.jsonl"}:
        return
    strings(record, ("sample_id",))
    if filename == "validation_errors.jsonl":
        strings(record, ("message", "error_type"))
        if "retry_scheduled" in record and not isinstance(record["retry_scheduled"], bool):
            raise ValueError("invalid connection retry flag")
        if record.get("retry_scheduled") and (
            type(record.get("attempt")) is not int or record["attempt"] < 1
        ):
            raise ValueError("missing or invalid connection attempt number")
        if "stage" in record:
            strings(record, ("stage",))
        if record.get("current_turn_id") is not None and not isinstance(record["current_turn_id"], str):
            raise ValueError("invalid error turn ID")
        if "affected_cores" in record and (not isinstance(record["affected_cores"], list)
                or any(not isinstance(core, str) for core in record["affected_cores"])):
            raise ValueError("invalid affected cores")
    elif filename == "activity_identification.jsonl":
        strings(record, ("current_turn_id",))
        for activity in rows(record.get("new_activities")):
            strings(activity, ("activity_id", "activity_label"))
    elif filename in {"turn_annotations.jsonl", "summary_annotations.jsonl"}:
        summary = filename == "summary_annotations.jsonl"
        if not summary:
            strings(record, ("current_turn_id",))
        for item in rows(record.get("results" if summary else "findings")):
            strings(item, ("dimension", "reason", "summary_status" if summary else "finding_status"))
            if "rating_label" not in item or (item["rating_label"] is not None and not isinstance(item["rating_label"], str)):
                raise ValueError("invalid saved category")
            for field in (("evidence", "counterevidence") if summary else ("evidence",)):
                for span in rows(item.get(field)):
                    strings(span, ("turn_id", "quote"))
            if summary:
                if "activity_reference" not in item:
                    raise ValueError("missing Activity reference")
                if item["activity_reference"] is not None:
                    strings(item["activity_reference"], ("activity_id", "activity_label"))


def run_statistics(
    manifest: dict[str, Any], turn_records: list[dict[str, Any]],
    summary_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Count failed requests and eligible result categories, not accuracy."""

    counts = manifest.get("counts") or {}
    usage = manifest.get("usage") or {}
    def count(mapping: dict[str, Any], key: str) -> int | None:
        value = mapping.get(key)
        return value if type(value) is int and value >= 0 else None

    requests = count(usage, "requests")
    failed_calls = count(counts, "failed_calls")
    samples = {row["sample_id"]: row for row in manifest.get("samples", [])
               if isinstance(row, dict) and isinstance(row.get("sample_id"), str)}
    excluded_results = False
    categories: Counter[tuple[str, str, str]] = Counter()
    for dimension in (*BOND_DIMENSIONS, *TASK_DIMENSIONS):
        core = "Bond" if dimension in BOND_DIMENSIONS else "Task"
        is_summary = dimension in SUMMARY_LEVEL_DIMENSIONS
        source = summary_records if is_summary else turn_records
        field = "results" if is_summary else "findings"
        allowed = {"rated", "insufficient_evidence", "no_evaluable_opportunity"} if is_summary else {"rated"}
        for record in source:
            sample_id = record.get("sample_id")
            items = record.get(field, [])
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or item.get("dimension") != dimension:
                    continue
                if sample_id not in samples:
                    excluded_results = True
                    continue
                if is_summary:
                    sample = samples[sample_id]
                    by_core = sample.get("summary_status_by_core")
                    annotations = sample.get("annotation_complete_by_core")
                    if (not isinstance(by_core, dict) or not isinstance(annotations, dict)
                            or by_core.get(core) != "complete" or annotations.get(core) is not True):
                        excluded_results = True
                        continue
                status = item.get("summary_status" if is_summary else "finding_status")
                if status not in allowed:
                    excluded_results = True
                    continue
                categories[(dimension, status, str(item.get("rating_label") or "—"))] += 1
    return {
        "requests": requests, "failed_calls": failed_calls,
        "failed_call_rate": failed_calls / requests
            if requests and failed_calls is not None and failed_calls <= requests else None,
        "excluded_results": excluded_results,
        "categories": [{"dimension": dimension, "status": status, "label": label, "results": number}
                       for (dimension, status, label), number in sorted(categories.items())],
    }


def sample_completion_display(counts: dict[str, Any]) -> tuple[str, str]:
    """Use the recorded plan denominator, never the number of processed samples."""

    completed = counts.get("completed_samples")
    planned = counts.get("planned_samples")
    if type(completed) is not int or completed < 0:
        return "Not recorded", "Completion rate unavailable"
    if type(planned) is not int or planned <= 0 or completed > planned:
        return str(completed), "Completion rate unavailable"
    rate = f"{completed / planned:.1%}".replace(".0%", "%")
    return f"{completed} / {planned}", f"{rate} of planned samples · execution completeness"


def render_run_statistics(
    statistics: dict[str, Any], *, preset: bool = False,
    sample_statistics: dict[str, Any] | None = None, sample_label: str = "",
) -> None:
    """Keep execution counts run-wide; optionally filter result categories by sample."""

    st.caption("Run-level execution · All samples in this run")
    rate = statistics["failed_call_rate"]
    value = f"{statistics['failed_calls']} / {statistics['requests']} ({rate:.1%})" if rate is not None else "—"
    st.metric(
        "Failed requests", "—" if preset else value,
        help=("Not applicable to the preset demo: 0 model requests." if preset else
              "Failed requests / all model requests, including call and output validation failures. "
              "Each attempt counts once, including connection failures recovered by retry. "
              "This is not the judgement error rate."),
    )
    st.subheader("Result categories")
    scope_label = "All samples in this run"
    if sample_statistics is not None:
        scope = st.radio(
            "Result scope", ("All samples in run", "Current sample"), horizontal=True,
            key="statistics_scope",
            help="Changes only the result category counts below. Execution metrics remain run-wide.",
            on_change=lambda: st.session_state.update(_statistics_open=True),
        )
        if scope == "Current sample":
            statistics = sample_statistics
            scope_label = f"Current sample · {sample_label}"
    if statistics["excluded_results"]:
        st.warning("Incomplete or unverified results are excluded from these counts. See the sample's Validation details.")
    dimensions = (*BOND_DIMENSIONS, *TASK_DIMENSIONS)
    available = {row["dimension"] for row in statistics["categories"]}
    selected_dimension = st.session_state.get(
        "statistics_dimension", next((name for name in dimensions if name in available), dimensions[0]),
    )
    dimension = st.selectbox(
        "Dimension", dimensions,
        index=dimensions.index(selected_dimension),
        format_func=display_dimension,
        key="statistics_dimension",
        help="Counts cover the selected result scope. They are not accuracy or scores.",
        on_change=lambda: st.session_state.update(_statistics_open=True),
    )
    level = ("Sample-level summary" if dimension in BOND_DIMENSIONS else "Activity-level summary"
             ) if dimension in SUMMARY_LEVEL_DIMENSIONS else "Turn-level results"
    st.caption(f"{level} · {scope_label}")
    status_labels = {
        "insufficient_evidence": "Insufficient evidence",
        "no_evaluable_opportunity": "No evaluable opportunity",
    }
    rows = [
        {"category": status_labels.get(row["status"], row["label"]), "results": row["results"]}
        for row in statistics["categories"] if row["dimension"] == dimension
    ]
    if not rows:
        st.info("No eligible results for this dimension in the selected scope. This does not establish absence or a low score.")
        return
    max_count = max(row["results"] for row in rows)
    tick_step = max(1, (max_count + 9) // 10)
    st.vega_lite_chart(rows, {
        "height": max(150, len(rows) * 36 + 70),
        "padding": {"left": 5, "top": 5, "bottom": 5,
                    "right": max(36, len(str(max_count)) * 8 + 12)},
        "encoding": {
            "y": {"field": "category", "type": "nominal", "sort": None,
                  "axis": {"title": None, "labelLimit": 360, "labelOverlap": False,
                           "labelFlush": False, "labelBaseline": "middle"}},
            "x": {"field": "results", "type": "quantitative",
                  "axis": {"title": "Result count", "format": "d",
                           "values": list(range(0, max_count + 1, tick_step))}},
            "tooltip": [{"field": "category", "type": "nominal", "title": "Category"},
                        {"field": "results", "type": "quantitative", "title": "Results"}],
        },
        "layer": [
            {"mark": {"type": "bar", "size": 24}},
            {"mark": {"type": "text", "align": "left", "baseline": "middle", "dx": 5},
             "encoding": {"text": {"field": "results", "type": "quantitative", "format": "d"}}},
        ],
    }, use_container_width=True)


def list_run_dirs(run_root: Path) -> list[Path]:
    """List runs by manifest modification time, newest first."""

    if not run_root.is_dir():
        return []
    return sorted(
        [path for path in run_root.iterdir() if (path / "manifest.json").exists()],
        key=lambda path: (path / "manifest.json").stat().st_mtime,
        reverse=True,
    )


def load_dialogue(
    source_file: str, dialogue_id: str, data_root: Path | None = None,
    *, expected_sha256: str | None = None,
) -> dict[str, Any] | None:
    """Read source bytes once and verify the recorded checksum before parsing or display."""

    path = Path(source_file)
    if not path.is_absolute():
        if data_root is None:
            raise ValueError("Relative source path requires an explicit data root or absolute recorded sample plan.")
        path = data_root / path
    content = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(
            "Source checksum mismatch: the file differs from the bytes recorded for this run; "
            "changed dialogue text is not displayed"
        )
    lines = content.decode("utf-8-sig").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for line in lines:
        record = parse_json_value(line)
        if record["dialogue_id"] == dialogue_id:
            return validate_dialogue(record)
    return None


def select_saved_dialogue(dialogue: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    """Apply exact recorded turn selection without inferring missing metadata."""

    if "selected_turn_ids" not in sample:
        return dialogue
    selected_ids = sample["selected_turn_ids"]
    if (not isinstance(selected_ids, list) or not selected_ids
            or any(not isinstance(turn_id, str) or not turn_id for turn_id in selected_ids)
            or len(set(selected_ids)) != len(selected_ids)
            or len(selected_ids) != sample.get("turn_count")):
        raise ValueError("Recorded turn selection is invalid or differs from the sample turn count")
    selected_set = set(selected_ids)
    positions = [index for index, turn in enumerate(dialogue["turns"])
                 if turn["turn_id"] in selected_set]
    turns = [dialogue["turns"][index] for index in positions]
    if ([turn["turn_id"] for turn in turns] != selected_ids
            or positions != list(range(positions[0], positions[0] + len(positions)))):
        raise ValueError("Recorded turn selection does not match a contiguous source excerpt")
    return {**dialogue, "turns": turns}


@st.cache_data(show_spinner=False, max_entries=64)
def _cached_dialogue(
    path: Path, dialogue_id: str, expected_sha256: str | None, version: tuple,
) -> dict[str, Any] | None:
    dialogue = load_dialogue(str(path), dialogue_id, expected_sha256=expected_sha256)
    _check_file_versions((version,))
    return dialogue


def _preview_dialogue(
    source_file: str, dialogue_id: str, data_root: Path | None = None,
    *, expected_sha256: str | None = None,
) -> dict[str, Any] | None:
    """Keep only selected dialogues in memory, never whole source datasets."""

    path = Path(source_file)
    if not path.is_absolute():
        if data_root is None:
            raise ValueError("Relative source path requires an explicit data root or absolute recorded sample plan.")
        path = data_root / path
    path = path.resolve()
    return _cached_dialogue(path, dialogue_id, expected_sha256, _file_version(path))


def recorded_data_root(manifest: dict[str, Any], explicit_root: Path | None) -> Path | None:
    """Use an explicit root or recorded absolute path without inferring a previous workspace."""

    if explicit_root is not None:
        return explicit_root
    recorded_root = manifest.get("data_root")
    if recorded_root and Path(recorded_root).is_absolute():
        return Path(recorded_root)
    recorded_plan = manifest.get("sample_plan")
    if recorded_plan and Path(recorded_plan).is_absolute():
        return Path(recorded_plan).parent
    return None


def evidence_by_turn(
    turn_records: list[dict[str, Any]],
    summary_records: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Collect exact turn and summary quotes for source-text highlighting."""

    collected: dict[str, list[str]] = {}
    for record in turn_records:
        for finding in record["findings"]:
            for span in finding["evidence"]:
                collected.setdefault(span["turn_id"], []).append(span["quote"])
    for record in summary_records:
        for result in record["results"]:
            for field in ("evidence", "counterevidence"):
                for span in result[field]:
                    collected.setdefault(span["turn_id"], []).append(span["quote"])
    return collected


def highlight_text(text: str, quotes: list[str]) -> str:
    """Escape source HTML before highlighting exact evidence validated by the pipeline."""

    rendered = html.escape(text)
    for quote in sorted(set(quotes), key=len, reverse=True):
        escaped = html.escape(quote)
        rendered = rendered.replace(escaped, f"<mark>{escaped}</mark>")
    return rendered


def turn_references_by_turn(
    records: list[dict[str, Any]],
) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    """Index findings that cite another turn, without duplicating same-turn results."""

    references: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for record in records:
        current_id = record["current_turn_id"]
        for finding in record["findings"]:
            for cited_id in dict.fromkeys(span["turn_id"] for span in finding["evidence"]):
                if cited_id != current_id:
                    references.setdefault(cited_id, []).append((current_id, finding))
    return references


def flatten_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten nested summary results for display only."""

    rows: list[dict[str, Any]] = []
    for record in records:
        for result in record["results"]:
            reference = result["activity_reference"] or {}
            rows.append(
                {
                    "dimension": result["dimension"],
                    "activity": reference.get("activity_label") or "—",
                    "result": display_result(result["summary_status"], result["rating_label"]),
                }
            )
    return rows


def display_dimension(dimension: str) -> str:
    """Identify a known dimension's Core without changing its saved name."""

    if dimension in BOND_DIMENSIONS:
        return f"Bond · {dimension}"
    if dimension in TASK_DIMENSIONS:
        return f"Task · {dimension}"
    return dimension


def display_result(status: str, label: str | None, *, badge: bool = False) -> str:
    """Keep status badges in details and a compact plain-text table result."""

    if badge:
        colour = {"rated": "green", "evidence_only": "orange", "insufficient_evidence": "red",
                  "no_evaluable_opportunity": "gray"}.get(status)
        text = status.replace("_", " ")
        status_badge = f":{colour}-badge[{text}]" if colour else (f"`{text}`" if text else "")
        return " · ".join(part for part in (status_badge, label) if part)
    return label or status.replace("_", " ")


def display_reason(reason: str, preset: bool) -> str:
    """Remove only the demo's presentation prefix, without rewriting saved text."""

    return reason.removeprefix("Preset illustration: ") if preset else reason


def records_by_turn(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index Activity or turn records by current_turn_id for inline inspection."""

    return {record["current_turn_id"]: record for record in records}


def _render_evidence_line(turn_id: str, quote: str) -> None:
    """Render an escaped quotation as a bullet with a neutral-coloured turn ID."""

    escaped_quote = html.escape(quote).replace("\n", "&#10;")
    st.markdown(
        '<ul><li>'
        '<code class="evidence-turn-id" style="color:#506784;">'
        f'{html.escape(turn_id)}</code> — '
        f'<span class="evidence-quote">{escaped_quote}</span></li></ul>',
        unsafe_allow_html=True,
    )


def _render_evidence(spans: list[dict[str, str]], heading: str) -> None:
    """Display exact turn evidence in finding or summary details."""

    if not spans:
        return
    st.markdown(f"**{heading}**")
    for span in spans:
        _render_evidence_line(span["turn_id"], span["quote"])


def render_turn_block(
    turn: dict[str, Any],
    activity_record: dict[str, Any] | None,
    turn_record: dict[str, Any] | None,
    highlights: list[str] | None = None,
    summary_references: list[dict[str, Any]] | None = None,
    prefix: str = "",
    errors: list[dict[str, Any]] | None = None,
    turn_references: list[tuple[str, dict[str, Any]]] | None = None,
    preset: bool = False,
    validation_scope: str = "",
    selection_known: bool = True,
) -> None:
    """Display a turn, its findings and other turn or summary results citing it."""

    errors = [error for error in errors or [] if not error.get("retry_scheduled")]
    colour = "#eef4ff" if turn["role"] == "counsellor" else "#f6f7f9"
    rendered_text = highlight_text(turn["text"], highlights or [])
    st.markdown(
        f"<div style='background:{colour};padding:0.75rem;"
        f"border-radius:0.5rem;margin-bottom:0.35rem'>"
        f"<b>{html.escape(turn['role'])} · {html.escape(turn['turn_id'])}</b>"
        f"<br>{rendered_text}</div>",
        unsafe_allow_html=True,
    )

    findings = turn_record["findings"] if turn_record else []
    activities = activity_record["new_activities"] if activity_record else []
    finding_count = f"{len(findings)} finding{'s' if len(findings) != 1 else ''}"
    if errors:
        label = (f"Annotations · {finding_count} retained" if turn_record is not None
                 else "Annotations unavailable")
        cores = list(dict.fromkeys(core for error in errors for core in error.get("affected_cores", [])))
        affected = "/".join(cores) + " annotation" if cores else "Result"
        if all(error.get("stage") == "activity_identification" for error in errors):
            affected = "Activity identification"
        anchor = validation_anchor(validation_scope, turn["turn_id"])
        st.button(
            f"{affected} unavailable · View reason", key=f"{prefix}view-{anchor}",
            on_click=lambda: st.session_state.update(_validation_focus=anchor),
        )
    elif turn_record is not None:
        label = f"Annotations · {finding_count}"
    elif activity_record is not None:
        label = "Activity checked · turn annotation unavailable"
    else:
        label = "Not processed" if selection_known else "No saved processing record"

    with st.expander(f"{prefix}{turn['turn_id']} · {label}"):
        if activity_record is None:
            st.caption("No validated Activity identification")
        elif activities:
            st.markdown("**Activity identification**")
            for activity in activities:
                st.markdown(f"- `{activity['activity_id']}` — {activity['activity_label']}")

        if turn_record is None:
            st.caption("No validated turn annotation")
        elif findings:
            st.markdown("**Turn findings**")
            for finding in findings:
                label_text = display_result(finding["finding_status"], finding["rating_label"], badge=True)
                st.markdown(
                    f"**{display_dimension(finding['dimension'])}** · {label_text}"
                )
                reference = finding.get("activity_reference")
                if reference:
                    st.caption(f"Activity: {reference['activity_id']} — {reference['activity_label']}")
                st.caption(display_reason(finding["reason"], preset))
                _render_evidence(finding["evidence"], "Evidence")
        elif not (activities or turn_references or summary_references or errors):
            st.caption("No findings")

        if turn_references or summary_references:
            st.caption("Cited by")
            for source_turn, finding in turn_references or []:
                st.caption(f"{source_turn} · {display_dimension(finding['dimension'])}")
            for result in summary_references or []:
                st.caption(f"Summary · {display_dimension(result['dimension'])}")


def render_summary_details(
    records: list[dict[str, Any]], *, preset: bool = False,
    sample: dict[str, Any] | None = None,
) -> None:
    """Display a summary overview with expandable reasons, evidence and counterevidence."""

    st.caption("Three Bond summary dimensions and engagement in each Activity; not an overview of all 14 dimensions.")
    rows = flatten_summary(records)
    if not rows:
        st.info("No summary")
        return
    if sample is not None:
        annotations = sample.get("annotation_complete_by_core")
        statuses = sample.get("summary_status_by_core")
        cores = {
            "Bond" if row["dimension"] in BOND_DIMENSIONS else "Task"
            for row in rows
        }
        affected = [
            core for core in sorted(cores)
            if not (isinstance(annotations, dict) and isinstance(statuses, dict)
                    and annotations.get(core) is True and statuses.get(core) == "complete")
        ]
        if affected:
            st.warning(
                f"{' / '.join(affected)} summary is incomplete or unconfirmed: "
                "shown for inspection, not as a final result."
            )
    st.dataframe(
        [{**row, "dimension": display_dimension(row["dimension"])} for row in rows],
        use_container_width=True, hide_index=True,
    )
    for record in records:
        for result in record["results"]:
            label = display_result(result["summary_status"], result["rating_label"], badge=True)
            with st.expander(f"{display_dimension(result['dimension'])} · {label}"):
                reference = result["activity_reference"]
                if reference:
                    st.markdown(
                        f"**Activity:** `{reference['activity_id']}` — "
                        f"{reference['activity_label']}"
                    )
                st.markdown(f"**Reason:** {display_reason(result['reason'], preset)}")
                _render_evidence(result["evidence"], "Evidence")
                _render_evidence(result["counterevidence"], "Counterevidence")


def render_core_mapping(records: list[dict[str, Any]]) -> None:
    """Present saved Core relationships and their source evidence, without rescoring."""

    records = [record for record in records if record.get("status") != "evidence_only"]
    if not records:
        st.info("No Core mapping")
        return
    st.caption("Predefined conceptual mappings organise dimension results under Bond / Task without additional scoring.")
    cores = dict.fromkeys(record.get("core", "Unspecified Core") for record in records)
    for core in sorted(cores, key=lambda core: {"Bond": 0, "Task": 1}.get(core, 2)):
        st.subheader(core)
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for record in records:
            if record.get("core", "Unspecified Core") != core:
                continue
            relationship = (
                record.get("source_dimension") or "Unspecified dimension",
                record.get("core_element") or core,
            )
            groups.setdefault(relationship, []).append(record)
        for (dimension, element), sources in groups.items():
            with st.expander(f"{dimension} → {element}"):
                if any(record.get("diagnostic_only") for record in sources):
                    st.caption("Diagnostic mapping — this Core's results are incomplete.")
                for record in sources:
                    source = record.get("current_turn_id") or record.get("source_level") or "Source unavailable"
                    result = display_result(record.get("status", ""), record.get("rating_label"), badge=True)
                    st.markdown(f"**{source}**" + (f" · {result}" if result else ""))
                    reference = record.get("activity_reference")
                    if reference:
                        st.caption(f"Activity: {reference['activity_id']} — {reference['activity_label']}")
                    _render_evidence(record.get("evidence", []), "Evidence")
                    _render_evidence(record.get("counterevidence", []), "Counterevidence")


def render_sample_validation_status(sample: dict[str, Any]) -> bool:
    """Show completeness warnings only for affected cores; do not infer missing legacy status."""

    by_core = sample.get("summary_status_by_core")
    if by_core is None:
        if sample.get("summary_status") in ("diagnostic", "partial", "failed"):
            st.warning("Sample results incomplete. See Validation details.")
        else:
            st.warning("Sample completeness cannot be confirmed. See Validation details.")
        return True
    if not isinstance(by_core, dict):
        st.warning("Sample completeness cannot be confirmed. See Validation details.")
        return True
    annotation = sample.get("annotation_complete_by_core")
    annotation = annotation if isinstance(annotation, dict) else {}
    issues = []
    for core in ("Bond", "Task"):
        complete = annotation.get(core)
        if complete is not True:
            issues.append(f"{core} annotations {'incomplete' if complete is False else 'unconfirmed'}")
        status = by_core.get(core)
        if status != "complete" and not (status == "diagnostic" and complete is False):
            label = ({"failed": "failed", "not_run": "not run", "diagnostic": "diagnostic only"}
                     .get(status, "unconfirmed") if isinstance(status, str) else "unconfirmed")
            issues.append(f"{core} summary {label}")
    if not issues and sample.get("summary_status") in ("diagnostic", "partial", "failed"):
        issues.append("Sample results incomplete")
    if issues:
        st.warning("; ".join(issues) + ". See Validation details.")
    return bool(issues)


def validation_anchor(scope: str, location: str) -> str:
    """Use a safe, sample-scoped DOM identifier, never raw saved text."""

    return "validation-" + hashlib.sha256(f"{scope}\0{location}".encode()).hexdigest()[:20]


def _quoted_error_text(value: str) -> str | None:
    """Read the quoted literal in a known saved error, without evaluating code."""

    try:
        result = ast.literal_eval(value)
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return None
    return result if isinstance(result, str) else None


def validation_issue(error: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
    """Explain recognised validation failures without rewriting saved records."""

    message = detail.get("message", "")
    message = re.sub(r"^(?:Bond|Task):\s*", "", message)
    issue: dict[str, Any] = {
        "title": "Model response could not be accepted",
        "reason": "The response did not meet the required output structure or evidence rules. "
                  "The saved record does not support a more specific explanation here.",
    }
    mismatch = re.fullmatch(r"Evidence is not an exact quote from (.+?): (.+)", message, re.DOTALL)
    summary = re.fullmatch(
        r"Summary evidence must come from a validated turn span for the same dimension and Activity: "
        r"(\([^\n]*?\)), ([^,\s]+), (.+)", message, re.DOTALL,
    )
    if mismatch:
        issue.update(title="Quoted evidence does not match the cited turn",
                     reason="The quoted words cannot be found exactly in the turn the model cited.",
                     turn_id=mismatch[1], quote=_quoted_error_text(mismatch[2]))
    elif summary:
        try:
            target = ast.literal_eval(summary[1])
        except (SyntaxError, ValueError, TypeError, RecursionError):
            target = None
        issue.update(title="Summary quote is not linked to accepted evidence",
                     reason="The summary used a quote that was not accepted for this same dimension "
                            "and activity in the earlier annotations.",
                     turn_id=summary[2], quote=_quoted_error_text(summary[3]), summary_target=target)
    elif message == "Each turn finding must cite an exact quote from the current turn":
        issue.update(title="Evidence from the current turn is missing",
                     reason="The model returned an annotation without citing the turn being assessed.",
                     current_evidence_missing=True)
    elif "Activity label does not match the registry:" in message:
        activity_id = message.rsplit(": ", 1)[-1]
        issue.update(title="Activity name does not match the registered activity",
                     reason="The model used an existing activity ID with a different name.",
                     activity_id=activity_id)
    elif message.startswith("Evidence refers to an unavailable turn:"):
        issue.update(title="Evidence refers to an unavailable turn",
                     reason="The model cited a turn that was not available at this step.")
    elif any(text in message for text in ("unregistered Activity", "unknown Activity")):
        issue.update(title="Annotation refers to an unknown activity",
                     reason="The model referred to an activity that had not been registered.")
    elif any(text in message for text in (
        "Summary includes an unrequested target", "Incomplete summary coverage",
    )):
        issue.update(title="Summary does not cover the requested targets",
                     reason="The model omitted a requested summary target or returned an unrequested one.")
    elif error.get("error_type") in {"ReadError", "ConnectError", "ReadTimeout", "ConnectTimeout", "RemoteProtocolError"}:
        issue.update(title="Model response was not received",
                     reason="The connection was interrupted or timed out before a complete response arrived.")
    if error.get("retry_scheduled"):
        issue.update(title="Connection attempt failed; retry scheduled",
                     reason=f"Connection attempt {error['attempt']} failed. A later attempt was scheduled; "
                            "this entry alone does not mean the final result failed.")
    return issue


def _validation_comparison(
    issue: dict[str, Any], error: dict[str, Any], core: str,
    dialogue: dict[str, Any] | None, activity_records: list[dict[str, Any]],
    turn_records: list[dict[str, Any]],
) -> None:
    """Show only the evidence needed for this failure, never the full response."""

    turns = {turn["turn_id"]: turn["text"] for turn in (dialogue or {}).get("turns", [])}
    quotes = []
    source_turn = issue.get("turn_id")
    if issue.get("quote") is not None:
        quotes = [{"turn_id": source_turn, "quote": issue["quote"]}]
    if issue.get("current_evidence_missing") or issue.get("activity_id"):
        if issue.get("current_evidence_missing"):
            source_turn = error.get("current_turn_id")
        try:
            response = parse_json_value(error.get("raw_response") or "")
        except (ValueError, TypeError):
            response = {}
        findings = response.get("findings", []) if isinstance(response, dict) else []
        dimensions = BOND_DIMENSIONS if core == "Bond" else TASK_DIMENSIONS if core == "Task" else ()
        for finding in findings if isinstance(findings, list) else []:
            if not isinstance(finding, dict) or finding.get("dimension") not in dimensions:
                continue
            spans = finding.get("evidence", [])
            if issue.get("current_evidence_missing") and isinstance(spans, list):
                if not any(isinstance(span, dict) and span.get("turn_id") == error.get("current_turn_id") for span in spans):
                    quotes = [span for span in spans if isinstance(span, dict)
                              and isinstance(span.get("quote"), str) and isinstance(span.get("turn_id"), str)]
                    source_turn = error.get("current_turn_id")
                    break
            reference = finding.get("activity_reference", finding.get("task_reference"))
            if issue.get("activity_id") and isinstance(reference, dict):
                activity_id = reference.get("activity_id", reference.get("task_id"))
                if activity_id == issue["activity_id"]:
                    st.caption(f"Activity: {activity_id}")
                    st.text("Model name: " + str(reference.get("activity_label", reference.get("task_label", "Not saved"))))
                    registered = next((activity["activity_label"] for record in activity_records
                                       for activity in record["new_activities"] if activity["activity_id"] == activity_id), None)
                    st.text("Registered name: " + (registered or "Unavailable in the saved results"))
                    return
    if quotes or issue.get("current_evidence_missing"):
        st.markdown("**Model evidence**")
        if not quotes:
            st.caption("No readable quote was saved for this issue.")
        for span in quotes:
            _render_evidence_line(span["turn_id"], span["quote"])
        target = issue.get("summary_target")
        if isinstance(target, tuple) and len(target) == 2:
            st.markdown("**Accepted evidence for this summary target at the cited turn**")
            accepted = list(dict.fromkeys(
                span["quote"] for record in turn_records for finding in record["findings"]
                if finding["dimension"] == target[0]
                and (finding.get("activity_reference") or {}).get("activity_id") == target[1]
                for span in finding["evidence"] if span["turn_id"] == source_turn
            ))
            st.caption(str(target[0]) + (f" · {target[1]}" if target[1] else ""))
            for quote in accepted:
                _render_evidence_line(source_turn, quote)
            if not accepted:
                st.caption("No accepted quote for this target at the cited turn.")
        else:
            st.markdown("**Original turn**")
            if source_turn in turns:
                _render_evidence_line(source_turn, turns[source_turn])
            else:
                st.caption("Original text unavailable: its source could not be verified or the turn is missing.")


def render_validation_details(
    sample: dict[str, Any], errors: list[dict[str, Any]], *,
    dialogue: dict[str, Any] | None = None,
    activity_records: list[dict[str, Any]] | None = None,
    turn_records: list[dict[str, Any]] | None = None,
    validation_scope: str = "",
) -> str | None:
    """Explain location, cause and evidence; raw records remain on disk for review."""

    incomplete = render_sample_validation_status(sample)
    if not (incomplete or errors):
        return
    if errors and not incomplete:
        if any(not error.get("retry_scheduled") for error in errors):
            st.warning("This sample has recorded errors. See Validation details.")
        else:
            st.caption("Connection retries were recorded; final results are complete. See Validation details.")
    anchors = {validation_anchor(validation_scope, error.get("current_turn_id") or sample["sample_id"]) for error in errors}
    focus = st.session_state.pop("_validation_focus", None)
    focus = focus if focus in anchors else None
    with st.expander("Validation details", expanded=focus is not None):
        if not errors:
            st.caption("No error detail was saved for this sample; completeness is determined from the saved status.")
        rendered_anchors = set()
        first_issue = True
        for error in errors:
            location = error.get("current_turn_id") or sample["sample_id"]
            anchor = validation_anchor(validation_scope, location)
            stage = {"turn_annotation": "Turn annotation", "summary_annotation": "Summary",
                     "activity_identification": "Activity identification"}.get(error.get("stage"), "Result")
            saved_details = error.get("group_errors")
            details = ([detail for detail in saved_details
                        if isinstance(detail, dict) and isinstance(detail.get("message"), str)]
                       if isinstance(saved_details, list) else []) or [error]
            for detail in details:
                if not first_issue:
                    st.divider()
                first_issue = False
                if anchor not in rendered_anchors:
                    st.markdown(f'<div id="{anchor}" style="scroll-margin-top:5rem"></div>', unsafe_allow_html=True)
                    rendered_anchors.add(anchor)
                core = detail.get("core") or "/".join(error.get("affected_cores", []))
                issue = validation_issue(error, detail)
                st.markdown(f"**{core + ' · ' if core else ''}{issue['title']}**")
                st.caption(f"{location} · {stage}")
                st.write(issue["reason"])
                _validation_comparison(issue, error, core, dialogue, activity_records or [], turn_records or [])
    return focus


def _queue_run(settings: dict[str, Any]) -> None:
    """Capture a one-use Run All configuration; keep credentials in the current session only."""

    st.session_state["_pending_run"] = settings


def _render_run_setup(runtime_ready: bool = True) -> tuple[Path | None, Path | None]:
    """Display idle settings and submit a snapshot for the next app rerun."""

    with st.container():
        st.header("Run setup")
        st.caption("No model configured? Run the fictional, preset demonstration locally.")
        if st.button("Run preset demo", disabled=not runtime_ready):
            from alliance_eval.demo import run_demo

            try:
                demo_run = run_demo(WORKSPACE_ROOT)
            except (OSError, ValueError, RuntimeError) as error:
                st.error(f"Demo could not run — {error}")
            else:
                st.success(f"Preset demonstration saved: {demo_run.name}. No model was called.")
        plans = list_sample_plans(WORKSPACE_ROOT / "data" / "plans")
        sample_plan = st.selectbox(
            "Sample plan", plans, format_func=lambda path: path.stem,
            disabled=not plans, placeholder="No prepared plans",
        )
        run_root = WORKSPACE_ROOT / "runs"
        data_root = None
        plan_valid = False
        try:
            if run_root.exists() and not run_root.is_dir():
                raise ValueError("Run directory points to a file.")
            if sample_plan is None:
                st.info("No valid sample plan found in data/plans/. The preset demo does not require one.")
            else:
                _preview_plan_input(sample_plan, data_root)
                plan_valid = True
        except (OSError, ValueError, KeyError, TypeError) as error:
            st.warning(f"Input not ready — {error}")
        connections_valid = True
        try:
            connections = load_custom_providers(WORKSPACE_ROOT)
        except (OSError, ValueError) as error:
            connections = {}
            connections_valid = False
            st.error(f"Provider configuration not ready — {error}")
        provider = st.selectbox("Provider", [*BUILTIN_PROVIDERS, *connections])
        connection = connections.get(provider)
        dialect = connection["dialect"] if connection is not None else provider
        # A changed connection must not inherit credentials or edits for the previous endpoint.
        widget_key = provider if connection is None else "custom_" + hashlib.sha256(
            json.dumps([provider, connection], sort_keys=True).encode("utf-8")
        ).hexdigest()
        api_key = ""
        response_format = "json_schema"
        parameters = dict(connection["parameters"]) if connection is not None else {}
        key_required = dialect in {"deepseek", "openai", "mistral"}
        if connection is not None:
            base_url = connection["base_url"]
            default_model = connection["model_id"]
            response_format = connection["response_format"]
            key_required = bool(connection["api_key_env"])
            st.text_input("Base URL", base_url, disabled=True, key=f"base_url_{widget_key}")
            api_key = st.text_input(
                "API key" if key_required else "API key (optional)", type="password",
                value=os.environ.get(connection["api_key_env"], "") if key_required else "",
                key=f"api_key_{widget_key}",
            )
            st.caption(f"Interface: {dialect} · {response_format} · Parameters from providers.json")
        elif dialect in {"deepseek", "openai", "mistral"}:
            base_url, default_model, key_env, key_label = {
                "deepseek": ("https://api.deepseek.com", "deepseek-flash", "DEEPSEEK_API_KEY", "DeepSeek API key"),
                "openai": ("https://api.openai.com/v1", "gpt-5.6-luna", "OPENAI_API_KEY", "OpenAI API key"),
                "mistral": (MISTRAL_BASE_URL, MISTRAL_MODEL_ID, "MISTRAL_API_KEY", "Mistral API key"),
            }[dialect]
            api_key = st.text_input(
                key_label, type="password", value=os.environ.get(key_env, ""),
                key=f"api_key_{widget_key}",
            )
            st.caption("Sent only when a run is started. Key is not saved to run files.")
        else:
            base_url = st.text_input("Base URL", "http://127.0.0.1:11434", key=f"base_url_{widget_key}")
            default_model = ""
            try:
                validate_model_url(base_url)
            except ValueError as error:
                connections_valid = False
                st.error(f"Model configuration not ready — {error}")
        model_id = st.text_input("Model ID", default_model, key=f"model_id_{widget_key}").strip()
        st.caption("Enter a model ID supported by your service. Opening this page does not query models.")
        thinking = False
        if dialect in {"deepseek", "openai", "mistral"}:
            thinking = st.toggle("Thinking", value=False, key=f"{dialect}_thinking")
        if dialect == "deepseek":
            parameters["thinking"] = {"type": "enabled" if thinking else "disabled"}
        elif dialect in {"openai", "mistral"}:
            parameters["reasoning_effort"] = "none"
        if thinking:
            # Use provider-specific effort options and widget keys to prevent settings crossing providers.
            efforts = (
                ["low", "medium", "high", "xhigh", "max"] if dialect == "openai"
                else ["high"] if dialect == "mistral" else ["low", "high", "max"]
            )
            parameters["reasoning_effort"] = st.selectbox(
                "Reasoning effort", efforts, index=min(1, len(efforts) - 1), key=f"{dialect}_reasoning_effort",
            )
        elif connection is None:
            temperature = st.number_input(
                "Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1
            )
            parameters["temperature"] = float(temperature)
        if dialect == "ollama" and connection is None:
            num_ctx = st.number_input(
                "Context window (num_ctx)", min_value=1024, value=16384, step=1024
            )
            seed = st.number_input("Seed", min_value=0, value=666, step=1)
            parameters.update(num_ctx=int(num_ctx), seed=int(seed), think=False)
        run_id = (
            # Include only the currently enabled reasoning level in the run name.
            build_run_id(
                model_id, sample_plan, run_root,
                reasoning_effort=parameters.get("reasoning_effort") if thinking else None,
            )
            if model_id and sample_plan is not None and run_root is not None
            else ""
        )
        st.text_input("Run ID", value=run_id, disabled=True)
        st.button(
            "Run All",
            type="primary",
            use_container_width=True,
            disabled=(not runtime_ready or not plan_valid or not connections_valid or not run_id
                      or (key_required and not api_key.strip())),
            on_click=_queue_run,
            args=({"sample_plan": sample_plan, "dialect": dialect,
                   "model_id": model_id, "base_url": base_url,
                   "api_key": api_key, "parameters": parameters, "run_id": run_id,
                   "response_format": response_format,
                   "run_root": run_root, "data_root": data_root},),
        )
        return run_root, data_root


def _run_with_progress(settings: dict[str, Any]) -> None:
    """Run the shared pipeline with settings disabled, retaining timing, progress and native Stop."""

    # Revalidate after consuming the snapshot to catch input changes before model calls.
    if not _render_runtime_status():
        return

    sample_plan = settings["sample_plan"]
    dialect = settings["dialect"]
    model_id = settings["model_id"]
    base_url = settings["base_url"]
    api_key = settings["api_key"]
    parameters = settings["parameters"]
    run_id = settings["run_id"]
    try:
        if dialect in {"deepseek", "openai"}:
            judge_class = DeepSeekJudge if dialect == "deepseek" else OpenAIJudge
            backend = judge_class(model_id=model_id, api_key=api_key, parameters=parameters)
        else:
            backend = LocalHTTPJudge(
                model_id=model_id, base_url=base_url,
                dialect=dialect, parameters=parameters, api_key=api_key or None,
                response_format=settings.get("response_format", "json_schema"),
            )
    except (ValueError, httpx.InvalidURL) as error:
        st.error(f"Model configuration not ready — {error}")
        return
    progress = st.progress(0)
    status = st.empty()
    timer = st.empty()

    with st.container():
        started_at = time.monotonic()
        with timer.container():
            components.html(running_timer_html(), height=45)
        def update_progress(event: dict[str, Any]) -> None:
            """Convert pipeline progress events into app progress and status."""

            if event["stage"] == "turn_annotation":
                value = event["turn_position"] / event["turn_total"]
                progress.progress(value)
                status.write(
                    f"{event['sample_id']} · {event['current_turn_id']} · "
                    f"{event['turn_position']}/{event['turn_total']}"
                )
            elif event["stage"] == "summary_annotation":
                status.write(f"{event['sample_id']} · summary")
            elif event["stage"] == "complete":
                progress.progress(1.0)
                completed_manifest = json.loads(
                    (Path(event["run_dir"]) / "manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                if completed_manifest["status"] == "completed":
                    status.success(f"Complete: {event['run_dir']}")
                else:
                    status.warning(
                        f"{completed_manifest['status']}: {event['run_dir']}"
                    )

        try:
            run_all(
                sample_plan_path=sample_plan,
                backend=backend,
                run_root=settings["run_root"],
                run_id=run_id,
                data_root=settings["data_root"],
                progress_callback=update_progress,
            )
        finally:
            close = getattr(backend, "close", None)
            if close is not None:
                close()
            # Stop the browser timer after success or an unexpected abort.
            elapsed = format_elapsed(time.monotonic() - started_at)
            timer.empty()
            timer.metric("Elapsed time", elapsed)


def _render_run_explorer(run_root: Path | None, data_root: Path | None = None) -> None:
    """Display the sample and turn evidence explorer while idle or after a run."""

    st.header("Run explorer")
    run_dirs = list_run_dirs(run_root) if run_root is not None else []
    if not run_dirs:
        st.info("No pipeline runs found.")
        return

    selected_run = st.selectbox("Run", run_dirs, format_func=lambda path: path.name)
    try:
        manifest = parse_json_value((selected_run / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be a JSON object")
        for field in ("model", "counts", "usage", "versions"):
            if field in manifest and not isinstance(manifest[field], dict):
                raise ValueError(f"manifest {field} must be an object")
        if not isinstance(manifest.get("samples", []), list):
            raise ValueError("manifest samples must be a list")
        if "status" in manifest and not isinstance(manifest["status"], str):
            raise ValueError("manifest status must be a string")
        if "outputs" in manifest and (not isinstance(manifest["outputs"], list)
                or any(not isinstance(name, str) for name in manifest["outputs"])):
            raise ValueError("manifest outputs must be a list of filenames")
    except (OSError, ValueError, TypeError) as error:
        st.error(f"Saved run cannot be read — {error}. No files were changed.")
        return
    preset = bool(manifest.get("demonstration") or manifest.get("model", {}).get("dialect") == "preset_demo")
    if preset:
        st.warning("Preset demonstration: fictional dialogue and fixed illustrative responses. No model inference.")
    output_warnings: list[str] = []
    output_paths = {
        name: selected_run / f"{name}.jsonl"
        for name in ("turn_annotations", "activity_identification", "summary_annotations", "core_evidence", "validation_errors")
    }
    recorded_outputs = manifest.get("outputs", [])
    if "activity_identification.jsonl" not in recorded_outputs and (
        "task_identification.jsonl" in recorded_outputs
        or (not output_paths["activity_identification"].exists()
            and (selected_run / "task_identification.jsonl").is_file())
    ):
        output_paths["activity_identification"] = selected_run / "task_identification.jsonl"
    outputs = {name: _preview_saved_output(path, output_warnings) for name, path in output_paths.items()}
    for path in output_paths.values():
        if path.name in recorded_outputs and not path.is_file():
            output_warnings.append(f"{path.name}: recorded output file is missing")
    for warning in output_warnings:
        st.warning(f"Saved output excluded from this view — {warning}. Counts may be incomplete; files are unchanged.")
    display_status = {
        "completed": "Complete",
        "completed_with_errors": "Errors",
        "running": "Running",
    }.get(manifest.get("status"), manifest.get("status", "Not recorded"))
    overview = st.columns(4)
    overview[0].metric(
        "Status", display_status,
        help=(
            "**Running** — Started; no end state recorded. May remain after a forced stop; "
            "not a live process check.\n\n"
            "**Complete** — Finished with no unresolved call or validation failures; "
            "connection retries may have occurred. This is "
            "not a guarantee of correct judgements.\n\n"
            "**Errors** — Finished with unresolved call or validation failures; valid results are retained.\n\n"
            "**aborted** — Stopped early after a fatal error.\n\n"
            "Check Validation and Bond/Task completeness when reviewing results."
        ),
    )
    overview[1].metric("Model", "demo" if preset else manifest.get("model", {}).get("model_id", "Not recorded"))
    overview[2].metric("Request", 0 if preset else manifest.get("usage", {}).get("requests", "Not recorded"))
    completion_value, completion_caption = sample_completion_display(manifest.get("counts", {}))
    overview[3].metric(
        "Fully validated samples", completion_value,
        help=(
            "Samples that finished Activity identification, turn annotations and summary "
            "with no unresolved call or validation failures in either Bond or Task. "
            "Recovered connection failures remain in the request statistics and log.\n\n"
            "One sample is one plan entry. The denominator is all planned samples, including "
            "any not yet processed. This is execution completeness, not judgement accuracy. "
            "Missing, zero or inconsistent counts do not produce a percentage."
        ),
    )
    overview[3].caption(completion_caption)
    statistics = run_statistics(manifest, outputs["turn_annotations"], outputs["summary_annotations"])
    sample_options = [sample for sample in manifest.get("samples", [])
                      if isinstance(sample, dict) and all(isinstance(sample.get(key), str)
                      for key in ("sample_id", "dialogue_id", "source_file"))]
    if len(sample_options) != len(manifest.get("samples", [])):
        st.warning("Some saved sample entries lack identifiers or source paths and cannot be displayed.")
    if not sample_options:
        with st.expander("Statistics", expanded=st.session_state.pop("_statistics_open", False)):
            render_run_statistics(statistics, preset=preset)
        st.info("No readable sample records in this run yet.")
        return
    sample = st.selectbox(
        "Sample",
        sample_options,
        format_func=lambda item: f"{item['sample_id']} · {item['dialogue_id']}",
        on_change=lambda: st.session_state.update(_statistics_open=True),
    )
    sample_id = sample["sample_id"]

    turn_records = [
        row
        for row in outputs["turn_annotations"]
        if row.get("sample_id") == sample_id
    ]
    activity_records = [
        row
        for row in outputs["activity_identification"]
        if row.get("sample_id") == sample_id
    ]
    summary_records = [
        row
        for row in outputs["summary_annotations"]
        if row.get("sample_id") == sample_id
    ]
    core_records = [
        row
        for row in outputs["core_evidence"]
        if row.get("sample_id") == sample_id
    ]
    errors = [
        row
        for row in outputs["validation_errors"]
        if row.get("sample_id") == sample_id
    ]

    dialogue = None
    source_problem = None
    source_turn_count = None
    try:
        dialogue = _preview_dialogue(
            sample["source_file"], sample["dialogue_id"],
            recorded_data_root(manifest, data_root),
            expected_sha256=sample.get("source_sha256"),
        )
        if dialogue is None:
            source_problem = "The recorded dialogue ID is missing from its source file. Saved results are unchanged."
        else:
            source_turn_count = len(dialogue["turns"])
            dialogue = select_saved_dialogue(dialogue, sample)
    except (OSError, ValueError, KeyError, TypeError) as error:
        dialogue = None
        source_problem = (
            f"Original dialogue unavailable — {error}. "
            "Saved results remain readable and are not modified."
        )
    validation_scope = f"{selected_run}:{sample_id}"
    selection_known = ("selected_turn_ids" in sample
                       or (source_turn_count is not None and sample.get("turn_count") == source_turn_count))
    validation_focus = render_validation_details(
        sample, errors, dialogue=dialogue if sample.get("source_sha256") else None,
        activity_records=activity_records, turn_records=turn_records,
        validation_scope=validation_scope,
    )
    with st.expander("Statistics", expanded=st.session_state.pop("_statistics_open", False)):
        render_run_statistics(
            statistics, preset=preset,
            sample_statistics=run_statistics(manifest, turn_records, summary_records),
            sample_label=f"{sample_id} · {sample['dialogue_id']}",
        )
    tabs = st.tabs(
        ["Dialogue & annotations", "Summary", "Core mapping"]
    )

    with tabs[0]:
        st.markdown(f"**Turn findings: {sum(len(record['findings']) for record in turn_records)}**")
        if dialogue:
            if "selected_turn_ids" in sample and len(dialogue["turns"]) < source_turn_count:
                turn_count = len(dialogue["turns"])
                st.caption(
                    f"Selected excerpt: {turn_count} turn{'s' if turn_count != 1 else ''} · "
                    f"{dialogue['turns'][0]['turn_id']} to {dialogue['turns'][-1]['turn_id']}"
                )
            elif not selection_known:
                st.caption(
                    "Selection boundary was not recorded in this run. Showing the complete source dialogue; "
                    "a turn without saved results may be outside the selected excerpt or unprocessed."
                )
        if sample.get("source_sha256") is None:
            st.caption(
                "This saved run has no source checksum. Whether the current source bytes "
                "match those used for the run cannot be verified."
            )
        if source_problem:
            st.warning(source_problem)
        highlights = evidence_by_turn(turn_records, summary_records)
        st.caption(
            "Yellow text is an exact quote cited by at least one validated turn or "
            "summary result. It does not mean that the text is positive, correct or "
            "high-scoring."
        )
        activity_by_turn = records_by_turn(activity_records)
        findings_by_turn = records_by_turn(turn_records)
        turn_references = turn_references_by_turn(turn_records)
        summary_by_turn: dict[str, list[dict[str, Any]]] = {}
        for record in summary_records:
            for result in record["results"]:
                cited_turn_ids = {
                    span["turn_id"]
                    for field in ("evidence", "counterevidence")
                    for span in result[field]
                }
                for turn_id in cited_turn_ids:
                    summary_by_turn.setdefault(turn_id, []).append(result)
        if dialogue:
            for turn in dialogue["turns"]:
                render_turn_block(
                    turn,
                    activity_by_turn.get(turn["turn_id"]),
                    findings_by_turn.get(turn["turn_id"]),
                    highlights=highlights.get(turn["turn_id"], []),
                    summary_references=summary_by_turn.get(turn["turn_id"], []),
                    turn_references=turn_references.get(turn["turn_id"], []),
                    errors=[
                        error for error in errors
                        if error.get("current_turn_id") == turn["turn_id"]
                    ],
                    preset=preset,
                    validation_scope=validation_scope,
                    selection_known=selection_known,
                )

    with tabs[1]:
        render_summary_details(summary_records, preset=preset, sample=sample)

    with tabs[2]:
        render_core_mapping(core_records)

    if validation_focus:
        components.html(
            "<script>let tries = 0; const locate = () => {"
            f"const target = parent.document.getElementById({json.dumps(validation_focus)});"
            "if (target) { target.scrollIntoView({block: 'start'}); }"
            "else if (++tries < 60) { requestAnimationFrame(locate); }"
            "}; requestAnimationFrame(locate);</script>", height=0,
        )


def main() -> None:
    """Hide app controls while running and restore them afterwards; retain Streamlit's native Stop."""

    st.set_page_config(page_title="Alliance Evaluation", layout="wide")
    # Use a readable yellow palette for intermediate evidence badges.
    st.markdown(
        """<style>
        span.is-badge[style*="background-color: rgba(255, 227, 18,"] {
            color: #806600 !important;
            background-color: #fff4bf !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )
    st.title("Alliance-oriented Evaluation Pipeline")
    setup_view = st.sidebar.empty()
    lock_view = st.sidebar.empty()
    progress_view = st.empty()
    explorer_view = st.empty()
    # Consume the snapshot first so a rerun after Stop or refresh cannot repeat the request.
    settings = st.session_state.pop("_pending_run", None)
    runtime_ready = _render_runtime_status()
    if settings is not None and runtime_ready:
        # Use separate placeholders for run status to prevent stale widgets from reused Streamlit deltas.
        # Do not reuse the cleared settings and explorer placeholders until model calls finish.
        setup_view.empty()
        explorer_view.empty()
        with lock_view.container():
            st.header("Run in progress")
            st.write(settings["run_id"])
            st.info(
                "Controls are locked while running. Use Stop in the top-right corner "
                "to stop. After stopping, refresh the page to return to setup."
            )
        with progress_view.container():
            _run_with_progress(settings)
        lock_view.empty()
    with setup_view.container():
        run_root, data_root = _render_run_setup(runtime_ready)
    with explorer_view.container():
        _render_run_explorer(run_root, data_root)


if __name__ == "__main__":
    main()
