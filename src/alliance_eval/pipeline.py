#!/usr/bin/env python3
"""Shared evaluation pipeline: Activity identification, turn evidence and ratings, bounded summary and Core mapping.

Store validated results and raw failures separately, with independent Bond/Task completeness.
Core mapping organises existing evidence without rescoring; summaries after partial failure are diagnostic."""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx
from pydantic import BaseModel

from alliance_eval.input_data import (
    LOADED_SOURCE_SHA256 as INPUT_SOURCE_SHA256,
    INPUT_SCHEMA_VERSION,
    dialogue_json_schema,
    parse_json_value,
    validate_dialogue,
)
from alliance_eval.resources import (
    LOADED_SOURCE_SHA256 as RESOURCES_SOURCE_SHA256,
    PACKAGE_DIR,
    code_identity,
    resource_directory,
)

from alliance_eval.evaluate import (
    LOADED_SOURCE_SHA256 as EVALUATE_SOURCE_SHA256,
    JudgeBackend,
    JudgeOutputError,
    build_summary_user_prompt,
    build_activity_identification_user_prompt,
    build_turn_user_prompt,
    load_prompt,
    validate_output_groups,
    validate_summary_output,
    validate_activity_identification_output,
    validate_turn_output,
)
from alliance_eval.evaluation_models import (
    LOADED_SOURCE_SHA256 as MODELS_SOURCE_SHA256,
    ClientBondOnlyTurnLLMOutput,
    ClientTurnLLMOutput,
    CounsellorBondOnlyTurnLLMOutput,
    CounsellorTurnLLMOutput,
    SamplePlan,
    SummaryLLMOutput,
    SummaryStoredRecord,
    ActivityIdentificationLLMOutput,
    ActivityIdentificationStoredRecord,
    TurnStoredRecord,
)


ProgressCallback = Callable[[dict[str, Any]], None]
UsageCallback = Callable[[dict[str, int]], None]


# Capture identity when the module loads; matching filenames do not establish identical execution.
LOADED_SOURCE_SHA256 = {
    "alliance_eval/pipeline.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "alliance_eval/evaluate.py": EVALUATE_SOURCE_SHA256,
    "alliance_eval/evaluation_models.py": MODELS_SOURCE_SHA256,
    "alliance_eval/input_data.py": INPUT_SOURCE_SHA256,
    "alliance_eval/resources.py": RESOURCES_SOURCE_SHA256,
}
PROMPT_FILES = {
    "activity_prompt": "identify_activity.txt",
    "turn_prompt": "judge_turn.txt",
    "summary_prompt": "judge_summary.txt",
}
SCHEMA_MODELS = {
    "activity_schema": ("activity_identification.schema.json", ActivityIdentificationLLMOutput),
    "turn_schema_counsellor": ("evaluation_turn_counsellor.schema.json", CounsellorTurnLLMOutput),
    "turn_schema_client": ("evaluation_turn_client.schema.json", ClientTurnLLMOutput),
    "turn_schema_counsellor_no_activity": ("evaluation_turn_counsellor_no_activity.schema.json", CounsellorBondOnlyTurnLLMOutput),
    "turn_schema_client_no_activity": ("evaluation_turn_client_no_activity.schema.json", ClientBondOnlyTurnLLMOutput),
    "summary_schema": ("evaluation_summary.schema.json", SummaryLLMOutput),
}
VALIDATION_POLICY = "bond_task_groups"
CONNECTION_RETRY_DELAYS = (1.0, 2.0)


def check_runtime_version() -> dict[str, Any]:
    """Check loaded source and all six output schemas without model calls; reject mismatched new runs."""

    for relative_path, loaded_hash in LOADED_SOURCE_SHA256.items():
        path = PACKAGE_DIR / Path(relative_path).name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != loaded_hash:
            raise RuntimeError(
                f"Loaded code differs from disk: {relative_path}. "
                "Restart the Streamlit service before starting a run; refreshing the page is not enough."
            )
    schema_hashes = {}
    for filename, model in SCHEMA_MODELS.values():
        path = resource_directory("schemas") / filename
        content = path.read_bytes()
        if json.loads(content) != model.model_json_schema():
            raise RuntimeError(
                f"Runtime schema differs from {filename}. Check the code/schema versions and restart the service."
            )
        schema_hashes[filename] = hashlib.sha256(content).hexdigest()
    input_schema_path = resource_directory("schemas") / f"{INPUT_SCHEMA_VERSION}.schema.json"
    input_schema_content = input_schema_path.read_bytes()
    if json.loads(input_schema_content) != dialogue_json_schema():
        raise RuntimeError("Runtime input schema differs from the packaged dialogue schema.")
    return {
        "source_sha256": dict(LOADED_SOURCE_SHA256),
        "schema_sha256": schema_hashes,
        "input_schema_sha256": {input_schema_path.name: hashlib.sha256(input_schema_content).hexdigest()},
    }


def _call_stage(
    backend: JudgeBackend, system_prompt: str, user_prompt: str,
    response_model: type[BaseModel], validator: Callable[[Any], Any],
    stage: str, sample_id: str, current_turn_id: str | None,
    errors: list[dict[str, Any]], usage_callback: UsageCallback | None,
    error_callback: ProgressCallback | None,
    group_validator: Callable[[Any, str], None] | None = None,
    active_cores: tuple[str, ...] = ("Bond", "Task"),
) -> BaseModel | None:
    """Call and validate; retry connection-establishment failures at most twice without converting failures into empty findings.

    Handle expected output and HTTP failures. Authentication, balance, endpoint or parameter errors abort the run;
    other call failures are recorded and return None. The run entry point handles unexpected code errors.
    """

    def record_failure(
        error_type: str, message: str, affected_cores: list[str],
        http_status: int | None = None,
        group_errors: list[dict[str, str]] | None = None,
        retry_scheduled: bool = False,
    ) -> None:
        """Record each failed call and its usage once, even when both groups fail."""

        failure = {
            "sample_id": sample_id,
            "current_turn_id": current_turn_id,
            "stage": stage,
            "error_type": error_type,
            "message": message,
            "http_status": http_status,
            "raw_response": raw_response,
            "usage": dict(usage),
            "affected_cores": affected_cores,
            "attempt": attempt,
        }
        if group_errors:
            failure["group_errors"] = group_errors
        if retry_scheduled:
            failure["retry_scheduled"] = True
        else:
            # Retain recovered connection failures in logs and request counts, not as unresolved summary gaps.
            errors.append(failure)
        if error_callback:
            error_callback(failure)

    for attempt in range(1, len(CONNECTION_RETRY_DELAYS) + 2):
        call = None
        raw_response = None
        usage = {"requests": 1, "input_tokens": 0, "output_tokens": 0}
        try:
            try:
                call = backend.generate(system_prompt, user_prompt, response_model)
            except JudgeOutputError as error:
                usage.update(error.usage)
                raw_response = error.raw_response
                if group_validator is None or not error.schema_validation_failed:
                    raise
            else:
                usage.update(call.usage)
                raw_response = call.raw_response or call.parsed.model_dump_json()

            if group_validator is not None:
                parsed, group_errors = validate_output_groups(
                    raw_response, response_model, group_validator, active_cores,
                )
                if group_errors:
                    record_failure(
                        "GroupValidationError",
                        "; ".join(f"{item['core']}: {item['message']}" for item in group_errors),
                        [item["core"] for item in group_errors],
                        group_errors=group_errors,
                    )
                return parsed

            validator(call.parsed)
            return call.parsed
        except (ValueError, httpx.HTTPError) as error:
            if isinstance(error, JudgeOutputError):
                usage.update(error.usage)
                raw_response = error.raw_response
            elif call is not None:
                raw_response = call.raw_response or call.parsed.model_dump_json()
            status_code = None
            if isinstance(error, httpx.HTTPStatusError):
                status_code = error.response.status_code
                raw_response = error.response.text
            retry_scheduled = (
                isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout))
                and attempt <= len(CONNECTION_RETRY_DELAYS)
            )
            record_failure(
                type(error).__name__, str(error),
                ["Task"] if stage == "activity_identification" else ["Bond", "Task"],
                http_status=status_code, retry_scheduled=retry_scheduled,
            )
            if status_code in {400, 401, 402, 403, 404, 422}:
                raise
            if not retry_scheduled:
                return None
        finally:
            if usage_callback:
                usage_callback(usage)
        # Retry only failed connections with no response, not structural or evidence validation failures.
        time.sleep(CONNECTION_RETRY_DELAYS[attempt - 1])


# See framework/bond.md and framework/task.md for mappings, sources and interpretation limits.
# Bordin (1979) and Horvath and Greenberg (1989) define the cores; other sources support observable evidence.
# This table connects saved results without rescoring text or producing core ratings or totals.
CORE_MAPPING: dict[str, list[dict[str, str | None]]] = {
    # Truax (1967) defines Accurate Empathy, not a WAI Bond subscale.
    # Retain its indirect Bond-level contribution rather than assign it to a specific core term.
    "Accurate Empathy": [
        {
            "core": "Bond",
            "core_element": None,
        }
    ],
    "Therapist accepts/validates Patient": [
        {
            "core": "Bond",
            "core_element": "acceptance",
        }
    ],
    "Grounded content uptake": [
        {
            "core": "Bond",
            "core_element": "mutual respect",
        }
    ],
    "Client confirmation/disconfirmation": [
        {
            "core": "Bond",
            "core_element": None,
        }
    ],
    "Client-perceived acceptance": [
        {
            "core": "Bond",
            "core_element": "acceptance",
        }
    ],
    # Eubanks and Muran (2022) WT2 explicitly defines mutual trust and respect as dyadic.
    "Mutual trust and respect": [
        {
            "core": "Bond",
            "core_element": "mutual trust",
        },
        {
            "core": "Bond",
            "core_element": "mutual respect",
        },
    ],
    # Task mappings use the core construct from Bordin (1979) and Horvath and Greenberg (1989).
    # Behavioural categories are source-bounded project codes, not original WAI scale items.
    "Collaborative activity negotiation": [
        {
            "core": "Task",
            "core_element": "agreement on the activities",
        }
    ],
    "Activity and responsibility clarification": [
        {
            "core": "Task",
            "core_element": "acceptance of responsibility to perform the activities",
        }
    ],
    "Activity–client relevance linkage": [
        {
            "core": "Task",
            "core_element": "activities perceived as relevant",
        }
    ],
    # Stiles et al. (1998) supplies the responsive/ballistic distinction, used here as indirect evidence
    # of maintaining or renegotiating Activity agreement; an adjustment is not automatically an improvement.
    "Activity-related responsiveness": [
        {
            "core": "Task",
            "core_element": "agreement on the activities",
        }
    ],
    "Activity agreement/disagreement": [
        {
            "core": "Task",
            "core_element": "agreement on the activities",
        }
    ],
    "Client engagement in the activity": [
        {
            "core": "Task",
            "core_element": "acceptance of responsibility to perform the activities",
        }
    ],
    "Client-expressed activity relevance": [
        {
            "core": "Task",
            "core_element": "activities perceived as relevant",
        }
    ],
    "Client-expressed activity efficacy": [
        {
            "core": "Task",
            "core_element": "activities perceived as efficacious",
        }
    ],
}


def _utc_now() -> str:
    """Return a timezone-aware UTC ISO timestamp for the run manifest."""

    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Write one UTF-8 JSON record with consistent indentation."""

    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    """Append a JSONL record so turn results can be saved immediately."""

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _load_source_dialogues(source_file: Path) -> tuple[dict[str, dict[str, Any]], str]:
    """Parse and hash the same source bytes to avoid mixing file versions within a run."""

    content = source_file.read_bytes()
    lines = content.decode("utf-8-sig").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(lines, start=1):
        try:
            record = parse_json_value(line)
        except ValueError as error:
            raise ValueError(f"{source_file}:{line_number} is not a valid JSON record: {error}") from error
        if not isinstance(record, dict) or not isinstance(record.get("dialogue_id"), str):
            raise ValueError(f"{source_file}:{line_number} is missing a string dialogue_id")
        dialogue_id = record["dialogue_id"]
        if dialogue_id in records:
            raise ValueError(f"Duplicate dialogue_id={dialogue_id} in {source_file}; cannot select an arbitrary record")
        records[dialogue_id] = record
    return records, hashlib.sha256(content).hexdigest()


def load_selected_dialogues(
    sample_plan: SamplePlan,
    data_root: Path,
) -> list[dict[str, Any]]:
    """Read planned dialogues or contiguous excerpts while preserving original stable turn IDs."""

    selected: list[dict[str, Any]] = []
    sources: dict[Path, tuple[dict[str, dict[str, Any]], str]] = {}
    for item in sample_plan.items:
        source_path = Path(item.source_file)
        if not source_path.is_absolute():
            source_path = data_root / source_path
        source_path = source_path.resolve()
        if source_path not in sources:
            sources[source_path] = _load_source_dialogues(source_path)
        records, source_sha256 = sources[source_path]
        if item.dialogue_id not in records:
            raise ValueError(f"Cannot find dialogue_id={item.dialogue_id} in {source_path}")
        dialogue = validate_dialogue(records[item.dialogue_id])
        turns = dialogue["turns"]
        if item.start_turn_index is not None:
            turns = [
                turn for turn in turns
                if turn["turn_index"] >= item.start_turn_index
            ]
        if item.end_turn_index is not None:
            turns = [
                turn for turn in turns
                if turn["turn_index"] <= item.end_turn_index
            ]
        if not turns:
            raise ValueError(f"Selected range for {item.dialogue_id} contains no turns")
        selected.append(
            {
                **dialogue,
                "turns": turns,
                "_source_file": str(source_path),
                "_source_sha256": source_sha256,
            }
        )
    return selected


def run_turn_annotation(
    dialogue: dict[str, Any],
    sample_id: str,
    backend: JudgeBackend,
    activity_prompt: str,
    turn_prompt: str,
    progress_callback: ProgressCallback | None = None,
    record_callback: Callable[[TurnStoredRecord], None] | None = None,
    activity_record_callback: (
        Callable[[ActivityIdentificationStoredRecord], None] | None
    ) = None,
    usage_callback: UsageCallback | None = None,
    errors: list[dict[str, Any]] | None = None,
    error_callback: ProgressCallback | None = None,
) -> tuple[list[TurnStoredRecord], dict[str, str], dict[str, int]]:
    """Identify Activities and annotate turns in order while maintaining the Activity registry.

    The other role contributes context only, not findings. Each validated canonical
    turn record can be saved immediately through record_callback before a long sample finishes.
    """

    records: list[TurnStoredRecord] = []
    activity_registry: dict[str, str] = {}
    usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0}
    errors = errors if errors is not None else []

    def record_usage(delta: dict[str, int]) -> None:
        """Accumulate attempted requests and known token usage, including failures."""

        for key, value in delta.items():
            usage[key] = usage.get(key, 0) + value
        if usage_callback:
            usage_callback(delta)

    for position, current_turn in enumerate(dialogue["turns"]):
        if progress_callback:
            progress_callback(
                {
                    "stage": "turn_annotation",
                    "sample_id": sample_id,
                    "current_turn_id": current_turn["turn_id"],
                    "turn_position": position + 1,
                    "turn_total": len(dialogue["turns"]),
                }
            )

        if current_turn["role"] == "other":
            activity_record = ActivityIdentificationStoredRecord(
                sample_id=sample_id,
                current_turn_id=current_turn["turn_id"],
                new_activities=[],
            )
            if activity_record_callback:
                activity_record_callback(activity_record)
            stored = TurnStoredRecord(
                sample_id=sample_id,
                current_turn_id=current_turn["turn_id"],
                findings=[],
            )
        else:
            # Identify new Activities separately, then pass the validated registry as
            # read-only input to the turn judge, which must not create Activities to fill Task dimensions.
            activity_user_prompt = build_activity_identification_user_prompt(
                dialogue, position, activity_registry
            )
            activity_output = _call_stage(
                backend, activity_prompt, activity_user_prompt, ActivityIdentificationLLMOutput,
                lambda output: validate_activity_identification_output(
                    output, dialogue, position, activity_registry
                ),
                "activity_identification", sample_id, current_turn["turn_id"],
                errors, record_usage, error_callback,
            )
            activity_record = None
            if activity_output is not None:
                activity_registry.update({
                    activity.activity_id: activity.activity_label for activity in activity_output.new_activities
                })
                activity_record = ActivityIdentificationStoredRecord(
                    sample_id=sample_id,
                    current_turn_id=current_turn["turn_id"],
                    new_activities=activity_output.new_activities,
                )
                if activity_record_callback:
                    activity_record_callback(activity_record)
            # On identification failure, retain the validated registry without concluding that no Activity exists.

            user_prompt = build_turn_user_prompt(dialogue, position, activity_registry)
            response_model = (
                (
                    CounsellorTurnLLMOutput
                    if activity_registry
                    else CounsellorBondOnlyTurnLLMOutput
                )
                if current_turn["role"] == "counsellor"
                else (
                    ClientTurnLLMOutput
                    if activity_registry
                    else ClientBondOnlyTurnLLMOutput
                )
            )
            output = _call_stage(
                backend, turn_prompt, user_prompt, response_model,
                lambda result: validate_turn_output(
                    result, dialogue, position, activity_registry
                ),
                "turn_annotation", sample_id, current_turn["turn_id"],
                errors, record_usage, error_callback,
                group_validator=lambda result, core: validate_turn_output(
                    result, dialogue, position, activity_registry
                ),
                active_cores=("Bond", "Task") if activity_registry else ("Bond",),
            )
            stored = (
                TurnStoredRecord(
                    sample_id=sample_id, current_turn_id=current_turn["turn_id"],
                    findings=output.findings,
                ) if output is not None else None
            )

        if stored is not None:
            records.append(stored)
            if record_callback:
                record_callback(stored)
        if progress_callback:
            # Save validated groups and report failed groups separately without additional calls.
            progress_callback(
                {
                    "stage": "turn_complete",
                    "sample_id": sample_id,
                    "turn": {
                        "turn_id": current_turn["turn_id"],
                        "role": current_turn["role"],
                        "text": current_turn["text"],
                    },
                    "activity_record": activity_record.model_dump() if activity_record else None,
                    "turn_record": stored.model_dump() if stored else None,
                    "errors": [
                        error for error in errors
                        if error["current_turn_id"] == current_turn["turn_id"]
                    ],
                }
            )

    return records, activity_registry, usage


def run_summary_annotation(
    dialogue: dict[str, Any],
    sample_id: str,
    turn_records: list[TurnStoredRecord],
    activity_registry: dict[str, str],
    backend: JudgeBackend,
    summary_prompt: str,
    progress_callback: ProgressCallback | None = None,
    usage_callback: UsageCallback | None = None,
    errors: list[dict[str, Any]] | None = None,
    error_callback: ProgressCallback | None = None,
) -> tuple[SummaryStoredRecord | None, dict[str, int]]:
    """Produce only the four summary dimensions after the complete sample."""

    if progress_callback:
        progress_callback({"stage": "summary_annotation", "sample_id": sample_id})
    errors = errors if errors is not None else []
    failed_annotations = [
        {
            "current_turn_id": error["current_turn_id"], "stage": error["stage"],
            "affected_cores": error["affected_cores"],
        }
        for error in errors if error["stage"] in {"activity_identification", "turn_annotation"}
    ]
    user_prompt = build_summary_user_prompt(
        dialogue, turn_records, activity_registry, failed_annotations
    )
    usage: dict[str, int] = {}

    def record_usage(delta: dict[str, int]) -> None:
        """Retain known summary usage and forward it to the run record."""

        for key, value in delta.items():
            usage[key] = usage.get(key, 0) + value
        if usage_callback:
            usage_callback(delta)

    output = _call_stage(
        backend, summary_prompt, user_prompt, SummaryLLMOutput,
        lambda result: validate_summary_output(
            result, dialogue, turn_records, activity_registry,
        ),
        "summary_annotation", sample_id, None, errors, record_usage, error_callback,
        group_validator=lambda result, core: validate_summary_output(
            result, dialogue, turn_records, activity_registry, core=core,
        ),
        active_cores=("Bond", "Task") if activity_registry else ("Bond",),
    )
    stored = (
        SummaryStoredRecord(sample_id=sample_id, results=output.results)
        if output is not None else None
    )
    return stored, usage


def run_core_mapping(
    turn_records: list[TurnStoredRecord],
    summary_record: SummaryStoredRecord | None,
) -> list[dict[str, Any]]:
    """Organise turn and summary results into a Core evidence view using fixed conceptual mappings.

    Mapping does not call models, change labels or sum Contributors and Indicators, nor establish Bond,
    Task or Alliance. A dimension linked to several core elements yields multiple traceable records.
    """

    mapped: list[dict[str, Any]] = []

    for record in turn_records:
        for finding in record.findings:
            if finding.finding_status == "evidence_only":
                continue
            for mapping in CORE_MAPPING[finding.dimension]:
                mapped.append(
                    {
                        "sample_id": record.sample_id,
                        "source_level": "turn-level",
                        "current_turn_id": record.current_turn_id,
                        "core": mapping["core"],
                        "core_element": mapping["core_element"],
                        "source_dimension": finding.dimension,
                        "activity_reference": (
                            finding.activity_reference.model_dump()
                            if finding.activity_reference is not None
                            else None
                        ),
                        "status": finding.finding_status,
                        "rating_label": finding.rating_label,
                        "evidence": [span.model_dump() for span in finding.evidence],
                        "counterevidence": [],
                        "reason": finding.reason,
                    }
                )

    for result in summary_record.results if summary_record is not None else []:
        source_level = (
            "activity-level summary"
            if result.activity_reference is not None
            else "sample-level summary"
        )
        for mapping in CORE_MAPPING[result.dimension]:
            mapped.append(
                {
                    "sample_id": summary_record.sample_id,
                    "source_level": source_level,
                    "current_turn_id": None,
                    "core": mapping["core"],
                    "core_element": mapping["core_element"],
                    "source_dimension": result.dimension,
                    "activity_reference": (
                        result.activity_reference.model_dump()
                        if result.activity_reference is not None
                        else None
                    ),
                    "status": result.summary_status,
                    "rating_label": result.rating_label,
                    "evidence": [span.model_dump() for span in result.evidence],
                    "counterevidence": [
                        span.model_dump() for span in result.counterevidence
                    ],
                    "reason": result.reason,
                }
            )
    return mapped


def run_all(
    sample_plan_path: Path,
    backend: JudgeBackend,
    run_root: Path,
    run_id: str | None = None,
    prompt_dir: Path | None = None,
    progress_callback: ProgressCallback | None = None,
    data_root: Path | None = None,
) -> Path:
    """Run turn annotation, summary and Core mapping for a sample plan and create a run directory.

    Continue after local failures; summaries for incomplete samples are diagnostic. Run-level errors abort execution.
    Save validated records and raw failed responses separately.
    """

    # Reject stale processes before reading samples, creating a run or calling a model.
    runtime = check_runtime_version()
    prompt_dir = prompt_dir or resource_directory("prompts")
    prompts = {key: load_prompt(prompt_dir / filename) for key, filename in PROMPT_FILES.items()}
    runtime["prompt_text_sha256"] = {
        PROMPT_FILES[key]: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for key, text in prompts.items()
    }
    sample_plan_path = sample_plan_path.resolve()
    sample_plan_content = sample_plan_path.read_bytes()
    plan = SamplePlan.model_validate(parse_json_value(sample_plan_content.decode("utf-8")))
    data_root = data_root.resolve() if data_root is not None else sample_plan_path.parent
    dialogues = load_selected_dialogues(plan, data_root)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_alliance_eval") if run_id is None else run_id
    if run_id in {".", ".."} or not run_id.strip() or any(char in run_id for char in "/\\"):
        raise ValueError("run_id must be a single non-empty directory name without paths or parent-directory references")
    run_dir = run_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    activity_prompt = prompts["activity_prompt"]
    turn_prompt = prompts["turn_prompt"]
    summary_prompt = prompts["summary_prompt"]

    activity_path = run_dir / "activity_identification.jsonl"
    turn_path = run_dir / "turn_annotations.jsonl"
    summary_path = run_dir / "summary_annotations.jsonl"
    core_path = run_dir / "core_evidence.jsonl"
    error_path = run_dir / "validation_errors.jsonl"
    # Keep empty output files so every run has the same inspectable structure.
    for output_path in (activity_path, turn_path, summary_path, core_path, error_path):
        output_path.touch()

    identity = code_identity()
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "stage": "alliance_evaluation",
        "status": "running",
        "started_at_utc": _utc_now(),
        "completed_at_utc": None,
        **identity,
        "runtime": runtime,
        "model": {
            "backend": backend.__class__.__name__,
            "model_id": backend.model_id,
            "dialect": getattr(backend, "dialect", "mock"),
            "base_url": getattr(backend, "base_url", None),
            "parameters": getattr(backend, "parameters", {}),
            "response_format": getattr(backend, "response_format", None),
        },
        "versions": {
            "framework": "bond_task",
            "validation_policy": VALIDATION_POLICY,
            "input_schema": INPUT_SCHEMA_VERSION,
            **PROMPT_FILES,
            **{key: filename for key, (filename, _) in SCHEMA_MODELS.items()},
            "sample": plan.sample_version,
            "context_policy": "past-context incremental turn annotation",
        },
        "connection_retry_policy": {
            "error_types": ["ConnectError", "ConnectTimeout"],
            "max_retries": len(CONNECTION_RETRY_DELAYS),
            "delays_seconds": list(CONNECTION_RETRY_DELAYS),
        },
        "sample_plan": str(sample_plan_path),
        "sample_plan_sha256": hashlib.sha256(sample_plan_content).hexdigest(),
        "data_root": str(data_root),
        "samples": [],
        "counts": {
            "planned_samples": len(dialogues),
            "processed_samples": 0,
            "completed_samples": 0,
            "technical_failures": 0,
            "failed_calls": 0,
        },
        "usage": {"requests": 0, "input_tokens": 0, "output_tokens": 0},
        "outputs": [
            activity_path.name,
            turn_path.name,
            summary_path.name,
            core_path.name,
            error_path.name,
        ],
    }
    _write_json(run_dir / "manifest.json", manifest)

    def record_usage(usage: dict[str, int]) -> None:
        """Immediately account for completed model calls in the in-memory run manifest."""

        for key, value in usage.items():
            manifest["usage"][key] = manifest["usage"].get(key, 0) + value

    for sample_number, dialogue in enumerate(dialogues, start=1):
        sample_id = f"S{sample_number:04d}"
        sample_state = {
            "sample_id": sample_id,
            "dialogue_id": dialogue["dialogue_id"],
            "source_file": dialogue["_source_file"],
            "source_sha256": dialogue["_source_sha256"],
            "turn_count": len(dialogue["turns"]),
            "selected_turn_ids": [turn["turn_id"] for turn in dialogue["turns"]],
            "annotation_complete": False,
            "summary_status": "not_run",
            "annotation_complete_by_core": {"Bond": False, "Task": False},
            "summary_status_by_core": {"Bond": "not_run", "Task": "not_run"},
        }
        manifest["samples"].append(sample_state)
        _write_json(run_dir / "manifest.json", manifest)
        errors: list[dict[str, Any]] = []

        def record_error(error: dict[str, Any]) -> None:
            """Retain every failed response; save valid groups separately and exclude failed groups from annotations."""

            _append_jsonl(error_path, {**error, "dialogue_id": dialogue["dialogue_id"]})
            manifest["counts"]["failed_calls"] += 1

        try:
            turn_records, activity_registry, _ = run_turn_annotation(
                dialogue,
                sample_id,
                backend,
                activity_prompt,
                turn_prompt,
                progress_callback=progress_callback,
                record_callback=lambda record: _append_jsonl(
                    turn_path, record.model_dump()
                ),
                activity_record_callback=lambda record: _append_jsonl(
                    activity_path, record.model_dump()
                ),
                usage_callback=record_usage,
                errors=errors,
                error_callback=record_error,
            )
            sample_state["annotation_complete"] = not errors
            sample_state["annotation_complete_by_core"] = {
                core: not any(core in error["affected_cores"] for error in errors)
                for core in ("Bond", "Task")
            }
            summary_record, _ = run_summary_annotation(
                dialogue,
                sample_id,
                turn_records,
                activity_registry,
                backend,
                summary_prompt,
                progress_callback=progress_callback,
                usage_callback=record_usage,
                errors=errors,
                error_callback=record_error,
            )
            summary_errors = [error for error in errors if error["stage"] == "summary_annotation"]
            sample_state["summary_status_by_core"] = {
                core: (
                    "failed" if any(core in error["affected_cores"] for error in summary_errors) else
                    "complete" if sample_state["annotation_complete_by_core"][core] else "diagnostic"
                )
                for core in ("Bond", "Task")
            }
            sample_state["summary_status"] = (
                "failed" if summary_record is None else
                "partial" if summary_errors else
                "complete" if sample_state["annotation_complete"] else "diagnostic"
            )
            if summary_record is not None:
                _append_jsonl(summary_path, summary_record.model_dump())
            for mapped_result in run_core_mapping(turn_records, summary_record):
                _append_jsonl(core_path, {
                    **mapped_result,
                    "diagnostic_only": sample_state["summary_status_by_core"][mapped_result["core"]] != "complete",
                })

            manifest["counts"]["processed_samples"] += 1
            if not errors:
                manifest["counts"]["completed_samples"] += 1
            else:
                # technical_failures counts failed samples, not failed calls.
                manifest["counts"]["technical_failures"] += 1
        except Exception as error:
            manifest["counts"]["technical_failures"] += 1
            if not isinstance(error, httpx.HTTPStatusError):
                _append_jsonl(error_path, {
                    "sample_id": sample_id,
                    "dialogue_id": dialogue["dialogue_id"],
                    "stage": "pipeline",
                    "affected_cores": ["Bond", "Task"],
                    "current_turn_id": None,
                    "error_type": type(error).__name__,
                    "message": str(error),
                })
            # Do not duplicate fatal HTTP errors already recorded at stage level; unexpected errors also stop further calls.
            manifest["status"] = "aborted"
            break
        _write_json(run_dir / "manifest.json", manifest)

    if manifest["status"] != "aborted":
        manifest["status"] = (
            "completed_with_errors"
            if manifest["counts"]["technical_failures"] else "completed"
        )
    manifest["completed_at_utc"] = _utc_now()
    _write_json(run_dir / "manifest.json", manifest)
    if progress_callback:
        progress_callback({"stage": "complete", "run_dir": str(run_dir)})
    return run_dir


def group_results_by_sample(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Read JSONL output grouped by sample ID for display or analysis."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return grouped
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            grouped[record["sample_id"]].append(record)
    return grouped
