"""Regressions for standalone entry points and source-evidence binding."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import shutil
import sys
import tomllib
from pathlib import Path

import pytest

from alliance_eval import cli, pipeline, resources
from alliance_eval.evaluate import MockJudge
from alliance_eval.evaluation_models import (
    BOND_DIMENSIONS,
    SUMMARY_LEVEL_DIMENSIONS,
    TASK_DIMENSIONS,
    SamplePlan,
)


@pytest.mark.parametrize("directory,extras", [("src", ("app",)), ("tests", ("app", "dev"))])
def test_direct_third_party_imports_have_declared_dependencies(directory, extras):
    """Direct imports must not rely only on another dependency installing them."""
    project = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((project / "pyproject.toml").read_text())["project"]
    requirements = list(metadata["dependencies"])
    for extra in extras:
        requirements.extend(metadata["optional-dependencies"][extra])
    declared = {re.match(r"[A-Za-z0-9_.-]+", requirement)[0].lower() for requirement in requirements}
    external = set()
    for path in (project / directory).rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            external.update(name.split(".")[0] for name in names
                            if name.split(".")[0] not in sys.stdlib_module_names | {"alliance_eval"})
    assert external <= declared, f"Undeclared direct imports: {sorted(external - declared)}"


@pytest.mark.parametrize("side, dimensions", [("bond", BOND_DIMENSIONS), ("task", TASK_DIMENSIONS)])
def test_framework_dimension_and_mapping_tables_match_runtime(side, dimensions):
    """Reader tables match the runtime contract; source archives are not separate active definitions."""
    project = Path(__file__).resolve().parents[1]
    definition, mapping = (project / "framework" / f"{side}.md").read_text(encoding="utf-8").split("## Core mappings", 1)
    definition_rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in definition.splitlines() if line.startswith("| ")
    ][1:]
    assert [row[0] for row in definition_rows] == list(dict.fromkeys(row[0] for row in definition_rows))
    assert {row[0] for row in definition_rows} == set(dimensions)
    assert {row[0] for row in definition_rows if "summary" in row[-1].lower()} == set(dimensions) & set(SUMMARY_LEVEL_DIMENSIONS)
    mapping_rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in mapping.split("## Source relationships", 1)[0].splitlines() if line.startswith("| ")
    ][1:]
    assert len(mapping_rows) == len(dimensions)
    assert {row[0] for row in mapping_rows} == set(dimensions)
    for dimension, element, rationale in mapping_rows:
        assert rationale
        for expected in pipeline.CORE_MAPPING[dimension]:
            assert (expected["core_element"] or "Bond as a whole") in element
            assert set(expected) == {"core", "core_element"}


def test_prompt_cards_match_runtime_dimension_groups():
    """Check active names and groups without duplicating or rewriting rating cards."""
    turn = resources.resource_directory("prompts").joinpath("judge_turn.txt").read_text(encoding="utf-8")
    summary = resources.resource_directory("prompts").joinpath("judge_summary.txt").read_text(encoding="utf-8")
    for prompt, expected in [(turn, BOND_DIMENSIONS + TASK_DIMENSIONS), (summary, SUMMARY_LEVEL_DIMENSIONS)]:
        cards = [heading.split(" (", 1)[0]
                 for heading in re.findall(r"^### ([^\n]+)$", prompt, flags=re.MULTILINE)]
        assert len(cards) == len(expected)
        assert set(cards) == set(expected)


def test_bond_definition_elements_match_core_mapping():
    """Core elements match the fixed mappings; anchor wording does not introduce additional elements."""
    project = Path(__file__).resolve().parents[1]
    paragraph = (project / "framework/bond.md").read_text(encoding="utf-8").split("\n\n")[1]
    documented = re.findall(r"`([^`]+)`", paragraph)
    mapped = {
        entry["core_element"] for entries in pipeline.CORE_MAPPING.values() for entry in entries
        if entry["core"] == "Bond" and entry["core_element"] is not None
    }
    assert len(documented) == len(set(documented))
    assert set(documented) == mapped


def write_plan(tmp_path: Path) -> tuple[Path, Path, dict]:
    record = {"dialogue_id": "original-fixture", "metadata": {"fictional": True}, "turns": [
        {"turn_id": "u-original", "turn_index": 4, "role": "client", "text": "I feel hesitant about small decisions."},
    ]}
    source = tmp_path / "dialogues.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    plan = {"sample_version": "engineering-fixture", "items": [{
        "source_file": source.name, "dialogue_id": record["dialogue_id"],
        "start_turn_index": None, "end_turn_index": None,
    }]}
    path = tmp_path / "sample_plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path, source, plan


def mock_results() -> list[dict]:
    return [{"new_activities": []}, {"findings": []}, {"results": [{
        "dimension": dimension, "activity_reference": None,
        "summary_status": "no_evaluable_opportunity", "rating_label": None,
        "evidence": [], "counterevidence": [], "reason": "Engineering fixture, not a semantic gold label.",
    } for dimension in ("Accurate Empathy", "Therapist accepts/validates Patient", "Mutual trust and respect")]}]


def test_full_mock_run_works_outside_source_checkout(tmp_path, monkeypatch):
    sample, source, _ = write_plan(tmp_path)
    elsewhere = tmp_path / "unrelated-workspace"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    run = pipeline.run_all(sample, MockJudge(mock_results()), elsewhere / "runs", "offline")
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["model"]["model_id"] == "mock"
    assert manifest["samples"][0]["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["sample_plan_sha256"] == hashlib.sha256(sample.read_bytes()).hexdigest()
    assert manifest["data_root"] == str(sample.parent)
    assert manifest["samples"][0]["summary_status_by_core"] == {"Bond": "complete", "Task": "complete"}
    assert set(manifest["runtime"]["schema_sha256"]) == {name for name, _ in pipeline.SCHEMA_MODELS.values()}
    assert "dialogue.schema.json" in manifest["runtime"]["input_schema_sha256"]
    assert manifest["versions"]["framework"] == "bond_task"
    assert manifest["versions"]["input_schema"] == "dialogue"
    assert manifest["versions"]["validation_policy"] == "bond_task_groups"
    with pytest.raises(FileExistsError):
        pipeline.run_all(sample, MockJudge(mock_results()), elsewhere / "runs", "offline")


def test_distributed_resource_names_and_exports_are_consistent(tmp_path, monkeypatch):
    prompt_names = {p.name for p in resources.resource_directory("prompts").glob("*.txt")}
    assert prompt_names == set(pipeline.PROMPT_FILES.values())
    output = tmp_path / "export"
    monkeypatch.setattr(sys, "argv", ["alliance-eval", "export-schema", "--output-dir", str(output)])
    assert cli.main() == 0
    stored = resources.resource_directory("schemas")
    assert {p.name for p in output.glob("*.json")} == {p.name for p in stored.glob("*.json")}
    for path in output.glob("*.json"):
        assert not re.search(r"(?:_|-)v\d", path.name)
        assert json.loads(path.read_text()) == json.loads((stored / path.name).read_text())


@pytest.mark.parametrize("start,end,expected", [
    (None, None, ["u1", "u3", "u5", "u7"]),
    (3, 5, ["u3", "u5"]),
    (None, 3, ["u1", "u3"]),
    (5, None, ["u5", "u7"]),
    (3, 3, ["u3"]),
])
def test_manifest_records_exact_selected_turns(tmp_path, start, end, expected):
    sample, source, plan = write_plan(tmp_path)
    record = json.loads(source.read_text())
    record["turns"] = [
        {"turn_id": f"u{index}", "turn_index": index, "role": "client",
         "text": f"Fictional source sentence {index}."}
        for index in (1, 3, 5, 7)
    ]
    source.write_text(json.dumps(record) + "\n")
    plan["items"][0].update(start_turn_index=start, end_turn_index=end)
    sample.write_text(json.dumps(plan))
    outputs = [{"new_activities": []}, {"findings": []}] * len(expected) + mock_results()[-1:]
    run = pipeline.run_all(sample, MockJudge(outputs), tmp_path / "runs", "selection")
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["samples"][0]["selected_turn_ids"] == expected
    assert manifest["samples"][0]["turn_count"] == len(expected)
    turns = [json.loads(line) for line in (run / "turn_annotations.jsonl").read_text().splitlines()]
    assert [turn["current_turn_id"] for turn in turns] == expected


def test_selection_is_saved_before_calls_and_survives_abort(tmp_path):
    sample, _, _ = write_plan(tmp_path)
    manifest_path = tmp_path / "runs" / "abort" / "manifest.json"
    saved_before_call = []

    class AbortJudge(MockJudge):
        def generate(self, *args):
            saved = json.loads(manifest_path.read_text())
            saved_before_call.append(saved["samples"])
            raise RuntimeError("Synthetic interruption before any accepted output")

    run = pipeline.run_all(sample, AbortJudge([]), tmp_path / "runs", "abort")
    saved = json.loads((run / "manifest.json").read_text())
    assert len(saved_before_call) == 1
    assert saved_before_call[0][0]["selected_turn_ids"] == ["u-original"]
    assert saved["status"] == "aborted"
    assert saved["samples"][0]["selected_turn_ids"] == ["u-original"]
    assert not (run / "turn_annotations.jsonl").read_text()


def test_selected_source_is_read_once_and_hashes_actual_bytes(tmp_path, monkeypatch):
    _, source, raw_plan = write_plan(tmp_path)
    original = source.read_bytes()
    raw_plan["items"].append(copy.deepcopy(raw_plan["items"][0]))
    read_bytes = Path.read_bytes
    reads = []

    def read_and_change(path):
        value = read_bytes(path)
        if path == source:
            reads.append(path)
            source.write_bytes(value.replace(b"hesitant", b"confident"))
        return value

    monkeypatch.setattr(Path, "read_bytes", read_and_change)
    selected = pipeline.load_selected_dialogues(SamplePlan.model_validate(raw_plan), tmp_path)
    assert len(reads) == 1
    assert len(selected) == 2
    assert all("hesitant" in item["turns"][0]["text"] for item in selected)
    assert all(item["_source_sha256"] == hashlib.sha256(original).hexdigest() for item in selected)


@pytest.mark.parametrize("value", [True, False, "1", 1.0, -1])
def test_sample_plan_never_coerces_turn_range(tmp_path, value):
    _, _, plan = write_plan(tmp_path)
    plan["items"][0]["start_turn_index"] = value
    with pytest.raises(ValueError):
        SamplePlan.model_validate(plan)


@pytest.mark.parametrize("bad_source", [
    '{"dialogue_id":"original-fixture","dialogue_id":"other","turns":[]}\n',
    '{"dialogue_id":"original-fixture","turns":[],"value":NaN}\n',
    '\n',
])
def test_direct_jsonl_rejects_ambiguous_or_nonstandard_json_before_run(tmp_path, bad_source):
    plan, source, _ = write_plan(tmp_path)
    source.write_text(bad_source, encoding="utf-8")
    with pytest.raises(ValueError):
        pipeline.run_all(plan, MockJudge(), tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_duplicate_dialogue_ids_do_not_select_first_silently(tmp_path):
    sample, source, _ = write_plan(tmp_path)
    source.write_bytes(source.read_bytes() * 2)
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        pipeline.run_all(sample, MockJudge(), tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("run_id", ["../outside", "nested/path", "bad\\name", ".", "..", "", " "])
def test_run_id_cannot_escape_its_output_root(tmp_path, run_id):
    sample, _, _ = write_plan(tmp_path)
    with pytest.raises(ValueError, match="run_id"):
        pipeline.run_all(sample, MockJudge(), tmp_path / "runs", run_id)
    assert not (tmp_path / "runs").exists()


def test_stale_loaded_code_blocks_before_reading_any_data(tmp_path, monkeypatch):
    monkeypatch.setitem(pipeline.LOADED_SOURCE_SHA256, "alliance_eval/pipeline.py", "invalid")
    with pytest.raises(RuntimeError, match="Loaded code differs"):
        pipeline.run_all(tmp_path / "missing-plan.json", MockJudge(), tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("filename", ["evaluation_summary.schema.json", "dialogue.schema.json"])
def test_schema_drift_blocks_runtime(tmp_path, monkeypatch, filename):
    copied = tmp_path / "schemas"
    shutil.copytree(resources.resource_directory("schemas"), copied)
    path = copied / filename
    data = json.loads(path.read_text())
    data["unexpected"] = "drift"
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(pipeline, "resource_directory", lambda name: copied)
    with pytest.raises(RuntimeError, match="schema differs"):
        pipeline.check_runtime_version()


def test_wheel_resources_do_not_depend_on_user_cwd(tmp_path, monkeypatch):
    installed = tmp_path / "site-packages" / "alliance_eval"
    (installed / "resources" / "prompts").mkdir(parents=True)
    monkeypatch.setattr(resources, "PACKAGE_DIR", installed)
    monkeypatch.chdir(tmp_path)
    assert resources.resource_directory("prompts") == installed / "resources" / "prompts"
    assert resources.code_identity()["git_commit"] is None


def test_checkout_resource_missing_does_not_fall_back_to_stale_copy(tmp_path, monkeypatch):
    package = tmp_path / "src" / "alliance_eval"
    (package / "resources" / "prompts").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("# fictional checkout", encoding="utf-8")
    monkeypatch.setattr(resources, "PACKAGE_DIR", package)
    with pytest.raises(FileNotFoundError, match="resource directory missing"):
        resources.resource_directory("prompts")


def test_cli_all_is_opt_in_before_backend_construction(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["alliance-eval", "all", "--sample-file", str(tmp_path / "missing.json")])
    monkeypatch.setattr(cli, "DeepSeekJudge", lambda **kwargs: pytest.fail("Unexpected backend construction"))
    assert cli.main() == 2
    assert "--allow-model-calls" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_cli_default_conversion_and_dry_run_in_empty_workspace(tmp_path, monkeypatch, capsys):
    _, source, _ = write_plan(tmp_path)
    monkeypatch.chdir(tmp_path)
    command = ["alliance-eval", "prepare-data", "--input", str(source)]
    monkeypatch.setattr(sys, "argv", command + ["--dry-run"])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "validated"
    assert not (tmp_path / "data").exists()
    monkeypatch.setattr(sys, "argv", command)
    assert cli.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "complete"
    processed = Path(result["files"]["dialogues.jsonl"])
    assert processed == tmp_path / "data" / "processed" / "dialogues.jsonl"
    assert not (processed.parent / "sample_plan.json").exists()
    monkeypatch.setattr(sys, "argv", ["alliance-eval", "create-plan", "--input", str(processed), "--dialogue-id", "original-fixture"])
    assert cli.main() == 0
    sample = Path(json.loads(capsys.readouterr().out)["sample_file"])
    assert sample == tmp_path / "data" / "plans" / "sample_plan.json"
    monkeypatch.setattr(sys, "argv", ["alliance-eval", "validate-input", "--sample-file", str(sample)])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["turns"] == 1


def test_cli_export_schema_never_overwrites_a_directory(tmp_path, monkeypatch, capsys):
    target = tmp_path / "exported"
    monkeypatch.setattr(sys, "argv", ["alliance-eval", "export-schema", "--output-dir", str(target)])
    assert cli.main() == 0
    assert len(list(target.glob("*.schema.json"))) == 7
    assert cli.main() == 2
    assert "File exists" in capsys.readouterr().err


def test_cli_explicit_call_routes_to_same_pipeline_with_mock(tmp_path, monkeypatch, capsys):
    sample, _, _ = write_plan(tmp_path)
    monkeypatch.chdir(tmp_path)
    backend = MockJudge(mock_results())
    assert not hasattr(backend, "client") and not hasattr(backend, "close")
    monkeypatch.setattr(cli, "DeepSeekJudge", lambda **kwargs: backend)
    monkeypatch.setattr(sys, "argv", ["alliance-eval", "all", "--sample-file", str(sample), "--allow-model-calls", "--run-id", "mock-only"])
    assert cli.main() == 0
    assert Path(capsys.readouterr().out.strip()).is_dir()


@pytest.mark.parametrize("invalid_result", [False, True])
def test_cli_closes_backend_once_and_reports_saved_result(tmp_path, monkeypatch, capsys, invalid_result):
    sample, _, _ = write_plan(tmp_path)
    responses = mock_results()
    if invalid_result:
        responses[1] = {"findings": [{"invalid": True}]}
    backend = MockJudge(responses)
    closed = []
    backend.close = lambda: closed.append(True)
    assert not hasattr(backend, "client")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "DeepSeekJudge", lambda **kwargs: backend)
    monkeypatch.setattr(sys, "argv", [
        "alliance-eval", "all", "--sample-file", str(sample), "--allow-model-calls", "--run-id", "close-fixture",
    ])
    assert cli.main() == (1 if invalid_result else 0)
    assert closed == [True]
    run = Path(capsys.readouterr().out.strip())
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == ("completed_with_errors" if invalid_result else "completed")


@pytest.mark.parametrize("with_close", [False, True])
def test_cli_pipeline_failure_preserves_error_and_optional_cleanup(tmp_path, monkeypatch, capsys, with_close):
    sample, _, _ = write_plan(tmp_path)
    backend = MockJudge()
    closed = []
    if with_close:
        backend.close = lambda: closed.append(True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "DeepSeekJudge", lambda **kwargs: backend)

    def failed_pipeline(**kwargs):
        raise RuntimeError("Fictional pipeline interruption")

    monkeypatch.setattr(cli, "run_all", failed_pipeline)
    monkeypatch.setattr(sys, "argv", [
        "alliance-eval", "all", "--sample-file", str(sample), "--allow-model-calls",
    ])
    assert cli.main() == 2
    assert "Fictional pipeline interruption" in capsys.readouterr().err
    assert closed == ([True] if with_close else [])
    assert not (tmp_path / "runs").exists()


def test_cli_mistral_is_gated_and_records_native_defaults_without_credentials(tmp_path, monkeypatch, capsys):
    from alliance_eval.evaluate import LocalHTTPJudge

    sample, _, _ = write_plan(tmp_path)
    captured = {}

    def backend(**kwargs):
        captured.update(kwargs)
        native = LocalHTTPJudge(**kwargs)
        result = MockJudge(mock_results())
        for name in ("model_id", "dialect", "base_url", "parameters", "response_format"):
            setattr(result, name, getattr(native, name))
        result.close = native.close
        return result

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MISTRAL_API_KEY", "fictional-mistral-key")
    monkeypatch.setattr(cli, "LocalHTTPJudge", backend)
    command = ["alliance-eval", "all", "--sample-file", str(sample),
               "--dialect", "mistral", "--run-id", "mistral-fixture"]
    monkeypatch.setattr(sys, "argv", command)
    assert cli.main() == 2 and not captured
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", command + ["--allow-model-calls"])
    assert cli.main() == 0
    assert captured["model_id"] == "mistral-small-2603"
    assert captured["base_url"] == "https://api.mistral.ai/v1"
    run_dir = Path(capsys.readouterr().out.strip())
    manifest = (run_dir / "manifest.json").read_text(encoding="utf-8")
    model = json.loads(manifest)["model"]
    assert "fictional-mistral-key" not in manifest
    assert model["dialect"] == "mistral" and model["response_format"] == "json_schema"
    assert model["parameters"] == {"temperature": 0.0, "max_tokens": 32768,
                                   "reasoning_effort": "none", "random_seed": 666}


def test_cli_compatible_key_is_read_from_named_environment_only(tmp_path, monkeypatch, capsys):
    sample, _, _ = write_plan(tmp_path)
    captured = {}

    def backend(**kwargs):
        captured.update(kwargs)
        result = MockJudge(mock_results())
        result.response_format = kwargs["response_format"]
        return result

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EXAMPLE_MODEL_KEY", "not-a-real-key")
    monkeypatch.setattr(cli, "LocalHTTPJudge", backend)
    command = ["alliance-eval", "all", "--sample-file", str(sample),
               "--dialect", "openai_compatible", "--base-url", "https://model.example/v1",
               "--model", "example-model", "--api-key-env", "EXAMPLE_MODEL_KEY",
               "--response-format", "json_object", "--run-id", "compatible-fixture"]
    monkeypatch.setattr(sys, "argv", command)
    assert cli.main() == 2
    assert not captured
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", command + ["--allow-model-calls"])
    assert cli.main() == 0
    assert captured["api_key"] == "not-a-real-key"
    assert captured["response_format"] == "json_object"
    run_dir = Path(capsys.readouterr().out.strip())
    manifest = (run_dir / "manifest.json").read_text(encoding="utf-8")
    assert "not-a-real-key" not in manifest
    assert json.loads(manifest)["model"]["response_format"] == "json_object"
