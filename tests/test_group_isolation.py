"""Offline regressions for Bond/Task group isolation and bounded connection retries."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alliance_eval.evaluate import JudgeOutputError, MockJudge, parse_judge_response
from alliance_eval import pipeline
from alliance_eval.pipeline import run_all, run_turn_annotation, run_summary_annotation


ACTIVITY = {"activity_id": "ACTIVITY_01", "activity_label": "Record thoughts"}
TURNS = [
    {"turn_id": "T0", "role": "counsellor", "text": "Please record your thoughts. I understand this is hard."},
    {"turn_id": "T1", "role": "client", "text": "I will record my thoughts. Thank you."},
]


def bond_finding() -> dict:
    """Structurally valid Bond evidence fixture, not a semantically adjudicated rating."""

    return {
        "dimension": "Accurate Empathy", "activity_reference": None,
        "finding_status": "evidence_only", "rating_label": None,
        "evidence": [{"turn_id": "T0", "quote": "I understand this is hard."}],
        "reason": "Offline evidence fixture.",
    }


def activity_finding() -> dict:
    """Valid Activity reference and quote fixture."""

    return {
        "dimension": "Activity and responsibility clarification", "activity_reference": dict(ACTIVITY),
        "finding_status": "rated", "rating_label": "Clear activity and applicable responsibilities",
        "evidence": [{"turn_id": "T0", "quote": "Please record your thoughts."}],
        "reason": "Offline Activity fixture.",
    }


def summary_result(dimension: str, activity: dict | None = None) -> dict:
    """Create valid summaries for three Bond targets or one Activity target."""

    if dimension == "Accurate Empathy":
        status, label, evidence = "rated", "AE-3", bond_finding()["evidence"]
    elif activity:
        status, label = "rated", "Somewhat salient"
        evidence = [{"turn_id": "T1", "quote": "I will record my thoughts."}]
    else:
        status, label, evidence = "no_evaluable_opportunity", None, []
    return {
        "dimension": dimension, "activity_reference": activity,
        "summary_status": status, "rating_label": label,
        "evidence": evidence, "counterevidence": [], "reason": "Offline summary fixture.",
    }


def responses() -> list[dict]:
    """Provide responses for Activity, turn, Activity, turn and summary calls."""

    return [
        {"new_activities": [dict(ACTIVITY)]},
        {"findings": [bond_finding(), activity_finding()]},
        {"new_activities": []},
        {"findings": [{
            "dimension": "Client engagement in the activity", "activity_reference": dict(ACTIVITY),
            "finding_status": "evidence_only", "rating_label": None,
            "evidence": [{"turn_id": "T1", "quote": "I will record my thoughts."}],
            "reason": "Offline engagement fixture.",
        }]},
        {"results": [
            summary_result("Accurate Empathy"),
            summary_result("Therapist accepts/validates Patient"),
            summary_result("Mutual trust and respect"),
            summary_result("Client engagement in the activity", dict(ACTIVITY)),
        ]},
    ]


class RecordingJudge(MockJudge):
    """Record full inputs and deterministic usage, rejecting unexpected additional calls."""

    def __init__(self, outputs: list[dict | Exception]) -> None:
        super().__init__(outputs)
        self.inputs = []

    def generate(self, system_prompt, user_prompt, response_model):
        self.inputs.append((system_prompt, json.loads(user_prompt), response_model.__name__))
        assert self.responses, "Unexpected extra model call"
        output = self.responses.pop(0)
        if isinstance(output, Exception):
            raise output
        return parse_judge_response(
            json.dumps(output),
            {"input_tokens": 10, "output_tokens": 2}, response_model,
        )


def write_sample_plan(tmp_path: Path, copies: int = 1) -> Path:
    """Create temporary plans from an original two-turn fixture to test continuation after failure."""

    dialogues = [{"dialogue_id": f"fixture-{index}", "turns": [
        {**copy.deepcopy(turn), "turn_index": index} for index, turn in enumerate(TURNS)
    ]} for index in range(copies)]
    source = tmp_path / "dialogue.jsonl"
    source.write_text("".join(json.dumps(dialogue) + "\n" for dialogue in dialogues), encoding="utf-8")
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps({
        "sample_version": "offline_group_isolation",
        "items": [{
            "source_file": str(source), "dialogue_id": dialogue["dialogue_id"],
            "start_turn_index": None, "end_turn_index": None,
        } for dialogue in dialogues],
    }), encoding="utf-8")
    return sample


def run_fixture(tmp_path: Path, outputs: list[dict]) -> tuple[Path, RecordingJudge]:
    """Create temporary input containing the complete two-turn fixture."""

    sample = write_sample_plan(tmp_path)
    judge = RecordingJudge(outputs)
    path = run_all(
        sample, judge, tmp_path / "runs", "fixture",
    )
    assert len(judge.inputs) == 5
    assert not judge.responses
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["usage"] == {"requests": 5, "input_tokens": 50, "output_tokens": 10}
    assert judge.inputs[3][1]["visible_turns"] == TURNS
    return path, judge


def read_rows(path: Path, name: str) -> list[dict]:
    """Read the temporary run created by this test."""

    return [json.loads(line) for line in (path / name).read_text().splitlines()]


@pytest.mark.parametrize("defect", ["quote", "unknown_activity", "changed_label", "rating", "status", "missing", "extra", "future", "speaker"])
def test_activity_turn_failure_keeps_whole_valid_bond_group(tmp_path, defect) -> None:
    """Activity structure or evidence errors reject Task only; Bond evidence remains available to summary."""

    outputs = responses()
    bad = outputs[1]["findings"][1]
    if defect == "quote":
        bad["evidence"][0]["quote"] = "not in source"
    elif defect == "unknown_activity":
        bad["activity_reference"]["activity_id"] = "ACTIVITY_99"
    elif defect == "changed_label":
        bad["activity_reference"]["activity_label"] = "Changed label"
    elif defect == "rating":
        bad["rating_label"] = "invented rating"
    elif defect == "status":
        bad["finding_status"] = "insufficient_evidence"
    elif defect == "missing":
        del bad["reason"]
    elif defect == "extra":
        bad["extra_field"] = True
    elif defect == "future":
        bad["evidence"] = [{"turn_id": "T1", "quote": "I will record my thoughts."}]
    else:
        bad.update(dimension="Client-expressed activity efficacy", rating_label="Clear client-perceived efficacy")
    # A valid finding in the rejected group must not be salvaged separately.
    outputs[1]["findings"].append(activity_finding())
    original = json.dumps(outputs[1])
    path, judge = run_fixture(tmp_path, outputs)
    assert read_rows(path, "turn_annotations.jsonl")[0]["findings"] == [bond_finding()]
    failure = read_rows(path, "validation_errors.jsonl")[0]
    assert failure["affected_cores"] == ["Task"]
    assert failure["raw_response"] == original
    assert len(failure["group_errors"]) == 1
    sample = json.loads((path / "manifest.json").read_text())["samples"][0]
    assert sample["annotation_complete_by_core"] == {"Bond": True, "Task": False}
    assert sample["summary_status_by_core"] == {"Bond": "complete", "Task": "diagnostic"}
    assert sample["annotation_complete"] is False
    assert sample["summary_status"] == "diagnostic"
    core_rows = read_rows(path, "core_evidence.jsonl")
    assert all(not row["diagnostic_only"] for row in core_rows if row["core"] == "Bond")
    assert all(row["diagnostic_only"] for row in core_rows if row["core"] == "Task")
    failures = judge.inputs[-1][1]["failed_annotations"]
    assert failures == [{"current_turn_id": "T0", "stage": "turn_annotation", "affected_cores": ["Task"]}]
    ae_target = judge.inputs[-1][1]["required_summary_targets"]["summary_level_ratings"][0]
    assert ae_target["validated_turn_annotations"][0]["evidence"] == bond_finding()["evidence"]


@pytest.mark.parametrize("status", ["rated", "insufficient_evidence", "no_evaluable_opportunity"])
def test_core_mapping_excludes_intermediate_evidence_without_changing_annotations(tmp_path, status):
    outputs = responses()
    summary = outputs[-1]["results"][0]
    summary["summary_status"] = status
    if status != "rated":
        summary["rating_label"] = None
    if status == "no_evaluable_opportunity":
        summary["evidence"] = []
    expected = copy.deepcopy(outputs)
    path, judge = run_fixture(tmp_path, outputs)
    turns = read_rows(path, "turn_annotations.jsonl")
    summaries = read_rows(path, "summary_annotations.jsonl")
    assert [row["findings"] for row in turns] == [expected[1]["findings"], expected[3]["findings"]]
    assert summaries[0]["results"] == expected[-1]["results"]
    assert len(judge.inputs) == 5
    targets = judge.inputs[-1][1]["required_summary_targets"]["summary_level_ratings"]
    assert targets[0]["validated_turn_annotations"][0]["evidence"] == bond_finding()["evidence"]
    assert targets[-1]["validated_turn_annotations"][0]["evidence"] == expected[3]["findings"][0]["evidence"]
    mapped = read_rows(path, "core_evidence.jsonl")
    assert mapped and all(row["status"] != "evidence_only" for row in mapped)
    assert all("mapping_type" not in row for row in mapped)
    assert {row["source_dimension"] for row in mapped if row["source_level"] == "turn-level"} == {
        activity_finding()["dimension"]
    }
    for result in expected[-1]["results"]:
        rows = [row for row in mapped if row["source_dimension"] == result["dimension"]]
        assert len(rows) == len(pipeline.CORE_MAPPING[result["dimension"]])
        assert all(row["status"] == result["summary_status"]
                   and row["rating_label"] == result["rating_label"]
                   and row["reason"] == result["reason"]
                   and row["activity_reference"] == result["activity_reference"]
                   and row["evidence"] == result["evidence"]
                   and row["counterevidence"] == result["counterevidence"] for row in rows)


def test_bond_turn_failure_does_not_invalidate_activity(tmp_path) -> None:
    """Apply symmetric isolation; a Bond error prevents its summary being marked complete."""

    outputs = responses()
    outputs[1]["findings"][0]["evidence"][0]["quote"] = "wrong Bond quote"
    outputs[1]["findings"].append(bond_finding())
    outputs[-1]["results"][0].update(summary_status="insufficient_evidence", rating_label=None, evidence=[])
    path, _ = run_fixture(tmp_path, outputs)
    assert read_rows(path, "turn_annotations.jsonl")[0]["findings"] == [activity_finding()]
    sample = json.loads((path / "manifest.json").read_text())["samples"][0]
    assert sample["annotation_complete_by_core"] == {"Bond": False, "Task": True}
    assert sample["summary_status_by_core"] == {"Bond": "diagnostic", "Task": "complete"}
    core_rows = read_rows(path, "core_evidence.jsonl")
    assert all(not row["diagnostic_only"] for row in core_rows if row["core"] == "Task")
    assert all(row["diagnostic_only"] for row in core_rows if row["core"] == "Bond")


@pytest.mark.parametrize("core", ["Bond", "Task"])
@pytest.mark.parametrize("defect", ["quote_pool", "missing", "duplicate", "schema", "reference"])
def test_summary_failure_is_atomic_within_one_core(tmp_path, core, defect) -> None:
    """Check Bond and Activity summary coverage independently and reject whole failed groups."""

    outputs = responses()
    index = 0 if core == "Bond" else 3
    bad = outputs[-1]["results"][index]
    if defect == "quote_pool":
        bad["evidence"] = [{"turn_id": "T1", "quote": "Thank you."}]
    elif defect == "missing":
        del outputs[-1]["results"][index]
    elif defect == "duplicate":
        outputs[-1]["results"].append(copy.deepcopy(bad))
    elif defect == "schema":
        bad["rating_label"] = "invalid label"
    else:
        bad["activity_reference"] = {"activity_id": "ACTIVITY_99", "activity_label": "Unknown"}
    original = json.dumps(outputs[-1])
    path, _ = run_fixture(tmp_path, outputs)
    kept = read_rows(path, "summary_annotations.jsonl")[0]["results"]
    assert len(kept) == (1 if core == "Bond" else 3)
    assert all((result["activity_reference"] is None) == (core == "Task") for result in kept)
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["counts"]["failed_calls"] == 1
    assert manifest["counts"]["completed_samples"] == 0
    assert manifest["samples"][0]["summary_status"] == "partial"
    other = "Task" if core == "Bond" else "Bond"
    assert manifest["samples"][0]["summary_status_by_core"] == {core: "failed", other: "complete"}
    failure = read_rows(path, "validation_errors.jsonl")[0]
    assert failure["affected_cores"] == [core]
    assert failure["raw_response"] == original


def test_both_group_errors_are_one_failed_call(tmp_path) -> None:
    """Retain both group errors while counting the shared request and usage once."""

    outputs = responses()
    for result in (outputs[-1]["results"][0], outputs[-1]["results"][3]):
        result["rating_label"] = "invalid"
    path, _ = run_fixture(tmp_path, outputs)
    assert read_rows(path, "summary_annotations.jsonl") == []
    errors = read_rows(path, "validation_errors.jsonl")
    assert len(errors) == 1
    assert errors[0]["affected_cores"] == ["Bond", "Task"]
    assert len(errors[0]["group_errors"]) == 2
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["counts"]["failed_calls"] == 1
    assert manifest["samples"][0]["summary_status_by_core"] == {"Bond": "failed", "Task": "failed"}


@pytest.mark.parametrize("defect", ["json", "unknown_dimension", "non_object", "extra_top_level", "truncated", "refusal"])
def test_shared_or_incomplete_response_is_not_salvaged(defect) -> None:
    """Reject the entire response when parsing, grouping or completion cannot be established."""

    raw = {"findings": [bond_finding(), activity_finding()]}
    if defect == "unknown_dimension":
        raw["findings"].append({"dimension": "Unknown dimension"})
    elif defect == "non_object":
        raw["findings"].append("invalid item")
    elif defect == "extra_top_level":
        raw["extra"] = True
    content = "{" if defect == "json" else json.dumps(raw)

    class RawJudge(MockJudge):
        def generate(self, system_prompt, user_prompt, response_model):
            if response_model.__name__ == "ActivityIdentificationLLMOutput":
                return parse_judge_response(json.dumps({"new_activities": [ACTIVITY]}), {}, response_model)
            if defect in {"truncated", "refusal"}:
                raise JudgeOutputError(defect, content, {"input_tokens": 10})
            return parse_judge_response(content, {"input_tokens": 10}, response_model)

    errors = []
    records, _, usage = run_turn_annotation(
        {"turns": TURNS[:1]}, "S1", RawJudge(), "activity", "turn", errors=errors,
    )
    assert records == []
    assert len(errors) == 1
    assert errors[0]["affected_cores"] == ["Bond", "Task"]
    assert errors[0]["raw_response"] == content
    assert usage["requests"] == 2
    assert usage["input_tokens"] == 10


def test_activity_identification_failure_does_not_mark_bond_diagnostic(tmp_path) -> None:
    """Preserve the registry after identification failure and report Bond completeness independently."""

    outputs = responses()
    outputs[0] = {"new_activities": [dict(ACTIVITY)], "invalid_extra": True}
    outputs[1] = {"findings": [bond_finding()]}
    outputs[3] = {"findings": []}
    outputs[-1]["results"].pop()
    path, _ = run_fixture(tmp_path, outputs)
    sample = json.loads((path / "manifest.json").read_text())["samples"][0]
    assert sample["annotation_complete_by_core"] == {"Bond": True, "Task": False}
    assert sample["summary_status_by_core"] == {"Bond": "complete", "Task": "diagnostic"}
    assert read_rows(path, "validation_errors.jsonl")[0]["affected_cores"] == ["Task"]
    assert all(not row["diagnostic_only"] for row in read_rows(path, "core_evidence.jsonl"))


def test_success_keeps_original_findings_order_and_contract(tmp_path) -> None:
    """Grouping preserves valid response order, prompts and input structure."""

    outputs = responses()
    outputs[1]["findings"].reverse()
    original = copy.deepcopy(outputs)
    path, judge = run_fixture(tmp_path, outputs)
    assert read_rows(path, "turn_annotations.jsonl")[0]["findings"] == original[1]["findings"]
    assert read_rows(path, "summary_annotations.jsonl")[0]["results"] == original[-1]["results"]
    assert read_rows(path, "validation_errors.jsonl") == []
    assert "failed_annotations" not in judge.inputs[-1][1]
    assert [call[2] for call in judge.inputs] == [
        "ActivityIdentificationLLMOutput", "CounsellorTurnLLMOutput",
        "ActivityIdentificationLLMOutput", "ClientTurnLLMOutput", "SummaryLLMOutput",
    ]
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["versions"]["validation_policy"] == "bond_task_groups"
    assert manifest["samples"][0]["summary_status_by_core"] == {"Bond": "complete", "Task": "complete"}


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ConnectTimeout])
@pytest.mark.parametrize("stage_index", [0, 1, 4])
@pytest.mark.parametrize("failure_count", [1, 2])
def test_connection_retry_preserves_input_results_and_counts(
    tmp_path, monkeypatch, error_type, stage_index, failure_count,
) -> None:
    outputs = responses()
    outputs[stage_index:stage_index] = [error_type("Offline connection failure") for _ in range(failure_count)]
    judge = RecordingJudge(outputs)
    waits = []
    monkeypatch.setattr(pipeline.time, "sleep", waits.append)
    path = run_all(write_sample_plan(tmp_path), judge, tmp_path / "runs", "retry")
    assert not judge.responses
    assert len(judge.inputs) == 5 + failure_count
    assert all(value == judge.inputs[stage_index] for value in
               judge.inputs[stage_index:stage_index + failure_count + 1])
    assert waits == list(pipeline.CONNECTION_RETRY_DELAYS[:failure_count])
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["usage"] == {"requests": 5 + failure_count, "input_tokens": 50, "output_tokens": 10}
    assert manifest["counts"]["failed_calls"] == failure_count
    assert manifest["counts"]["completed_samples"] == 1
    assert manifest["counts"]["technical_failures"] == 0
    assert manifest["connection_retry_policy"]["max_retries"] == 2
    assert manifest["samples"][0]["summary_status_by_core"] == {"Bond": "complete", "Task": "complete"}
    assert "failed_annotations" not in judge.inputs[-1][1]
    failures = read_rows(path, "validation_errors.jsonl")
    assert [row["attempt"] for row in failures] == list(range(1, failure_count + 1))
    assert all(row["retry_scheduled"] is True for row in failures)
    assert all(row["usage"] == {"requests": 1, "input_tokens": 0, "output_tokens": 0} for row in failures)
    assert len(read_rows(path, "activity_identification.jsonl")) == 2
    assert len(read_rows(path, "turn_annotations.jsonl")) == 2
    assert read_rows(path, "turn_annotations.jsonl")[0]["findings"] == responses()[1]["findings"]
    assert read_rows(path, "summary_annotations.jsonl")[0]["results"] == responses()[-1]["results"]


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ConnectTimeout])
@pytest.mark.parametrize("stage_index", [0, 1, 4])
def test_exhausted_connection_retries_continue_later_calls_and_samples(
    tmp_path, monkeypatch, error_type, stage_index,
) -> None:
    outputs = responses()
    # After upstream failure, return only fixture results with remaining evidence support.
    if stage_index in (0, 1):
        outputs[-1]["results"][0].update(summary_status="insufficient_evidence", rating_label=None, evidence=[])
    if stage_index == 0:
        outputs[1] = {"findings": [bond_finding()]}
        outputs[-1]["results"][0] = summary_result("Accurate Empathy")
        outputs[3] = {"findings": []}
        outputs[-1]["results"].pop()
    outputs[stage_index:stage_index + 1] = [error_type("Offline persistent connection failure") for _ in range(3)]
    outputs += responses()
    judge = RecordingJudge(outputs)
    waits = []
    monkeypatch.setattr(pipeline.time, "sleep", waits.append)
    path = run_all(write_sample_plan(tmp_path, copies=2), judge, tmp_path / "runs", "exhausted")
    assert not judge.responses
    assert len(judge.inputs) == 12
    assert waits == [1.0, 2.0]
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["status"] == "completed_with_errors"
    assert manifest["counts"] == {
        "planned_samples": 2, "processed_samples": 2, "completed_samples": 1,
        "technical_failures": 1, "failed_calls": 3,
    }
    assert manifest["usage"] == {"requests": 12, "input_tokens": 90, "output_tokens": 18}
    assert manifest["samples"][1]["summary_status"] == "complete"
    failures = read_rows(path, "validation_errors.jsonl")
    assert [row["attempt"] for row in failures] == [1, 2, 3]
    assert [row.get("retry_scheduled", False) for row in failures] == [True, True, False]
    if stage_index != 4:
        assert len(judge.inputs[6][1]["failed_annotations"]) == 1
    assert "failed_annotations" not in judge.inputs[-1][1]


def test_summary_usage_accumulates_connection_attempts(monkeypatch) -> None:
    monkeypatch.setattr(pipeline.time, "sleep", lambda delay: None)
    outputs = responses()
    turn_records, registry, _ = run_turn_annotation(
        {"turns": TURNS}, "S1", RecordingJudge(outputs[:-1]), "activity", "turn",
    )
    failures, errors = [], []
    summary, usage = run_summary_annotation(
        {"turns": TURNS}, "S1", turn_records, registry,
        RecordingJudge([httpx.ConnectTimeout("Offline timeout"), outputs[-1]]),
        "summary", errors=errors, error_callback=failures.append,
    )
    assert summary is not None and errors == []
    assert len(failures) == 1 and failures[0]["retry_scheduled"]
    assert usage == {"requests": 2, "input_tokens": 10, "output_tokens": 2}


@pytest.mark.parametrize("error_type", [httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout, httpx.ReadError])
def test_other_transport_errors_are_not_retried(tmp_path, monkeypatch, error_type) -> None:
    waits = []
    monkeypatch.setattr(pipeline.time, "sleep", waits.append)
    outputs = responses()
    outputs[-1] = error_type("Offline non-connection error")
    path, _ = run_fixture_with_failed_summary(tmp_path, outputs)
    assert waits == []
    assert read_rows(path, "validation_errors.jsonl")[0]["attempt"] == 1


def run_fixture_with_failed_summary(tmp_path, outputs):
    judge = RecordingJudge(outputs)
    path = run_all(write_sample_plan(tmp_path), judge, tmp_path / "runs", "failed-summary")
    manifest = json.loads((path / "manifest.json").read_text())
    assert len(judge.inputs) == 5 and not judge.responses
    assert manifest["counts"]["failed_calls"] == 1
    assert manifest["usage"]["requests"] == 5
    assert manifest["samples"][0]["summary_status"] == "failed"
    return path, judge


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422, 429, 500, 503])
def test_http_status_errors_keep_existing_policy_without_retry(tmp_path, monkeypatch, status) -> None:
    waits = []
    monkeypatch.setattr(pipeline.time, "sleep", waits.append)
    response = httpx.Response(status, request=httpx.Request("POST", "https://offline.invalid"))
    error = httpx.HTTPStatusError("Offline HTTP error", request=response.request, response=response)
    outputs = responses()
    outputs[0] = error
    fatal = status in {400, 401, 402, 403, 404, 422}
    if not fatal:
        outputs[1] = {"findings": [bond_finding()]}
        outputs[3] = {"findings": []}
        outputs[-1]["results"].pop()
    judge = RecordingJudge(outputs)
    path = run_all(write_sample_plan(tmp_path), judge, tmp_path / "runs", "http-error")
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["status"] == ("aborted" if fatal else "completed_with_errors")
    assert len(judge.inputs) == (1 if fatal else 5)
    assert waits == []
    failures = read_rows(path, "validation_errors.jsonl")
    assert len(failures) == 1 and failures[0]["http_status"] == status


def test_schema_error_after_connection_retry_is_not_retried_again(tmp_path, monkeypatch) -> None:
    waits = []
    monkeypatch.setattr(pipeline.time, "sleep", waits.append)
    outputs = responses()
    outputs[-1:] = [httpx.ConnectError("Offline connection failure"), {"results": "invalid"}]
    judge = RecordingJudge(outputs)
    path = run_all(write_sample_plan(tmp_path), judge, tmp_path / "runs", "invalid-output")
    assert len(judge.inputs) == 6 and not judge.responses
    assert waits == [1.0]
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["counts"]["failed_calls"] == 2
    assert manifest["counts"]["completed_samples"] == 0
    failures = read_rows(path, "validation_errors.jsonl")
    assert [row["attempt"] for row in failures] == [1, 2]
    assert not failures[-1].get("retry_scheduled")
