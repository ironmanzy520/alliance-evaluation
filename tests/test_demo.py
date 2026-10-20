"""The labelled, no-network demonstration uses the real pipeline and guards."""

import hashlib
import json
import sys

import pytest

from alliance_eval import cli, demo
from alliance_eval.evaluation_models import ActivityIdentificationLLMOutput
from alliance_eval.resources import resource_directory


def presets():
    source = resource_directory("examples") / "demo"
    return (
        records(source / "dialogue.jsonl"),
        json.loads((source / "responses.json").read_text()),
    )


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_demo_full_pipeline_offline_with_persistent_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = demo.run_demo(tmp_path, "illustration")
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["counts"]["completed_samples"] == 3
    assert manifest["counts"]["failed_calls"] == 0
    assert manifest["usage"]["requests"] == 43
    assert manifest["model"]["dialect"] == "preset_demo"
    assert manifest["model"]["parameters"]["network_calls"] is False
    inputs = tmp_path / "data" / "demo" / "illustration"
    assert {p.name for p in inputs.iterdir()} == {"dialogue.jsonl", "sample_plan.json", "responses.json"}
    assert manifest["sample_plan_sha256"] == hashlib.sha256((inputs / "sample_plan.json").read_bytes()).hexdigest()
    assert manifest["model"]["parameters"]["responses_sha256"] == hashlib.sha256((inputs / "responses.json").read_bytes()).hexdigest()
    dialogues, _ = presets()
    assert len(dialogues) == len(manifest["samples"]) == 3
    assert len({dialogue["dialogue_id"] for dialogue in dialogues}) == 3
    assert len({dialogue["turns"][0]["text"] for dialogue in dialogues}) == 3
    for sample, dialogue in zip(manifest["samples"], dialogues, strict=True):
        assert sample["dialogue_id"] == dialogue["dialogue_id"]
        assert sample["turn_count"] == len(dialogue["turns"])
        assert sample["source_sha256"] == hashlib.sha256((inputs / "dialogue.jsonl").read_bytes()).hexdigest()
        assert sample["annotation_complete_by_core"] == {"Bond": True, "Task": True}
        assert sample["summary_status_by_core"] == {"Bond": "complete", "Task": "complete"}
    turns = records(run / "turn_annotations.jsonl")
    summaries = records(run / "summary_annotations.jsonl")
    activities = records(run / "activity_identification.jsonl")
    assert len(activities) == 20
    assert not (run / "task_identification.jsonl").exists()
    for sample in manifest["samples"]:
        registered = [activity for row in activities if row["sample_id"] == sample["sample_id"]
                      for activity in row["new_activities"]]
        assert len(registered) == 1
        assert registered[0]["activity_id"] == "ACTIVITY_01"
        references = [result["activity_reference"] for row in turns + summaries
                      if row["sample_id"] == sample["sample_id"]
                      for result in row.get("findings", row.get("results", []))
                      if result["activity_reference"] is not None]
        assert references and all(reference == registered[0] for reference in references)
    assert len(turns) == 20
    assert len(summaries) == 3
    assert all(len(summary["results"]) == 4 for summary in summaries)
    core_records = records(run / "core_evidence.jsonl")
    assert core_records
    assert {row["core"] for row in core_records} == {"Bond", "Task"}
    assert not records(run / "validation_errors.jsonl")
    for sample, dialogue in zip(manifest["samples"], dialogues, strict=True):
        texts = {t["turn_id"]: t["text"] for t in dialogue["turns"]}
        findings = [f for row in turns if row["sample_id"] == sample["sample_id"] for f in row["findings"]]
        summary = next(row for row in summaries if row["sample_id"] == sample["sample_id"])
        assert findings
        for finding in findings + summary["results"]:
            assert finding["reason"].startswith("Preset illustration:")
            for span in finding["evidence"] + finding.get("counterevidence", []):
                assert span["quote"] in texts[span["turn_id"]]


def test_repeat_demo_preserves_previous_runs(tmp_path):
    first = demo.run_demo(tmp_path)
    before = {p.name: p.read_bytes() for p in first.iterdir()}
    second = demo.run_demo(tmp_path)
    assert first != second
    assert {p.name: p.read_bytes() for p in first.iterdir()} == before
    with pytest.raises(FileExistsError):
        demo.run_demo(tmp_path, first.name)
    assert {p.name: p.read_bytes() for p in first.iterdir()} == before


@pytest.mark.parametrize("run_id", ["", " ", ".", "..", "../escape", "nested/path", "bad\\name"])
def test_demo_rejects_unsafe_or_empty_names_before_writing(tmp_path, run_id):
    with pytest.raises(ValueError, match="run_id"):
        demo.run_demo(tmp_path, run_id)
    assert not list(tmp_path.iterdir())


def test_demo_never_overwrites_existing_input_directory(tmp_path):
    existing = tmp_path / "data" / "demo" / "previous"
    existing.mkdir(parents=True)
    (existing / "marker").write_text("preserve")
    with pytest.raises(FileExistsError):
        demo.run_demo(tmp_path, "previous")
    assert (existing / "marker").read_text() == "preserve"
    assert not (tmp_path / "runs").exists()


def test_presets_reject_changed_text_call_order_and_missing_entries():
    dialogues, responses = presets()
    first_id = dialogues[0]["dialogue_id"]
    with pytest.raises(ValueError, match="two presets"):
        demo.DemoJudge(dialogues, {**responses, first_id: responses[first_id][:-1]})
    backend = demo.DemoJudge(dialogues, responses)
    with pytest.raises(RuntimeError, match="every preset"):
        backend.ensure_consumed()
    with pytest.raises(RuntimeError, match="call order"):
        backend.generate("", json.dumps({"current_turn_id": "wrong"}), ActivityIdentificationLLMOutput)
    with pytest.raises(RuntimeError, match="modified dialogue"):
        backend.generate("", json.dumps({"current_turn_id": "demo-01", "visible_turns": []}), ActivityIdentificationLLMOutput)


def test_presets_reject_mismatched_dialogue_groups_and_cross_sample_text():
    dialogues, responses = presets()
    with pytest.raises(ValueError, match="match exactly"):
        demo.DemoJudge(dialogues[:-1], responses)
    with pytest.raises(ValueError, match="match exactly"):
        demo.DemoJudge([dialogues[0], dialogues[0]], responses)
    backend = demo.DemoJudge(dialogues, responses)
    with pytest.raises(RuntimeError, match="modified dialogue"):
        backend.generate("", json.dumps({
            "current_turn_id": dialogues[0]["turns"][0]["turn_id"],
            "visible_turns": [{key: dialogues[1]["turns"][0][key] for key in ("turn_id", "role", "text")}],
        }), ActivityIdentificationLLMOutput)


def test_cli_demo_without_permission_or_model_credentials(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["alliance-eval", "demo", "--workspace", str(tmp_path), "--run-id", "cli-demo"])
    assert cli.main() == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "runs" / "cli-demo")
