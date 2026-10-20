"""Offline App tests for paths, explicit execution, saved results and statistics.

Fixtures are fictional and network access is blocked. These tests check software
behaviour, not the accuracy or clinical validity of model judgements.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import socket
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import requests
from streamlit.testing.v1 import AppTest

from alliance_eval import app, evaluate, pipeline


# Allow bounded time for the initial table dependencies to load.
APP_TEST_TIMEOUT_SECONDS = 10


@pytest.fixture(autouse=True)
def offline_app(monkeypatch, tmp_path):
    """App tests use no real credentials and reject HTTP or socket connections."""

    for name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "ALLIANCE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ALLIANCE_EVAL_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(app, "WORKSPACE_ROOT", tmp_path)

    def deny_network(*args, **kwargs):
        raise AssertionError("Network is forbidden in product App tests")

    monkeypatch.setattr(httpx.Client, "send", deny_network)
    monkeypatch.setattr(httpx.AsyncClient, "send", deny_network)
    monkeypatch.setattr(requests.Session, "send", deny_network)
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(
        pipeline, "check_runtime_version",
        lambda: {"source_sha256": {"alliance_eval/pipeline.py": "a" * 64}},
    )


@pytest.fixture
def prepared_plan(tmp_path):
    """Original test sentences, not research samples or gold-standard annotations."""

    prepared = tmp_path / "data" / "processed"
    prepared.mkdir(parents=True)
    dialogue = {
        "dialogue_id": "fictional-app-dialogue",
        "turns": [
            {"turn_id": "u1", "turn_index": 0, "role": "client", "text": "I would like to organise my notes."},
            {"turn_id": "u2", "turn_index": 1, "role": "counsellor", "text": "Which part would you like to begin with?"},
        ],
    }
    (prepared / "dialogues.jsonl").write_text(json.dumps(dialogue) + "\n", encoding="utf-8")
    plan = tmp_path / "data" / "plans" / "sample_plan.json"
    plan.parent.mkdir()
    plan.write_text(json.dumps({
        "sample_version": "fictional-app-fixture",
        "items": [{"source_file": "../processed/dialogues.jsonl", "dialogue_id": dialogue["dialogue_id"],
                   "start_turn_index": None, "end_turn_index": None}],
    }), encoding="utf-8")
    return plan


def prepared_source(plan: Path) -> Path:
    return plan.parent.parent / "processed" / "dialogues.jsonl"


def labelled(elements, label):
    return next(element for element in elements if element.label == label)


def displayed_evidence_quotes(elements):
    return [html.unescape(quote) for element in elements for quote in
            re.findall(r'<span class="evidence-quote"[^>]*>(.*?)</span>', element.value, re.DOTALL)]


@pytest.mark.parametrize("heading", ["Evidence", "Counterevidence"])
def test_evidence_bullets_escape_text_and_use_neutral_id_colour(heading):
    def page(heading, spans):
        from alliance_eval.app import _render_evidence, display_result
        import streamlit as st
        _render_evidence(spans, heading)
        st.markdown(display_result("rated", "Example", badge=True))

    spans = [{"turn_id": 'u1</code><script>bad</script>',
              "quote": 'A literal <b>quote</b> & [not a link](https://example.invalid).\nSecond line.'}]
    before = json.dumps(spans)
    at = AppTest.from_function(page, args=(heading, spans)).run()
    assert not at.exception
    lines = [item.value for item in at.markdown if 'class="evidence-turn-id"' in item.value]
    assert len(lines) == 1
    assert lines[0].startswith('<ul><li>') and lines[0].endswith('</li></ul>')
    assert re.findall(r'style="([^"]*)"', lines[0]) == ['color:#506784;']
    assert '</code> — <span class="evidence-quote"' in lines[0]
    assert '<br' not in lines[0] and '<script>' not in lines[0] and '<b>' not in lines[0]
    assert html.escape(spans[0]["turn_id"]) in lines[0]
    assert displayed_evidence_quotes(at.markdown) == [spans[0]["quote"]]
    assert any(':green-badge[rated]' in item.value for item in at.markdown)
    assert json.dumps(spans) == before


def test_legacy_activity_view_preserves_evidence_and_core(tmp_path):
    legacy = {
        "sample_id": "s1", "core": "Task", "core_element": "agreement on the tasks",
        "source_dimension": "Task agreement/disagreement", "source_level": "task-level summary",
        "task_reference": {"task_id": "TASK_01", "task_label": "Discuss the task list"},
        "rating_label": "Clear task agreement", "status": "rated",
        "reason": "The original reason says Task and TASK_01.",
        "evidence": [{"turn_id": "u1", "quote": "My task list is long."}],
        "counterevidence": [],
    }
    path = tmp_path / "core_evidence.jsonl"
    path.write_text(json.dumps(legacy) + "\n")
    before = path.read_bytes()
    shown = app.read_jsonl(path)[0]
    assert shown["core"] == "Task"
    assert shown["core_element"] == "agreement on the activities"
    assert shown["source_dimension"] == "Activity agreement/disagreement"
    assert shown["source_level"] == "activity-level summary"
    assert shown["rating_label"] == "Clear activity agreement"
    assert shown["activity_reference"] == {
        "activity_id": "ACTIVITY_01", "activity_label": "Discuss the task list",
    }
    for field in ("reason", "evidence", "counterevidence", "status"):
        assert shown[field] == legacy[field]
    assert path.read_bytes() == before
    assert app._saved_activity_terms(shown) == shown


def test_view_does_not_merge_legacy_and_current_reference_fields(tmp_path):
    path = tmp_path / "turn_annotations.jsonl"
    path.write_text(json.dumps({"sample_id": "s1", "current_turn_id": "u1", "findings": [
        {"task_reference": None, "activity_reference": None},
    ]}) + "\n")
    before = path.read_bytes()
    warnings = []
    assert app.read_jsonl(path, warnings=warnings) == []
    assert "conflicting legacy and current" in warnings[0]
    assert path.read_bytes() == before


@pytest.mark.parametrize("declared_name", ["task_identification.jsonl", "activity_identification.jsonl"])
def test_legacy_run_app_uses_current_terms_without_rewriting(prepared_plan, tmp_path, declared_name):
    source = prepared_source(prepared_plan)
    run = tmp_path / "runs" / "legacy"
    run.mkdir(parents=True)
    reference = {"task_id": "TASK_01", "task_label": "Organise notes"}
    result = {"dimension": "Client engagement in the work", "task_reference": reference,
              "summary_status": "rated", "rating_label": "Very salient", "reason": "Saved reason.",
              "evidence": [{"turn_id": "u1", "quote": "I would like to organise my notes."}],
              "counterevidence": []}
    files = {
        "manifest.json": {
            "status": "completed", "counts": {}, "usage": {"requests": 0},
            "outputs": [declared_name, "turn_annotations.jsonl", "summary_annotations.jsonl"],
            "samples": [{"sample_id": "s1", "dialogue_id": "fictional-app-dialogue",
                         "source_file": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                         "annotation_complete_by_core": {"Bond": True, "Task": True},
                         "summary_status_by_core": {"Bond": "complete", "Task": "complete"}}],
        },
        "task_identification.jsonl": {"sample_id": "s1", "current_turn_id": "u1", "new_tasks": [reference]},
        "turn_annotations.jsonl": {"sample_id": "s1", "current_turn_id": "u1", "findings": []},
        "summary_annotations.jsonl": {"sample_id": "s1", "results": [result]},
        "validation_errors.jsonl": {
            "sample_id": "s1", "current_turn_id": "u2", "stage": "task_identification",
            "affected_cores": ["Task"], "error_type": "SavedError", "message": "Old Task error",
            "raw_response": '{"new_tasks": []}',
        },
    }
    for name, content in files.items():
        (run / name).write_text(json.dumps(content) + "\n")
    before = {path: path.read_bytes() for path in run.iterdir()}
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    assert not at.exception
    summary = labelled(at.tabs, "Summary")
    assert summary.dataframe[0].value.to_dict("records") == [{
        "dimension": "Task · Client engagement in the activity", "activity": "Organise notes", "result": "Very salient",
    }]
    assert any("ACTIVITY_01" in item.value for item in summary.markdown)
    assert any("Model response could not be accepted" in item.value for item in at.markdown)
    assert not any(item.value == "Old Task error" for item in at.markdown)
    assert not at.code
    if declared_name == "task_identification.jsonl":
        assert any(item.value == "**Activity identification**" for item in at.markdown)
        assert not any("recorded output file is missing" in item.value for item in at.warning)
    else:
        assert not any(item.value == "**Activity identification**" for item in at.markdown)
        assert any("activity_identification.jsonl: recorded output file is missing" in item.value for item in at.warning)
    assert {path: path.read_bytes() for path in run.iterdir()} == before


def test_default_plan_base_and_empty_run_directory(prepared_plan, tmp_path):
    assert app.validate_plan_input(prepared_plan, None) == 1
    assert app.list_run_dirs(tmp_path / "not-created") == []
    assert not (tmp_path / "not-created").exists()


def test_page_reruns_reuse_input_precheck(prepared_plan, monkeypatch):
    calls = []
    original = pipeline.load_selected_dialogues

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "load_selected_dialogues", counted)
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    assert not at.exception
    labelled(at.selectbox, "Provider").select("ollama").run()
    labelled(at.text_input, "Model ID").input("fictional-model").run()
    at.run()
    assert not at.exception
    assert calls == [1]
    source = prepared_source(prepared_plan)
    source.write_text("invalid source\n", encoding="utf-8")
    at.run()
    assert not at.exception
    assert calls == [1, 1]
    assert any("Input not ready" in warning.value for warning in at.warning)
    assert labelled(at.button, "Run All").disabled


@pytest.mark.parametrize("change", ["plan", "source", "replace", "delete"])
def test_precheck_cache_invalidates_on_file_changes(prepared_plan, monkeypatch, change):
    calls = []
    original = app.validate_plan_input

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(app, "validate_plan_input", counted)
    assert app._preview_plan_input(prepared_plan, None) == 1
    assert app._preview_plan_input(prepared_plan, None) == 1
    assert calls == [1]
    source = prepared_source(prepared_plan)
    target = prepared_plan if change == "plan" else source
    if change == "delete":
        source.unlink()
        with pytest.raises(FileNotFoundError):
            app._preview_plan_input(prepared_plan, None)
        return
    before = target.stat()
    if change == "replace":
        replacement = source.with_suffix(".replacement")
        replacement.write_bytes(source.read_bytes())
        replacement.replace(source)
    else:
        text = target.read_text(encoding="utf-8")
        target.write_text(text.replace("fixture", "updated") if change == "plan"
                          else text.replace("notes", "plans"), encoding="utf-8")
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert target.stat().st_size == before.st_size
    assert app._preview_plan_input(prepared_plan, None) == 1
    assert calls == [1, 1]


def test_precheck_does_not_cache_files_changed_during_read(prepared_plan, monkeypatch):
    original = app.validate_plan_input

    def changed_during_read(*args):
        count = original(*args)
        source = prepared_source(prepared_plan)
        source.write_text(source.read_text().replace("notes", "plans"))
        return count

    monkeypatch.setattr(app, "validate_plan_input", changed_during_read)
    with pytest.raises(OSError, match="changed while being read"):
        app._preview_plan_input(prepared_plan, None)


def test_run_all_revalidates_inputs_after_warm_page_precheck(prepared_plan, tmp_path):
    assert app._preview_plan_input(prepared_plan, None) == 1
    assert app._preview_plan_input(prepared_plan, None) == 1
    prepared_source(prepared_plan).write_text("invalid source\n")
    with pytest.raises(ValueError):
        pipeline.run_all(prepared_plan, SimpleNamespace(), tmp_path / "runs", "must-not-start")
    assert not (tmp_path / "runs").exists()


def test_saved_output_cache_preserves_warnings_copies_and_refreshes(tmp_path, monkeypatch):
    path = tmp_path / "saved.jsonl"
    path.write_text('{"sample_id": "first"}\ninvalid\n')
    calls = []
    original = app.read_jsonl

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(app, "read_jsonl", counted)
    warnings = []
    records = app._preview_saved_output(path, warnings)
    assert records == [{"sample_id": "first"}]
    assert len(warnings) == 1
    records[0]["sample_id"] = "mutated copy"
    repeated_warnings = []
    assert app._preview_saved_output(path, repeated_warnings) == [{"sample_id": "first"}]
    assert repeated_warnings == warnings
    assert calls == [1]
    path.write_text('{"sample_id": "first"}\n{"sample_id": "second"}\n')
    repaired_warnings = []
    assert len(app._preview_saved_output(path, repaired_warnings)) == 2
    assert repaired_warnings == []
    assert calls == [1, 1]
    path.unlink()
    assert app._preview_saved_output(path, []) == []


def test_dialogue_cache_retains_checksum_binding_and_refreshes(prepared_plan, monkeypatch):
    source = prepared_source(prepared_plan)
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    calls = []
    original = app.load_dialogue

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(app, "load_dialogue", counted)
    record = app._preview_dialogue(str(source), "fictional-app-dialogue", expected_sha256=expected)
    record["turns"][0]["text"] = "mutated copy"
    assert app._preview_dialogue(str(source), "fictional-app-dialogue", expected_sha256=expected) != record
    assert calls == [1]
    with pytest.raises(ValueError, match="Source checksum mismatch"):
        app._preview_dialogue(str(source), "fictional-app-dialogue", expected_sha256="0" * 64)
    assert calls == [1, 1]
    before = source.stat()
    source.write_text(source.read_text().replace("notes", "plans"))
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="Source checksum mismatch"):
        app._preview_dialogue(str(source), "fictional-app-dialogue", expected_sha256=expected)
    assert calls == [1, 1, 1]
    updated = app._preview_dialogue(str(source), "fictional-app-dialogue")
    assert "plans" in updated["turns"][0]["text"]
    source.unlink()
    with pytest.raises(FileNotFoundError):
        app._preview_dialogue(str(source), "fictional-app-dialogue", expected_sha256=expected)


def test_plan_dropdown_only_lists_prepared_plans(prepared_plan):
    (prepared_plan.parent / "manifest.json").write_text('{"datasets": {}}', encoding="utf-8")
    (prepared_plan.parent / "damaged.json").write_text('{"broken":', encoding="utf-8")
    assert app.list_sample_plans(prepared_plan.parent) == [prepared_plan]
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    assert labelled(at.selectbox, "Sample plan").value == prepared_plan
    assert labelled(at.selectbox, "Sample plan").options == ["sample_plan"]
    assert not labelled(at.selectbox, "Sample plan").disabled
    assert not any(field.label == "Custom plan path" for field in at.text_input)
    assert not any("Prepared plan:" in caption.value for caption in at.caption)
    assert not any(field.label == "Data root (optional)" for field in at.text_input)
    assert not any("Paths are on this computer" in caption.value for caption in at.caption)
    assert not any("Local input validated" in success.value for success in at.success)
    assert not any("Input not ready" in warning.value for warning in at.warning)
    assert not any("Runtime verified" in caption.value for caption in at.caption)
    second_plan = prepared_plan.with_name("second.json")
    second_plan.write_bytes(prepared_plan.read_bytes())
    at.run()
    labelled(at.selectbox, "Sample plan").select(second_plan).run()
    assert labelled(at.selectbox, "Sample plan").value == second_plan
    assert not at.exception


def test_no_prepared_plan_blocks_model_run_but_keeps_demo_available(tmp_path):
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    assert not at.exception
    assert any("sample plan" in info.value.lower() and "preset demo" in info.value.lower()
               for info in at.info)
    assert not any("Input not ready" in warning.value for warning in at.warning)
    assert labelled(at.button, "Run All").disabled
    assert labelled(at.selectbox, "Sample plan").options == []
    assert labelled(at.selectbox, "Sample plan").disabled
    assert not any(field.label == "Custom plan path" for field in at.text_input)
    assert not labelled(at.button, "Run preset demo").disabled
    labelled(at.selectbox, "Provider").select("ollama").run()
    labelled(at.text_input, "Model ID").input("fictional-model").run()
    assert not at.exception
    assert labelled(at.button, "Run All").disabled
    assert not (tmp_path / "runs").exists()


def test_statistics_split_levels_diagnostics_and_call_denominator():
    manifest = {
        "counts": {"planned_samples": 3, "processed_samples": 2, "completed_samples": 1, "failed_calls": 1},
        "usage": {"requests": 8},
        "samples": [
            {"sample_id": "s1", "annotation_complete_by_core": {"Bond": True, "Task": True},
             "summary_status_by_core": {"Bond": "complete", "Task": "complete"}},
            {"sample_id": "s2", "annotation_complete_by_core": {"Bond": True, "Task": False},
             "summary_status_by_core": {"Bond": "complete", "Task": "diagnostic"}},
            {"sample_id": "s3", "summary_status": "complete"},
        ],
    }
    turns = [{"sample_id": "s2", "findings": [
        {"dimension": "Grounded content uptake", "finding_status": "rated", "rating_label": "Grounded uptake"},
        {"dimension": "Accurate Empathy", "finding_status": "evidence_only", "rating_label": None},
    ]}, {"sample_id": "s1", "findings": []}]
    summary = [{"sample_id": sample, "results": [
        {"dimension": "Accurate Empathy", "summary_status": "insufficient_evidence", "rating_label": None},
        {"dimension": "Client engagement in the activity", "summary_status": "no_evaluable_opportunity", "rating_label": None},
    ]} for sample in ("s1", "s2", "s3")]
    statistics = app.run_statistics(manifest, turns, summary)
    assert statistics["failed_call_rate"] == 1 / 8
    assert statistics["failed_calls"] == 1
    assert statistics["requests"] == 8
    assert statistics["excluded_results"] is True
    assert statistics["categories"] == [
        {"dimension": "Accurate Empathy", "status": "insufficient_evidence", "label": "—", "results": 2},
        {"dimension": "Client engagement in the activity", "status": "no_evaluable_opportunity", "label": "—", "results": 1},
        {"dimension": "Grounded content uptake", "status": "rated", "label": "Grounded uptake", "results": 1},
    ]
    assert not {"dimensions", "cores", "errors", "completion_rate"}.intersection(statistics)


def test_statistics_count_result_records_not_unique_samples(prepared_plan, tmp_path):
    plan = json.loads(prepared_plan.read_text(encoding="utf-8"))
    original_item = plan["items"][0]
    plan["items"] = [
        {**original_item, "start_turn_index": 0, "end_turn_index": 0},
        {**original_item, "start_turn_index": 1, "end_turn_index": 1},
        {**original_item, "start_turn_index": 0, "end_turn_index": 0},
    ]
    prepared_plan.write_text(json.dumps(plan), encoding="utf-8")
    assert app.validate_plan_input(prepared_plan, None) == 3
    manifest = {
        "status": "completed",
        "counts": {"planned_samples": 3, "processed_samples": 3, "completed_samples": 3, "failed_calls": 0},
        "usage": {"requests": 3},
        "samples": [
            {"sample_id": sample_id, "dialogue_id": original_item["dialogue_id"],
             "source_file": str(prepared_source(prepared_plan)),
             "annotation_complete_by_core": {"Bond": True, "Task": True},
             "summary_status_by_core": {"Bond": "complete", "Task": "complete"}}
            for sample_id in ("S1", "S2", "S3")
        ],
    }
    finding = {"dimension": "Grounded content uptake", "finding_status": "rated", "rating_label": "Grounded uptake"}
    turns = [{"sample_id": sample_id, "findings": [finding.copy()]} for sample_id in ("S1", "S1", "S3")]
    statistics = app.run_statistics(manifest, turns, [])
    assert statistics["categories"] == [
        {"dimension": "Grounded content uptake", "status": "rated", "label": "Grounded uptake", "results": 3},
    ]
    assert statistics["excluded_results"] is False
    assert statistics["failed_call_rate"] == 0
    run = tmp_path / "runs" / "repeated-plan-items"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    assert not at.exception
    assert not any(metric.label == "Planned samples" for metric in at.metric)
    assert labelled(at.metric, "Failed requests").value == "0 / 3 (0.0%)"
    assert labelled(at.metric, "Fully validated samples").value == "3 / 3"
    assert any(item.value == "100% of planned samples · execution completeness" for item in at.caption)
    assert labelled(at.metric, "Request").value == "3"
    assert not any(metric.label == "Planned dialogues" for metric in at.metric)
    assert not any("Local input validated" in success.value for success in at.success)


@pytest.mark.parametrize("counts,usage", [({}, {}), ({"planned_samples": 0, "completed_samples": 0}, {"requests": 0}),
                                          ({"planned_samples": 1, "completed_samples": 2, "failed_calls": 3}, {"requests": 2})])
def test_statistics_do_not_invent_denominators_or_invalid_rates(counts, usage):
    result = app.run_statistics({"counts": counts, "usage": usage}, [], [])
    assert result["failed_call_rate"] is None
    assert result["categories"] == []


@pytest.mark.parametrize("counts,value,caption", [
    ({"completed_samples": 14, "planned_samples": 40, "processed_samples": 40},
     "14 / 40", "35% of planned samples · execution completeness"),
    ({"completed_samples": 1, "planned_samples": 40, "processed_samples": 5},
     "1 / 40", "2.5% of planned samples · execution completeness"),
    ({"completed_samples": 0, "planned_samples": 40},
     "0 / 40", "0% of planned samples · execution completeness"),
    ({}, "Not recorded", "Completion rate unavailable"),
    ({"completed_samples": 3}, "3", "Completion rate unavailable"),
    ({"completed_samples": 0, "planned_samples": 0}, "0", "Completion rate unavailable"),
    ({"completed_samples": 2, "planned_samples": 1}, "2", "Completion rate unavailable"),
    ({"completed_samples": -1, "planned_samples": 4}, "Not recorded", "Completion rate unavailable"),
    ({"completed_samples": True, "planned_samples": 4}, "Not recorded", "Completion rate unavailable"),
    ({"completed_samples": 1, "planned_samples": "4"}, "1", "Completion rate unavailable"),
])
def test_sample_completion_uses_recorded_plan_and_rejects_invalid_rates(counts, value, caption):
    assert app.sample_completion_display(counts) == (value, caption)


def test_statistics_scope_filters_categories_but_not_execution_counts():
    manifest = {
        "counts": {"failed_calls": 1}, "usage": {"requests": 8},
        "samples": [
            {"sample_id": "a", "annotation_complete_by_core": {"Bond": True, "Task": True},
             "summary_status_by_core": {"Bond": "complete", "Task": "complete"}},
            {"sample_id": "b", "annotation_complete_by_core": {"Bond": False, "Task": True},
             "summary_status_by_core": {"Bond": "diagnostic", "Task": "complete"}},
        ],
    }
    finding = {"dimension": "Grounded content uptake", "finding_status": "rated",
               "rating_label": "Substantive grounded uptake"}
    turns = [{"sample_id": sample_id, "findings": [finding]} for sample_id in ("a", "a", "b")]
    turns.append({"sample_id": "b", "findings": [
        {"dimension": "Accurate Empathy", "finding_status": "evidence_only", "rating_label": None},
    ]})
    summaries = [{"sample_id": sample_id, "results": [
        {"dimension": "Accurate Empathy", "summary_status": "rated", "rating_label": "AE-5"},
        *[{"dimension": "Client engagement in the activity", "summary_status": "rated",
           "rating_label": "Very salient", "activity_reference": {"activity_id": activity}}
          for activity in activities],
    ]} for sample_id, activities in (("a", ["ACTIVITY_01", "ACTIVITY_02"]), ("b", ["ACTIVITY_01"]))]
    at = AppTest.from_string(
        "import streamlit as st\nfrom alliance_eval.app import run_statistics, render_run_statistics\n"
        f"manifest = {manifest!r}\nturns = {turns!r}\nsummaries = {summaries!r}\n"
        "selected = st.selectbox('Sample', ['a', 'b'])\n"
        "render_run_statistics(run_statistics(manifest, turns, summaries), sample_label=selected,\n"
        "    sample_statistics=run_statistics(manifest, [r for r in turns if r['sample_id'] == selected],\n"
        "        [r for r in summaries if r['sample_id'] == selected]))\n",
        default_timeout=APP_TEST_TIMEOUT_SECONDS,
    ).run()

    def maximum_tick():
        return max(json.loads(at.get("arrow_vega_lite_chart")[0].proto.spec)["encoding"]["x"]["axis"]["values"])

    labelled(at.selectbox, "Dimension").select("Grounded content uptake").run()
    assert maximum_tick() == 3
    assert len(at.warning) == 1
    labelled(at.radio, "Result scope").set_value("Current sample").run()
    assert maximum_tick() == 2
    assert not at.warning
    assert any(item.value == "Turn-level results · Current sample · a" for item in at.caption)
    labelled(at.selectbox, "Sample").select("b").run()
    assert labelled(at.selectbox, "Dimension").value == "Grounded content uptake"
    assert maximum_tick() == 1
    assert len(at.warning) == 1
    assert any(item.value == "Turn-level results · Current sample · b" for item in at.caption)
    assert labelled(at.metric, "Failed requests").value == "1 / 8 (12.5%)"
    assert any(item.value == "Run-level execution · All samples in this run" for item in at.caption)
    labelled(at.selectbox, "Dimension").select("Accurate Empathy").run()
    assert not at.get("arrow_vega_lite_chart")
    assert any("does not establish absence or a low score" in item.value for item in at.info)
    labelled(at.selectbox, "Dimension").select("Client engagement in the activity").run()
    assert maximum_tick() == 1
    assert any("Activity-level summary · Current sample · b" == item.value for item in at.caption)
    labelled(at.selectbox, "Sample").select("a").run()
    assert maximum_tick() == 2
    labelled(at.radio, "Result scope").set_value("All samples in run").run()
    assert maximum_tick() == 3
    assert not at.exception


def test_statistics_preserve_partial_turn_results_without_counting_evidence_only():
    manifest = {"samples": [{"sample_id": "s1", "annotation_complete_by_core": {"Bond": False},
                             "summary_status_by_core": {"Bond": "diagnostic"}}]}
    turns = [{"sample_id": "s1", "findings": [
        {"dimension": "Grounded content uptake", "finding_status": "rated", "rating_label": "Substantive grounded uptake"},
        {"dimension": "Accurate Empathy", "finding_status": "evidence_only", "rating_label": None},
    ]}]
    stats = app.run_statistics(manifest, turns, [])
    assert sum(row["results"] for row in stats["categories"]) == 1
    assert stats["excluded_results"] is False
    summaries = [{"sample_id": "s1", "results": [
        {"dimension": "Accurate Empathy", "summary_status": "rated", "rating_label": "Accurate empathy"},
    ]}]
    excluded = app.run_statistics(manifest, turns, summaries)
    assert excluded["categories"] == stats["categories"]
    assert excluded["excluded_results"] is True


@pytest.mark.parametrize("annotation,summary,expected", [
    ({"Bond": True, "Task": True}, {"Bond": "complete", "Task": "complete"}, None),
    ({"Bond": False, "Task": True}, {"Bond": "diagnostic", "Task": "failed"},
     "Bond annotations incomplete; Task summary failed. See Validation details."),
    ({"Bond": True, "Task": True}, {"Bond": "complete", "Task": "failed"},
     "Task summary failed. See Validation details."),
    ({"Bond": True, "Task": False}, {"Bond": "complete", "Task": "not_run"},
     "Task annotations incomplete; Task summary not run. See Validation details."),
    ({"Bond": True}, {"Bond": "complete"},
     "Task annotations unconfirmed; Task summary unconfirmed. See Validation details."),
    ({"Bond": True, "Task": True}, {"Bond": "complete", "Task": "diagnostic"},
     "Task summary diagnostic only. See Validation details."),
    ({"Bond": True, "Task": True}, {"Bond": "complete", "Task": []},
     "Task summary unconfirmed. See Validation details."),
    ({}, "unreadable", "Sample completeness cannot be confirmed. See Validation details."),
])
def test_sample_status_hides_complete_and_reports_only_affected_groups(annotation, summary, expected):
    sample = {"annotation_complete_by_core": annotation, "summary_status_by_core": summary}
    at = AppTest.from_string(
        f"from alliance_eval.app import render_sample_validation_status\nrender_sample_validation_status({sample!r})",
        default_timeout=APP_TEST_TIMEOUT_SECONDS,
    ).run()
    assert not at.exception
    assert not at.caption
    assert not at.info
    assert [warning.value for warning in at.warning] == ([] if expected is None else [expected])


def test_legacy_diagnostic_sample_keeps_short_warning_without_inventing_affected_group():
    at = AppTest.from_string(
        "from alliance_eval.app import render_sample_validation_status\n"
        "render_sample_validation_status({'summary_status': 'diagnostic'})",
        default_timeout=APP_TEST_TIMEOUT_SECONDS,
    ).run()
    assert not at.exception
    assert [warning.value for warning in at.warning] == ["Sample results incomplete. See Validation details."]


@pytest.mark.parametrize("max_count,row_count", [(1, 1), (15, 3), (250, 3), (276, 3), (732, 3), (12345, 4)])
def test_statistics_chart_centres_labels_and_reserves_space_for_counts(max_count, row_count):
    statistics = {
        "requests": 0, "failed_calls": 0, "failed_call_rate": None, "excluded_results": False,
        "categories": [
            {"dimension": "Activity and responsibility clarification", "status": "rated",
             "label": f"Clear activity and applicable responsibilities {index}", "results": max_count - index}
            for index in range(row_count)
        ],
    }
    at = AppTest.from_string(
        f"from alliance_eval.app import render_run_statistics\nrender_run_statistics({statistics!r})",
        default_timeout=APP_TEST_TIMEOUT_SECONDS,
    ).run()
    assert not at.exception
    chart = at.get("arrow_vega_lite_chart")[0]
    spec = json.loads(chart.proto.spec)
    assert spec["encoding"]["y"]["axis"]["labelFlush"] is False
    assert spec["encoding"]["y"]["axis"]["labelBaseline"] == "middle"
    assert spec["encoding"]["y"]["axis"]["labelOverlap"] is False
    assert spec["layer"][0]["mark"] == {"type": "bar", "size": 24}
    assert spec["layer"][1]["mark"] == {"type": "text", "align": "left", "baseline": "middle", "dx": 5}
    assert spec["padding"]["right"] >= len(str(max_count)) * 8 + 12
    assert spec["height"] >= row_count * 36 + 70
    assert spec["encoding"]["x"]["axis"]["values"] == list(range(0, max_count + 1, max(1, (max_count + 9) // 10)))


def test_statistics_page_switches_dimension_without_extra_tables():
    statistics = {
        "requests": 35, "failed_calls": 2, "failed_call_rate": 2 / 35, "excluded_results": True,
        "categories": [
            {"dimension": "Grounded content uptake", "status": "rated", "label": "Substantive grounded uptake", "results": 6},
            {"dimension": "Grounded content uptake", "status": "rated", "label": "Mechanical or surface uptake", "results": 2},
            {"dimension": "Client engagement in the activity", "status": "no_evaluable_opportunity", "label": "—", "results": 1},
        ],
    }
    at = AppTest.from_string(
        f"from alliance_eval.app import render_run_statistics\nrender_run_statistics({statistics!r})",
        default_timeout=APP_TEST_TIMEOUT_SECONDS,
    ).run()
    assert not at.exception
    assert labelled(at.metric, "Failed requests").value == "2 / 35 (5.7%)"
    assert len(at.metric) == 1
    assert not at.dataframe
    assert len(at.get("arrow_vega_lite_chart")) == 1
    axis = json.loads(at.get("arrow_vega_lite_chart")[0].proto.spec)["encoding"]["x"]["axis"]
    assert axis["values"] == list(range(7))
    assert axis["format"] == "d"
    assert json.loads(at.get("arrow_vega_lite_chart")[0].proto.spec)["encoding"]["y"]["axis"]["labelOverlap"] is False
    assert labelled(at.selectbox, "Dimension").value == "Grounded content uptake"
    assert any("Turn-level results" in caption.value for caption in at.caption)
    assert len(at.warning) == 1
    labelled(at.selectbox, "Dimension").select("Client engagement in the activity").run()
    assert not at.exception
    assert any("Activity-level summary" in caption.value for caption in at.caption)
    assert len(at.get("arrow_vega_lite_chart")) == 1
    axis = json.loads(at.get("arrow_vega_lite_chart")[0].proto.spec)["encoding"]["x"]["axis"]
    assert axis["values"] == [0, 1]
    labelled(at.selectbox, "Dimension").select("Accurate Empathy").run()
    assert not at.exception
    assert any("Sample-level summary" in caption.value for caption in at.caption)
    assert len(at.get("arrow_vega_lite_chart")) == 0
    assert any("No eligible results" in info.value for info in at.info)


@pytest.mark.parametrize("saved_status,display_status", [
    ("running", "Running"), ("completed", "Complete"),
    ("completed_with_errors", "Errors"), ("aborted", "aborted"),
    (None, "Not recorded"), ("legacy-status", "legacy-status"),
])
def test_run_status_help_explains_saved_states_without_rewriting(tmp_path, saved_status, display_status):
    run = tmp_path / "runs" / "fictional-status-check"
    run.mkdir(parents=True)
    manifest = run / "manifest.json"
    content = {"counts": {}, "samples": []}
    if saved_status is not None:
        content["status"] = saved_status
    manifest.write_text(json.dumps(content), encoding="utf-8")
    before = manifest.read_bytes()
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    assert not at.exception
    status = labelled(at.metric, "Status")
    assert status.value == display_status
    for label in ("Running", "Complete", "Errors", "aborted"):
        assert f"**{label}**" in status.help
    assert "Not recorded" not in status.help
    assert "Other values" not in status.help
    assert "not a live process check" in status.help
    assert "not a guarantee of correct judgements" in status.help
    assert "valid results are retained" in status.help
    assert manifest.read_bytes() == before


def test_broken_run_and_output_files_warn_without_rewriting(tmp_path):
    run = tmp_path / "runs" / "broken-run"
    run.mkdir(parents=True)
    manifest = run / "manifest.json"
    manifest.write_text('{"status":', encoding="utf-8")
    before = manifest.read_bytes()
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    assert not at.exception
    assert any("Saved run cannot be read" in error.value for error in at.error)
    assert manifest.read_bytes() == before
    manifest.write_text(json.dumps({"status": "running", "counts": {}, "samples": []}), encoding="utf-8")
    output = run / "turn_annotations.jsonl"
    output.write_text('not json\n{"sample_id":"s1","findings":null}\n', encoding="utf-8")
    before = output.read_bytes()
    at.run()
    assert not at.exception
    assert sum("Saved output excluded" in warning.value for warning in at.warning) == 2
    assert output.read_bytes() == before


@pytest.mark.parametrize("dialect,response_format,parameters", [
    ("openai_compatible", "json_object", {"reasoning_effort": "low"}),
    ("ollama", "json_schema", {"think": True, "seed": 123, "num_ctx": 8192}),
])
def test_named_connection_credentials_and_parameters_stay_in_snapshot(
    prepared_plan, tmp_path, monkeypatch, dialect, response_format, parameters,
):
    (tmp_path / "providers.json").write_text(json.dumps({"My model": {
        "base_url": "https://model.example/v1", "model_id": "fixture-model",
        "dialect": dialect, "response_format": response_format, "parameters": parameters,
    }}), encoding="utf-8")
    settings = []
    monkeypatch.setattr(evaluate, "LocalHTTPJudge", lambda **kwargs: (
        settings.append(kwargs) or SimpleNamespace(close=lambda: None)
    ))
    monkeypatch.setattr(pipeline, "run_all", lambda **kwargs: None)
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    labelled(at.selectbox, "Provider").select("ollama").run()
    labelled(at.number_input, "Context window (num_ctx)").set_value(8192).run()
    labelled(at.number_input, "Seed").set_value(123).run()
    assert labelled(at.selectbox, "Provider").options == ["deepseek", "openai", "mistral", "ollama", "My model"]
    labelled(at.selectbox, "Provider").select("My model").run()
    assert not at.number_input
    assert not at.toggle
    assert labelled(at.text_input, "Base URL").disabled
    assert labelled(at.text_input, "Base URL").value == "https://model.example/v1"
    assert labelled(at.text_input, "API key (optional)").value == ""
    labelled(at.text_input, "API key (optional)").input("fictional-key-for-test").run()
    assert labelled(at.text_input, "Model ID").value == "fixture-model"
    labelled(at.text_input, "Model ID").input("").run()
    assert labelled(at.button, "Run All").disabled
    labelled(at.text_input, "Model ID").input("edited-model").run()
    at.run()
    assert settings == []
    labelled(at.button, "Run All").click().run()
    assert not at.exception
    assert settings[0]["api_key"] == "fictional-key-for-test"
    assert settings[0]["response_format"] == response_format
    assert settings[0]["parameters"] == parameters
    assert settings[0]["model_id"] == "edited-model"
    assert settings[0]["dialect"] == dialect
    assert settings[0]["base_url"] == "https://model.example/v1"
    at.run()
    assert len(settings) == 1
    assert not (tmp_path / "runs").exists()


def test_custom_connection_defaults_and_credential_isolation(prepared_plan, tmp_path, monkeypatch):
    monkeypatch.setenv("FICTIONAL_MODEL_KEY", "fictional-env-key")
    connections = {
        "Connection A": {"base_url": "https://a.example/v1", "api_key_env": "FICTIONAL_MODEL_KEY"},
        "Connection B": {"base_url": "https://b.example/v1"},
    }
    config_file = tmp_path / "providers.json"
    config_file.write_text(json.dumps(connections), encoding="utf-8")
    loaded = app.load_custom_providers(tmp_path)
    assert loaded["Connection B"] == {
        "base_url": "https://b.example/v1", "dialect": "openai_compatible",
        "model_id": "", "api_key_env": "", "parameters": {}, "response_format": "json_schema",
    }
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    labelled(at.selectbox, "Provider").select("Connection A").run()
    assert labelled(at.text_input, "Model ID").value == ""
    assert labelled(at.button, "Run All").disabled
    assert labelled(at.text_input, "API key").value == "fictional-env-key"
    labelled(at.text_input, "Model ID").input("fictional-model").run()
    assert not labelled(at.button, "Run All").disabled
    labelled(at.text_input, "API key").input("").run()
    assert labelled(at.button, "Run All").disabled
    labelled(at.text_input, "API key").input("fictional-entered-key-a").run()
    assert not labelled(at.button, "Run All").disabled
    labelled(at.selectbox, "Provider").select("Connection B").run()
    assert labelled(at.text_input, "API key (optional)").value == ""
    assert labelled(at.text_input, "Model ID").value == ""
    labelled(at.text_input, "API key (optional)").input("fictional-entered-key-b").run()
    labelled(at.text_input, "Model ID").input("fictional-model-b").run()
    connections["Connection B"]["base_url"] = "https://changed.example/v1"
    config_file.write_text(json.dumps(connections), encoding="utf-8")
    at.run()
    assert not at.exception
    assert labelled(at.text_input, "Base URL").value == "https://changed.example/v1"
    assert labelled(at.text_input, "API key (optional)").value == ""
    assert labelled(at.text_input, "Model ID").value == ""
    assert "fictional-entered-key" not in config_file.read_text(encoding="utf-8")
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("configuration", [
    [], {"openai": {"base_url": "https://model.example"}},
    {" My model": {"base_url": "https://model.example"}},
    {"My model": []}, {"My model": {}},
    *[{"My model": {"base_url": "https://model.example", **changes}} for changes in (
        {"dialect": "unsupported"}, {"api_key": "do-not-store-keys"},
        {"api_key_env": "not-an-environment-variable"}, {"parameters": []}, {"model_id": 42},
        {"parameters": {"Authorization": "do-not-store-keys"}},
        {"response_format": "auto"}, {"dialect": "ollama", "response_format": "json_object"},
        {"base_url": "file:///tmp/model"}, {"base_url": "https://user:password@model.example"},
        {"base_url": "https://model.example?key=do-not-store-keys"},
        {"base_url": "http://[invalid"}, {"base_url": "http://model.example:invalid"},
        {"base_url": "http://127.0.0.1:65536"}, {"base_url": "http://127.0.0.1:-1"},
    )],
])
def test_invalid_custom_connections_are_rejected(tmp_path, configuration):
    (tmp_path / "providers.json").write_text(json.dumps(configuration), encoding="utf-8")
    with pytest.raises(ValueError):
        app.load_custom_providers(tmp_path)


@pytest.mark.parametrize("content", ['{"broken":', '{"same": {}, "same": {}}', '{"My model": {}}'])
def test_broken_provider_configuration_blocks_run_but_not_viewer(prepared_plan, tmp_path, content):
    (tmp_path / "providers.json").write_text(content, encoding="utf-8")
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    labelled(at.selectbox, "Provider").select("ollama").run()
    labelled(at.text_input, "Model ID").input("fictional-model").run()
    assert not at.exception
    assert any("Provider configuration not ready" in error.value for error in at.error)
    assert labelled(at.button, "Run All").disabled
    assert not labelled(at.button, "Run preset demo").disabled
    assert not (tmp_path / "runs").exists()


def test_preset_demo_button_runs_without_model_credentials_and_opens_results(tmp_path, monkeypatch):
    def no_real_backend(*args, **kwargs):
        raise AssertionError("The preset demo must not construct a real model backend")

    for name in ("DeepSeekJudge", "OpenAIJudge", "LocalHTTPJudge"):
        monkeypatch.setattr(evaluate, name, no_real_backend)
    # Streamlit emits this warning once per process; do not let earlier tests mask it.
    monkeypatch.setattr("streamlit.elements.lib.policies._shown_default_value_warning", False)
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    assert not labelled(at.button, "Run preset demo").disabled
    assert labelled(at.button, "Run All").disabled
    assert not any("I have permission" in checkbox.label for checkbox in at.checkbox)
    labelled(at.button, "Run preset demo").click().run()
    assert not at.exception
    assert any("No model was called" in success.value for success in at.success)
    assert any("Preset demonstration" in warning.value and "No model inference" in warning.value
               for warning in at.warning)
    assert not any("Selected excerpt" in caption.value or "Selection boundary" in caption.value
                   for caption in at.caption)
    assert not any("default value" in warning.value and "Session State API" in warning.value
                   for warning in at.warning)
    run_dirs = app.list_run_dirs(tmp_path / "runs")
    assert len(run_dirs) == 1
    run = run_dirs[0]
    assert run.parent == tmp_path / "runs"
    assert not any(field.label == "Run directory" for field in at.text_input)
    assert labelled(at.selectbox, "Run").value == run
    assert labelled(at.selectbox, "Sample").value["dialogue_id"] == "fictional-scripted-demo-001"
    assert [tab.label for tab in at.tabs] == [
        "Dialogue & annotations", "Summary", "Core mapping",
    ]
    assert any(item.value == "**Turn findings: 11**" for item in at.tabs[0].markdown)
    assert any(expander.label == "Statistics" for expander in at.expander)
    assert not any(expander.label == "Moved source files (read-only)" for expander in at.expander)
    assert any("When my supervisor asks for an update" in text.value for text in at.markdown)
    assert len(labelled(at.selectbox, "Dimension").options) == 14
    summary_tab = next(tab for tab in at.tabs if tab.label == "Summary")
    assert summary_tab.dataframe[0].value.columns.tolist() == ["dimension", "activity", "result"]
    assert summary_tab.dataframe[0].value["dimension"].tolist() == [
        "Bond · Accurate Empathy", "Bond · Therapist accepts/validates Patient",
        "Bond · Mutual trust and respect", "Task · Client engagement in the activity",
    ]
    assert any(expander.label.startswith("Task · Client engagement in the activity · ")
               for expander in summary_tab.expander)
    assert any(item.value.startswith("**Task · Activity agreement/disagreement** · ")
               for item in at.tabs[0].markdown)
    assert not any("Preset illustration:" in item.value for item in summary_tab.markdown)
    assert not any("Preset illustration:" in item.value for item in at.caption)
    assert not any(expander.label == "Validation details" for expander in at.expander)
    assert not any("No validation errors" in item.value for item in at.success)
    assert len(at.get("arrow_vega_lite_chart")) == 1
    assert not any("excluded from these counts" in warning.value for warning in at.warning)
    assert not any(metric.label in {"Planned samples", "Processed samples", "Complete samples", "Execution completion"}
                   for metric in at.metric)
    assert labelled(at.metric, "Model").value == "demo"
    assert labelled(at.metric, "Request").value == "0"
    validated = labelled(at.metric, "Fully validated samples")
    assert validated.value == "3 / 3"
    assert "Activity identification, turn annotations and summary" in validated.help
    assert "no unresolved call or validation failures in either Bond or Task" in validated.help
    assert "Recovered connection failures remain in the request statistics and log" in validated.help
    assert "One sample is one plan entry" in validated.help
    assert "not judgement accuracy" in validated.help
    assert labelled(at.metric, "Failed requests").value == "—"
    assert "0 model requests" in labelled(at.metric, "Failed requests").help
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["model"]["dialect"] == "preset_demo"
    assert manifest["model"]["parameters"]["network_calls"] is False
    assert manifest["usage"]["requests"] == 43
    assert manifest["usage"]["input_tokens"] == manifest["usage"]["output_tokens"] == 0
    assert not any("I have permission" in checkbox.label for checkbox in at.checkbox)
    before = {path: path.read_bytes() for path in run.iterdir() if path.is_file()}
    first_turn = next(expander for expander in at.expander if expander.label.startswith("demo-01 ·"))
    assert first_turn.label == "demo-01 · Annotations · 0 findings"
    assert [item.value for item in first_turn.caption] == [
        "Cited by", "demo-02 · Bond · Grounded content uptake",
    ]
    fourth_turn = next(expander for expander in at.expander if expander.label.startswith("demo-04 ·"))
    assert fourth_turn.label == "demo-04 · Annotations · 0 findings"
    assert any("Activity identification" in item.value for item in fourth_turn.markdown)
    assert any(item.value == "demo-05 · Task · Activity agreement/disagreement" for item in fourth_turn.caption)
    assert not any("No new Activity identified" in item.value for item in at.caption)
    assert len(labelled(at.selectbox, "Sample").options) == 3
    labelled(at.radio, "Result scope").set_value("Current sample").run()
    for sample, expected_findings in zip(manifest["samples"][1:], (7, 8), strict=True):
        labelled(at.selectbox, "Sample").select(sample).run()
        assert not at.exception
        assert not any("Selected excerpt" in caption.value or "Selection boundary" in caption.value
                       for caption in at.caption)
        assert labelled(at.selectbox, "Sample").value["dialogue_id"] == sample["dialogue_id"]
        dialogue = app.load_dialogue(sample["source_file"], sample["dialogue_id"], app.recorded_data_root(manifest, None))
        assert any(
            dialogue["turns"][0]["text"] in html.unescape(text.value.replace("<mark>", "").replace("</mark>", ""))
            for text in at.markdown
        )
        assert not any("When my supervisor asks for an update" in text.value for text in at.markdown)
        assert not any(expander.label.startswith("demo-01 ·") for expander in at.expander)
        assert any(item.value == f"**Turn findings: {expected_findings}**" for item in at.tabs[0].markdown)
        assert len(at.dataframe[0].value) == 4
        summary_tab = next(tab for tab in at.tabs if tab.label == "Summary")
        assert summary_tab.dataframe[0].value.columns.tolist() == ["dimension", "activity", "result"]
        assert not any("Preset illustration:" in item.value for item in summary_tab.markdown)
        assert not any("Preset illustration:" in item.value for item in at.caption)
        assert not any(expander.label == "Validation details" for expander in at.expander)
        assert labelled(at.metric, "Fully validated samples").value == "3 / 3"
        assert labelled(at.metric, "Request").value == "0"
        assert labelled(at.radio, "Result scope").value == "Current sample"
        assert any(f"Current sample · {sample['sample_id']} · {sample['dialogue_id']}" in item.value
                   for item in at.caption)
    labelled(at.selectbox, "Dimension").select("Grounded content uptake").run()
    assert not at.exception
    assert next(expander for expander in at.expander if expander.label == "Statistics").proto.expanded
    at.run()
    assert not at.exception
    assert app.list_run_dirs(tmp_path / "runs") == [run]
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("dimension,core", [
    *((dimension, "Bond") for dimension in app.BOND_DIMENSIONS),
    *((dimension, "Task") for dimension in app.TASK_DIMENSIONS),
])
def test_dimension_prefix_uses_framework_membership(dimension, core):
    assert app.display_dimension(dimension) == f"{core} · {dimension}"


def test_unrecognised_dimension_is_not_assigned_to_a_core():
    assert app.display_dimension("Unrecognised dimension") == "Unrecognised dimension"


@pytest.mark.parametrize("status,colour", [
    ("rated", "green"), ("evidence_only", "orange"), ("insufficient_evidence", "red"),
    ("no_evaluable_opportunity", "gray"),
])
def test_result_status_badges_preserve_plain_table_text(status, colour):
    text = status.replace("_", " ")
    label = "Existing category" if status == "rated" else None
    suffix = f" · {label}" if label else ""
    assert app.display_result(status, label, badge=True) == f":{colour}-badge[{text}]{suffix}"
    assert app.display_result(status, label) == (label or text)
    assert app.display_result("", None, badge=True) == ""


def test_evidence_badge_uses_readable_yellow_style(prepared_plan, monkeypatch):
    monkeypatch.setenv("ALLIANCE_EVAL_WORKSPACE", str(prepared_plan.parents[2]))
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    assert not at.exception
    assert any('span.is-badge[style*="background-color: rgba(255, 227, 18,"]' in item.value
               and 'color: #806600' in item.value and 'background-color: #fff4bf' in item.value
               for item in at.markdown)


@pytest.mark.parametrize("preset", [False, True])
def test_summary_reason_prefix_is_display_only_and_demo_only(preset):
    def summary_page(records, preset):
        from alliance_eval.app import render_summary_details
        render_summary_details(records, preset=preset)

    reason = "Preset illustration: The saved evidence supports this result."
    records = [{"results": [{
        "dimension": "Accurate Empathy", "activity_reference": None,
        "summary_status": "insufficient_evidence", "rating_label": None,
        "reason": reason, "evidence": [], "counterevidence": [],
    }]}]
    before = json.dumps(records)
    at = AppTest.from_function(summary_page, args=(records, preset)).run()
    assert not at.exception
    expected = "The saved evidence supports this result." if preset else reason
    assert at.dataframe[0].value.columns.tolist() == ["dimension", "activity", "result"]
    assert at.dataframe[0].value["result"].tolist() == ["insufficient evidence"]
    assert at.dataframe[0].value["dimension"].tolist() == ["Bond · Accurate Empathy"]
    assert at.expander[0].label == "Bond · Accurate Empathy · :red-badge[insufficient evidence]"
    assert any(item.value == f"**Reason:** {expected}" for item in at.markdown)
    assert json.dumps(records) == before


@pytest.mark.parametrize("preset", [False, True])
def test_turn_display_simplifies_status_without_rewriting_saved_findings(preset):
    def turn_page(turn, activities, annotation, preset):
        from alliance_eval.app import render_turn_block
        render_turn_block(turn, activities, annotation, preset=preset)

    reason = "Preset illustration: The speaker acknowledges the concern."
    turn = {"turn_id": "u2", "role": "counsellor", "text": "It sounds difficult."}
    annotation = {"findings": [{
        "dimension": "Accurate Empathy", "finding_status": status,
        "rating_label": "AE-7" if status == "rated" else None, "reason": reason,
        "evidence": [{"turn_id": "u2", "quote": "It sounds difficult."}],
    } for status in ("evidence_only", "insufficient_evidence", "rated")]}
    before = json.dumps(annotation)
    at = AppTest.from_function(
        turn_page, args=(turn, {"new_activities": []}, annotation, preset),
    ).run()
    assert not at.exception
    assert at.expander[0].label == "u2 · Annotations · 3 findings"
    displayed = "\n".join(item.value for item in at.markdown)
    assert "**Bond · Accurate Empathy** · :orange-badge[evidence only]" in displayed
    assert "**Bond · Accurate Empathy** · :red-badge[insufficient evidence]" in displayed
    assert "**Bond · Accurate Empathy** · :green-badge[rated] · AE-7" in displayed
    assert ":orange-badge[evidence only] ·" not in displayed and ":red-badge[insufficient evidence] ·" not in displayed
    assert "evidence_only" not in displayed and "insufficient_evidence" not in displayed
    assert "no turn-level label" not in displayed
    assert "Activity identification" not in displayed
    assert not any("No new Activity" in item.value for item in at.caption)
    expected = reason.removeprefix("Preset illustration: ") if preset else reason
    assert [item.value for item in at.caption] == [expected, expected, expected]
    assert json.dumps(annotation) == before


@pytest.mark.parametrize("has_turn_references,has_summary_references", [
    (False, False), (True, False), (False, True), (True, True),
])
def test_empty_turn_keeps_only_compact_citation_locations(has_turn_references, has_summary_references):
    def turn_page(has_turn_references, has_summary_references):
        from alliance_eval.app import render_turn_block
        render_turn_block(
            {"turn_id": "u1", "role": "client", "text": "I am uncertain."},
            {"new_activities": []}, {"findings": []},
            turn_references=[
                ("u2", {"dimension": "Grounded content uptake", "finding_status": "rated",
                        "rating_label": "Substantive grounded uptake"}),
                ("u2", {"dimension": "Accurate Empathy", "finding_status": "evidence_only",
                        "rating_label": None}),
            ] if has_turn_references else [],
            summary_references=[{"dimension": "Accurate Empathy",
                "summary_status": "insufficient_evidence", "rating_label": None}]
                if has_summary_references else [],
        )

    at = AppTest.from_function(turn_page, args=(has_turn_references, has_summary_references)).run()
    assert not at.exception
    assert at.expander[0].label == "u1 · Annotations · 0 findings"
    captions = [item.value for item in at.caption]
    displayed = "\n".join([item.value for item in at.markdown] + captions)
    assert "Activity identification" not in displayed and "Turn findings" not in displayed
    assert "no turn-level label" not in displayed and "no summary label" not in displayed
    assert "Referenced by" not in displayed
    assert not any(text in displayed for text in (
        "rated", "evidence only", "insufficient evidence", "evidence_only",
        "insufficient_evidence", "Substantive grounded uptake",
    ))
    if has_turn_references or has_summary_references:
        expected = ["Cited by"]
        if has_turn_references:
            expected += ["u2 · Bond · Grounded content uptake", "u2 · Bond · Accurate Empathy"]
        if has_summary_references:
            expected += ["Summary · Bond · Accurate Empathy"]
        assert captions == expected
    else:
        assert captions == ["No findings"]


@pytest.mark.parametrize("count,has_error,expected", [
    (1, False, "Annotations · 1 finding"),
    (2, False, "Annotations · 2 findings"),
    (0, True, "Annotations · 0 findings retained"),
    (1, True, "Annotations · 1 finding retained"),
    (2, True, "Annotations · 2 findings retained"),
    (None, True, "Annotations unavailable"),
    (None, False, "Activity checked · turn annotation unavailable"),
])
def test_turn_finding_counts_keep_partial_and_unavailable_results_distinct(count, has_error, expected):
    def turn_page(count, has_error):
        from alliance_eval.app import render_turn_block
        finding = {"dimension": "Accurate Empathy", "finding_status": "evidence_only",
                   "rating_label": None, "reason": "A saved reflection.", "evidence": []}
        render_turn_block(
            {"turn_id": "u2", "role": "counsellor", "text": "That feels uncertain."},
            {"new_activities": []}, {"findings": [finding] * count} if count is not None else None,
            errors=[{"affected_cores": ["Task"], "stage": "turn_annotation",
                     "message": "The other group failed."}] if has_error else [],
        )

    at = AppTest.from_function(turn_page, args=(count, has_error)).run()
    assert not at.exception
    assert at.expander[0].label == f"u2 · {expected}"
    assert not at.warning
    assert bool(at.button) is has_error
    if has_error:
        assert at.button[0].label == "Task annotation unavailable · View reason"
    if count:
        assert sum("**Bond · Accurate Empathy**" in item.value for item in at.markdown) == count
    if count is None:
        assert any(item.value == "No validated turn annotation" for item in at.caption)


def test_sample_total_counts_only_saved_turn_findings_in_a_partial_run(prepared_plan, tmp_path):
    run = tmp_path / "runs" / "fictional-partial-findings"
    run.mkdir(parents=True)
    source = prepared_source(prepared_plan)
    evidence = [{"turn_id": "u1", "quote": "I would like to organise my notes."}]
    findings = [{"dimension": "Grounded content uptake", "finding_status": "rated",
                 "rating_label": "Substantive grounded uptake", "reason": "A saved grounded response.",
                 "evidence": evidence},
                {"dimension": "Accurate Empathy", "finding_status": "evidence_only",
                 "rating_label": None, "reason": "A saved reflection.", "evidence": evidence}]
    (run / "manifest.json").write_text(json.dumps({
        "status": "completed_with_errors", "counts": {}, "usage": {"requests": 0},
        "samples": [{"sample_id": "s1", "dialogue_id": "fictional-app-dialogue", "source_file": str(source),
                     "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                     "annotation_complete_by_core": {"Bond": True, "Task": False},
                     "summary_status_by_core": {"Bond": "complete", "Task": "failed"}}],
    }), encoding="utf-8")
    (run / "turn_annotations.jsonl").write_text("\n".join(json.dumps(record) for record in [
        {"sample_id": "s1", "current_turn_id": "u1", "findings": []},
        {"sample_id": "s1", "current_turn_id": "u2", "findings": findings},
        {"sample_id": "other-sample", "current_turn_id": "u2", "findings": findings},
    ]) + "\n", encoding="utf-8")
    (run / "summary_annotations.jsonl").write_text(json.dumps({
        "sample_id": "s1", "results": [{"dimension": "Accurate Empathy", "activity_reference": None,
          "summary_status": "insufficient_evidence", "rating_label": None, "reason": "Limited evidence.",
          "evidence": evidence, "counterevidence": []}],
    }) + "\n", encoding="utf-8")
    (run / "core_evidence.jsonl").write_text(json.dumps({
        "sample_id": "s1", "core": "Bond", "core_element": "mutual respect",
        "source_dimension": "Grounded content uptake", "source_level": "turn-level",
        "current_turn_id": "u2", "mapping_type": "conditional project mapping",
        "activity_reference": None, "status": "rated", "rating_label": "Substantive grounded uptake",
        "reason": "A saved grounded response.", "evidence": evidence, "counterevidence": [],
    }) + "\n", encoding="utf-8")
    (run / "validation_errors.jsonl").write_text(json.dumps({
        "sample_id": "s1", "current_turn_id": "u2", "stage": "turn_annotation",
        "affected_cores": ["Task"], "error_type": "fictional_test_error", "message": "The Task group failed.",
    }) + "\n", encoding="utf-8")
    before = {path: path.read_bytes() for path in run.iterdir()}
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    assert not at.exception
    dialogue_tab = labelled(at.tabs, "Dialogue & annotations")
    assert [item.value for item in dialogue_tab.markdown if item.value.startswith("**Turn findings:")] == [
        "**Turn findings: 2**",
    ]
    first_turn = labelled(dialogue_tab.expander, "u1 · Annotations · 0 findings")
    assert [item.value for item in first_turn.caption] == [
        "No validated Activity identification", "Cited by",
        "u2 · Bond · Grounded content uptake", "u2 · Bond · Accurate Empathy",
        "Summary · Bond · Accurate Empathy",
    ]
    partial_turn = labelled(dialogue_tab.expander, "u2 · Annotations · 2 findings retained")
    assert any(":green-badge[rated]" in item.value for item in partial_turn.markdown)
    assert any(":orange-badge[evidence only]" in item.value for item in partial_turn.markdown)
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("complete", [True, False, None])
def test_summary_marks_only_unconfirmed_or_incomplete_results_as_not_final(complete):
    def summary_page(records, sample):
        from alliance_eval.app import render_summary_details
        render_summary_details(records, sample=sample)

    records = [{"results": [{
        "dimension": "Accurate Empathy", "activity_reference": None,
        "summary_status": "rated", "rating_label": "AE-7",
        "reason": "Saved explanation.", "evidence": [], "counterevidence": [],
    }]}]
    sample = ({"annotation_complete_by_core": {"Bond": complete, "Task": True},
               "summary_status_by_core": {"Bond": "complete", "Task": "complete"}}
              if complete is not None else {})
    before = json.dumps(records)
    at = AppTest.from_function(summary_page, args=(records, sample)).run()
    assert not at.exception
    assert at.dataframe[0].value["result"].tolist() == ["AE-7"]
    assert at.expander[0].label == "Bond · Accurate Empathy · :green-badge[rated] · AE-7"
    assert bool(at.warning) is (complete is not True)
    if complete is not True:
        assert any("not as a final result" in warning.value.lower() for warning in at.warning)
    assert json.dumps(records) == before


@pytest.mark.parametrize("legacy_type", [None, "indirect evidence"])
def test_core_mapping_groups_relationships_and_renders_saved_evidence_without_raw_table(legacy_type):
    def core_page(records):
        from alliance_eval.app import render_core_mapping
        render_core_mapping(records)

    shared = {
        "sample_id": "s1", "core": "Task", "core_element": "acceptance of responsibility to perform the activities",
        "source_dimension": "Client engagement in the activity",
        "activity_reference": {"activity_id": "ACTIVITY01", "activity_label": "Write a short note"},
        "reason": "Do not repeat this saved explanation.", "diagnostic_only": False,
    }
    if legacy_type is not None:
        shared["mapping_type"] = legacy_type
    records = [{**shared, "source_level": "turn-level", "current_turn_id": "u2", "diagnostic_only": True,
                "status": "evidence_only", "rating_label": None,
                "evidence": [{"turn_id": "u2", "quote": "I will bring my note."}], "counterevidence": []},
               {**shared, "source_level": "activity-level summary", "current_turn_id": None,
                "status": "rated", "rating_label": "Somewhat salient", "diagnostic_only": True,
                "evidence": [{"turn_id": "u3", "quote": "Here is my written note."}],
                "counterevidence": [{"turn_id": "u1", "quote": "I have not begun yet."}]}]
    before = json.dumps(records)
    at = AppTest.from_function(core_page, args=(records,)).run()
    assert not at.exception
    assert not at.dataframe
    assert len(at.expander) == 1
    assert "Client engagement in the activity" in at.expander[0].label
    displayed = "\n".join(item.value for item in [*at.markdown, *at.caption])
    for expected in ("activity-level summary", "u3", "ACTIVITY01",
                     "Write a short note", "Somewhat salient",
                     "Here is my written note.", "Counterevidence", "I have not begun yet."):
        assert expected in displayed
    assert "Do not repeat this saved explanation" not in displayed
    assert "indirect evidence" not in displayed
    assert "**Mapping:**" not in displayed
    assert "Mapping relationship unavailable" not in displayed
    assert ":green-badge[rated] · Somewhat salient" in displayed
    assert "evidence only" not in displayed and "I will bring my note." not in displayed
    assert "**u2**" not in displayed
    assert "[object Object]" not in displayed and "no turn-level label" not in displayed
    assert [item.value for item in at.caption].count(
        "Diagnostic mapping — this Core's results are incomplete."
    ) == 1
    assert at.expander[0].caption[0].value == "Diagnostic mapping — this Core's results are incomplete."
    assert json.dumps(records) == before


def test_core_mapping_groups_records_independently_of_legacy_types():
    def core_page(records):
        from alliance_eval.app import render_core_mapping
        render_core_mapping(records)

    shared = {
        "core": "Bond", "core_element": "mutual respect",
        "source_dimension": "Grounded content uptake", "source_level": "turn-level",
        "status": "rated", "rating_label": "Substantive grounded uptake",
        "activity_reference": None, "evidence": [], "counterevidence": [],
    }
    records = [
        {**shared, "current_turn_id": "u1", "mapping_type": "conditional project mapping"},
        {**shared, "current_turn_id": "u2"},
    ]
    before = json.dumps(records)
    at = AppTest.from_function(core_page, args=(records,)).run()
    assert not at.exception
    assert len(at.expander) == 1
    assert at.expander[0].label == "Grounded content uptake → mutual respect"
    displayed = "\n".join(item.value for item in [*at.markdown, *at.caption])
    assert "**u1**" in displayed and "**u2**" in displayed
    assert "conditional project mapping" not in displayed
    assert "Mapping relationship unavailable" not in displayed
    assert json.dumps(records) == before


@pytest.mark.parametrize("cores, expected", [
    (["Task", "Bond", "Task"], ["Bond", "Task"]),
    (["Bond", "Task", "Bond"], ["Bond", "Task"]),
    (["Task"], ["Task"]),
    (["Bond"], ["Bond"]),
    (["Unspecified Core", "Task", "Bond"], ["Bond", "Task", "Unspecified Core"]),
])
def test_core_mapping_displays_bond_before_task_without_reordering_records(cores, expected):
    def core_page(records):
        from alliance_eval.app import render_core_mapping
        render_core_mapping(records)

    records = [{"core": core, "source_dimension": f"Dimension {index}",
                "status": "rated", "rating_label": "Example category",
                "evidence": [], "counterevidence": []}
               for index, core in enumerate(cores)]
    before = json.dumps(records)
    at = AppTest.from_function(core_page, args=(records,)).run()
    assert not at.exception
    assert [item.value for item in at.subheader] == expected
    assert len(at.expander) == len(records)
    assert json.dumps(records) == before


def test_core_mapping_with_only_intermediate_evidence_is_empty_without_mutation():
    def core_page(records):
        from alliance_eval.app import render_core_mapping
        render_core_mapping(records)

    records = [{"core": "Bond", "source_dimension": "Accurate Empathy",
                "source_level": "turn-level", "current_turn_id": "u1",
                "status": "evidence_only", "rating_label": None,
                "evidence": [{"turn_id": "u1", "quote": "That sounds difficult."}]}]
    before = json.dumps(records)
    at = AppTest.from_function(core_page, args=(records,)).run()
    assert not at.exception
    assert not at.expander
    assert [item.value for item in at.info] == ["No Core mapping"]
    assert json.dumps(records) == before


@pytest.mark.parametrize("state", ["complete", "incomplete", "unknown", "error", "recovered"])
def test_validation_details_are_conditional_and_never_show_raw_response(prepared_plan, tmp_path, state):
    run = tmp_path / "runs" / "fictional-validation-display"
    run.mkdir(parents=True)
    source = prepared_source(prepared_plan)
    sample = {"sample_id": "s1", "dialogue_id": "fictional-app-dialogue", "source_file": str(source),
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    if state != "unknown":
        sample.update({"annotation_complete_by_core": {"Bond": state != "incomplete", "Task": True},
                       "summary_status_by_core": {"Bond": "complete", "Task": "complete"}})
    (run / "manifest.json").write_text(json.dumps({
        "status": "completed", "counts": {}, "usage": {"requests": 0}, "samples": [sample],
    }), encoding="utf-8")
    if state in ("error", "recovered"):
        (run / "validation_errors.jsonl").write_text(json.dumps({
            "sample_id": "s1", "current_turn_id": "u2", "stage": "turn_annotation",
            "affected_cores": ["Bond"], "error_type": "fictional_test_error",
            "message": "The saved evidence does not match.", "raw_response": '{"fictional": "response"}',
            **({"error_type": "ConnectError", "message": "Offline connection failure",
                "raw_response": None, "attempt": 1, "retry_scheduled": True}
               if state == "recovered" else {}),
        }) + "\n", encoding="utf-8")
    before = {path: path.read_bytes() for path in run.iterdir()}
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    assert not at.exception
    assert [tab.label for tab in at.tabs] == ["Dialogue & annotations", "Summary", "Core mapping"]
    details = [expander for expander in at.expander if expander.label == "Validation details"]
    assert bool(details) is (state != "complete")
    assert not any("No validation errors" in success.value for success in at.success)
    if state == "unknown":
        assert any("cannot be confirmed" in warning.value for warning in at.warning)
    if state == "error":
        assert any("Model response could not be accepted" in item.value for item in details[0].markdown)
        assert not any("The saved evidence does not match." in item.value for item in at.markdown)
        assert not details[0].get("popover")
        assert not details[0].code
    if state == "recovered":
        assert not at.warning
        assert any("Connection retries were recorded; final results are complete" in item.value for item in at.caption)
        assert any("attempt 1 failed" in item.value for item in details[0].markdown)
        assert not any(item.value == "Offline connection failure" for item in details[0].markdown)
        assert not any("Annotation error" in item.label for item in at.expander)
    assert {path: path.read_bytes() for path in before} == before


def test_retry_attempts_remain_in_failed_request_statistics():
    statistics = app.run_statistics({
        "status": "completed", "counts": {
            "planned_samples": 1, "processed_samples": 1, "completed_samples": 1, "failed_calls": 2,
        }, "usage": {"requests": 7}, "samples": [],
    }, [], [])
    assert statistics["failed_calls"] == 2
    assert statistics["requests"] == 7
    assert statistics["failed_call_rate"] == 2 / 7


@pytest.mark.parametrize("message,expected", [
    ("Each turn finding must cite an exact quote from the current turn", "Evidence from the current turn is missing"),
    ("Activity label does not match the registry: ACTIVITY_01", "Activity name does not match the registered activity"),
    ("Evidence refers to an unavailable turn: u8", "Evidence refers to an unavailable turn"),
    ("Turn judge refers to an unregistered Activity: ACTIVITY_09", "Annotation refers to an unknown activity"),
    ("Summary includes an unrequested target: ('dimension', None)", "Summary does not cover the requested targets"),
    ("unknown private diagnostic", "Model response could not be accepted"),
])
def test_validation_explanations_only_describe_recognised_saved_errors(message, expected):
    error = {"message": message, "error_type": "ValueError"}
    assert app.validation_issue(error, error)["title"] == expected


@pytest.mark.parametrize("prefix", ["", "Bond: ", "Task: "])
@pytest.mark.parametrize("message,fields", [
    ("Evidence is not an exact quote from u2: 'Quoted words.'",
     {"turn_id": "u2", "quote": "Quoted words."}),
    ("Evidence is not an exact quote from u2: 'First line.\\nSecond line.'",
     {"turn_id": "u2", "quote": "First line.\nSecond line."}),
    ("Summary evidence must come from a validated turn span for the same dimension and Activity: "
     "('Client engagement in the activity', 'ACTIVITY_01'), u2, 'Evidence.'",
     {"summary_target": ("Client engagement in the activity", "ACTIVITY_01"), "turn_id": "u2", "quote": "Evidence."}),
    ("Summary evidence must come from a validated turn span for the same dimension and Activity: "
     "('Accurate Empathy', None), u2, 'Evidence.'",
     {"summary_target": ("Accurate Empathy", None), "turn_id": "u2", "quote": "Evidence."}),
    ("Each turn finding must cite an exact quote from the current turn",
     {"current_evidence_missing": True}),
    ("Activity label does not match the registry: ACTIVITY_01",
     {"activity_id": "ACTIVITY_01"}),
    ("Evidence refers to an unavailable turn: u8", {}),
    ("Turn judge refers to an unregistered Activity: ACTIVITY_09", {}),
    ("Summary refers to an unknown Activity: ACTIVITY_09", {}),
    ("Summary includes an unrequested target: ('dimension', None)", {}),
    ("Incomplete summary coverage: missing target", {}),
])
def test_saved_validation_errors_extract_fields_without_mutation(prefix, message, fields):
    error = {"message": prefix + message, "error_type": "ValueError", "raw_response": "unchanged"}
    before = json.dumps(error)
    actual = app.validation_issue(error, error)
    assert actual["title"] != "Model response could not be accepted"
    assert all(actual[key] == value for key, value in fields.items())
    assert json.dumps(error) == before


@pytest.mark.parametrize("message", [
    "Unrecognised diagnostic", "Evidence is not an exact quote from u2", "Summary evidence cannot be identified",
])
def test_unknown_errors_keep_general_explanation(message):
    error = {"message": message, "error_type": "ValueError"}
    issue = app.validation_issue(error, error)
    assert issue["title"] == "Model response could not be accepted"
    assert not {"quote", "turn_id", "activity_id", "summary_target"} & issue.keys()


def test_saved_quote_literals_are_not_evaluated():
    message = "Evidence is not an exact quote from u2: __import__('os').getcwd()"
    error = {"message": message, "error_type": "ValueError"}
    issue = app.validation_issue(error, error)
    assert issue["turn_id"] == "u2"
    assert issue["quote"] is None


@pytest.mark.parametrize("source_state", ["verified", "missing", "changed", "unverified"])
def test_validation_quote_comparison_and_one_click_location_preserve_files(prepared_plan, tmp_path, source_state):
    run = tmp_path / "runs" / "fictional-evidence-error"
    run.mkdir(parents=True)
    source = prepared_source(prepared_plan)
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_state == "changed":
        source.write_text(source.read_text().replace("Which part", "What section"))
    sample = {
        "sample_id": "s1", "dialogue_id": "fictional-app-dialogue",
        "source_file": str(source if source_state != "missing" else source.with_name("absent.jsonl")),
        "annotation_complete_by_core": {"Bond": False, "Task": True},
        "summary_status_by_core": {"Bond": "diagnostic", "Task": "complete"},
    }
    if source_state != "unverified":
        sample["source_sha256"] = original_hash
    (run / "manifest.json").write_text(json.dumps({
        "status": "completed_with_errors", "counts": {}, "usage": {}, "samples": [sample],
    }))
    quote = 'A fictional "quote" with <b>literal markup</b>.\nNot the original turn.'
    message = "Evidence is not an exact quote from u2: " + repr(quote)
    error = {
        "sample_id": "s1", "current_turn_id": "u2", "stage": "turn_annotation",
        "affected_cores": ["Bond"], "error_type": "GroupValidationError",
        "message": "Bond: " + message,
        "group_errors": [{"core": "Bond", "message": message}],
        "raw_response": '{"private": "raw-response-marker"}',
    }
    (run / "validation_errors.jsonl").write_text(json.dumps(error) + "\n")
    before = {path: path.read_bytes() for path in [source, *run.iterdir()]}
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    assert not at.exception
    details = labelled(at.expander, "Validation details")
    assert details.proto.expanded is False
    assert quote in displayed_evidence_quotes(details.markdown)
    evidence_lines = [item.value for item in details.markdown if 'class="evidence-turn-id"' in item.value]
    assert all(line.startswith('<ul><li>') and line.endswith('</li></ul>') for line in evidence_lines)
    assert not at.code and not at.json and not details.get("popover")
    assert not any(marker in item.value for item in at.markdown
                   for marker in ("raw-response-marker", "Evidence is not an exact quote"))
    if source_state == "verified":
        assert "Which part would you like to begin with?" in displayed_evidence_quotes(details.markdown)
        labelled(at.button, "Bond annotation unavailable · View reason").click().run()
        assert not at.exception
        assert labelled(at.expander, "Validation details").proto.expanded is True
        anchor = app.validation_anchor(f"{run}:s1", "u2")
        assert any(anchor in item.value for item in labelled(at.expander, "Validation details").markdown)
        assert any(anchor in item.proto.srcdoc and "scrollIntoView" in item.proto.srcdoc for item in at.get("iframe"))
    else:
        assert any("Original text unavailable" in item.value for item in details.caption)
        assert len(displayed_evidence_quotes(details.markdown)) == 1
    assert {path: path.read_bytes() for path in before} == before


def test_validation_comparison_current_turn_missing_and_activity_identity():
    def page(errors, dialogue, activities):
        from alliance_eval.app import render_validation_details
        render_validation_details(
            {"sample_id": "s1", "summary_status": "diagnostic"}, errors,
            dialogue=dialogue, activity_records=activities,
        )

    findings = [
        {"dimension": "Grounded content uptake", "evidence": [{"turn_id": "u1", "quote": "Bond evidence must not be selected"}]},
        {"dimension": "Collaborative activity negotiation", "evidence": [{"turn_id": "u1", "quote": "A prior-turn quote."}],
         "activity_reference": {"activity_id": "ACTIVITY_01", "activity_label": "A different name"}},
    ]
    messages = (
        "Each turn finding must cite an exact quote from the current turn", "Activity label does not match the registry: ACTIVITY_01",
    )
    errors = [{"sample_id": "s1", "current_turn_id": "u2", "stage": "turn_annotation", "affected_cores": ["Task"],
               "message": message, "raw_response": json.dumps({"findings": findings})} for message in messages]
    dialogue = {"turns": [{"turn_id": "u2", "text": "The current turn to compare."}]}
    activities = [{"new_activities": [{"activity_id": "ACTIVITY_01", "activity_label": "The registered name"}]}]
    before = json.dumps([errors, dialogue, activities])
    at = AppTest.from_function(page, args=(errors, dialogue, activities)).run()
    assert not at.exception
    assert displayed_evidence_quotes(at.markdown) == ["A prior-turn quote.", "The current turn to compare."]
    assert [item.value for item in at.text] == ["Model name: A different name", "Registered name: The registered name"]
    assert json.dumps([errors, dialogue, activities]) == before


def test_validation_summary_compares_only_same_target_accepted_evidence():
    def page(error, records):
        from alliance_eval.app import render_validation_details
        render_validation_details({"sample_id": "s1", "summary_status": "diagnostic"}, [error], turn_records=records)

    error = {"sample_id": "s1", "stage": "summary_annotation", "affected_cores": ["Task"], "message":
             "Summary evidence must come from a validated turn span for the same dimension and Activity: "
             "('Client engagement in the activity', 'ACTIVITY_01'), u2, 'A shortened quote.'"}
    records = [{"findings": [{"dimension": "Client engagement in the activity",
                              "activity_reference": {"activity_id": activity_id},
                              "evidence": [{"turn_id": turn_id, "quote": quote}]}
                             for activity_id, turn_id, quote in [
                                 ("ACTIVITY_01", "u2", "The accepted quote for this target."),
                                 ("ACTIVITY_02", "u2", "Wrong activity."),
                                 ("ACTIVITY_01", "u1", "Wrong turn."),
                             ]]}]
    at = AppTest.from_function(page, args=(error, records)).run()
    assert not at.exception
    assert displayed_evidence_quotes(at.markdown) == ["A shortened quote.", "The accepted quote for this target."]
    assert any("not linked to accepted evidence" in item.value for item in at.markdown)


@pytest.mark.parametrize("raw", [None, "invalid JSON", "[]", '{"findings": [null, 5]}'])
def test_validation_missing_evidence_does_not_fabricate_comparison(raw):
    def page(raw):
        from alliance_eval.app import render_validation_details
        render_validation_details({"sample_id": "s1"}, [{
            "sample_id": "s1", "current_turn_id": "u2", "affected_cores": ["Bond"],
            "message": "Each turn finding must cite an exact quote from the current turn", "raw_response": raw,
        }])

    at = AppTest.from_function(page, args=(raw,)).run()
    assert not at.exception
    assert not at.text and not at.code
    assert any("No readable quote" in item.value for item in at.caption)
    assert any("Original text unavailable" in item.value for item in at.caption)


@pytest.mark.parametrize("issue_counts", [[], [1], [1, 1], [2], [2, 1]])
def test_validation_issues_have_only_between_item_dividers_and_no_repeated_notice(issue_counts):
    def page(issue_counts):
        from alliance_eval.app import render_validation_details
        errors = [{
            "current_turn_id": f"u{index}", "stage": "turn_annotation",
            "message": "A saved failure", "affected_cores": ["Bond"],
            "group_errors": [{"core": core, "message": "Each turn finding must cite an exact quote from the current turn"}
                             for core in ("Bond", "Task")[:count]],
        } for index, count in enumerate(issue_counts)]
        render_validation_details({"sample_id": "s1"}, errors)

    at = AppTest.from_function(page, args=(issue_counts,)).run()
    assert not at.exception
    details = labelled(at.expander, "Validation details")
    assert len(details.get("divider")) == max(0, sum(issue_counts) - 1)
    visible_text = "\n".join(item.value for item in [*details.markdown, *details.caption])
    assert "was not accepted" not in visible_text
    assert "Other validated results remain available" not in visible_text


@pytest.mark.parametrize("attempt", [None, 0, "1", True])
def test_retry_view_record_requires_attempt_number(attempt):
    with pytest.raises(ValueError, match="attempt number"):
        app._validate_view_record({
            "sample_id": "s1", "message": "Offline failure", "error_type": "ConnectError",
            "retry_scheduled": True, "attempt": attempt,
        }, "validation_errors.jsonl")


def test_cross_turn_references_are_deduplicated_without_relabelling_current_turn():
    finding = {"dimension": "Grounded content uptake", "evidence": [
        {"turn_id": "previous", "quote": "first"},
        {"turn_id": "previous", "quote": "second"},
        {"turn_id": "current", "quote": "response"},
    ]}
    assert app.turn_references_by_turn([
        {"current_turn_id": "previous", "findings": []},
        {"current_turn_id": "current", "findings": [finding]},
    ]) == {"previous": [("current", finding)]}


def test_plan_can_use_explicit_legacy_data_root(prepared_plan, tmp_path):
    outside_plan = tmp_path / "old-plan.json"
    outside_plan.write_bytes(prepared_plan.read_bytes())
    with pytest.raises(FileNotFoundError):
        app.validate_plan_input(outside_plan, None)
    assert app.validate_plan_input(outside_plan, prepared_plan.parent) == 1


def test_plan_rejects_duplicate_json_keys_in_local_validation(prepared_plan):
    content = prepared_plan.read_text(encoding="utf-8")
    prepared_plan.write_text(
        content.replace('"sample_version":', '"sample_version": "discarded", "sample_version":', 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate JSON field"):
        app.validate_plan_input(prepared_plan, None)
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    assert not at.exception
    assert labelled(at.selectbox, "Sample plan").options == []
    assert any("No valid sample plan" in info.value for info in at.info)
    assert labelled(at.button, "Run All").disabled


def test_dialogue_sources_never_guess_current_working_directory(prepared_plan):
    record = app.load_dialogue(str(prepared_source(prepared_plan)), "fictional-app-dialogue")
    assert record["dialogue_id"] == "fictional-app-dialogue"
    with pytest.raises(ValueError, match="explicit data root"):
        app.load_dialogue("dialogues.jsonl", "fictional-app-dialogue")
    assert app.load_dialogue("../processed/dialogues.jsonl", "fictional-app-dialogue", prepared_plan.parent) == record
    assert app.recorded_data_root({"sample_plan": "relative/plan.json"}, None) is None
    assert app.recorded_data_root({"sample_plan": str(prepared_plan)}, None) == prepared_plan.parent
    assert app.recorded_data_root({"data_root": str(prepared_plan.parent)}, None) == prepared_plan.parent
    assert app.recorded_data_root({"sample_plan": str(prepared_plan)}, Path("/explicit")) == Path("/explicit")


def test_source_checksum_uses_the_same_single_read_as_dialogue_parsing(prepared_plan, monkeypatch):
    source = prepared_source(prepared_plan)
    original = source.read_bytes()
    expected_sha256 = hashlib.sha256(original).hexdigest()
    reads = []
    returned_bytes = [original]
    real_read_bytes = Path.read_bytes

    def read_source_once(path):
        if path == source:
            reads.append(path)
            return returned_bytes[0]
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_source_once)
    record = app.load_dialogue(str(source), "fictional-app-dialogue", expected_sha256=expected_sha256)
    assert record["dialogue_id"] == "fictional-app-dialogue"
    assert reads == [source]
    returned_bytes[0] = b"Changed bytes are not valid JSON and must not be parsed."
    with pytest.raises(ValueError, match="Source checksum mismatch"):
        app.load_dialogue(str(source), "fictional-app-dialogue", expected_sha256=expected_sha256)
    assert reads == [source, source]


def test_initial_page_and_local_provider_do_not_query_network(tmp_path):
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    assert not at.exception
    assert labelled(at.selectbox, "Provider").options == ["deepseek", "openai", "mistral", "ollama"]
    assert not any("Local workspace:" in caption.value for caption in at.caption)
    assert not any("I have permission" in checkbox.label for checkbox in at.checkbox)
    assert not any("Run All sends selected dialogue text" in caption.value for caption in at.caption)
    assert labelled(at.button, "Run All").disabled
    assert labelled(at.text_input, "DeepSeek API key").value == ""
    assert not any(field.label == "Run directory" for field in at.text_input)
    assert not (tmp_path / "runs").exists()
    labelled(at.selectbox, "Provider").select("ollama").run()
    assert not at.exception
    assert labelled(at.text_input, "Model ID").value == ""
    assert labelled(at.button, "Run All").disabled


def test_run_requires_button_click_and_consumes_snapshot_once(prepared_plan, tmp_path, monkeypatch):
    calls = []
    closed = []
    monkeypatch.setattr(
        evaluate, "LocalHTTPJudge",
        lambda **kwargs: SimpleNamespace(model_id=kwargs["model_id"], close=lambda: closed.append(True)),
    )
    monkeypatch.setattr(pipeline, "run_all", lambda **kwargs: calls.append(kwargs))
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    labelled(at.selectbox, "Provider").select("ollama").run()
    assert labelled(at.button, "Run All").disabled
    labelled(at.text_input, "Model ID").input("fictional-model").run()
    assert not labelled(at.button, "Run All").disabled
    labelled(at.text_input, "Model ID").input("changed-fictional-model").run()
    at.run()
    assert not labelled(at.button, "Run All").disabled
    assert calls == []
    assert closed == []
    labelled(at.button, "Run All").click().run()
    assert not at.exception
    assert len(calls) == 1
    assert calls[0]["sample_plan_path"] == prepared_plan
    assert calls[0]["run_root"] == tmp_path / "runs"
    assert calls[0]["data_root"] is None
    assert calls[0]["backend"].model_id == "changed-fictional-model"
    assert "project_root" not in calls[0]
    assert closed == [True]
    at.run()
    assert len(calls) == 1
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("base_url", [
    "model.example", "file:///tmp/model", "http://[invalid", "http://model.example:invalid",
    "https://user:password@model.example", "http://model.example?key=fictional-key",
    "http://127.0.0.1:65536", "http://127.0.0.1:-1",
])
def test_invalid_ollama_url_blocks_run_and_can_be_corrected(prepared_plan, tmp_path, monkeypatch, base_url):
    monkeypatch.setattr(evaluate, "LocalHTTPJudge", lambda **kwargs: pytest.fail("Unexpected backend construction"))
    monkeypatch.setattr(pipeline, "run_all", lambda **kwargs: pytest.fail("Unexpected pipeline execution"))
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    labelled(at.selectbox, "Provider").select("ollama").run()
    labelled(at.text_input, "Model ID").input("fictional-model").run()
    assert not labelled(at.button, "Run All").disabled
    labelled(at.text_input, "Base URL").input(base_url).run()
    assert not at.exception
    assert any("Model configuration not ready" in error.value for error in at.error)
    assert labelled(at.button, "Run All").disabled
    assert not labelled(at.button, "Run preset demo").disabled
    labelled(at.text_input, "Base URL").input("http://127.0.0.1:11434").run()
    assert not at.exception
    assert not labelled(at.button, "Run All").disabled
    assert not any("Model configuration not ready" in error.value for error in at.error)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("provider,backend_name,error_class", [
    ("deepseek", "DeepSeekJudge", ValueError),
    ("openai", "OpenAIJudge", ValueError),
    ("ollama", "LocalHTTPJudge", ValueError),
    ("ollama", "LocalHTTPJudge", httpx.InvalidURL),
])
def test_backend_configuration_error_does_not_start_pipeline(
    prepared_plan, tmp_path, monkeypatch, provider, backend_name, error_class,
):
    constructed = []

    def invalid_backend(**kwargs):
        constructed.append(kwargs)
        raise error_class("Fictional invalid model configuration")

    monkeypatch.setattr(evaluate, backend_name, invalid_backend)
    monkeypatch.setattr(pipeline, "run_all", lambda **kwargs: pytest.fail("Unexpected pipeline execution"))
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    labelled(at.selectbox, "Provider").select(provider).run()
    if provider == "ollama":
        labelled(at.text_input, "Model ID").input("fictional-model").run()
    else:
        key_label = "OpenAI API key" if provider == "openai" else "DeepSeek API key"
        labelled(at.text_input, key_label).input("fictional-key").run()
    assert not labelled(at.button, "Run All").disabled
    labelled(at.button, "Run All").click().run()
    assert not at.exception
    assert len(constructed) == 1
    assert any("Model configuration not ready" in error.value for error in at.error)
    assert any("Fictional invalid model configuration" in error.value for error in at.error)
    assert labelled(at.selectbox, "Provider").value == provider
    at.run()
    assert len(constructed) == 1
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("with_close,invalid_result", [(False, False), (True, False), (True, True)])
def test_run_displays_saved_results_with_optional_backend_cleanup(
    prepared_plan, tmp_path, monkeypatch, with_close, invalid_result,
):
    responses = [
        {"new_activities": []},
        {"findings": [{"invalid": True}]} if invalid_result else {"findings": []},
        {"new_activities": []}, {"findings": []},
        {"results": [{
            "dimension": dimension, "activity_reference": None,
            "summary_status": "no_evaluable_opportunity", "rating_label": None,
            "evidence": [], "counterevidence": [], "reason": "Fictional offline fixture.",
        } for dimension in ("Accurate Empathy", "Therapist accepts/validates Patient", "Mutual trust and respect")]},
    ]
    backend = evaluate.MockJudge(responses)
    closed = []
    if with_close:
        backend.close = lambda: closed.append(True)
    assert not hasattr(backend, "client")
    assert hasattr(backend, "close") is with_close
    monkeypatch.setattr(evaluate, "LocalHTTPJudge", lambda **kwargs: backend)
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    labelled(at.selectbox, "Provider").select("ollama").run()
    labelled(at.text_input, "Model ID").input("fictional-model").run()
    labelled(at.button, "Run All").click().run()
    assert not at.exception
    assert closed == ([True] if with_close else [])
    run = labelled(at.selectbox, "Run").value
    assert run.parent == tmp_path / "runs"
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == ("completed_with_errors" if invalid_result else "completed")
    assert labelled(at.metric, "Status").value == ("Errors" if invalid_result else "Complete")
    assert labelled(at.metric, "Model").value == "mock"
    assert labelled(at.selectbox, "Sample").value["dialogue_id"] == "fictional-app-dialogue"
    assert not any(expander.label == "Run details" for expander in at.expander)
    before = {path: path.read_bytes() for path in run.iterdir()}
    at.run()
    assert not at.exception
    assert closed == ([True] if with_close else [])
    assert {path: path.read_bytes() for path in before} == before


def test_backend_is_closed_once_when_pipeline_raises(prepared_plan, tmp_path, monkeypatch):
    closed = []
    backend = SimpleNamespace(model_id="fictional-model", close=lambda: closed.append(True))
    monkeypatch.setattr(evaluate, "LocalHTTPJudge", lambda **kwargs: backend)

    def failed_pipeline(**kwargs):
        raise RuntimeError("Fictional pipeline interruption")

    monkeypatch.setattr(pipeline, "run_all", failed_pipeline)
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    labelled(at.selectbox, "Provider").select("ollama").run()
    labelled(at.text_input, "Model ID").input("fictional-model").run()
    labelled(at.button, "Run All").click().run()
    assert len(at.exception) == 1
    assert "Fictional pipeline interruption" in at.exception[0].message
    assert closed == [True]
    assert labelled(at.metric, "Elapsed time").value
    at.run()
    assert not at.exception
    assert closed == [True]
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("provider,default_model", [
    ("deepseek", "deepseek-flash"), ("openai", "gpt-5.6-luna"),
    ("mistral", "mistral-small-2603"),
    ("ollama", ""),
])
def test_model_id_defaults_are_editable_and_required(prepared_plan, provider, default_model):
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    labelled(at.selectbox, "Provider").select(provider).run()
    assert labelled(at.text_input, "Model ID").value == default_model
    if provider == "openai":
        for enabled in (False, True, False):
            labelled(at.toggle, "Thinking").set_value(enabled).run()
            assert not any("Output limit:" in caption.value or "JSON mode" in caption.value for caption in at.caption)
    assert bool(labelled(at.text_input, "Run ID").value) == bool(default_model)
    assert labelled(at.button, "Run All").disabled
    labelled(at.text_input, "Model ID").input("").run()
    assert labelled(at.text_input, "Model ID").value == ""
    assert labelled(at.text_input, "Run ID").value == ""
    if provider in {"deepseek", "openai", "mistral"}:
        key_label = {"openai": "OpenAI API key", "deepseek": "DeepSeek API key", "mistral": "Mistral API key"}[provider]
        labelled(at.text_input, key_label).input("fictional-test-key").run()
    assert labelled(at.button, "Run All").disabled
    labelled(at.text_input, "Model ID").input("fictional-model").run()
    assert not labelled(at.button, "Run All").disabled
    labelled(at.text_input, "Model ID").input("   ").run()
    assert labelled(at.button, "Run All").disabled
    assert labelled(at.text_input, "Run ID").value == ""
    assert not at.exception


@pytest.mark.parametrize("thinking", [False, True])
def test_mistral_page_snapshot_and_credentials_are_isolated(prepared_plan, tmp_path, monkeypatch, thinking):
    settings = []
    monkeypatch.setenv("MISTRAL_API_KEY", "fictional-env-key")
    monkeypatch.setattr(evaluate, "LocalHTTPJudge", lambda **kwargs: (
        settings.append(kwargs) or SimpleNamespace(close=lambda: None)
    ))
    monkeypatch.setattr(pipeline, "run_all", lambda **kwargs: None)
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    labelled(at.text_input, "DeepSeek API key").input("fictional-deepseek-key").run()
    labelled(at.selectbox, "Provider").select("mistral").run()
    assert labelled(at.text_input, "Model ID").value == "mistral-small-2603"
    assert labelled(at.text_input, "Mistral API key").value == "fictional-env-key"
    assert labelled(at.toggle, "Thinking").value is False
    assert not any(field.label == "Context window (num_ctx)" for field in at.number_input)
    labelled(at.text_input, "Mistral API key").input("fictional-session-key").run()
    labelled(at.toggle, "Thinking").set_value(thinking).run()
    if thinking:
        assert labelled(at.selectbox, "Reasoning effort").options == ["high"]
        assert "thinking_high" in labelled(at.text_input, "Run ID").value
    else:
        assert not any(field.label == "Reasoning effort" for field in at.selectbox)
        assert "thinking" not in labelled(at.text_input, "Run ID").value
    assert settings == []
    labelled(at.button, "Run All").click().run()
    assert not at.exception and len(settings) == 1
    assert settings[0]["dialect"] == "mistral"
    assert settings[0]["base_url"] == "https://api.mistral.ai/v1"
    assert settings[0]["model_id"] == "mistral-small-2603"
    assert settings[0]["api_key"] == "fictional-session-key"
    assert settings[0]["response_format"] == "json_schema"
    assert settings[0]["parameters"]["reasoning_effort"] == ("high" if thinking else "none")
    labelled(at.selectbox, "Provider").select("openai").run()
    assert labelled(at.text_input, "OpenAI API key").value == ""
    at.run()
    assert len(settings) == 1 and not (tmp_path / "runs").exists()


def test_invalid_source_stays_blocked_with_model_configured(prepared_plan):
    source = prepared_source(prepared_plan)
    dialogue = json.loads(source.read_text(encoding="utf-8"))
    dialogue["turns"][0]["role"] = "unmapped-role"
    source.write_text(json.dumps(dialogue) + "\n", encoding="utf-8")
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    labelled(at.selectbox, "Provider").select("ollama").run()
    labelled(at.text_input, "Model ID").input("fictional-model").run()
    assert not at.exception
    assert any("Input not ready" in warning.value for warning in at.warning)
    assert labelled(at.button, "Run All").disabled


def test_runtime_mismatch_cannot_construct_backend(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A mismatched runtime must not construct a judge")

    monkeypatch.setattr(app, "DeepSeekJudge", forbidden)
    monkeypatch.setattr(app, "OpenAIJudge", forbidden)
    monkeypatch.setattr(app, "LocalHTTPJudge", forbidden)
    monkeypatch.setattr(app, "_render_runtime_status", lambda: False)
    app._run_with_progress({})


@pytest.mark.parametrize("selected_ids,count", [
    (None, 1), ([], 0), ("u1", 1), ([1], 1), ([""], 1),
    (["u1", "u1"], 2), (["missing"], 1), (["u1"], 2),
    (["u3", "u1"], 2), (["u1", "u3"], 2),
])
def test_saved_selection_rejects_invalid_missing_reordered_or_disjoint_ids(selected_ids, count):
    dialogue = {"turns": [{"turn_id": f"u{index}"} for index in (1, 2, 3)]}
    with pytest.raises(ValueError, match="Recorded turn selection"):
        app.select_saved_dialogue(dialogue, {"selected_turn_ids": selected_ids, "turn_count": count})


def test_saved_selection_keeps_source_order_and_does_not_mutate_cached_dialogue():
    dialogue = {"dialogue_id": "fictional", "turns": [
        {"turn_id": f"u{index}", "turn_index": index * 2} for index in (1, 2, 3)
    ]}
    before = json.dumps(dialogue)
    selected = app.select_saved_dialogue(dialogue, {"selected_turn_ids": ["u2", "u3"], "turn_count": 2})
    assert [turn["turn_id"] for turn in selected["turns"]] == ["u2", "u3"]
    assert json.dumps(dialogue) == before
    assert app.select_saved_dialogue(dialogue, {"turn_count": 1}) is dialogue


def test_app_switches_between_saved_excerpts_of_same_source(prepared_plan, tmp_path):
    plan = json.loads(prepared_plan.read_text())
    item = plan["items"][0]
    plan["items"] = [
        {**item, "start_turn_index": index, "end_turn_index": index}
        for index in (0, 1)
    ]
    prepared_plan.write_text(json.dumps(plan))
    summary = {"results": [{
        "dimension": dimension, "activity_reference": None,
        "summary_status": "no_evaluable_opportunity", "rating_label": None,
        "evidence": [], "counterevidence": [], "reason": "Offline selection fixture.",
    } for dimension in app.SUMMARY_LEVEL_DIMENSIONS[:3]]}
    run = pipeline.run_all(
        prepared_plan, evaluate.MockJudge([{"new_activities": []}, {"findings": []}, summary] * 2),
        tmp_path / "runs", "excerpts",
    )
    before = {path: path.read_bytes() for path in [prepared_plan, prepared_source(prepared_plan), *run.iterdir()]}
    samples = json.loads((run / "manifest.json").read_text())["samples"]
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    for index, turn_id in enumerate(("u1", "u2", "u1")):
        options = labelled(at.selectbox, "Sample")
        if index:
            options.select(samples[index % 2]).run()
        assert not at.exception
        turn_labels = [entry.label for entry in at.expander if " · Annotations" in entry.label]
        assert turn_labels == [f"{turn_id} · Annotations · 0 findings"]
        assert any(f"Selected excerpt: 1 turn · {turn_id} to {turn_id}" == caption.value for caption in at.caption)
        assert not any("Selection boundary" in caption.value for caption in at.caption)
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("selection", ["legacy", "complete_source", "valid", "invalid"])
def test_app_selection_without_annotations_is_explicit_and_read_only(prepared_plan, tmp_path, selection):
    source = prepared_source(prepared_plan)
    sample = {"sample_id": "s1", "dialogue_id": "fictional-app-dialogue",
              "source_file": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "turn_count": 1, "summary_status": "not_run"}
    if selection == "complete_source":
        sample["turn_count"] = 2
    elif selection != "legacy":
        sample["selected_turn_ids"] = ["u2"] if selection == "valid" else ["missing"]
    run = tmp_path / "runs" / "selection-fixture"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({
        "status": "aborted", "counts": {}, "usage": {}, "samples": [sample],
    }))
    (run / "summary_annotations.jsonl").write_text(json.dumps({
        "sample_id": "s1", "results": [{
            "dimension": "Accurate Empathy", "activity_reference": None,
            "summary_status": "insufficient_evidence", "rating_label": None,
            "reason": "Saved synthetic summary remains readable", "evidence": [], "counterevidence": [],
        }],
    }) + "\n")
    before = {path: path.read_bytes() for path in [source, *run.iterdir()]}
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    assert not at.exception
    labels = [entry.label for entry in at.expander]
    if selection == "legacy":
        assert any("Selection boundary was not recorded" in caption.value for caption in at.caption)
        assert {label for label in labels if "No saved processing record" in label} == {
            "u1 · No saved processing record", "u2 · No saved processing record",
        }
        assert not any("Not processed" in label for label in labels)
    elif selection == "complete_source":
        assert not any("Selection boundary" in caption.value or "Selected excerpt" in caption.value
                       for caption in at.caption)
        assert {"u1 · Not processed", "u2 · Not processed"}.issubset(labels)
    elif selection == "valid":
        assert "u2 · Not processed" in labels
        assert not any(label.startswith("u1 ·") for label in labels)
    else:
        assert any("Recorded turn selection" in warning.value for warning in at.warning)
        assert not any(label.startswith(("u1 ·", "u2 ·")) for label in labels)
        assert not any("I would like to organise my notes." in item.value for item in at.markdown)
    assert any("Saved synthetic summary remains readable" in item.value for item in at.markdown)
    assert {path: path.read_bytes() for path in before} == before


def test_old_run_missing_source_stays_read_only(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "old-run"
    run.mkdir(parents=True)
    manifest = run / "manifest.json"
    manifest.write_text(json.dumps({
        "status": "completed", "model": {"model_id": "fictional-saved-model"},
        "usage": {"requests": 0}, "counts": {"completed_samples": 1}, "versions": {},
        "sample_plan": "old/relative-plan.json",
        "samples": [{"sample_id": "s1", "dialogue_id": "missing-fictional-dialogue",
                     "source_file": "missing.jsonl", "summary_status": "complete"}],
    }), encoding="utf-8")
    before = manifest.read_bytes()
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    assert not at.exception
    assert any("Original dialogue unavailable" in warning.value for warning in at.warning)
    assert any("has no source checksum" in caption.value for caption in at.caption)
    assert labelled(at.selectbox, "Run").value == run
    assert not any(expander.label == "Run details" for expander in at.expander)
    assert not at.json
    assert manifest.read_bytes() == before
    assert list(run.iterdir()) == [manifest]

    def runtime_mismatch():
        raise RuntimeError("Simulated package mismatch")

    monkeypatch.setattr(pipeline, "check_runtime_version", runtime_mismatch)
    at.run()
    assert not at.exception
    assert any("Simulated package mismatch" in error.value for error in at.error)
    assert labelled(at.button, "Run All").disabled
    assert labelled(at.selectbox, "Run").value == run
    assert manifest.read_bytes() == before


@pytest.mark.parametrize("source_available", [True, False])
def test_unavailable_or_changed_source_keeps_saved_results_readable(prepared_plan, tmp_path, source_available):
    source = prepared_source(prepared_plan)
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    dialogue = json.loads(source.read_text(encoding="utf-8"))
    dialogue["turns"][0]["text"] = "Changed source sentence that must not be displayed."
    source.write_text(json.dumps(dialogue) + "\n", encoding="utf-8")
    run = tmp_path / "runs" / "checksum-run"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({
        "status": "completed_with_errors", "model": {"model_id": "fictional-saved-model"},
        "usage": {"requests": 0}, "counts": {"completed_samples": 0}, "versions": {},
        "sample_plan": str(prepared_plan),
        "samples": [{"sample_id": "s1", "dialogue_id": dialogue["dialogue_id"],
                     "source_file": str(source if source_available else source.with_name("missing.jsonl")),
                     "source_sha256": original_hash,
                     "summary_status": "diagnostic"}],
    }), encoding="utf-8")
    (run / "summary_annotations.jsonl").write_text(json.dumps({
        "sample_id": "s1", "results": [{
            "dimension": "Accurate Empathy", "activity_reference": None,
            "summary_status": "not_applicable", "rating_label": None,
            "reason": "Saved synthetic summary marker", "evidence": [], "counterevidence": [],
        }],
    }) + "\n", encoding="utf-8")
    (run / "core_evidence.jsonl").write_text(json.dumps({
        "sample_id": "s1", "core": "Bond", "core_element": "mutual respect",
        "source_dimension": "Grounded content uptake", "source_level": "turn-level",
        "current_turn_id": "u2", "mapping_type": "conditional project mapping",
        "activity_reference": None, "status": "rated", "rating_label": "Grounded uptake",
        "reason": "Saved synthetic reason", "counterevidence": [],
        "evidence": [{"turn_id": "u1", "quote": "Saved synthetic core marker"}],
    }) + "\n", encoding="utf-8")
    (run / "validation_errors.jsonl").write_text(json.dumps({
        "sample_id": "s1", "stage": "turn_annotation", "error_type": "fictional_test_error",
        "message": "Saved synthetic error marker",
    }) + "\n", encoding="utf-8")
    before = {path: path.read_bytes() for path in [source, *run.iterdir()]}
    at = AppTest.from_file(app.__file__, default_timeout=APP_TEST_TIMEOUT_SECONDS).run()
    assert not at.exception
    assert any(("Source checksum mismatch" if source_available else "Original dialogue unavailable")
               in warning.value for warning in at.warning)
    assert not any("Changed source sentence" in entry.value for entry in at.markdown)
    assert any("Saved synthetic summary marker" in entry.value for entry in at.markdown)
    assert any("Model response could not be accepted" in entry.value for entry in at.markdown)
    assert not any("Saved synthetic error marker" in entry.value for entry in at.markdown)
    assert any("Saved synthetic core marker" in entry.value for entry in at.markdown)
    assert not any("has no source checksum" in caption.value for caption in at.caption)
    assert {path: path.read_bytes() for path in before} == before
