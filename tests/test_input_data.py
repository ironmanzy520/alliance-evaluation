"""Local engineering tests for generic input."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alliance_eval.input_data import (
    LOADED_SOURCE_SHA256,
    create_sample_plan,
    dialogue_json_schema,
    export_input_schema,
    parse_json_value,
    prepare_data,
    validate_dialogue,
)


def dialogue() -> dict:
    return {
        "dialogue_id": "fictional-1",
        "turns": [
            {"turn_id": "original-a", "turn_index": 0, "role": "counsellor", "text": " Shall we pause?\n"},
            {"turn_id": "original-z", "turn_index": 4, "role": "client", "text": "Yes, briefly."},
        ],
    }


def write_input(tmp_path: Path, value: object, name: str = "input.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def read_output(directory: Path) -> list[dict]:
    return [json.loads(line) for line in (directory / "input.jsonl").read_text(encoding="utf-8").split("\n") if line]


def test_validation_is_pure_and_preserves_unrestricted_source_metadata() -> None:
    record = dialogue()
    record.update(schema_version="source-format", dataset="independent-source", therapy="unspecified", access={"raw_text_restricted": True}, metadata={"nested": [1, False, None]}, quality_flags=["fictional"])
    record["turns"][0].update(original_role="helper", original_text=" Original text. ", annotations={"author": "fictional"})
    before = copy.deepcopy(record)
    checked = validate_dialogue(record)
    assert checked == before == record
    checked["metadata"]["nested"].append("new")
    assert record == before


@pytest.mark.parametrize("existing_format", ["processed", "processed-v1"])
def test_prepare_uses_fixed_manifest_identifier_and_keeps_existing_entries(tmp_path: Path, existing_format: str) -> None:
    source = write_input(tmp_path, dialogue())
    destination = tmp_path / "prepared"
    destination.mkdir()
    retained = {"prior": {"source": "preserved", "sha256": "unchanged"}}
    manifest = destination / "manifest.json"
    manifest.write_text(json.dumps({"manifest_version": existing_format, "datasets": retained}))
    prepare_data(source, destination)
    result = json.loads(manifest.read_text())
    assert result["manifest_version"] == "processed"
    assert result["datasets"]["prior"] == retained["prior"]
    assert "input" in result["datasets"]


@pytest.mark.parametrize("field,value", [("turn_id", " "), ("turn_id", 1), ("turn_index", True), ("turn_index", "1"), ("turn_index", -1), ("role", "therapist"), ("text", "\n\t"), ("text", None)])
def test_invalid_turn_fields_are_rejected(field: str, value: object) -> None:
    record = dialogue()
    record["turns"][0][field] = value
    with pytest.raises(ValueError):
        validate_dialogue(record)


@pytest.mark.parametrize("mutation", ["missing", "empty", "duplicate_id", "duplicate_index", "reversed"])
def test_validation_does_not_repair_turns(mutation: str) -> None:
    record = dialogue()
    if mutation == "missing":
        record["turns"][0].pop("turn_index")
    elif mutation == "empty":
        record["turns"] = []
    elif mutation == "duplicate_id":
        record["turns"][1]["turn_id"] = record["turns"][0]["turn_id"]
    elif mutation == "duplicate_index":
        record["turns"][1]["turn_index"] = record["turns"][0]["turn_index"]
    else:
        record["turns"].reverse()
    with pytest.raises(ValueError):
        validate_dialogue(record)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), (1, 2), {1: "not a JSON key"}])
def test_metadata_must_be_losslessly_json_serialisable(value: object) -> None:
    record = dialogue()
    record["metadata"] = value
    with pytest.raises(ValueError):
        validate_dialogue(record)


def test_dry_run_has_no_writes_and_reports_all_dialogues(tmp_path: Path) -> None:
    second = dialogue()
    second["dialogue_id"] = "fictional-2"
    source = write_input(tmp_path, [dialogue(), second])
    before = source.read_bytes()
    destination = tmp_path / "prepared"
    preview = prepare_data(source, destination, dry_run=True)
    assert preview["status"] == "validated"
    assert preview["dialogue_count"] == 2 and preview["turn_count"] == 4
    assert not destination.exists()
    assert source.read_bytes() == before


def test_missing_technical_fields_are_generated_and_traceable(tmp_path: Path) -> None:
    source = write_input(tmp_path, {"turns": [{"role": "other", "text": "A bell rings.", "note": "fictional"}]})
    destination = tmp_path / "prepared"
    result = prepare_data(source, destination)
    record = read_output(destination)[0]
    assert record["dialogue_id"] == "generated-dialogue-000001"
    assert record["turns"][0]["turn_id"] == "generated-turn-000001"
    assert record["turns"][0]["turn_index"] == 1
    assert result["generated_fields"] == {"dialogue_id": 1, "turn_id": 1, "turn_index": 1}
    assert record["turns"][0]["_conversion"]["location"] == {"record": 1, "turn": 1}
    assert record["turns"][0]["note"] == "fictional"


def test_csv_mapping_preserves_text_ids_extra_fields_and_original_role(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    source.write_bytes(b'session,uid,order,speaker,utterance,note\r\ns1,u1,0,H," Space, kept.\r\nNext line. ",fictional\r\ns1,u9,7,P,Okay.,source-note\r\n')
    destination = tmp_path / "prepared"
    result = prepare_data(source, destination, columns={"dialogue_id": "session", "turn_id": "uid", "turn_index": "order", "role": "speaker", "text": "utterance"}, role_map={"H": "counsellor", "P": "client"})
    record = read_output(destination)[0]
    first = record["turns"][0]
    assert first["text"] == " Space, kept.\r\nNext line. "
    assert first["turn_id"] == "u1" and record["turns"][1]["turn_id"] == "u9"
    assert first["note"] == "fictional"
    assert first["_conversion"]["original_values"] == {"order": "0", "speaker": "H"}
    assert "utterance" not in first
    assert result["role_counts"] == {"client": 1, "counsellor": 1, "other": 0}


def test_nested_alias_mapping_does_not_duplicate_dialogue_text(tmp_path: Path) -> None:
    source = write_input(tmp_path, {"id": "custom", "messages": [{"speaker": "H", "content": "A fictional pause."}], "metadata": {"x": 1}})
    destination = tmp_path / "prepared"
    prepare_data(source, destination, columns={"dialogue_id": "id", "turns": "messages", "role": "speaker", "text": "content"}, role_map={"H": "counsellor"})
    record = read_output(destination)[0]
    assert record["metadata"] == {"x": 1}
    assert "messages" not in record and "id" not in record
    assert (destination / "input.jsonl").read_text().count("A fictional pause.") == 1
    assert record["_conversion"]["field_map"] == {"dialogue_id": "id", "turns": "messages"}


def test_overwritten_metadata_and_previous_provenance_are_not_lost(tmp_path: Path) -> None:
    source = write_input(tmp_path, {"turns": [{"role": "client", "text": "metadata value", "utterance": "Actual fictional words.", "_conversion": {"source": "earlier"}}]})
    destination = tmp_path / "prepared"
    prepare_data(source, destination, columns={"text": "utterance"})
    provenance = read_output(destination)[0]["turns"][0]["_conversion"]
    assert provenance["replaced_fields"] == {"text": "metadata value"}
    assert provenance["previous"] == {"source": "earlier"}


def test_flat_json_rows_preserve_order_and_all_extra_metadata(tmp_path: Path) -> None:
    rows = [{"dialogue_id": key, "role": "other", "text": text, "source_meta": {"index": index}} for index, (key, text) in enumerate([("B", "First."), ("A", "Second."), ("B", "Third.")])]
    source = write_input(tmp_path, rows)
    destination = tmp_path / "prepared"
    prepare_data(source, destination)
    records = read_output(destination)
    assert [record["dialogue_id"] for record in records] == ["B", "A"]
    assert [turn["text"] for turn in records[0]["turns"]] == ["First.", "Third."]
    assert records[0]["turns"][1]["source_meta"] == {"index": 2}


def test_plan_is_relative_to_its_directory_and_manifest_hashes_match(tmp_path: Path) -> None:
    source = write_input(tmp_path, dialogue())
    destination = tmp_path / "prepared"
    prepare_data(source, destination)
    assert not (destination / "sample_plan.json").exists()
    plan_path = tmp_path / "plans" / "selected.json"
    create_sample_plan(destination / "input.jsonl", plan_path, dialogue_ids=["fictional-1"])
    plan = json.loads(plan_path.read_text())
    assert plan["items"][0]["source_file"] == "../prepared/input.jsonl"
    assert plan["items"][0]["start_turn_index"] is None
    manifest = json.loads((destination / "manifest.json").read_text())["datasets"]["input"]
    assert manifest["status"] == "complete"
    assert manifest["sources"][0]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert hashlib.sha256((destination / manifest["output_file"]).read_bytes()).hexdigest() == manifest["output_sha256"]


@pytest.mark.parametrize("filename,content", [("input.json", '{"turns":[],"turns":[]}'), ("input.json", '{"turns":[],"extra":NaN}'), ("input.jsonl", '{}\n\n{}\n'), ("input.csv", "role,text,text\nclient,One.,Two.\n"), ("input.csv", "role,text\nclient,Words.,extra\n"), ("input.csv", "role,text\n\nclient,Words.\n")])
def test_malformed_input_is_rejected_before_any_output(tmp_path: Path, filename: str, content: str) -> None:
    source = tmp_path / filename
    source.write_text(content)
    destination = tmp_path / "prepared"
    with pytest.raises(ValueError):
        prepare_data(source, destination)
    assert not destination.exists()


def test_jsonl_unicode_separator_is_part_of_original_text(tmp_path: Path) -> None:
    source = tmp_path / "input.jsonl"
    record = dialogue()
    record["turns"][0]["text"] = "A\u2028B"
    source.write_text(json.dumps(record, ensure_ascii=False) + "\n")
    destination = tmp_path / "prepared"
    prepare_data(source, destination)
    assert read_output(destination)[0]["turns"][0]["text"] == "A\u2028B"


@pytest.mark.parametrize("payload,options", [([dialogue(), dialogue()], {}), ({"turns": [{"role": "helper", "text": "Fiction."}]}, {}), ({"turns": [{"role": "H", "text": "Fiction."}]}, {"role_map": {"P": "client"}}), ({"turns": [{"role": "other", "text": "Fiction."}]}, {"columns": {"turn_id": "missing_id"}}), ({"turns": [{"role": "other", "text": "Fiction.", "turn_id": ""}]}, {}), ([{"dialogue_id": "a", "role": "client", "text": "One."}, {"role": "client", "text": "Two."}], {})])
def test_ambiguous_or_invalid_mapping_is_not_repaired(tmp_path: Path, payload: object, options: dict) -> None:
    source = write_input(tmp_path, payload)
    destination = tmp_path / "prepared"
    with pytest.raises(ValueError):
        prepare_data(source, destination, **options)
    assert not destination.exists()


def test_existing_output_and_input_are_never_overwritten(tmp_path: Path) -> None:
    source = write_input(tmp_path, dialogue())
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "saved.txt"
    marker.write_text("keep")
    before = source.read_bytes()
    prepare_data(source, destination)
    for target in (destination, source):
        with pytest.raises(FileExistsError):
            prepare_data(source, target)
    assert marker.read_text() == "keep" and source.read_bytes() == before


def test_mapping_cannot_reuse_a_default_source_field(tmp_path: Path) -> None:
    source = write_input(tmp_path, {"turns": [{"text": "counsellor"}]})
    with pytest.raises(ValueError, match="collision"):
        prepare_data(source, columns={"role": "text"}, dry_run=True)


@pytest.mark.parametrize("text", ['{"id":1,"id":2}', '{"metadata":{"a":1,"a":2}}', '{"value":1e400}'])
def test_shared_json_parser_rejects_overwrite_and_overflow(text: str) -> None:
    with pytest.raises(ValueError):
        parse_json_value(text)


def test_empty_columns_uses_canonical_fields(tmp_path: Path) -> None:
    source = write_input(tmp_path, dialogue())
    preview = prepare_data(source, columns={}, dry_run=True)
    assert preview["turn_count"] == 2 and preview["generated_fields"] == {}


@pytest.mark.parametrize("filename,content", [
    ("input.csv", "role,text\nclient,One fictional line.\n"),
    ("input.json", '[{"role":"client","text":"One fictional line."}]'),
    ("input.jsonl", '{"role":"client","text":"One fictional line."}\n'),
])
def test_flat_rows_without_dialogue_identity_cannot_be_merged(tmp_path: Path, filename: str, content: str) -> None:
    source = tmp_path / filename
    source.write_text(content)
    destination = tmp_path / "new-parent" / "prepared"
    with pytest.raises(ValueError, match="dialogue boundaries"):
        prepare_data(source, destination)
    assert not destination.parent.exists()


def test_nested_output_parents_are_created_only_after_validation(tmp_path: Path) -> None:
    source = write_input(tmp_path, dialogue())
    destination = tmp_path / "new-workspace" / "data" / "prepared"
    prepare_data(source, destination, dry_run=True)
    assert not destination.parents[1].exists()
    result = prepare_data(source, destination)
    assert result["status"] == "complete"
    assert (destination / "manifest.json").is_file()


def test_failed_write_retains_explicit_incomplete_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = write_input(tmp_path, dialogue())
    destination = tmp_path / "prepared"
    real_open = Path.open

    def fail_output(path: Path, *args, **kwargs):
        if path == destination / "input.jsonl":
            raise OSError("simulated local write failure")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_output)
    with pytest.raises(RuntimeError, match="incomplete directory retained"):
        prepare_data(source, destination)
    manifest = json.loads((destination / "manifest.json").read_text())["datasets"]["input"]
    assert manifest["status"] == "failed"
    assert manifest["completed_files"] == []
    assert "simulated local write failure" in manifest["error"]


def test_schema_is_current_and_export_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "schema.json"
    export_input_schema(path)
    assert json.loads(path.read_text()) == dialogue_json_schema()
    with pytest.raises(FileExistsError):
        export_input_schema(path)
    stored = Path(__file__).resolve().parents[1] / "schemas" / "dialogue.schema.json"
    assert json.loads(stored.read_text()) == dialogue_json_schema()
    assert len(LOADED_SOURCE_SHA256) == 64
