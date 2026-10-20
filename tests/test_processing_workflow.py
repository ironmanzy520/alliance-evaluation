"""Offline tests for flat conversion output and XLSX input."""

from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pytest

from alliance_eval import cli, input_data
from alliance_eval.input_data import _read_xlsx_records, create_sample_plan, prepare_data


def xlsx_bytes(rows: list[list], *, sheet_names=("Sheet1",), formula=False, merged=False, gap=False) -> bytes:
    """Minimal OOXML fixture for parsing tests, not a deliverable workbook."""
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    fragments = []
    for number, row in enumerate(rows, 1):
        row_number = number + (1 if gap and number > 1 else 0)
        cells = []
        for index, value in enumerate(row):
            address = f"{chr(65 + index)}{row_number}"
            if value is None:
                continue
            if type(value) in (int, float):
                cells.append(f'<c r="{address}"><v>{value}</v></c>')
            else:
                formula_xml = "<f>1+1</f>" if formula and number == 2 and index == 0 else ""
                cells.append(f'<c r="{address}" t="inlineStr">{formula_xml}<is><t xml:space="preserve">{escape(str(value))}</t></is></c>')
        fragments.append(f'<row r="{row_number}">{"".join(cells)}</row>')
    merge = '<mergeCells><mergeCell ref="A1:B1"/></mergeCells>' if merged else ""
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        sheets = "".join(f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>' for i, name in enumerate(sheet_names, 1))
        archive.writestr("xl/workbook.xml", f'<workbook xmlns="{namespace}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>{sheets}</sheets></workbook>')
        relationships = "".join(f'<Relationship Id="rId{i}" Target="worksheets/sheet{i}.xml"/>' for i in range(1, len(sheet_names) + 1))
        archive.writestr("xl/_rels/workbook.xml.rels", f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{relationships}</Relationships>')
        for i in range(1, len(sheet_names) + 1):
            archive.writestr(f"xl/worksheets/sheet{i}.xml", f'<worksheet xmlns="{namespace}"><sheetData>{"".join(fragments)}</sheetData>{merge}</worksheet>')
    return data.getvalue()


def generic_source(tmp_path: Path) -> Path:
    path = tmp_path / "fiction.json"
    path.write_text(json.dumps({"dialogue_id": "fictional-a", "turns": [{"role": "client", "text": " A quiet moment. "}]}))
    return path


def test_xlsx_mapping_preserves_stored_values_and_location(tmp_path):
    source = tmp_path / "input.xlsx"
    source.write_bytes(xlsx_bytes([["session", "role", "text", "turn_index", "duration"], ["f1", "client", " Space kept.\nNext line. ", 4, 0.0125]]))
    before = source.read_bytes()
    output = tmp_path / "processed"
    result = prepare_data(source, output, columns={"dialogue_id": "session"})
    turn = json.loads((output / "input.jsonl").read_text())["turns"][0]
    assert turn["text"] == " Space kept.\nNext line. "
    assert turn["duration"] == 0.0125 and turn["turn_index"] == 4
    assert turn["_conversion"]["location"] == {"sheet": "Sheet1", "row": 2}
    assert result["rules"]["sheet"] == "Sheet1"
    assert source.read_bytes() == before


@pytest.mark.parametrize("options,match", [({"formula": True}, "formula"), ({"merged": True}, "merged"), ({"gap": True}, "blank rows"), ({"sheet_names": ("One", "Two")}, "multiple sheets")])
def test_xlsx_ambiguous_structures_are_rejected(options, match):
    with pytest.raises(ValueError, match=match):
        _read_xlsx_records(xlsx_bytes([["role", "text"], ["client", "A quiet moment."]], **options))


def test_xlsx_multisheet_requires_explicit_selection():
    records, name = _read_xlsx_records(xlsx_bytes([["role", "text"], ["client", "Fiction."]], sheet_names=("One", "Two")), "Two")
    assert name == "Two" and records[0][1]["sheet"] == "Two"


@pytest.mark.parametrize("headers", [["role", "role"], ["role", ""]])
def test_xlsx_headers_cannot_overwrite_or_hide_fields(headers):
    with pytest.raises(ValueError, match="headers"):
        _read_xlsx_records(xlsx_bytes([headers, ["client", "Fiction."]]))


def test_manifest_accumulates_without_overwriting_prior_dataset(tmp_path):
    source = generic_source(tmp_path)
    output = tmp_path / "processed"
    prepare_data(source, output, dataset="one")
    first_bytes = (output / "one.jsonl").read_bytes()
    first_entry = json.loads((output / "manifest.json").read_text())["datasets"]["one"]
    prepare_data(source, output, dataset="two")
    entries = json.loads((output / "manifest.json").read_text())["datasets"]
    assert set(entries) == {"one", "two"}
    assert entries["one"] == first_entry and (output / "one.jsonl").read_bytes() == first_bytes
    with pytest.raises(FileExistsError):
        prepare_data(source, output, dataset="one")


@pytest.mark.parametrize("name", ["../escape", "a/b", "a\\b", ".hidden", "", "a space"])
def test_output_names_cannot_escape_processed(tmp_path, name):
    source = generic_source(tmp_path)
    with pytest.raises(ValueError):
        prepare_data(source, tmp_path / "processed", dataset=name)


def test_existing_lock_and_symlink_do_not_modify_other_files(tmp_path):
    source = generic_source(tmp_path)
    output = tmp_path / "processed"
    output.mkdir()
    lock = output / ".manifest.lock"
    lock.write_text("another process")
    with pytest.raises(FileExistsError):
        prepare_data(source, output)
    assert lock.read_text() == "another process"
    target = output / "linked.jsonl"
    target.symlink_to(source)
    with pytest.raises(FileExistsError):
        prepare_data(source, output, dataset="linked", dry_run=True)
    assert target.is_symlink()


def test_failed_write_keeps_other_manifest_entries(tmp_path, monkeypatch):
    source = generic_source(tmp_path)
    output = tmp_path / "processed"
    prepare_data(source, output, dataset="good")
    before = json.loads((output / "manifest.json").read_text())["datasets"]["good"]
    real_open = Path.open

    def fail_target(path, *args, **kwargs):
        if path == output / "failed.jsonl":
            raise OSError("fictional failure")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_target)
    with pytest.raises(RuntimeError):
        prepare_data(source, output, dataset="failed")
    entries = json.loads((output / "manifest.json").read_text())["datasets"]
    assert entries["good"] == before
    assert entries["failed"]["status"] == "failed"
    assert not (output / ".manifest.lock").exists()


def test_preflight_and_write_must_be_identical(tmp_path, monkeypatch):
    source = generic_source(tmp_path)
    output = tmp_path / "processed"
    original = input_data._jsonl_bytes

    def changing(record):
        payload = original(record)
        if output.exists():
            return payload.replace(b"quiet", b"calm")
        return payload

    monkeypatch.setattr(input_data, "_jsonl_bytes", changing)
    with pytest.raises(RuntimeError, match="differs"):
        prepare_data(source, output)
    assert (output / "fiction.jsonl").exists()
    assert json.loads((output / "manifest.json").read_text())["datasets"]["fiction"]["status"] == "failed"


def test_plan_requires_selection_and_matching_completed_source(tmp_path):
    source = generic_source(tmp_path)
    output = tmp_path / "processed"
    prepare_data(source, output)
    processed = output / "fiction.jsonl"
    plan = tmp_path / "plans" / "selected.json"
    for options in ({}, {"dialogue_ids": ["missing"]}, {"dialogue_ids": ["fictional-a", "fictional-a"]}):
        with pytest.raises(ValueError):
            create_sample_plan(processed, plan, **options)
        assert not plan.parent.exists()
    create_sample_plan(processed, plan, dialogue_ids=["fictional-a"])
    assert len(json.loads(plan.read_text())["items"]) == 1
    with pytest.raises(FileExistsError):
        create_sample_plan(processed, plan, select_all=True)
    processed.write_bytes(processed.read_bytes().replace(b"quiet", b"calm"))
    with pytest.raises(ValueError, match="checksum"):
        create_sample_plan(processed, plan.with_name("second.json"), select_all=True)


@pytest.mark.parametrize("input_format", ["csv", "json", "jsonl", "xlsx"])
def test_generic_formats_are_repeatable_and_loadable_as_complete_dialogues(tmp_path, input_format):
    """All formats share one field contract; output names do not alter data and plans remain readable."""
    from alliance_eval.evaluation_models import SamplePlan
    from alliance_eval.pipeline import load_selected_dialogues

    rows = [
        {"session": "independent-B", "uid": "b0", "position": 0, "speaker": "H", "utterance": "  A fictional question,\nkept intact.  ", "note": "original"},
        {"session": "independent-A", "uid": "a1", "position": 1, "speaker": "O", "utterance": "A bell rings.", "note": "fictional context"},
        {"session": "independent-B", "uid": "b8", "position": 8, "speaker": "P", "utterance": "I wrote a list.", "note": "fictional response"},
    ]
    source = tmp_path / f"source.{input_format}"
    if input_format == "csv":
        with source.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    elif input_format == "json":
        source.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    elif input_format == "jsonl":
        source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    else:
        source.write_bytes(xlsx_bytes([list(rows[0]), *[list(row.values()) for row in rows]], sheet_names=("User dialogue",)))
    source_bytes = source.read_bytes()
    options = {"columns": {"dialogue_id": "session", "turn_id": "uid", "turn_index": "position", "role": "speaker", "text": "utterance"}, "role_map": {"H": "counsellor", "P": "client", "O": "other"}}
    output = tmp_path / "processed"
    preview = prepare_data(source, output, dataset="custom-label", dry_run=True, **options)
    assert not output.exists()
    result = prepare_data(source, output, dataset="custom-label", **options)
    repeated = prepare_data(source, output, dataset="unrelated-label", **options)
    processed = output / "custom-label.jsonl"
    assert processed.read_bytes() == (output / "unrelated-label.jsonl").read_bytes()
    assert preview["output_sha256"] == result["output_sha256"] == repeated["output_sha256"]
    assert result["converter_sha256"] == {"input_data.py": input_data.sha256_file(Path(input_data.__file__))}
    records = [json.loads(line) for line in processed.read_text().splitlines()]
    assert [record["dialogue_id"] for record in records] == ["independent-B", "independent-A"]
    assert [turn["turn_index"] for turn in records[0]["turns"]] == [0, 8]
    assert records[0]["turns"][0]["text"] == rows[0]["utterance"]
    assert records[0]["turns"][0]["note"] == "original"
    assert records[1]["turns"][0]["role"] == "other"
    assert result["generated_fields"] == {}
    assert source.read_bytes() == source_bytes
    plan_file = tmp_path / "plans" / "selected.json"
    create_sample_plan(processed, plan_file, dialogue_ids=["independent-B"])
    plan = SamplePlan.model_validate(json.loads(plan_file.read_text()))
    selected = load_selected_dialogues(plan, plan_file.parent)
    assert len(selected) == 1 and len(selected[0]["turns"]) == 2
    assert selected[0]["turns"][0]["text"] == rows[0]["utterance"]


def test_nested_user_data_generates_only_missing_fields_and_preserves_metadata(tmp_path):
    source = tmp_path / "bespoke.json"
    records = [{"conversation": f"own-{number}", "messages": [{"speaker": "H", "words": "Can we check the list?"}, {"speaker": "P", "words": "I marked one item.", "annotations": {"source_label": "custom"}}], "metadata": {"dataset": "user-defined", "therapy": "user-defined", "language": "en"}} for number in range(23)]
    source.write_text(json.dumps(records), encoding="utf-8")
    result = prepare_data(source, tmp_path / "processed", columns={"dialogue_id": "conversation", "turns": "messages", "role": "speaker", "text": "words"}, role_map={"H": "counsellor", "P": "client"})
    assert result["dialogue_count"] == 23 and result["turn_count"] == 46
    assert len(result["dialogues"]) == 20 and result["dialogues_not_shown"] == 3
    assert result["generated_fields"] == {"turn_id": 46, "turn_index": 46}
    saved = [json.loads(line) for line in (tmp_path / "processed/bespoke.jsonl").read_text().splitlines()]
    assert len(saved) == 23
    assert saved[-1]["metadata"] == records[-1]["metadata"]
    assert saved[-1]["turns"][1]["annotations"] == {"source_label": "custom"}


def test_stale_input_module_blocks_xlsx_conversion_before_writing(tmp_path, monkeypatch):
    """XLSX reading is covered by the shared input module's loaded-source fingerprint."""
    source = tmp_path / "input.xlsx"
    source.write_bytes(xlsx_bytes([["session", "role", "text"], ["f1", "client", "A fictional response."]]))
    before = source.read_bytes()
    monkeypatch.setattr(input_data, "LOADED_SOURCE_SHA256", "stale")
    with pytest.raises(RuntimeError, match="restart the process: input_data.py"):
        prepare_data(source, tmp_path / "processed", columns={"dialogue_id": "session"})
    assert not (tmp_path / "processed").exists()
    assert source.read_bytes() == before


@pytest.mark.parametrize("kind", ["directory", "unsupported-file"])
def test_non_file_or_unsupported_format_fails_before_output(tmp_path, kind):
    source = tmp_path / "raw"
    if kind == "directory":
        source.mkdir()
    else:
        source.write_text("H: A fictional sentence.\nP: A response.")
    with pytest.raises(ValueError, match="file|format"):
        prepare_data(source, tmp_path / "processed")
    assert not (tmp_path / "processed").exists()


def test_cli_has_no_source_profile_switch(tmp_path, monkeypatch, capsys):
    source = generic_source(tmp_path)
    monkeypatch.setattr(sys, "argv", ["alliance-eval", "prepare-data", "--input", str(source), "--profile", "custom", "--output-dir", str(tmp_path / "processed")])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
    assert not (tmp_path / "processed").exists()


def test_existing_completed_entry_remains_usable_with_its_own_provenance(tmp_path):
    """New conversions preserve source rules and runtime fingerprints in previous complete entries."""
    source = generic_source(tmp_path)
    output = tmp_path / "processed"
    prepare_data(source, output, dataset="previous")
    manifest_file = output / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    entry = manifest["datasets"]["previous"]
    entry["rules"] = {"source_method": "user-supplied-import", "notes": ["retain exactly"]}
    entry["converter_sha256"] = {"previous-converter.py": "a" * 64}
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    previous_bytes = (output / "previous.jsonl").read_bytes()
    plan = tmp_path / "plans" / "previous.json"
    create_sample_plan(output / "previous.jsonl", plan, select_all=True)
    assert json.loads(plan.read_text())["items"][0]["dialogue_id"] == "fictional-a"
    prepare_data(source, output, dataset="new")
    assert json.loads(manifest_file.read_text())["datasets"]["previous"] == entry
    assert (output / "previous.jsonl").read_bytes() == previous_bytes


def test_tsv_reuses_mapping_preserves_multiline_values_and_original_indices(tmp_path):
    source = tmp_path / "transcript.tsv"
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\r\n")
    writer.writerows([
        ["visit", "speaker", "words", "position", "note"],
        ["one", "H", '  A fictional pause.\r\nTab\tand "quotes" kept.  ', "004", "0"],
        ["two", "P", "Another fictional visit.", "2", "false"],
        ["one", "P", "A fictional answer.", "9", ""],
    ])
    original = b"\xef\xbb\xbf" + buffer.getvalue().encode("utf-8")
    source.write_bytes(original)
    output = tmp_path / "processed"
    result = prepare_data(source, output, columns={"dialogue_id": "visit", "role": "speaker", "text": "words", "turn_index": "position"}, role_map={"H": "counsellor", "P": "client"})
    records = [json.loads(line) for line in (output / "transcript.jsonl").read_text().splitlines()]
    turn = records[0]["turns"][0]
    assert [record["dialogue_id"] for record in records] == ["one", "two"]
    assert [turn["turn_index"] for turn in records[0]["turns"]] == [4, 9]
    assert turn["text"] == '  A fictional pause.\r\nTab\tand "quotes" kept.  '
    assert turn["note"] == "0" and turn["_conversion"]["original_values"]["position"] == "004"
    assert turn["_conversion"]["location"] == {"line_start": 2, "line_end": 3}
    assert result["rules"]["delimiter"] == "\t" and source.read_bytes() == original


@pytest.mark.parametrize("suffix,first,last,column", [
    ("txt", "H:", "P：", 3),
    ("md", "**H:**", "**P**：", 7),
    ("md", "**H**:", "P:", 7),
])
@pytest.mark.parametrize("ending", ["\n", "\r\n", "\r"])
def test_transcript_preserves_body_lines_locations_and_repeatability(tmp_path, suffix, first, last, column, ending):
    source = tmp_path / f"transcript.{suffix}"
    original = (first + "  Fictional opening.  " + ending + ending + "  More: words\tand \u2028kept." + ending + last + " Fictional reply. " + ending).encode("utf-8")
    source.write_bytes(original)
    options = {"dialogue_id": "explicit-visit", "role_map": {"H": "counsellor", "P": "client"}}
    first_output, second_output = tmp_path / "one", tmp_path / "two"
    preview = prepare_data(source, first_output, dry_run=True, **options)
    assert not first_output.exists() and preview["turn_count"] == 2
    result = prepare_data(source, first_output, **options)
    prepare_data(source, second_output, **options)
    saved = first_output / "transcript.jsonl"
    payload = json.loads(saved.read_text())
    turn = payload["turns"][0]
    assert payload["dialogue_id"] == "explicit-visit"
    assert turn["text"] == "  Fictional opening.  " + ending * 2 + "  More: words\tand \u2028kept."
    assert turn["_conversion"]["location"] == {"line_start": 1, "line_end": 3, "text_column": column}
    assert turn["_conversion"]["original_values"]["role"] == "H"
    assert payload["turns"][1]["text"] == " Fictional reply. "
    assert result["rules"]["dialogue_id"] == "explicit-visit"
    assert result["generated_fields"] == {"turn_id": 2, "turn_index": 2}
    before = {file.name: file.read_bytes() for file in first_output.iterdir()}
    with pytest.raises(FileExistsError):
        prepare_data(source, first_output, **options)
    assert {file.name: file.read_bytes() for file in first_output.iterdir()} == before
    assert saved.read_bytes() == (second_output / "transcript.jsonl").read_bytes()
    assert source.read_bytes() == original
    plan = tmp_path / "plans" / "chosen.json"
    create_sample_plan(saved, plan, dialogue_ids=["explicit-visit"])
    assert json.loads(plan.read_text())["items"][0]["dialogue_id"] == "explicit-visit"


@pytest.mark.parametrize("suffix,body,options,match", [
    ("txt", "client: Fiction.", {}, "dialogue_id"),
    ("txt", "client: Fiction.", {"dialogue_id": " "}, "dialogue_id"),
    ("txt", "", {"dialogue_id": "visit"}, "at least one"),
    ("txt", "\nclient: Fiction.", {"dialogue_id": "visit"}, "[Ll]ine 1"),
    ("txt", "Preface\nclient: Fiction.", {"dialogue_id": "visit"}, "[Ll]ine 1"),
    ("txt", "client: Fiction.\nUnknown: New speaker.", {"dialogue_id": "visit"}, "[Ll]ine 2"),
    ("txt", "client: \ncounsellor: Fiction.", {"dialogue_id": "visit"}, "text"),
    ("txt", "client: Fiction.", {"dialogue_id": "visit", "columns": {"text": "words"}}, "columns"),
    ("txt", "client: Fiction.", {"dialogue_id": "visit", "sheet": "One"}, "sheet"),
    ("md", "# Visit\n**client:** Fiction.", {"dialogue_id": "visit"}, "document structure"),
    ("md", "client: Fiction.\n```text\nMore", {"dialogue_id": "visit"}, "document structure"),
    ("md", "client: Fiction.\n|role|text|", {"dialogue_id": "visit"}, "document structure"),
    ("md", "client: Fiction.\n---\nclient: Next visit.", {"dialogue_id": "visit"}, "document structure"),
    ("tsv", "dialogue_id\trole\ttext\nv\tclient\tFiction.", {"dialogue_id": "v"}, "applies only"),
    ("tsv", "dialogue_id\trole\ttext\nv\tclient", {}, "field count"),
    ("tsv", "dialogue_id\trole\trole\nv\tclient\tFiction.", {}, "headers must not be duplicated"),
    ("tsv", "role\ttext\nclient\tFiction.", {}, "dialogue_id"),
    ("tsv", "dialogue_id\trole\ttext\n\n", {}, "field count"),
])
def test_text_inputs_reject_ambiguous_or_invalid_structure_without_writes(tmp_path, suffix, body, options, match):
    source = tmp_path / f"input.{suffix}"
    source.write_bytes(body.encode("utf-8"))
    before = source.read_bytes()
    with pytest.raises(ValueError, match=match):
        prepare_data(source, tmp_path / "processed", **options)
    assert source.read_bytes() == before and not (tmp_path / "processed").exists()


@pytest.mark.parametrize("suffix", ["txt", "md", "tsv"])
def test_new_text_formats_use_cli_and_same_write_path(tmp_path, monkeypatch, capsys, suffix):
    source = tmp_path / f"source.{suffix}"
    content = "client: A fictional observation.\n"
    extra = ["--dialogue-id", "visit"]
    if suffix == "tsv":
        content = "dialogue_id\trole\ttext\nvisit\tclient\tA fictional observation.\n"
        extra = []
    source.write_bytes(content.encode("utf-8"))
    output = tmp_path / "processed"
    args = ["alliance-eval", "prepare-data", "--input", str(source), "--output-dir", str(output), *extra]
    monkeypatch.setattr(sys, "argv", [*args, "--dry-run"])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "validated"
    assert not output.exists()
    monkeypatch.setattr(sys, "argv", args)
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    assert json.loads((output / "source.jsonl").read_text())["dialogue_id"] == "visit"


def test_transcript_rejects_stale_loaded_converter_before_write(tmp_path, monkeypatch):
    source = tmp_path / "input.txt"
    source.write_text("client: Fiction.", encoding="utf-8")
    monkeypatch.setattr(input_data, "LOADED_SOURCE_SHA256", "stale")
    with pytest.raises(RuntimeError, match="restart the process"):
        prepare_data(source, tmp_path / "processed", dialogue_id="visit")
    assert not (tmp_path / "processed").exists()
