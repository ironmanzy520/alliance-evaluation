"""Offline JSONL reader regressions using original fictional fixtures only.

These tests check text preservation and display counts, not rating accuracy.
The shared conftest blocks network connections and removes model credentials.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from alliance_eval import app, pipeline


def _dialogue(dialogue_id: str, text: str) -> dict:
    return {
        "dialogue_id": dialogue_id,
        "turns": [
            {"turn_id": "u1", "turn_index": 0, "role": "client", "text": text},
        ],
    }


def _saved_record(kind: str, separator: str, turn_id: str = "u1") -> dict:
    item = {
        "dimension": "Accurate Empathy" if kind == "summary" else "Client confirmation/disconfirmation",
        "activity_reference": None,
        "rating_label": "AE-5" if kind == "summary" else "Clear confirmation",
        "evidence": [{"turn_id": turn_id, "quote": f"First{separator}second."}],
        "reason": f"Fictional reader fixture: preserve{separator}this text.",
    }
    if kind == "summary":
        item.update(summary_status="rated", counterevidence=[])
        return {"sample_id": "s1", "results": [item]}
    item["finding_status"] = "rated"
    return {"sample_id": "s1", "current_turn_id": turn_id, "findings": [item]}


def _manifest() -> dict:
    return {
        "counts": {"failed_calls": 0},
        "usage": {"requests": 1},
        "samples": [{
            "sample_id": "s1",
            "annotation_complete_by_core": {"Bond": True, "Task": True},
            "summary_status_by_core": {"Bond": "complete", "Task": "complete"},
        }],
    }


@pytest.mark.parametrize("separator", ["\u2028", "\u2029"], ids=["line-separator", "paragraph-separator"])
@pytest.mark.parametrize("with_bom", [False, True], ids=["utf8", "utf8-bom"])
@pytest.mark.parametrize("final_newline", [False, True], ids=["no-final-newline", "final-newline"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_source_preview_matches_pipeline_text_and_checksum(
    tmp_path: Path, separator: str, with_bom: bool, final_newline: bool, newline: str,
) -> None:
    records = [
        _dialogue("fictional-first", "A separate fictional dialogue."),
        _dialogue("fictional-target", f"First{separator}second."),
    ]
    text = newline.join(json.dumps(record, ensure_ascii=False) for record in records)
    if final_newline:
        text += newline
    content = (b"\xef\xbb\xbf" if with_bom else b"") + text.encode("utf-8")
    assert separator.encode("utf-8") in content
    source = tmp_path / "dialogues.jsonl"
    source.write_bytes(content)
    expected_sha256 = hashlib.sha256(content).hexdigest()

    pipeline_records, pipeline_sha256 = pipeline._load_source_dialogues(source)
    assert pipeline_records == {record["dialogue_id"]: record for record in records}
    assert pipeline_sha256 == expected_sha256
    for record in records:
        assert app.load_dialogue(
            str(source), record["dialogue_id"], expected_sha256=expected_sha256,
        ) == record
    assert app.load_dialogue(
        str(source), "missing-dialogue", expected_sha256=expected_sha256,
    ) is None
    assert source.read_bytes() == content


def test_source_checksum_is_checked_on_the_same_bytes_before_decoding(
    tmp_path: Path, monkeypatch,
) -> None:
    record = _dialogue("fictional-target", "First\u2028second\u2029third.")
    content = b"\xef\xbb\xbf" + (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    source = tmp_path / "dialogues.jsonl"
    source.write_bytes(content)
    expected_sha256 = hashlib.sha256(content).hexdigest()
    reads = []
    original_read_bytes = Path.read_bytes

    def counted_read_bytes(path):
        if path == source:
            reads.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    assert app.load_dialogue(
        str(source), record["dialogue_id"], expected_sha256=expected_sha256,
    ) == record
    assert reads == [source]
    assert original_read_bytes(source) == content

    changed = b"\xffChanged bytes are neither UTF-8 nor JSON."
    source.write_bytes(changed)
    with pytest.raises(ValueError, match="Source checksum mismatch"):
        app.load_dialogue(str(source), record["dialogue_id"], expected_sha256=expected_sha256)
    assert reads == [source, source]
    assert original_read_bytes(source) == changed


def test_source_preview_does_not_skip_an_internal_empty_record(tmp_path: Path) -> None:
    first = _dialogue("fictional-first", "A fictional first record.")
    target = _dialogue("fictional-target", "A fictional target record.")
    content = (json.dumps(first) + "\n\n" + json.dumps(target) + "\n").encode("utf-8")
    source = tmp_path / "dialogues.jsonl"
    source.write_bytes(content)
    with pytest.raises(ValueError):
        pipeline._load_source_dialogues(source)
    with pytest.raises(ValueError):
        app.load_dialogue(str(source), target["dialogue_id"])
    assert source.read_bytes() == content


@pytest.mark.parametrize("separator", ["\u2028", "\u2029"], ids=["line-separator", "paragraph-separator"])
@pytest.mark.parametrize("kind", ["turn", "summary"])
@pytest.mark.parametrize("final_newline", [False, True], ids=["no-final-newline", "final-newline"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_saved_unicode_evidence_and_category_counts_are_preserved(
    tmp_path: Path, separator: str, kind: str, final_newline: bool, newline: str,
) -> None:
    record = _saved_record(kind, separator)
    text = json.dumps(record, ensure_ascii=False) + (newline if final_newline else "")
    content = text.encode("utf-8")
    assert separator.encode("utf-8") in content
    path = tmp_path / f"{kind}_annotations.jsonl"
    path.write_bytes(content)
    warnings = []

    records = app.read_jsonl(path, warnings=warnings)
    assert records == [record]
    assert warnings == []
    statistics = app.run_statistics(
        _manifest(), records if kind == "turn" else [], records if kind == "summary" else [],
    )
    item = record["results" if kind == "summary" else "findings"][0]
    assert statistics["categories"] == [{
        "dimension": item["dimension"],
        "status": "rated",
        "label": item["rating_label"],
        "results": 1,
    }]
    assert statistics["excluded_results"] is False
    assert path.read_bytes() == content


@pytest.mark.parametrize("separator", ["\u2028", "\u2029"], ids=["line-separator", "paragraph-separator"])
def test_saved_reader_reports_real_bad_lines_without_losing_valid_records(
    tmp_path: Path, separator: str,
) -> None:
    first = _saved_record("turn", separator)
    second = _saved_record("turn", separator, turn_id="u2")
    content = (
        json.dumps(first, ensure_ascii=False) + "\nnot valid JSON\n\n"
        + json.dumps(second, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    path = tmp_path / "turn_annotations.jsonl"
    path.write_bytes(content)
    warnings = []

    records = app.read_jsonl(path, warnings=warnings)
    assert records == [first, second]
    assert len(warnings) == 1
    assert warnings[0].startswith("turn_annotations.jsonl:2:")
    statistics = app.run_statistics(_manifest(), records, [])
    assert statistics["categories"] == [{
        "dimension": "Client confirmation/disconfirmation",
        "status": "rated",
        "label": "Clear confirmation",
        "results": 2,
    }]
    with pytest.raises(ValueError):
        app.read_jsonl(path)
    assert path.read_bytes() == content
