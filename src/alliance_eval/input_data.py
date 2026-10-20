"""Counselling dialogue input with explicit mappings, text preservation, validation and conversion provenance."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import os
import posixpath
import re
import tempfile
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from xml.etree import ElementTree as ET

from pydantic import BaseModel, ConfigDict, Field, model_validator


INPUT_SCHEMA_VERSION = "dialogue"
LOADED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ROLES = {"counsellor", "client", "other"}
FIELDS = {"dialogue_id", "turns", "turn_id", "turn_index", "role", "text"}
_MISSING = object()
_XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_XLSX_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


class InputTurn(BaseModel):
    """Minimal turn contract; extra source annotations and original roles do not define ratings."""

    model_config = ConfigDict(extra="allow", strict=True)

    turn_id: str = Field(min_length=1, pattern=r"\S")
    turn_index: int = Field(ge=0)
    role: Literal["counsellor", "client", "other"]
    text: str = Field(min_length=1, pattern=r"\S")


class InputDialogue(BaseModel):
    """A complete dialogue, independent of dataset, language or therapeutic approach."""

    model_config = ConfigDict(extra="allow", strict=True)

    dialogue_id: str = Field(min_length=1, pattern=r"\S")
    turns: list[InputTurn] = Field(min_length=1)

    @model_validator(mode="after")
    def check_turn_identity_and_order(self) -> "InputDialogue":
        ids = [turn.turn_id for turn in self.turns]
        if len(set(ids)) != len(ids):
            raise ValueError("turn_id must be unique within a dialogue")
        indices = [turn.turn_index for turn in self.turns]
        if any(right <= left for left, right in zip(indices, indices[1:])):
            raise ValueError("turn_index must strictly increase in input order; no automatic sorting or renumbering")
        return self


def _check_json_value(value: Any) -> None:
    """Extra source fields must also be losslessly serialisable JSON values."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _check_json_value(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _check_json_value(item)
        return
    raise ValueError("Input and extra metadata must use standard JSON types without NaN, Infinity or non-string keys")


def validate_dialogue(record: dict[str, Any]) -> dict[str, Any]:
    """Validate and deep-copy without filling fields, editing text, mapping roles or stripping metadata."""
    _check_json_value(record)
    InputDialogue.model_validate(record)
    return copy.deepcopy(record)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"JSON does not permit {value}")


def parse_json_value(text: str) -> Any:
    """Shared JSON loader for CLI, conversion and pipeline; reject duplicate keys and non-finite numbers."""
    value = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    _check_json_value(value)
    return value


def _read_records(content: str, input_format: str) -> list[tuple[dict[str, Any], dict[str, int]]]:
    records: list[tuple[Any, dict[str, int]]] = []
    if input_format in {"csv", "tsv"}:
        label = input_format.upper()
        reader = csv.reader(io.StringIO(content, newline=""), delimiter="\t" if input_format == "tsv" else ",", strict=True)
        headers = next(reader, None)
        if not headers or any(not header.strip() for header in headers):
            raise ValueError(f"{label} requires non-empty headers")
        if len(set(headers)) != len(headers):
            raise ValueError(f"{label} headers must not be duplicated")
        previous_line = reader.line_num
        for row in reader:
            if len(row) != len(headers):
                raise ValueError(f"{label} line {previous_line + 1} has an incorrect field count; blank rows and truncated records are not skipped")
            records.append((dict(zip(headers, row)), {"line_start": previous_line + 1, "line_end": reader.line_num}))
            previous_line = reader.line_num
    elif input_format == "jsonl":
        lines = content.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        for number, line in enumerate(lines, 1):
            if not line.strip():
                raise ValueError(f"JSONL line {number} is empty; records are not silently skipped")
            records.append((parse_json_value(line), {"line": number}))
    else:
        payload = parse_json_value(content)
        values = payload if isinstance(payload, list) else [payload]
        records = [(value, {"record": index}) for index, value in enumerate(values, 1)]
    if not records or any(not isinstance(record, dict) for record, _ in records):
        raise ValueError("Input must contain at least one object record")
    return records


def _read_transcript(
    content: str, input_format: str, dialogue_id: str, role_map: dict[str, str] | None,
) -> list[tuple[dict[str, Any], dict[str, int]]]:
    """Use line-start speaker labels as turn boundaries and preserve text; do not infer dialogue boundaries."""
    patterns = [r"(?P<role>[^\s:：][^:：]*?)[：:](?P<text>.*)"]
    if input_format == "md":
        patterns = [
            r"\*\*(?P<role>[^*\s:：][^*:：]*?)[：:]\*\*(?P<text>.*)",
            r"\*\*(?P<role>[^*\s:：][^*:：]*?)\*\*[：:](?P<text>.*)",
            *patterns,
        ]
    labels = role_map if role_map is not None else ROLES
    records: list[tuple[dict[str, Any], dict[str, int]]] = []
    previous_ending = ""
    # Only CR/LF delimit physical lines; Unicode separators within text do not start new turns.
    lines = re.findall(r"([^\r\n]*)(\r\n|\r|\n|$)", content)
    for number, (line, ending) in enumerate(lines, 1):
        if not line and not ending:  # The final zero-length findall match is not a source blank line.
            continue
        if input_format == "md" and re.match(r"(?:#{1,6}(?:\s|$)|`{3}|~{3}|\||---\s*$)", line):
            raise ValueError(f"MD line {number} contains unsupported document structure; use line-start speaker labels, not headings, tables or code blocks")
        match = next((match for pattern in patterns if (match := re.fullmatch(pattern, line))), None)
        if match:
            role = match["role"]
            if role not in labels:
                raise ValueError(f"Speaker label {role!r} on line {number} has no explicit mapping; a line-start colon label cannot be merged as continuation text")
            records.append((
                {"dialogue_id": dialogue_id, "role": role, "text": match["text"]},
                {"line_start": number, "line_end": number, "text_column": match.start("text") + 1},
            ))
        elif records:
            raw, location = records[-1]
            raw["text"] += previous_ending + line
            location["line_end"] = number
        else:
            raise ValueError(f"Line {number} must start with an explicit speaker label; prefaces and blank lines are not discarded and roles are not guessed")
        previous_ending = ending
    if not records:
        raise ValueError("TXT/MD must contain at least one speaker record with text")
    return records


def _read_xlsx_records(content: bytes, sheet: str | None = None) -> tuple[list[tuple[dict, dict]], str]:
    """Read the selected XLSX value table, preserving row numbers and rejecting formulae or ambiguous structure.

    Read stored values without calculating formulae, interpreting date styles or modifying the workbook.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            workbook = ET.fromstring(archive.read("xl/workbook.xml"))
            sheets = workbook.findall(f"{_XLSX_NS}sheets/{_XLSX_NS}sheet")
            names = [entry.attrib["name"] for entry in sheets]
            if sheet is None:
                if len(sheets) != 1:
                    raise ValueError(f"XLSX has multiple sheets; specify a sheet explicitly: {names}")
                sheet = names[0]
            if sheet not in names:
                raise ValueError(f"XLSX sheet does not exist: {sheet}; available: {names}")
            selected = sheets[names.index(sheet)]
            relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            target = next((r.attrib["Target"] for r in relationships if r.attrib.get("Id") == selected.attrib[f"{_XLSX_REL}id"] and r.attrib.get("TargetMode") != "External"), None)
            if target is None:
                raise ValueError("XLSX worksheet relationship is missing or external")
            target = posixpath.normpath(target.lstrip("/") if target.startswith("/") else posixpath.join("xl", target))
            if not target.startswith("xl/"):
                raise ValueError("XLSX worksheet path is outside xl")
            shared: list[str] = []
            if "xl/sharedStrings.xml" in archive.namelist():
                shared = ["".join(t.text or "" for t in si.iter(f"{_XLSX_NS}t")) for si in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
            root = ET.fromstring(archive.read(target))
            if root.find(f"{_XLSX_NS}mergeCells") is not None:
                raise ValueError("XLSX input must not contain merged cells")
            rows: list[tuple[int, dict[int, object]]] = []
            for row in root.findall(f"{_XLSX_NS}sheetData/{_XLSX_NS}row"):
                values: dict[int, object] = {}
                for cell in row.findall(f"{_XLSX_NS}c"):
                    address = cell.attrib.get("r", "")
                    match = re.fullmatch(r"([A-Z]+)([1-9][0-9]*)", address)
                    if not match:
                        raise ValueError("XLSX cell address is missing or invalid")
                    if cell.find(f"{_XLSX_NS}f") is not None:
                        raise ValueError(f"XLSX formula is not an explicit input value: {sheet}!{address}; provide an export of values")
                    column = 0
                    for letter in match[1]:
                        column = column * 26 + ord(letter) - ord("A") + 1
                    kind = cell.attrib.get("t", "n")
                    value_node = cell.find(f"{_XLSX_NS}v")
                    value = value_node.text if value_node is not None else None
                    if kind == "inlineStr":
                        value = "".join(t.text or "" for t in cell.iter(f"{_XLSX_NS}t"))
                    elif kind == "s" and value is not None:
                        value = shared[int(value)]
                    elif kind == "b" and value is not None:
                        if value not in {"0", "1"}:
                            raise ValueError(f"Invalid XLSX boolean: {sheet}!{address}")
                        value = value == "1"
                    elif kind == "n" and value is not None:
                        number = float(value)
                        if not math.isfinite(number):
                            raise ValueError(f"Non-finite XLSX number: {sheet}!{address}")
                        value = int(number) if number.is_integer() else number
                    elif kind not in {"str", "d", "n", "s", "b"}:
                        raise ValueError(f"Unsupported XLSX cell type {kind}: {sheet}!{address}")
                    if column in values:
                        raise ValueError(f"Duplicate XLSX cell: {sheet}!{address}")
                    values[column] = value
                if any(value is not None and value != "" for value in values.values()):
                    rows.append((int(row.attrib["r"]), values))
    except (zipfile.BadZipFile, ET.ParseError, KeyError, IndexError) as error:
        raise ValueError(f"Cannot read standard XLSX structure: {error}") from error
    if not rows:
        raise ValueError("XLSX has no non-empty headers and records")
    first_row, header = rows[0]
    width = max(header)
    headers = [header.get(index) for index in range(1, width + 1)]
    if any(not isinstance(value, str) or not value.strip() for value in headers):
        raise ValueError("XLSX requires non-empty string headers")
    if len(set(headers)) != len(headers):
        raise ValueError("XLSX headers must not be duplicated")
    records = []
    previous = first_row
    for number, values in rows[1:]:
        if number != previous + 1:
            raise ValueError(f"XLSX row order or internal blank rows are ambiguous: {sheet}!{number}")
        if any(index > width and value is not None and value != "" for index, value in values.items()):
            raise ValueError(f"XLSX has data columns without headers: {sheet}!{number}")
        records.append((dict(zip(headers, [values.get(index) for index in range(1, width + 1)])), {"sheet": sheet, "row": number}))
        previous = number
    if not records:
        raise ValueError("XLSX has no records after its headers")
    return records, sheet


def _field(raw: dict[str, Any], name: str, columns: dict[str, str]) -> Any:
    source = columns.get(name, name)
    if source not in raw and name in columns:
        raise ValueError(f"Explicitly mapped field does not exist: {name} -> {source}")
    return raw.get(source, _MISSING)


def _converted_record(
    raw: dict[str, Any], values: dict[str, Any], columns: dict[str, str],
    generated: list[str], location: dict[str, int],
) -> dict[str, Any]:
    """Preserve original values only when canonical fields cannot reconstruct them, avoiding duplicated text."""
    sources = {key: columns.get(key, key) for key in values if key not in generated}
    if len(set(sources.values())) != len(sources):
        raise ValueError("Field mapping collision with a default field; canonical fields must not share a source field")
    result = copy.deepcopy(raw)
    provenance: dict[str, Any] = {"location": location}
    renamed = {key: source for key, source in sources.items() if key != source}
    if renamed:
        provenance["field_map"] = renamed
    original_values = {
        source: raw[source] for key, source in sources.items()
        if key != "turns" and source in raw and raw[source] != values[key]
    }
    if original_values:
        provenance["original_values"] = copy.deepcopy(original_values)
    replaced = {
        key: raw[key] for key in values
        if key in raw and key not in sources.values() and raw[key] != values[key]
    }
    if replaced:
        provenance["replaced_fields"] = copy.deepcopy(replaced)
    for key, source in renamed.items():
        result.pop(source, None)
    if generated:
        provenance["generated_fields"] = generated
    if "_conversion" in raw:
        provenance["previous"] = copy.deepcopy(raw["_conversion"])
    result.update(values)
    result["_conversion"] = provenance
    return result


def _convert_turn(
    raw: dict[str, Any], number: int, location: dict[str, int],
    columns: dict[str, str], role_map: dict[str, str] | None, is_delimited: bool,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"turn must be an object: {location}")
    values = {key: _field(raw, key, columns) for key in ("turn_id", "turn_index", "role", "text")}
    generated: list[str] = []
    for key, replacement in (("turn_id", f"generated-turn-{number:06d}"), ("turn_index", number)):
        if values[key] is _MISSING:
            values[key] = replacement
            generated.append(key)
    if values["role"] is _MISSING or values["text"] is _MISSING:
        raise ValueError(f"Missing role/text fields: {location}; supply an explicit mapping")
    if role_map is not None:
        role = values["role"]
        if not isinstance(role, str) or role not in role_map:
            raise ValueError(f"role_map does not explicitly cover role {role!r}: {location}")
        values["role"] = role_map[role]
    if is_delimited and "turn_index" not in generated:
        value = values["turn_index"]
        if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
            raise ValueError(f"CSV/TSV turn_index must be a non-negative decimal integer: {location}")
        values["turn_index"] = int(value)
    result = _converted_record(raw, values, columns, generated, location)
    InputTurn.model_validate(result)
    return result


def _convert_dialogues(
    records: list[tuple[dict[str, Any], dict[str, int]]], columns: dict[str, str],
    role_map: dict[str, str] | None, input_format: str,
) -> list[dict[str, Any]]:
    turns_key = columns.get("turns", "turns")
    nested = [turns_key in record for record, _ in records]
    if input_format in {"csv", "tsv", "xlsx"} and any(nested):
        raise ValueError("Each CSV/TSV/XLSX row must represent one turn and must not contain a turns column")
    if any(nested) and not all(nested):
        raise ValueError("Dialogue records and per-turn rows must not be mixed")
    if "turns" in columns and not all(nested):
        raise ValueError(f"Explicitly mapped field does not exist: turns -> {turns_key}")
    result: list[dict[str, Any]] = []
    if all(nested):
        for number, (raw, location) in enumerate(records, 1):
            raw_turns = raw[turns_key]
            if not isinstance(raw_turns, list) or not raw_turns:
                raise ValueError(f"turns must be a non-empty array: {location}")
            turns = [
                _convert_turn(turn, index, {**location, "turn": index}, columns, role_map, False)
                for index, turn in enumerate(raw_turns, 1)
            ]
            dialogue_id = _field(raw, "dialogue_id", columns)
            generated = ["dialogue_id"] if dialogue_id is _MISSING else []
            if generated:
                dialogue_id = f"generated-dialogue-{number:06d}"
            result.append(_converted_record(raw, {"dialogue_id": dialogue_id, "turns": turns}, columns, generated, location))
    else:
        identities = [_field(raw, "dialogue_id", columns) for raw, _ in records]
        if any(identity is _MISSING for identity in identities):
            raise ValueError("Per-turn rows require explicit dialogue_id values; dialogue boundaries cannot be inferred or merged")
        grouped: dict[str, list[dict[str, Any]]] = {}
        locations: dict[str, dict[str, int]] = {}
        for (raw, location), identity in zip(records, identities):
            if not isinstance(identity, str) or not identity.strip():
                raise ValueError(f"dialogue_id must be a non-empty string: {location}")
            turns = grouped.setdefault(identity, [])
            locations.setdefault(identity, location)
            turns.append(_convert_turn(raw, len(turns) + 1, location, columns, role_map, input_format in {"csv", "tsv"}))
        for identity, turns in grouped.items():
            result.append(_converted_record({}, {"dialogue_id": identity, "turns": turns}, {}, [], locations[identity]))
    validated = [validate_dialogue(record) for record in result]
    ids = [record["dialogue_id"] for record in validated]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate dialogue_id in input, including generated ID collisions; dialogues are not merged or overwritten")
    return validated


def sha256_file(path: Path) -> str:
    """Compute the local file checksum in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_bytes(record: dict) -> bytes:
    return (json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


def _summarise(dialogues: Iterable[dict]) -> dict:
    identities: set[str] = set()
    roles: Counter = Counter()
    generated: Counter = Counter()
    flags: Counter = Counter()
    preview = []
    digest = hashlib.sha256()
    size = 0
    for record in dialogues:
        validate_dialogue(record)
        identity = record["dialogue_id"]
        if identity in identities:
            raise ValueError(f"Duplicate dialogue_id in input: {identity}; no merging or overwriting")
        identities.add(identity)
        turns = record["turns"]
        roles.update(turn["role"] for turn in turns)
        record_flags = record.get("quality_flags", [])
        if isinstance(record_flags, list) and all(isinstance(flag, str) for flag in record_flags):
            flags.update(record_flags)
        for item in [record, *turns]:
            generated.update(item.get("_conversion", {}).get("generated_fields", []))
        if len(preview) < 20:
            preview.append({"dialogue_id": identity, "turn_count": len(turns), "first_turn_index": turns[0]["turn_index"], "last_turn_index": turns[-1]["turn_index"]})
        payload = _jsonl_bytes(record)
        digest.update(payload)
        size += len(payload)
    if not identities:
        raise ValueError("No dialogues to convert")
    return {
        "dialogue_count": len(identities), "turn_count": sum(roles.values()),
        "role_counts": {role: roles[role] for role in sorted(ROLES)},
        "quality_flag_counts": dict(sorted(flags.items())), "generated_fields": dict(generated),
        "dialogues": preview, "dialogues_not_shown": len(identities) - len(preview),
        "output_sha256": digest.hexdigest(), "output_bytes": size,
    }


def _load_manifest(directory: Path) -> dict:
    path = directory / "manifest.json"
    if path.is_symlink():
        raise FileExistsError(f"The manifest must not be a symbolic link: {path}")
    if not path.exists():
        return {"manifest_version": "processed", "datasets": {}}
    manifest = parse_json_value(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("manifest_version") not in {"processed", "processed-v1"} or not isinstance(manifest.get("datasets"), dict):
        raise ValueError("The existing manifest is not a valid shared processed manifest")
    manifest["manifest_version"] = "processed"
    return manifest


def _check_destination(directory: Path, dataset: str) -> dict:
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise FileExistsError(f"The destination directory must not be overwritten: {directory}")
    target = directory / f"{dataset}.jsonl"
    manifest = _load_manifest(directory)
    if target.exists() or target.is_symlink() or dataset in manifest["datasets"]:
        raise FileExistsError(f"The dataset destination or manifest entry already exists; no overwrite: {target}")
    return manifest


def _save_manifest(path: Path, manifest: dict) -> None:
    """Atomically replace the shared manifest under a lock, preserving other datasets on failure."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".manifest-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _write_processed(directory: Path, dataset: str, preview: dict, factory: Callable[[], Iterable[dict]], verify_sources: Callable[[], None]) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".manifest.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write("conversion in progress\n")
    target = directory / f"{dataset}.jsonl"
    manifest_path = directory / "manifest.json"
    try:
        manifest = _check_destination(directory, dataset)
        entry = {**preview, "created_at": datetime.now(timezone.utc).isoformat(), "status": "writing", "completed_files": []}
        manifest["datasets"][dataset] = entry
        _save_manifest(manifest_path, manifest)
        try:
            digest = hashlib.sha256()
            with target.open("xb") as handle:
                for record in factory():
                    payload = _jsonl_bytes(record)
                    handle.write(payload)
                    digest.update(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if digest.hexdigest() != preview["output_sha256"] or sha256_file(target) != preview["output_sha256"]:
                raise ValueError("Written content differs from the full preview; conversion cannot be marked complete")
            verify_sources()
            entry.update(status="complete", completed_files=[target.name])
            _save_manifest(manifest_path, manifest)
        except Exception as error:
            entry.update(status="failed", error=f"{type(error).__name__}: {error}")
            try:
                _save_manifest(manifest_path, manifest)
            except OSError:
                pass
            raise RuntimeError(f"Conversion write failed; incomplete directory retained at {directory}; check dataset status: {error}") from error
    finally:
        lock.unlink()
    return {**preview, "status": "complete", "files": {target.name: str(target), "manifest.json": str(manifest_path)}}


def prepare_data(
    input_path: Path, output_dir: Path | None = None, *, input_format: str | None = None,
    columns: dict[str, str] | None = None, role_map: dict[str, str] | None = None,
    dry_run: bool = False, dataset: str | None = None, sheet: str | None = None,
    dialogue_id: str | None = None,
) -> dict[str, Any]:
    """Convert explicit structures and mappings without dataset-name-specific rules."""
    input_path = Path(input_path).expanduser().resolve()
    if not input_path.is_file():
        raise ValueError("input must be an existing CSV/TSV/JSON/JSONL/XLSX/TXT/MD file; choose a file explicitly")
    dataset = input_path.stem if dataset is None else dataset
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", dataset):
        raise ValueError("dataset must start with a letter or digit and contain only letters, digits, underscores, hyphens or dots")
    output_dir = Path(output_dir).expanduser().absolute() if output_dir is not None else None
    if output_dir is not None:
        _check_destination(output_dir, dataset)
    if not dry_run and output_dir is None:
        raise ValueError("output_dir is required unless dry_run is enabled")
    columns = dict(columns or {})
    if set(columns) - FIELDS or any(not isinstance(value, str) or not value.strip() or value == "_conversion" for value in columns.values()):
        raise ValueError(f"columns supports only {sorted(FIELDS)} mapped to non-empty source field names; _conversion cannot be mapped")
    if len(set(columns.values())) != len(columns):
        raise ValueError("Different canonical fields must not map to the same source field")
    if role_map is not None and (not role_map or any(not isinstance(key, str) or not key.strip() or not isinstance(value, str) or value not in ROLES for key, value in role_map.items())):
        raise ValueError("role_map must explicitly map non-empty source roles to counsellor/client/other")
    input_format = input_format or input_path.suffix.lower().lstrip(".")
    if input_format not in {"csv", "tsv", "json", "jsonl", "xlsx", "txt", "md"}:
        raise ValueError("Supported formats are csv/tsv/json/jsonl/xlsx/txt/md; other structures need explicit adaptation")
    if sheet is not None and input_format != "xlsx":
        raise ValueError("sheet applies only to XLSX")
    if input_format in {"txt", "md"}:
        if not isinstance(dialogue_id, str) or not dialogue_id.strip():
            raise ValueError("TXT/MD requires a non-empty dialogue_id (--dialogue-id), confirming that the whole file is one dialogue")
        if columns:
            raise ValueError("TXT/MD does not use columns; use line-start labels and an explicit role_map")
    elif dialogue_id is not None:
        raise ValueError("dialogue_id applies only to TXT/MD; other formats must define dialogue boundaries in their source structure")
    paths, source_root = [input_path], input_path.parent
    rules = {"columns": columns, "role_map": role_map, "text": "preserve exactly; no trimming or sorting"}
    sources = [{"path": path.relative_to(source_root).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in sorted(paths)]
    source_set = hashlib.sha256()
    for source in sources:
        source_set.update(source["path"].encode("utf-8"))
        source_set.update(source["sha256"].encode("ascii"))

    def verify_sources() -> None:
        for source in sources:
            if sha256_file(source_root / source["path"]) != source["sha256"]:
                raise ValueError(f"Source file changed during conversion: {source['path']}")

    source_bytes = input_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != sources[0]["sha256"]:
        raise ValueError("Source file changed during reading")
    if input_format == "xlsx":
        records, selected_sheet = _read_xlsx_records(source_bytes, sheet)
        rules.update(sheet=selected_sheet, cells="stored values only; formulas rejected; date/time styles not interpreted; trailing blank formatting ignored")
    elif input_format in {"txt", "md"}:
        records = _read_transcript(source_bytes.decode("utf-8-sig"), input_format, dialogue_id, role_map)
        rules.update(
            dialogue_id=dialogue_id, transcript="speaker-prefix; one explicit dialogue per file",
            markers="column-one label followed by : or ：; MD also accepts **label:** and **label**:",
            text="preserve body after marker and inter-line endings; exclude final line ending of each turn; no trimming",
        )
    else:
        records = _read_records(source_bytes.decode("utf-8-sig"), input_format)
        if input_format == "tsv":
            rules["delimiter"] = "\t"
    dialogues = _convert_dialogues(records, columns, role_map, input_format)
    factory = lambda: iter(dialogues)
    summary = _summarise(factory())
    verify_sources()
    # One input-module fingerprint covers all formats; historical manifest fingerprints remain unchanged.
    code_modules = {"input_data.py": LOADED_SOURCE_SHA256}
    if sha256_file(Path(__file__)) != LOADED_SOURCE_SHA256:
        raise RuntimeError("Loaded converter code differs from disk; restart the process: input_data.py")
    preview = {
        "input_schema": INPUT_SCHEMA_VERSION, "dataset": dataset,
        "input": {"path": str(input_path), "format": input_format},
        "source_root": str(source_root), "sources": sources, "source_file_count": len(sources),
        "source_set_sha256": source_set.hexdigest(), "rules": rules,
        "converter_sha256": code_modules, **summary,
        "output_file": f"{dataset}.jsonl", "output_dir": str(output_dir) if output_dir else None,
    }
    if dry_run:
        return {**preview, "status": "validated", "files": {}}
    assert output_dir is not None
    return _write_processed(output_dir, dataset, preview, factory, verify_sources)


def create_sample_plan(input_path: Path, output_path: Path, *, dialogue_ids: list[str] | None = None, select_all: bool = False) -> dict:
    """Select complete dialogues by explicit IDs or an explicit all-dialogues option; no slicing or model calls."""
    from alliance_eval.evaluation_models import SamplePlan

    if bool(dialogue_ids) == select_all:
        raise ValueError("Specify exactly one of dialogue_ids or select_all")
    if dialogue_ids and len(set(dialogue_ids)) != len(dialogue_ids):
        raise ValueError("Selection contains duplicate dialogue_id values")
    source = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Plan already exists; no overwrite: {output}")
    if source.parent == output.parent.resolve():
        raise ValueError("Store plans separately from processed data, for example data/plans/sample_plan.json")
    manifest = _load_manifest(source.parent)
    entry = manifest["datasets"].get(source.stem)
    if not isinstance(entry, dict) or entry.get("status") != "complete" or entry.get("output_file") != source.name or entry.get("output_sha256") != sha256_file(source):
        raise ValueError("Source requires a matching complete conversion manifest and checksum")
    wanted = set(dialogue_ids or [])
    selected = []
    seen = set()
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            record = validate_dialogue(parse_json_value(line))
            identity = record["dialogue_id"]
            if identity in seen:
                raise ValueError(f"Duplicate dialogue_id: {identity}")
            seen.add(identity)
            if select_all or identity in wanted:
                selected.append({"source_file": Path(os.path.relpath(source, output.parent)).as_posix(), "dialogue_id": identity, "start_turn_index": None, "end_turn_index": None})
    if wanted - seen:
        raise ValueError(f"Selected dialogue_id values do not exist: {sorted(wanted - seen)}")
    if not selected:
        raise ValueError("No dialogues selected")
    plan = {"sample_version": "selected", "items": selected}
    SamplePlan.model_validate(plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    return {"status": "complete", "sample_file": str(output), "dialogue_count": len(selected), "source_sha256": entry["output_sha256"]}


def dialogue_json_schema() -> dict[str, Any]:
    """Return the input schema for export and runtime identity checks."""
    schema = InputDialogue.model_json_schema()
    schema.update({"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "urn:alliance-evaluation:dialogue", "$comment": "Runtime validation additionally requires unique turn_id values and strictly increasing turn_index values in input order. Batch conversion requires unique dialogue_id values. Additional source metadata is preserved."})
    return schema


def export_input_schema(output_path: Path) -> None:
    """Exclusively create the input schema without changing the six model output schemas."""
    with Path(output_path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(dialogue_json_schema(), ensure_ascii=False, indent=2) + "\n")
