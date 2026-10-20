# Input contract for your dialogues

Dialogue input follows `dialogue.schema.json`, separately from model-evaluation output. Conversion is local. Select one source file at a time and make fields, roles and dialogue boundaries explicit.

Use the [original fictional templates](../examples/README.md) for format checks. For the complete model-free workflow, start the App from the [root README](../README.md) and click `Run preset demo`. Conversion does not create a run plan:

```bash
alliance-eval prepare-data --input examples/input-template.json --dataset example --dry-run
alliance-eval prepare-data --input examples/input-template.json --dataset example
```

Inspect the preview before removing `--dry-run`. Default outputs are `data/processed/example.jsonl` and shared `manifest.json`; existing targets or entries are not overwritten. CLI relative paths resolve from the command's working directory. Inputs and outputs may be outside the installation.

## Canonical dialogues

This original fictional dialogue illustrates the input format:

```json
{
  "dialogue_id": "fictional-1",
  "turns": [
    {"turn_id": "source-a", "turn_index": 0, "role": "counsellor", "text": "Shall we pause?"},
    {"turn_id": "source-z", "turn_index": 4, "role": "client", "text": "Yes, briefly."}
  ],
  "metadata": {"fictional": true}
}
```

- `dialogue_id`, `turn_id` and `text` must be strings containing non-whitespace characters. Numeric IDs are not converted to strings. Dialogue IDs are unique within a conversion; turn IDs are unique within each dialogue.
- `turn_index` is a non-negative integer, strictly increasing in input order. Zero-based indices and gaps are allowed; turns are not sorted or renumbered automatically.
- `role` is `counsellor`, `client` or `other`. Other turns remain context, not substitute rating targets. Map roles from explicit meanings, never names or inferred content.
- Preserve source spaces, line breaks, punctuation, existing IDs, order and extra metadata. Do not merge turns, discard dialogues or invent text. Retaining metadata does not verify its licence or quality claims.
- Source fields `dataset`, `therapy` and `schema_version` are not required. The input contract is recorded separately in the manifest. See the [input schema](../schemas/dialogue.schema.json); uniqueness, ordering and JSON ambiguity also receive runtime checks.

## Supported structures

Text files must be UTF-8; an initial BOM is accepted.

| Format | Accepted structure |
|:---|:---|
| JSON | One dialogue object, an array of dialogue objects or an array of turn objects |
| JSONL | One complete dialogue per line or one turn per line, without mixing the two |
| CSV / TSV | A header and one turn per row; quoted delimiters and multiline text are preserved |
| XLSX | One turn-value table with unique non-empty string headers; select `--sheet` when multiple sheets exist |
| TXT / Markdown (`.md`) | One complete dialogue with speaker labels at line starts and an explicit `--dialogue-id`; Markdown also accepts bold labels |

Every flat turn row requires an explicit dialogue ID. Grouping preserves first appearance of dialogues and each dialogue's turn order, with source-row provenance. Missing IDs do not imply a single dialogue. Only nested objects with an explicit `turns` array may receive generated dialogue IDs. Field mappings use direct field names, not dotted paths or deep queries.

XLSX reads stored values without recalculating formulas, interpreting date/time formatting or converting numeric IDs to strings. Formulas, error cells, merged cells, internal blank rows and data columns without headers are rejected. Trailing empty formatting is not data. Provenance includes source sheet and row numbers.

### Labelled TXT and Markdown transcripts

Confirm that **one file contains one complete dialogue**, then supply its existing or agreed technical ID with `--dialogue-id`. Multiple sessions, separate timestamp/comment fields and complex layout are outside this mode; filenames and blank lines do not determine session boundaries.

Each turn begins in column 1 with `label:text`; a fullwidth colon is also accepted. Use `--role-map` to define labels, unless they already use canonical roles. Markdown additionally accepts `**label:**text` and `**label**:text`. Original fictional example:

```text
Therapist:Which issue would you like to discuss first?
Client:I would like to talk about work.
It has been bothering me for a while.
```

```bash
alliance-eval prepare-data --input conversation.txt --dataset example-text \
  --dialogue-id fictional-visit-1 \
  --role-map '{"Therapist":"counsellor","Client":"client"}' --dry-run
```

- Subsequent unlabelled lines, including blank lines, belong to the current turn. A colon label at the start of a line is reserved syntax; unmapped labels fail. If body text conflicts with this syntax, choose an explicit alternative representation.
- Labels, colons and Markdown label-bold markers are structural. All subsequent body spaces, punctuation and within-turn line breaks are preserved without normalising CRLF/LF/CR. The final line ending of a turn separates records and is not part of its text.
- Empty files, empty utterances, preambles or blank lines before the first label, and unmapped labels are rejected. Markdown headings, tables, code fences and `---` separators at line starts are also rejected.
- Turn `_conversion.location` records source-line ranges and one-based body-start columns. The manifest stores the file hash and parsing rules. Generated IDs/indices are labelled as generated. `--dialogue-id` applies only to TXT/MD; it cannot repair missing dialogue IDs in flat turn tables.

### Unsupported structures and adaptation gaps

Check existing options first: `--columns` for field names, `--role-map` for roles and `--sheet` for sheets. Use `--format` when actual content matches a supported format but has a different extension; renaming an unsupported file is not conversion.

Ask for missing role or boundary definitions. For unsupported structures, identify the location and gap, such as deeply nested sessions or whole unlabelled dialogues inside cells. Distinguish invalid data from missing parser capability. Preserve originals and use a structured export or a separately agreed verifiable adapter. Do not silently flatten content, drop records, ignore decoding errors or use a model to fill gaps. New parser capability requires separate implementation and verification, not a temporary converter inside a preparation skill.

## Explicit mappings and preview

`--columns` maps **canonical field to source field**, supporting `dialogue_id/turns/turn_id/turn_index/role/text`. Unspecified mappings use identical names; explicitly mapped missing fields fail rather than falling back. Different canonical fields cannot explicitly map to the same source field. `--role-map` maps **source role value to canonical role** and, when supplied, must cover every source role.

```bash
alliance-eval prepare-data --input data/my-dialogues.csv --dataset my-data \
  --output-dir data/processed \
  --columns '{"dialogue_id":"session","turn_id":"uid","role":"speaker","text":"utterance"}' \
  --role-map '{"H":"counsellor","P":"client"}' --dry-run
```

This example assumes the meanings of H/P are known. Preview parses and validates every record, reporting totals, roles, generated fields, expected output checksum and index summaries for the first 20 dialogues without writing directories or source text. The 20 summaries are not a conversion limit. After checking, use the same source and options without `--dry-run`. Correct counts do not establish correct mapping semantics.

Only absent technical fields are generated: `generated-turn-000001`, one-based `turn_index` in input order, or `generated-dialogue-000001` within explicit nested boundaries. Flat turn rows never receive generated dialogue IDs. Empty IDs/indices, unknown roles, missing text, invalid order, duplicate IDs and generated-value collisions fail. JSON indices must already be integers; CSV/TSV decimal index strings may be converted with originals retained. Output names identify files and manifest entries, not parsing rules; they default to the source stem or explicit `--dataset`.

## Outputs, plans and failures

Successful conversion writes `<dataset>.jsonl`, one complete dialogue per line, and updates shared `manifest.json`. Its `datasets.<dataset>` entry records source path/size/hash, contract, mappings, code fingerprints, counts, output hash, timestamp and status; other entries remain. Dialogue/turn `_conversion` metadata records source locations, mappings and generated fields. Changed values retain `original_values`; replaced metadata retains `replaced_fields`; existing provenance is nested under `previous`.

**An existing usable run plan does not need `create-plan`.** When no plan exists and scope is explicit, this helper records real dialogue IDs without random, length-based or stratified sampling and without model calls:

```bash
alliance-eval create-plan --input data/processed/example.jsonl \
  --dialogue-id fictional-input-template-001 --output data/plans/sample_plan.json
alliance-eval validate-input --sample-file data/plans/sample_plan.json
```

Repeat `--dialogue-id` for multiple selections; use `--all-dialogues` only for an explicit all-dialogue request. The helper requires a `complete` conversion entry and matching hash, selects whole dialogues in source order, and stores the plan outside the processed directory with source paths relative to the plan. It neither copies nor crops the corpus. Underlying plans can reference turn ranges, but this helper offers whole-dialogue selection only. Preserve original plans and sources; recorded hashes are not copies.

Original files are unchanged. Existing output files, manifest entries and target symlinks are not overwritten. Missing directories are created only after full preflight validation. Shared manifests use an exclusive lock and atomic replacement; source/output bytes are rechecked before marking `complete`. Write failures retain artefacts and record `failed` where possible. Hard interruption may leave `writing` and a lock; no automatic retry occurs. Failed entries cannot produce plans as completed conversions. Investigate and use an explicit new name or directory without overwriting evidence.

Duplicate JSON keys, NaN/Infinity, internal empty JSONL lines, duplicate headers, wrong field counts and metadata not losslessly representable in standard JSON are rejected. Parsing retains records in memory; arbitrary large-file streaming is not promised. Format validation verifies neither licensing nor anonymisation and does not authorise external model processing.
