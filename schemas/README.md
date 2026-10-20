# Input and model-output contracts

This directory contains one dialogue-input schema and six model-output schemas for the evaluation method.

## General input

`dialogue.schema.json` is generated from `InputDialogue` and `InputTurn` in [input_data.py](../src/alliance_eval/input_data.py). Required content includes dialogue ID, turns, and each turn's ID, index, role and original text. Extra source metadata is preserved without restricting dataset, language or therapy approach.

Runtime code also checks unique IDs, strictly increasing turn indices and duplicate JSON keys. JSON Schema validation alone does not establish every input constraint. See the [input contract](../docs/data-format.md) for formats, explicit mappings, traceable technical IDs and conversion failures.

## Model outputs

| File | Responsibility |
|:---|:---|
| `activity_identification.schema.json` | New Activity references established at the current turn |
| `evaluation_turn_counsellor.schema.json` | Allowed counsellor-turn findings |
| `evaluation_turn_client.schema.json` | Allowed client-turn findings |
| `evaluation_turn_counsellor_no_activity.schema.json` | Bond-only counsellor findings when the Activity registry is empty |
| `evaluation_turn_client_no_activity.schema.json` | Bond-only client findings when the Activity registry is empty |
| `evaluation_summary.schema.json` | Four summary target types and their allowed states, evidence and categories |

Output schemas are generated from [evaluation_models.py](../src/alliance_eval/evaluation_models.py), constraining dimensions, states, labels, Activity references and extra fields. Runtime validation additionally checks quotation-to-turn binding and summary evidence provenance.

## Check and export

```bash
alliance-eval check-runtime
alliance-eval export-schema --output-dir schema-export
```

`check-runtime` checks loaded code and current schema identity without calling models. Input and output schema fingerprints are recorded separately.

`export-schema` writes all seven schemas to a **new, nonexistent directory**. Choose a separate destination, not the project's `schemas/` directory.
