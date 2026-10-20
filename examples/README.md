# Examples

The [complete preset demo](demo/README.md) provides three original fictional dialogues with clearly labelled preset responses. Start the App using the [quick start](../README.md), then click `Run preset demo`. The normal pipeline creates one run with three samples. Switch `Sample` to inspect each dialogue's Activities, turn evidence, Summary and Core mapping, with run-wide statistics. No model or API key is required.

## Input templates

[input-template.json](input-template.json) and [input-template.csv](input-template.csv) contain the same two original fictional utterances for format and conversion checks. Both provide dialogue ID, turn IDs, indices, canonical roles and fictional markers. The CSV requires no field or role mapping.

These short templates contain no evaluation results. Use the preset demo for the complete evaluation workflow.

After installation, preview from the project root:

```bash
alliance-eval prepare-data --input examples/input-template.json --dataset example --dry-run
```

Expect one dialogue, two turns, one `counsellor` and one `client`, with no generated IDs or indices. Dry-run writes no directories or files. After checking, convert locally and create a plan:

```bash
alliance-eval prepare-data --input examples/input-template.json --dataset example
alliance-eval create-plan --input data/processed/example.jsonl \
  --dialogue-id fictional-input-template-001 --output data/plans/sample_plan.json
alliance-eval validate-input --sample-file data/plans/sample_plan.json
```

These commands make no model calls. They produce `data/processed/example.jsonl`, shared `manifest.json` and a separate plan in `data/plans/`. Existing targets are not overwritten; choose an explicit new name when repeating. Preview CSV separately:

```bash
alliance-eval prepare-data --input examples/input-template.csv --dataset example-csv --dry-run
```

Canonical dialogue content is identical between formats. The fictional marker is dialogue metadata in JSON and extra row metadata in CSV, so converted provenance metadata has different nesting. See the [input contract](../docs/data-format.md).

Examples were created from scratch, not extracted, anonymised or rewritten from research data. Preset responses, engineering mocks and real model output must remain distinctly labelled. Real output must record its actual execution version; presets and mocks cannot substitute for it.
