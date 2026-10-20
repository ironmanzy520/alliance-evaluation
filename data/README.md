# User data

Except for this guide, this directory is excluded from Git and installation packages. Preserve original files. Conversion does not establish permission for model transmission or redistribution.

```text
data/
├── raw/         Authorised original files and provenance notes
├── processed/   Converted dialogue JSONL and manifest.json
└── plans/       Explicit run sample plans
```

Reuse an existing plan when available. For new inputs, follow the [input contract](../docs/data-format.md) for inspection, mapping and conversion.
`create-plan` records already selected whole-dialogue IDs; it neither chooses a sampling method nor calls a model.
Inputs may be outside the repository. Relative source paths resolve from the plan directory by default.
Provenance or sample-selection notes accompanying inputs may be kept in this directory; reserve `plans/` for runnable sample plans rather than those explanatory JSON files.

The [preset demo](../examples/README.md) creates independent local fictional inputs without reading user originals.
