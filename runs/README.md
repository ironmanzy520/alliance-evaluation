# Local run artefacts

Each run creates a new `runs/<run_id>/` and refuses to overwrite an existing directory. Except for this guide, contents are excluded from Git; the repository contains no real research runs or model outputs. The App reads and writes the workspace's `runs/`; CLI `--run-root` can select another local location.

```text
runs/<run_id>/
├── manifest.json
├── activity_identification.jsonl
├── turn_annotations.jsonl
├── summary_annotations.jsonl
├── core_evidence.jsonl
└── validation_errors.jsonl
```

## Saved records

The manifest records package version, available source Git identity, actual code/prompt/schema fingerprints, input-plan path and source checksums, selected dialogue and turn counts, model parameters, usage, failures and completeness. Each sample also records ordered `selected_turn_ids`, including turns whose processing later fails, so the App can display exactly the selected excerpt. Preserve the original sample plan and inputs as well: the manifest does not copy their contents, and paths/hashes cannot reconstruct removed or changed files. Wheel installations without source Git record a null commit rather than borrowing the working directory's identity. Input schema fingerprints are separate from the six output schemas.

Bond and Task groups from the same model response are validated separately. A valid side may survive failure on the other; read their respective completeness rather than treating global completion as semantic correctness. Original failures are preserved without automatic repair. Core mapping organises evidence without re-rating.

`turn_annotations.jsonl` retains both `rated` and `evidence_only` findings, the latter remaining candidates for summary evidence. `core_evidence.jsonl` maps turn ratings and summary results, including insufficient-evidence and no-opportunity states, but not intermediate `evidence_only`.

Core records retain `core`, `core_element`, source dimension and result level, Activity reference, category, status, rationale, evidence and counterevidence. Conceptual mapping rationales are in the [method's mapping tables](../framework/README.md), not stored as result types.

The manifest's `connection_retry_policy` records retry scope, count and delays. In `validation_errors.jsonl`, `attempt` identifies the call attempt within the same sample/turn/stage. `retry_scheduled: true` means another attempt was planned, not that it succeeded. Use sample completeness and the final result/error at that location to determine resolution. Every failed attempt contributes to `failed_calls`; every call attempt contributes to `usage.requests`. Only unresolved failures affect completeness.

## Preservation and sharing

Results containing source text or identifying information remain controlled data. Logs, manifests and absolute input paths can also disclose information. Check permissions separately before sharing; do not upload this directory.

Review results read-only, preserving saved ratings, failures and execution records. When sources are missing or checksums differ, the App retains saved results but withholds source text and highlighting. Invalid recorded turn selections also withhold source display rather than showing a different excerpt. Missing source hashes or selection metadata are identified explicitly, not inferred from result availability.
