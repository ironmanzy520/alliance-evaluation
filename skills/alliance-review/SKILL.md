---
name: alliance-review
description: Explain saved alliance-evaluation Bond/Task results and statistics, or review completeness, source evidence and reproducibility within the requested scope. Read-only; not new model scoring, automatic repair or clinical judgement.
---

# Review saved results

Read input and run files without calling models or rewriting results. Resolve links from this skill's real directory, following any discovery symlink.

Task is the core construct; Activity is the specific therapeutic work. Interpret saved records using the [output guide](../../runs/README.md), preserving their content and execution identity.

Identify the requested run and question, then read only the relevant material. Inputs and runs need not be moved into the code repository.

## Choose the review scope

- **Explain a result or statistic:** inspect the saved fields and the relevant section of [result review](../../docs/review.md). Use the output guide for unclear fields. Preserve request denominators, turn/summary levels and exclusion rules; do not automatically audit every hash or evidence span.
- **Review evidence, completeness or reproducibility:** follow the relevant checks in result review. Read source context, selected turns, Activity records, per-core status and version/source hashes as needed. Consult the [method definitions](../../framework/README.md) and relevant dimensions for semantic questions. Exact string matching or an isolated quotation does not establish semantic support, and mismatched source files cannot validate a quote.

Tie findings to the actual scope inspected. Locate issues by file and sample/turn/Activity or field, distinguishing deterministic inconsistencies, semantic concerns and missing information. Do not infer why an unsaved rating is absent or extract new evidence to fill gaps.

## Interpretation and hand-off

No rating does not mean poor performance; an empty finding does not establish failure or no evaluable opportunity. Repeated quotations are not independent evidence. Execution completeness and category counts are not accuracy, and preset demonstrations are not model performance. Do not combine dimensions into a total score or infer treatment effects.

Report read-only findings and their basis. Describe the scope needed for any correction or rerun without performing it. Save a separate report only when explicitly requested, leaving input and run files unchanged.
