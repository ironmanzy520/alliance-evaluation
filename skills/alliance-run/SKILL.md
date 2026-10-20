---
name: alliance-run
description: Demonstrate alliance-evaluation without a model, or check prepared input, configure a model and run authorised Bond/Task evaluation through its shared pipeline. Not for changing rating rules, research sampling, automatic reruns or reviewing saved results only.
---

# Run an evaluation

Use the existing CLI or local app; do not add a separate scorer, caller or default model in the skill. Resolve relative links from this skill's real directory, following any discovery symlink. Input, installation and run directories may be separate.

## Demonstration or model run

For a demonstration, follow the [quick start](../../README.md), select `Run preset demo` and inspect the dialogues using `Sample`. Existing runs can be viewed directly. The demo uses original fictional dialogue and preset responses, not model output or a human gold standard; no key is needed.

For a model run, read [model configuration](../../docs/models.md). Confirm the prepared plan, data scope, provider/model, parameters and new run location. Reuse an existing plan rather than resampling. Adapting a non-compatible service is a separate development task.

```bash
alliance-eval check-runtime
alliance-eval validate-input --sample-file "<plan-file>"
```

Relative sources resolve from the plan directory unless an explicit `--data-root` is appropriate. Retain the plan, input JSONL and conversion manifest; hashes are not backups. Stop new runs on a runtime mismatch. Python changes require restarting the app process, not merely refreshing the browser.

Run `alliance-eval all ... --allow-model-calls` or click `Run All` only when authorisation covers the selected data, service, parameters and possible cost. The button starts the run immediately. Do not ask again for already explicit, matching authorisation; otherwise stop at local checks. Skill selection or an enabled button is not authorisation. Keep keys in environment variables or the app session.

## Execution and hand-off

Use the shared `pipeline.run_all`, preserving the [execution architecture](../../docs/README.md) and [output contract](../../runs/README.md). Follow the built-in bounded connection retry policy in “Results and failures” in [model configuration](../../docs/models.md); do not add retries, fill failures or generate a separate model summary. Do not overwrite existing runs or repair responses. A new run requires an explicit scope and a new output location.

Reopen the manifest and outputs when execution ends. Report the actual run path, execution version, input and processed scope, global status, Bond/Task completeness and failures. Process exit or visible output is not proof of complete evaluation; a diagnostic summary is incomplete, while an empty finding need not be a failure.

For content checks, follow [result review](../../docs/review.md). Do not rewrite results, calculate a total score or treat engineering success as evidence of model accuracy or the method's validity.
