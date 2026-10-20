# Workflow and saved results

Start with the [root README](../README.md) to install and try the preset demo. Continue with [input preparation](data-format.md), [model configuration](models.md) or [result review](review.md) as needed. Scoring definitions remain in [framework/](../framework/README.md).

## One pipeline

The CLI, App and demo call `alliance_eval.pipeline.run_all`. A sample plan selects prepared dialogues. Each turn identifies new Activities before evaluating the current turn. Structure, roles and identities, exact quotations and evidence ownership are validated before results are saved separately for Bond and Task. After the dialogue, summary covers three Bond dimensions and engagement in each registered Activity. Turn ratings and summary results then receive fixed Core mappings; intermediate `evidence_only` records remain available to summary but are not separately mapped.

The model identifies Activities and judges evidence and categories. The program controls allowed targets, structure, evidence binding, persistence and completeness. `other` turns remain context rather than counsellor/client evaluation targets. Turn calls receive neither future text nor earlier model ratings. Summary does not fill the other ten dimensions or explain missing findings. Detailed evidence constraints follow the method definitions and operational prompts and schemas.

Completeness is reported separately for Bond and Task. Valid output on one side survives failure on the other, without establishing complete evaluation. Diagnostic summaries after upstream failures are not complete results. An absent Activity registry, empty findings, insufficient evidence and execution errors have different meanings.

Only connection-establishment failures receive [limited retries](models.md#results-and-failures). Every failed attempt is retained; only unresolved failures affect completeness. Continuation and termination rules for other errors remain unchanged. There is no automatic resume.

## Inputs, installation and run identity

- `prepare-data` performs local structural conversion. `create-plan` registers explicitly selected whole dialogues; it is not a sampling method. Reuse existing plans. Relative `source_file` paths resolve from the plan directory unless `--data-root` overrides this.
- Development installations read the single top-level prompt/schema sources. Regular installations use packaged copies of the same resources. Data workspaces are independent of code location; execution identity is not inferred from the user's working-directory Git repository.
- Runs record package version, actual code fingerprints, prompt/schema hashes, plan and source hashes, model parameters, usage, errors and completeness by side. The same source bytes are parsed and checksummed.
- A Git commit does not replace an execution fingerprint. Explain dirty states and non-Git installations accurately. Restart the App process after changing loaded Python code; refreshing the browser is insufficient.
- Hashes identify files but do not back them up. Reproduction needs the original plan, input bytes, conversion manifest, model configuration and matching implementation. Service changes and nondeterminism may still change outputs.

## What demos, statistics and tests establish

The preset demo sends original fictional dialogues and responses through the normal pipeline to illustrate input, validation, persistence, evidence review and statistics. It does not test model understanding. Offline tests check engineering invariants and failure boundaries. Real-model integration checks actual execution for the selected service and configuration; these are distinct forms of evidence.

Statistics describe saved results in the selected run. See [result review](review.md) for inclusion rules, denominators and interpretation, and the [method definitions](../framework/README.md#outputs-and-limitations) for validity limits.
