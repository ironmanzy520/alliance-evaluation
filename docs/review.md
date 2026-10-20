# Reviewing results

For a question about a field or statistic, inspect that result and its explanation below; a full run audit is not always necessary. For evidence, completeness or reproducibility reviews, read the original input, Activity records, turn/summary results, Core mappings, errors and manifest within the requested scope. See [run artefacts](../runs/README.md) for files and [method definitions](../framework/README.md) for meaning. Visible results or valid JSON do not establish correct judgements.

1. **Version and scope.** Check actual code and prompt/schema fingerprints, model parameters, selected turns, plan and source hashes. Preset demos are not model results. Preserve saved files when reviewing them.
2. **Completeness by side.** Check Bond/Task upstream annotations and summary status separately. Distinguish failure, missing output and insufficient evidence. Diagnostic output is incomplete; success on one side cannot hide failure on the other.
3. **Activity and source.** Was the work already present, duplicated or assigned to the wrong target? Does each quotation belong to its bound turn, include current-turn evidence and genuinely support the dimension and category? Exclude future information; string matching does not replace semantic review.
4. **Summary and Core.** Evidence and counterevidence must come from saved valid spans for the same dimension and Activity. Contiguous shortening must preserve negation and conditions. Core follows predefined relationships without new summaries to fill gaps. Repeated quotations are not independent evidence.
5. **Conclusions.** Missing ratings do not imply low performance; empty findings establish neither execution failure nor absence of opportunity. Distinguish mechanical acceptance, model reliability and research validity. Do not calculate totals or claim therapeutic effectiveness.

## Explore the App

Select `Run` and `Sample`. Task denotes the core construct; Activity identifies specific work. See [run artefacts](../runs/README.md) for the saved files.

Dimension labels use `Bond ·` or `Task ·` in turn results, Summary, citations and Statistics. Core mapping groups by core.

- **Dialogue & annotations:** source text, turn findings and cross-turn citations. Activity sections appear only where new work was identified. `Turn findings` counts saved findings for the current sample, including `rated` and `evidence_only`, without counting Summary, Core or citations again. Turn expanders show their own count; zero-finding turns can still have Activity or citation information. Counts do not establish quality or completeness.
- **Summary:** dimension, Activity and result, with expandable rationales, evidence and counterevidence. Only Accurate Empathy, Therapist accepts/validates Patient, Mutual trust and respect, and engagement in each Activity are summarised, not all 14 dimensions.
- **Core mapping:** predefined links from dimension results to Bond/Task elements or scope, with source quotations. Summary insufficient-evidence and no-opportunity states remain, without new ratings or totals. Intermediate `evidence_only` records stay in turn annotations and are not separately mapped. Summary quotations need not exhaust all intermediate evidence. A missing summary cannot be replaced with intermediate evidence.

Checking whether a quotation supports a judgement is distinct from validating the conceptual dimension-to-Core relationship. The [method's mapping tables](../framework/README.md) document those relationships.

Result badges use a consistent font, size and spacing to distinguish states: `rated` (green) has a category, `evidence only` (yellow) is turn evidence awaiting summary, `insufficient evidence` (red) means insufficient summary evidence, and `no evaluable opportunity` (grey) means no evaluable opportunity for that summary target. A rating, when present, follows its state. Colour identifies the evidence state, not favourable performance or execution success.

Evidence retains bullet lists, indentation and line spacing. Only the turn-ID text is blue-grey, without using the state colours. Quotations on all result pages and in Validation begin with “• ID — source text” on the same line; long quotations wrap naturally rather than placing the ID on a separate line.

## Run status and validation

`Status` reads the selected run's `manifest.json`; it describes execution, not judgement quality. The help icon provides a short explanation. Streamlit's temporary `Running` indicator during page updates is separate from evaluation execution. Unchanged files can reuse in-memory reads; opening a large file for the first time or after a change still requires reading and validation. Changing selections or reading cached results does not call a model; only `Run All` submits evaluation.

| Display | Saved value | Meaning |
|:---|:---|:---|
| `Running` | `running` | Started without a saved end state; not live process detection, so forced termination may leave this value |
| `Complete` | `completed` | Finished without unresolved call or validation failures; recovered retries may have occurred, and judgements may still be wrong |
| `Errors` | `completed_with_errors` | Reached the end with unresolved call or validation failures; validated results remain |
| `aborted` | `aborted` | Stopped early on a non-recoverable error, such as a fatal API error or unexpected exception; saved artefacts remain |

For errors or aborts, inspect `Validation details`, `validation_errors.jsonl` and completeness by side. Do not replace side-specific review with global status or edit old records to show success.

When a sample is fully complete and error-free, no extra confirmation panel is shown. Errors, incomplete results or uncertainty produce a notice below `Sample` and `Validation details`. Dividers separate individual issues, showing only their title, location, reason and evidence without repeatedly stating that the result was not accepted or other results were retained. Quotation mismatches compare the returned quote with the specified source turn; summary-ownership errors compare accepted evidence for the same target; Activity-label errors compare returned and registered labels. Missing or unverifiable sources are stated explicitly, without using changed text as proof or guessing an unrecognised error's cause.

Beside an affected turn, only the affected component and `View reason` are shown. Clicking opens and locates the corresponding detail without repeating a long error message. The App does not display raw exceptions or entire model JSON; original records remain in `validation_errors.jsonl` for scoped Agent review, without deleting or changing failure evidence. Unrecognised errors receive a general explanation rather than a guessed specific cause. Diagnostic results remain viewable but incomplete, with the underlying records by side unchanged.

Recovered connection failures retain retry notices and attempt records without marking the turn as an annotation error or excluding an otherwise complete result.

## Fully validated samples

This metric is `counts.completed_samples`: planned sample entries whose Activity identification, turn annotation and summary completed with no unresolved Bond/Task call or validation failures. Each plan entry is one sample. A successful connection retry does not reduce completeness, but failed attempts remain in request statistics and logs.

Samples with errors are excluded from this count even though their validated partial results remain. The metric measures execution completeness, not accuracy. When counts are valid, the App shows completed/planned samples and a percentage, using `counts.planned_samples`, not the number processed. For example, three complete samples out of five planned gives 3/5 and 60% process completeness, not 60% correct judgements. Unprocessed samples remain in the denominator during execution or after an abort. Missing/zero denominators or completed counts exceeding planned counts yield no inferred percentage. This percentage is not the complement of the failed-request proportion.

## Statistics

Select `Sample`, then expand `Statistics`. Execution statistics always cover the selected run. `Result scope` changes only the category charts: `All samples in run` or `Current sample`. Charts identify result level and scope; current-sample charts include sample and dialogue IDs. Changing samples retains the selected dimension.

- **Failed requests:** `counts.failed_calls / usage.requests`. The denominator includes every actual call attempt; the numerator includes failures before successful connection retries and other interface/output-validation failures. Fully complete runs can therefore have failed requests. This is not a model judgement error rate. No ratio is calculated for zero requests; preset demos make zero model requests.
- **Category counts:** valid results for one dimension, not sample counts or independent evidence counts. The ten turn dimensions use saved validated findings, including valid partial findings from incomplete samples. The four summary dimensions require complete upstream annotations and `complete` summary status on their respective Bond/Task side. Intermediate `evidence_only` findings are not counted again as final categories.
- **Exclusions and empty results:** diagnostic summaries or summaries whose completeness cannot be confirmed are excluded, with a notice and details where available. No countable results does not imply a low rating, missed detection or no opportunity. Valid `insufficient_evidence` and `no_evaluable_opportunity` results remain separate categories, not execution failures.

Both chart scopes use the same inclusion rules, filtering samples without generating an overall dialogue rating. Each sample normally contributes one result per Bond summary dimension; engagement is counted per Activity rather than collapsed to one dialogue category. See Summary for Activity-specific evidence.

Turn and summary levels are labelled and not pooled across dimensions. No means or total scores are calculated. Counts describe saved results, not accuracy, opportunity-detection rates or therapeutic effects.

## Source text and citations

Yellow highlighting means valid saved output quotes that text, not that the turn necessarily has its own rating. Expand its annotations to see current findings. `Cited by` lists referring turns or Summary and their dimensions without duplicating ratings or adding findings. Open the corresponding result for full details; final categories for `evidence_only` appear in Summary.

The viewer reads original source paths recorded by the run, resolving relative `source_file` using recorded `data_root` or, if absent, the original plan directory. Missing files or changed checksums withhold source text and highlighting while saved results and errors remain accessible. Missing hashes are identified as unconfirmed source identity. Viewing changes neither inputs nor saved runs.

The manifest records `selected_turn_ids` for each sample. The viewer shows only those turns; failed or unprocessed selected turns remain visible. For an excerpt, a caption identifies its first and last IDs. Complete dialogues need no additional scope caption. Invalid selection records withhold source display. If selection IDs are missing and the recorded turn count does not establish a complete dialogue, the viewer labels the selection boundary as unknown. Consult the original sample plan to establish the range.
