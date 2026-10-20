# Implementation structure

| File | Responsibility |
|:---|:---|
| `evaluation_models.py` | Sample plans, allowed dimensions/categories, output and storage contracts |
| `evaluate.py` | Model interfaces, request construction, structural and exact-evidence validation |
| `pipeline.py` | Shared Activity → turn → summary orchestration, fixed Core mappings, failures and execution records |
| `input_data.py` | Input conversion, mapping, validation, provenance manifests and explicit run selection |
| `resources.py` | Source/installed resource resolution and actual code identity |
| `demo.py` | Three original fictional dialogues and per-dialogue preset responses through the same pipeline, producing three selectable samples in one run |
| `cli.py`, `__main__.py` | Command-line entry and model-call authorisation gate |
| `app_service.py` | Local App lifecycle, verifying installation, user, workspace and port before managing a process |
| `app.py` | Local configuration, read-only dialogue/Summary/Core viewer, on-demand validation details, planned-sample completion rate, failed-request ratio and dimension-category charts for the run or current sample |
| `__init__.py` | Package version |

User input and runs are not installation resources; the fictional demo is packaged. Relative plan sources resolve from the plan directory unless CLI `--data-root` overrides this. Prompt and schema filenames are fixed. `pipeline.py` records runtime resources; `evaluation_models.py` and `input_data.py` generate the corresponding schemas, using the same names as CLI export.

The App selects plans only from workspace `data/plans/`, uses default source resolution, and reads/writes `runs/`. It does not expose arbitrary plan, source-root or run-directory fields. Missing plans or invalid inputs prevent model execution; the preset demo needs no existing plan. Only `Run All` submits an evaluation snapshot; changing settings or refreshing does not call a model.

Validation details explain saved issues, locations and causes, separated by dividers, with relevant quotations, verified source text or accepted target evidence. They do not re-rate or repair results. `View reason` beside a turn opens and locates the existing detail. Raw exceptions and complete responses remain in run files for scoped Agent review; the App has no JSON or technical-details panel.

Result pages and validation comparisons share the existing bullet-list style. The turn ID and quotation start on the same line; only the ID text is blue-grey, without overriding font size, background, indentation or spacing. Rendered text is HTML-escaped without changing state badges or source text. The four result states use native badges with the same font, size and spacing: green for `rated`, yellow for `evidence_only`, red for `insufficient_evidence` and grey for `no_evaluable_opportunity`.

Core view shows only turn ratings and summary results, grouped by Bond then Task, dimension and Core element. If only one side has results, only that side is shown, without empty groups. Viewing does not rewrite saved files or affect turn evidence, summary inputs or Statistics.

Each manifest sample stores ordered `selected_turn_ids` before processing starts. The viewer validates their count, source order and contiguity, then limits source display to those turns. Without IDs, a matching source and sample turn count establishes a complete dialogue; otherwise the boundary is marked unknown. Selection is never inferred from missing annotations. Cached source dialogues remain complete and are not mutated by sample-specific selection.

Input preflight, saved results and individually viewed source dialogues use bounded in-memory caches, not disk caches or credential caches. Whole source datasets are not cached. Changes in path, size, nanosecond modification/status times or file identity trigger rereading; files changing during reads are not cached. Source checksum checks and result warnings remain. `run_all` independently rereads and validates actual input before execution rather than trusting cached page preflight.

Statistics category labels and bar-end counts are vertically centred on the bars, explicitly disabling the theme's first/last-label edge alignment. Chart height grows with the category count while bar width remains consistent; right padding accounts for the longest count. Both result scopes use the same layout.

The shared pipeline applies bounded retries only to connection-establishment failures, counting and recording each attempt. Exhausted retries follow the existing failure handling without automatic resume; see [model configuration](../../docs/models.md#results-and-failures).

Built-in providers are DeepSeek, OpenAI, Mistral and Ollama. Optional workspace `providers.json` registers custom connections through existing backends, not another scoring entry. DeepSeek/OpenAI/Mistral use reference-model defaults; Ollama requires a model ID. Mistral shares the `LocalHTTPJudge` Chat Completions implementation. Custom connections use their own supported parameters; empty model IDs cannot run. See [model configuration](../../docs/models.md), including completion checks and bounded connection retries.

`check_runtime_version()` verifies loaded core code against disk and checks input/output schemas. Restart the process after code changes. Runs record actual source, prompt/schema and input hashes; installed packages do not invent Git identities. See the [method definitions](../../framework/README.md) and [review guide](../../docs/review.md) for interpretation boundaries.
