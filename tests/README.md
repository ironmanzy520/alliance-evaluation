# Offline engineering tests

Tests cover input, output contracts, execution entry points, resources, evidence binding, group isolation, persistence and local App behaviour. They use hand-written payloads, original fictional dialogue, deterministic `MockJudge` instances and interface mocks. They neither read real research data nor call real models.

| File | Coverage |
|:---|:---|
| `test_evaluation_models.py` | Dimensions, labels, states, role-specific output, Activity references and sample-plan ranges |
| `test_evaluate.py` | Request construction, interface mocks, schema/evidence validation, JSON-object schema guidance, completion metadata, truncation rejection, retry inputs, Mistral parameters and response blocks |
| `test_group_isolation.py` | Bond/Task isolation in the shared pipeline; retry counts, records, recovery and continuation after exhaustion; persistence and Core mapping without intermediate evidence or mapping-type fields, preserving summary inputs and three result states |
| `test_input_data.py` | CSV/JSON/JSONL mapping, source/metadata preservation, IDs/order, read-only preview, overwrite prevention and write-failure state |
| `test_processing_workflow.py` | Original cross-format/XLSX/TSV/TXT/MD fixtures, text and line provenance, CLI, repeatability, input fingerprints, plan creation and reading, and failure preservation |
| `test_product_runtime.py` | Direct-import dependency declarations, independent workspaces, package resources and fixed filenames, matching CLI schema exports, code/schema/input fingerprints, strict source reads, saved turn selections and pre-call boundaries; method dimensions, Core targets, Bond definition elements and prompt-card groups aligned with runtime contracts |
| `test_app_service.py` | Management CLI, process identity, workspace/port isolation, occupied-port refusal, normal/forced stop, restart ordering and Ctrl+C cleanup; signals are mocked |
| `test_product_app.py` | Local paths, custom providers, credential isolation, source reads/checksums, preflight cache invalidation, execution revalidation, single-click submission, sample and excerpt selection, missing selection metadata, cross-turn citations, Summary/Core rendering, validation error recognition, exact evidence comparison and navigation, hidden raw errors/JSON, run/current-sample counts and incomplete-result exclusions, completion denominators and missing/invalid boundaries |
| `test_demo.py` | Complete three-dialogue preset demo, per-dialogue binding through the shared pipeline, input preservation and repeat runs |
| `conftest.py` | Shared offline boundary: removes common model credentials, including Mistral, and blocks network connections |

Install development dependencies and run from the project root:

```bash
python -m pytest tests -q
```

The test count is determined by the current suite. Passing establishes the tested software behaviour. Evaluation quality and validity require separate evidence; see the [method's limitations](../framework/README.md#outputs-and-limitations).

Statistics layout assertions cover centred labels/counts, all category labels, count-dependent spacing and row height. Fonts, clipping and narrow/wide layouts still require browser inspection when the interface changes.

Real-model integration is outside offline testing and requires separate data and service authorisation.
