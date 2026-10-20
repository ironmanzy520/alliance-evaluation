# Complete preset workflow demo

Three original fictional dialogues and preset responses demonstrate the complete pipeline. One run contains three samples selectable in the App; Statistics can describe the whole run. No model, API key or network is required. The presets illustrate output structures, not model performance or expert gold-standard annotations.

Start the App from the [quick start](../../README.md) and click `Run preset demo`. The program copies the dialogue, plan and preset-response files into workspace `data/demo/<run_id>/`, then uses the normal evaluation pipeline to create `runs/<run_id>/`. Repeated demos use new directories without overwriting inputs or results. The copied inputs remain available for source verification even if the installation is later removed.

## Three dialogues

| Sample | Dialogue ID | Content and demonstration |
|:---|:---|:---|
| S0001 | `fictional-scripted-demo-001` | Eight turns: drafting a work-extension request; conditional acceptance, negotiation, writing engagement and uncertainty about helpfulness |
| S0002 | `fictional-scripted-demo-002` | Six turns: sorting bedtime worries into tonight and tomorrow; client-proposed Activity, counsellor acceptance, clear steps and perceived help |
| S0003 | `fictional-scripted-demo-003` | Six turns: rehearsing a sentence to end a call with a friend; explicit disconfirmation, corrected understanding and adjustment following rehearsal feedback |

Each dialogue registers one Activity. Activity IDs are local to their sample, not merged across dialogues. Highlighted text is quoted by a saved turn finding or summary, potentially as earlier context for a later finding. It does not imply that the highlighted turn has an independent rating; its expander shows referring results.

## The first dialogue, end to end

1. `demo-01` describes a difficulty without establishing an Activity. `demo-02` takes up the concern, producing a grounded-uptake turn result and empathy/acceptance evidence for summary.
2. `demo-03` explicitly confirms the earlier understanding and expresses feeling understood. This differs from inferring acceptance merely because the client continues talking.
3. `demo-04` proposes drafting an extension request and first registers `ACTIVITY_01`. Later writing choices and sentences refer to that same work rather than creating new Activities.
4. `demo-05` expresses conditional acceptance of writing first and the work's relevance. `demo-06` uses that preference to specify the writing approach and responsibilities, demonstrating negotiation.
5. `demo-07` actually drafts a sentence, supplying engagement evidence while retaining uncertainty about helpfulness. Participation does not establish efficacy.
6. `demo-08` closes the excerpt. Summary covers only the three Bond targets and this Activity's engagement; it does not re-rate all turn-level dimensions. Preset acceptance and engagement categories illustrate the result structure. Empathy retains evidence without choosing a representative anchor; mutual trust retains an empty insufficient-evidence result because sufficient dyadic evidence was not registered.
7. `Core mapping` organises saved dimension evidence under Bond/Task with expandable provenance and quotations. It neither re-rates nor totals results, and does not assert a good alliance.

Empty findings mean the demo supplies no rating at that turn, not a low score or identification failure. Dialogues need not populate all 14 dimensions, and other experts or models need not make the same judgements.

Preset responses pass the same structural, exact-quotation, Activity-ownership and summary-provenance validation as model responses. No model or network requests are made, and the run explicitly records its demonstration nature.
