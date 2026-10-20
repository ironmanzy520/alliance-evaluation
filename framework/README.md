# Evaluation method, units of observation and evidence rules

This directory describes the evaluation method: dimensions, categories and evidence rules. The [prompts](../prompts/README.md) contain the operational instructions; the [schemas](../schemas/README.md) define input and output structures.

## What the method describes

**Bond** and **Task** are the core constructs. **Activity** (therapeutic activity) denotes specific therapeutic work identified and tracked in a dialogue. Activity is the project's object terminology, not a third Core. Identifying an Activity establishes neither agreement on that work nor the Task core. Activity wording in dimensions, categories and mappings adapts source terminology; it is not a verbatim quotation.

| Level | Meaning and boundary |
|:---|:---|
| Core | The definitional scope of Bond or Task; not separately quantified or automatically established by a single behaviour |
| Counsellor-side Contributor | Observable counsellor behaviour that may support or undermine a Core; not proof of a mutual relationship |
| Client-side Indicator | The client's explicitly expressed experience, response or position; not inferred from continued talking or apparent compliance |
| Dyadic Indicator | Connected positions, responses or reciprocity from both participants; not a rating of the current speaker alone |

[Bond](bond.md) has six dimensions and [Task](task.md) has eight. Evidence subject and result level are independent: client-side or dyadic identifies whose evidence is required; turn, Activity summary or sample summary identifies when the result is formed.

## From turns to summaries

Each turn first identifies specific therapeutic work explicitly present in the current text and not already registered, then evaluates the current turn. The model receives the original dialogue up to that turn, the current speaker and turn ID, the Activity registry and allowed dimensions. It receives neither future turns nor previous model ratings. Task dimensions are unavailable when the Activity registry is empty.

The implementation registers at most three new Activities per turn; later turns can register more. This is an implementation limit, not an Activity definition or a limit on the total number of Activities in a dialogue.

Turn-level dimensions produce `rated` findings when current evidence supports an allowed category. The four summary-level dimensions accumulate turn evidence as `evidence_only`. Otherwise, no finding is emitted for that dimension: there are no placeholder low ratings or ambiguous process states.

After each dialogue, summary covers only three Bond dimensions and engagement in each registered Activity. It neither re-evaluates the other ten turn dimensions nor explains absent ratings. An empty registry produces no Activity summary; this does not prove the text contained no identifiable Activity.

## Evidence and fields

A turn finding stores `dimension`, `activity_reference`, `finding_status`, `rating_label`, `evidence` and `reason`. Each finding must quote at least one exact span from the current turn; directly relevant earlier turns may also be quoted. Each span is bound to its source by `turn_id` and `quote`. A rationale cannot replace source text.

Bond findings have `activity_reference: null`. Task findings copy a registered Activity's ID and label. An Activity label may use original wording or a faithful short description: it identifies the work and is not rating evidence. Ordinary elaboration, a topic change or different wording alone does not establish a new Activity. Records preserve order of appearance without revising earlier judgements.

Summary results additionally contain `summary_status` and `counterevidence`:

| Status | Permitted meaning |
|:---|:---|
| `rated` | Saved evidence for the same target supports an allowed category; a non-null label and evidence are required |
| `insufficient_evidence` | An evaluable opportunity exists but saved evidence is insufficient; the label is null, available evidence should still be cited, and empty evidence is permitted only when none has been registered, with the gap explained |
| `no_evaluable_opportunity` | The complete sample offers no evaluable opportunity for this summary target; label, evidence and counterevidence are empty |

Summary evidence and counterevidence must reuse a single validated span for the same dimension, Activity and turn, or a contiguous substring within that span. Spans cannot be extended, joined, paraphrased or borrowed across targets. The complete dialogue supports context, source checking and opportunity judgements, not additional rating evidence extraction. The meaning retained by a shortened quotation still requires human review.

## Outputs and limitations

Final results have three levels: turn-level, Activity-level summary and sample-level summary. Core mapping organises saved turn ratings and summary results, including insufficient-evidence and no-opportunity states. It preserves Contributor, client-side and dyadic provenance without re-rating. Intermediate `evidence_only` records remain available for summary evidence and source review; they are neither separately mapped to Core nor a fourth final result level.

The [Bond](bond.md) and [Task](task.md) mapping tables give the complete correspondences and rationales. These conceptual mappings were predefined from Core definitions, relevant literature and the study's interpretation of each dimension. Their construct validity has not been independently tested.

The same source text may support multiple dimensions only when it separately satisfies each definition; repeated citations are not independent improvements. Results retain original turn order. They are not summed, averaged or weighted across turns, Activities or dimensions, and no Bond, Task or Alliance total score is produced.

The program validates structure, allowed categories, exact quotations and evidence ownership. These checks do not establish correct model identification, semantic judgement or the method's validity. Execution failure is recorded separately from insufficient textual evidence. Bond and Task completeness are reported independently; success on one side cannot conceal failure on the other. See [result review](../docs/review.md).

## Shared sources

These sources support Core definitions and summary observation boundaries. Additional dimension sources appear on the [Bond](bond.md) and [Task](task.md) pages. The project changes observation material and operational procedure: it does not use WAI scores directly or inherit the source instruments' reliability and validity.

- Bordin, E. S. (1979). [The generalizability of the psychoanalytic concept of the working alliance](https://doi.org/10.1037/h0085885). *Psychotherapy: Theory, Research & Practice, 16*(3), 252–260.
- Horvath, A. O., & Greenberg, L. S. (1989). [Development and validation of the Working Alliance Inventory](https://doi.org/10.1037/0022-0167.36.2.223). *Journal of Counseling Psychology, 36*(2), 223–233.
- Truax, C. B. (n.d.). *[Toward a tentative measurement of the central therapeutic ingredients](https://files.eric.ed.gov/fulltext/ED020532.pdf)* (ERIC ED020532). Arkansas Rehabilitation Research and Training Center and University of Arkansas. This consulted extract credits Truax alone and carries no publication date. The Accurate Empathy scale is on printed pp. 46–57 (PDF pp. 6–17).
- Truax, C. B., & Carkhuff, R. R. (1967). *[Toward effective counseling and psychotherapy: Training and practice](https://books.google.com/books/about/Toward_Effective_Counseling_and_Psychoth.html?id=UHdHAAAAMAAJ)*. Aldine Publishing Company. References to “Truax (1967)” in this project identify Truax's Accurate Empathy scale in this volume, consulted through the undated ERIC extract above.
- Eubanks, C. F., & Muran, J. C. (2022). *[Rupture Resolution Rating System (3RS): Manual Version 2022](https://doi.org/10.13140/RG.2.2.29780.17282)*. The project draws on Working Together items WT 2, WT 3 and WT 5 for the corresponding dimensions, without introducing rupture or repair ratings.
