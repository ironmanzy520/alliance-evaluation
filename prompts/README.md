# Evaluation prompts

| File | Responsibility |
|:---|:---|
| `identify_activity.txt` | Identify specific therapeutic work explicitly present in the current turn and not already registered |
| `judge_turn.txt` | Extract evidence or form turn categories using dialogue up to the current turn and the Activity registry |
| `judge_summary.txt` | Summarise three Bond dimensions and engagement in each Activity |

These three prompts define identification, turn evaluation and summary rules. Source and packaged installations use the same content. See the [method definitions](../framework/README.md) for definitions and sources and [schemas](../schemas/README.md) for output contracts.
