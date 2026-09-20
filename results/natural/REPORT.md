# Natural-text validation of the retention/promotion decomposition (Paper 1, Option B item 2)

**Question.** Does the protection-versus-oracle gap seen on the synthetic credential task (0.423 vs 0.943 at
retention 0.25) appear on natural text?

## What was and was not reused

`nat_v1.jsonl` (Paper 3) could not be used as is. Its contexts are 4,099 tokens (Qwen render), and this harness
prefills the whole first turn in one forward with `output_attentions=True`, which needs about 28 layers x 12
heads x 4,099^2 in bf16 (about 11 GB) for the attention weights alone, on a 12 GB card. Chunked prefill would be a
change to the engine that every Paper 1 result depends on, so it was not attempted.

Instead `kvcache_harness/build_natural_dataset.py` reuses Paper 3's generator, Gutenberg prose sources, fact
template and scorer **unchanged**, and changes only the context-length target (2,048 +/- 32 tokens) and the
distractor count (D = 16, so 20 fact sentences of which H = 4 are queried). This is a **new dataset**,
`data/natural_l2048/nat_l2048_v1.jsonl` (100 instances, sha256 `9a64bc84f268c8ec275b541a003793f06a4f8255f81579016827a3e168940a59`),
tokenised with the Qwen tokenizer only. It is not `nat_v1` and results here are not comparable to Paper 3's numbers.

## Mapping onto this experiment (an adaptation, not the original task)

| credential experiment | natural-text version |
|---|---|
| 6 credential lines | the H = 4 queried fact sentences ("The audit of the X programme was completed by P ...") |
| 20 distractor lines | the D = 16 other fact sentences, same template |
| structural pattern (`LABEL:` lines) | regex `The audit of the ... depot.`, the cue the task's own instruction states; 21 matches (20 facts plus the worked example), one atomic group each; about 1,008 protected tokens (48% of the context) |
| retention oracle | gold token span of each queried fact at level 1 (sentence start to end of the person name), about 66 tokens |
| 6 shuffled turns | 4 turns, one per queried fact, shuffled per instance, question "Who completed the audit of the X programme? Answer with the name only." |
| score | Paper 3's scorer (whole-token, case-insensitive, no other person named), fraction of 4 turns correct |

Model Qwen2.5-1.5B-Instruct, bf16, eager attention, same engine, `SCORE_MODE="sum"`, recency window 64, sink kept,
8-bit tier (iso-token). n = 100 instances, budgets **256 (0.12 retention)** and **512 (0.25 retention)**.
A 10-instance pilot at 256, 512 and 1,024 was run first to check the gates; 1,024 was dropped because protection
already reached 0.875 against a full-cache 0.875 there (no headroom). The budget choice was made from the pilot,
not from the main run.

## Results (n = 100 per cell, no errors)

| arm | budget 256 | budget 512 |
|---|---|---|
| 1 no protection | 0.000 | 0.010 |
| 2 structural protection | 0.133 | 0.350 |
| 4 protection + recoverable tier | 0.133 | 0.350 |
| 5 retention oracle | 0.835 | 0.868 |
| 6 full cache (budget-free) | 0.900 | 0.900 |

Paired by instance, mean [95% CI], instances differing:

| contrast | 256 | 512 |
|---|---|---|
| structural − none | +0.133 [+0.103, +0.163], 47 | +0.340 [+0.305, +0.375], 91 |
| recoverable − structural (arm 4 − arm 2) | +0.000 [−0.008, +0.008], 2 | +0.000 [−0.008, +0.008], 2 |
| oracle − structural | **+0.703 [+0.655, +0.748]**, 97 | **+0.518 [+0.473, +0.563]**, 95 |
| oracle − full cache | −0.065 [−0.113, −0.020], 43 | −0.033 [−0.073, +0.005], 39 |

## Findings

1. **The retention gap appears on natural text.** Oracle minus structural protection is +0.52 at retention 0.25
   and +0.70 at 0.12, with intervals far from zero and 95 to 97 of 100 instances differing. The credential task gave
   +0.52 at retention 0.25. The gap is not specific to the synthetic credential format.
2. **Structural protection helps but leaves most of the headroom**: +0.133 and +0.340 over no protection, against
   an oracle at 0.835 to 0.868 (near the 0.900 full-cache ceiling).
3. **Recoverability adds nothing over protection under matched tokens** (arm 4 = arm 2 to four decimals at both
   budgets, differing on 2 of 100 instances). This matches the Phase 1 null.

## Caveats, including where the fit is weak

- **The null in finding 3 is partly structural.** With about 1,008 protected tokens against budgets of 256 and 512,
  the protection wrapper seats groups first and hands the tier almost no budget, so the recoverable tier has nothing to
  do (the composition trap documented in `cache/protected_split_tier.py`). Finding 3 is an 8-bit,
  iso-token result on this composition, and the Phase 1 null had the same property. It should not be read as
  independent evidence about promotion. Only findings 1 and 2, the retention gap, are the point of this item.
- The promotion-signal experiment (Phase 2 at 4-bit) was not run on natural text.
- **New dataset, adapted task.** Level-1 person-name questions only; queries are sequential turns, not the parallel
  variants of Paper 3's Experiment C; the structural pattern is a regex on the task's own cue and is a reasonable
  analogue, not a claim that it is what a deployed system would use. The oracle keeps only the gold sentence prefixes
  (about 66 tokens) plus sink and recency window, so the preamble and worked example are not retained.
- The fact-to-context ratio differs from the credential task (4 queried of 20 sentences, about 48% of the context
  matched by the pattern, against 6 of 26 lines). Protection fills half the context and competes hard for the budget,
  like the 6:26 ratio, but the ratios are not the same.
- Oracle is below full cache at 256 (−0.065, CI excludes zero). It is a retention oracle, not a full-cache
  replacement; the gap to full cache is small next to the gap to protection.
- One model (Qwen2.5-1.5B), one 2,048-token length, two budgets. Contexts are longer than the 1,029-token credential
  task, so retention fractions, not token counts, are the comparable quantity.

Files: `raw_results.jsonl` (900 rows), `pilot.jsonl` (130), `analysis.json`; code
`kvcache_harness/build_natural_dataset.py`, `tasks/natural_text.py`, `run_natural.py`, `analyze_natural.py`;
dataset `data/natural_l2048/nat_l2048_v1.jsonl` with `.sha256`.
