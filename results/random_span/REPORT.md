# Random-span control (Paper 1, Option B item 1)

**Question.** Does structural protection identify useful positions, or does it only reserve capacity?

**Design.** Arm 7 uses the same `StructuralProtectionWrapper` over `PermanentEvictPolicy` as arm 2 (same
budget, atomic groups seated by best-member attention). Only the protected set changes. The set is one random
contiguous token span per structural group, with that group's exact token count, placed uniformly over the dump
without overlap. Protected tokens: 311.2 on average, identical to arm 2 in all 450 rows. Group count 26,
same size distribution. Random protection covers 32.2% of credential tokens on average (range 1% to 72%
per prompt); structural covers 100%.

Same task, engine, `SCORE_MODE="sum"`, recency window, sink and seeds 3000 to 3149 as Phase 1, n = 150, budgets
154 (PARTIAL), 257 and 514 (VALID). Budgets 51 and 103 (VOID) were not run. Arms 1, 2 and 5 are the existing
Phase 1 iso-token rows. A rerun of arm 2 at budget 257 reproduced Phase 1 exactly on 25 of 25 seeds
(`determinism_check.jsonl`), so those rows are a like-for-like reference.

"Attention-selected protection (existing)" was read as arm 1, attention-ranked retention with no reserved set.
The wrapper's own within-protected ranking is attention, and it is held fixed in arm 2 and arm 7.

Statistics: paired by seed, mean difference, 10,000 prompt-resampled bootstrap (rng seed 0). Equivalence means the
whole 95% CI is inside ±0.05, as in PREREG §3.

## Accuracy (fraction retrieved)

| retention (budget) | 1 none | 7 random span | 2 structural | 5 oracle |
|---|---|---|---|---|
| 0.15 (154, partial) | 0.011 | **0.027** | 0.196 | 0.648 |
| 0.25 (257) | 0.010 | **0.049** | 0.423 | 0.943 |
| 0.50 (514) | 0.010 | **0.076** | 0.964 | 0.958 |

## Contrasts

| retention | random − none | structural − random | structural − none |
|---|---|---|---|
| 0.15 | +0.0156 [+0.0078, +0.0244], 13/150 differ | +0.1689 [+0.1522, +0.1844], 128/150 | +0.1844 [+0.1689, +0.1989] |
| 0.25 | +0.0389 [+0.0256, +0.0522], 30/150 | +0.3744 [+0.3456, +0.4022], 144/150 | +0.4133 [+0.3900, +0.4367] |
| 0.50 | +0.0656 [+0.0489, +0.0833], 47/150 | +0.8889 [+0.8678, +0.9089], 150/150 | +0.9544 [+0.9411, +0.9667] |

## Findings

1. **Structural protection is distinguishable from random protection at every budget**, by 0.17, 0.37 and 0.89,
   with the arms differing on 128 to 150 of 150 prompts. Matching the protected token count and group sizes does
   not reproduce its effect.
2. **Random protection is distinguishable from no protection, but only slightly.** All three intervals exclude
   zero (+0.016, +0.039, +0.066). Only the 0.15 interval is inside ±0.05, so equivalence to no protection is declared
   there and not at 0.25 (upper end +0.052) or 0.50. The gain grows with budget and stays under 0.08 in
   absolute terms.
3. **Reserved capacity alone accounts for at most about 0.07 of a 0.42 (at 0.25) or 0.95 (at 0.50) effect.**
   The random arm's small gain tracks how many credential tokens the random set happens to cover
   (per-prompt correlation of coverage with accuracy 0.27, 0.28, 0.30; accuracy in the lower-coverage half of prompts
   0.009, 0.022, 0.037 against the upper half 0.045, 0.077, 0.115). That is what a content effect predicts.
4. At retention 0.50 structural protection (0.964) equals the retention oracle (0.958; difference −0.0067,
   [−0.0167, +0.0033], inside ±0.05). Below that the oracle gap remains: 0.452 at 0.15 and 0.520 at 0.25.

## Scope and caveats

- One model, one task, one distractor format. AMD's ablation (distractor format changes what protection competes
  against) implies the size of the structural effect depends on the target-to-pattern-match ratio (6:26 here). This
  control shows the effect is about what is protected. It does not show that the pattern generalises.
- Random spans are contiguous token runs, not line-aligned, and can cut a credential line partway. That is part of
  what "no information about credentials" means here, but it is one random-protection design of several.
- Retention 0.15 is a qualified budget, so the clean evidence is 0.25 and 0.50.
- The protected set (about 311 tokens) exceeds the budget at 154 and 257, so there the wrapper seats groups by
  attention until the budget is used up, in both arm 2 and arm 7. At 514 the whole set fits (the competitive room is
  449 tokens), so seating is not selective and the result there is mainly about which positions were protected. The
  control holds the seating mechanism fixed by construction.

Files: `raw_results.jsonl` (450 rows), `determinism_check.jsonl` (25), `analysis.json`; code
`kvcache_harness/run_random_span.py`, `analyze_random_span.py`.
