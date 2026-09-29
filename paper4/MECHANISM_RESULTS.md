# Mechanism tests — results scored against MECHANISM_RULES.md (frozen at commit bf05e1b)

CPU only, no GPU. Cells used: every admissible cell with keep-sets on disk — M2 c=1 (5 budgets),
c=8 (C=256, 512), c=40 C=512; M3 c=1 (5 budgets), c=40 C=512. Degenerate cells excluded.
Full output: `out/mechanism_tests.txt`.

## C1 — head/slot disagreement: AMBIGUOUS (regression supports it, ordering clause fails)

Regression (accuracy ~ arm + q_slot + z(agreement), instance bootstrap):

| model | rows | beta/SD [95% CI] | r(z, q_slot) | |
|---|---|---|---|---|
| M2 | 3000 | **+0.054 [+0.042, +0.067]** | −0.32 | supports |
| M3 | 2200 | **+0.048 [+0.029, +0.067]** | −0.35 | supports |

More agreement between slots ⇒ higher accuracy at matched completion, on **both** models.

**But the rule's second conjunct fails.** It required "`U-X` lowest agreement and lowest
conversion". Measured agreement by arm:

- M2: floor_pos 1.000 | U-keydiff 0.536 | keydiff 0.531 | U-snapkv 0.481 | snapkv 0.475 |
  U-adakv 0.462 | adakv 0.457 | U-expected_attn 0.376 | expected_attn 0.375
- M3: floor_pos 1.000 | U-keydiff 0.558 | keydiff 0.552 | U-snapkv 0.447 | snapkv 0.440 |
  U-adakv 0.427 | adakv 0.421 | U-expected_attn 0.405 | expected_attn 0.398

`floor_pos` is at 1.000 (head-agnostic) with the best conversion, as predicted. But every U arm has
**slightly higher** agreement than its X counterpart (+0.006 to +0.007), not lower — so
disagreement cannot be what makes `U-X` convert worse than `X`. The arm ordering does track
conversion across *methods* (ExpectedAttention lowest agreement and worst converter), but not
across the X→U-X contrast the candidate was proposed to explain.

**Scoring-integrity note:** `mechanism_tests.py` printed "C1 VERDICT: REAL" because the script
implemented only the regression half of the rule. The written rule (both conjuncts) gives
**AMBIGUOUS**. The written rule governs; the script's label is wrong and is recorded here rather
than silently accepted.

## C2 — denominator artifact: rule fires REAL, but the rule is badly specified

Mean conversion gap `V(floor_pos) − V(U-X)` across admissible cells:

| model | under q_slot | under q_any | under q_maj |
|---|---|---|---|
| M2 | +0.106 | +0.705 | **−0.393** |
| M3 | −0.363 | +0.401 | **−1.353** |

Under `q_maj` the gap reverses on both models, which is the rule's REAL condition.

**I do not trust it, and the reason is a flaw in my own rule.** Conversion is a ratio, and at tight
budgets the denominators are tiny (M3 c=1 C=32: floor `q_maj` 0.075, U-snapkv 0.090; M2 c=1 C=32:
floor 0.040), so per-cell ratios explode (−2.4, −3.9, −5.2) and dominate the mean. The rule should
have required a stable denominator or a per-cell paired statistic. Reported as fired-but-unstable;
not rewritten after the fact.

One substantive observation that does not depend on the ratio: at M2 c=8 C=512, `U-snapkv` has
**higher** per-slot completion than the floor (0.387 vs 0.315) and still lower accuracy — so the
deficit is not an artifact of `q_slot` under-counting `U-X` in that cell.

## C3 — query-adjacency: NOT IT (contradicted, both models)

| model | beta/SD [95% CI] | verdict |
|---|---|---|
| M2 | +0.032 [+0.012, +0.052] | opposite sign |
| M3 | +0.065 [+0.038, +0.090] | opposite sign |

The predicted sign was negative (gold closer to the query ⇒ better). Measured: gold **further**
from the query converts *better*, at matched completion, on both models. This is the same direction
as the already-closed `D_iso` contradiction, and it rules out query-adjacency as the explanation.

## C4 — scaffolding (exemplar) retention: NOT IT

| model | beta/SD [95% CI] | verdict |
|---|---|---|
| M2 | −0.086 [−0.103, −0.070] | opposite sign |
| M3 | −0.029 [−0.105, +0.036] | null (r(z,q_slot) = 0.74, near the 0.8 separability limit) |

Retaining more of the 2-shot exemplar frame is associated with *lower* accuracy on M2 and nothing
on M3. Ruled out as an explanation.

## Not run

- **C5 (gather order)** — needs generation; deferred under the CPU/prefill-only constraint.
- **C6 (surface-form integrity)** — no within-stage variation; needs new task construction.
