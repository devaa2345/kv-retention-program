# Lever 2 — coverage-aware selection, design (no GPU work yet)

Per `PAPER4_STRATEGY_BIG_THREE.md`'s Lever 2: spend budget on distinct facts rather than clusters
of similar ones, targeting the 75% "which query" share rather than the 25% ranking share Papers
1-4 have otherwise tuned. This is a next-session task; nothing below has been run.

## 1. Mechanism — decided from existing evidence, not re-litigated

**MMR-style selection**: `score = relevance − λ × similarity-to-already-selected`, applied unit by
unit during greedy allocation, in place of plain top-k-by-relevance.

**The similarity term is score-space proximity between candidate units' base scorer values, NOT
field-content similarity.** This is decided, not a free parameter to explore later, for a reason
already established in this program: `CONTAMINATION_UNEXPLAINED_CONTEXT.md` §2-3 (commit `230039a`
and its sources) showed field-level similarity is a corpus CONSTANT here — all 20 genuinely-
separated contamination cases have 8/8 "similar-format" fields (`d41ccf6`), because every LEDGER-C
record shares the same schema. A similarity term built on field content would be measuring
something with zero variance in this corpus and could never discriminate anything. What DID
discriminate real contamination pairs was the base scorer's own competition: both the queried and
substituted record sit in the top half of the per-instance ranking (recency bias, rho≈+0.61-0.71,
`6d65f79`), and the substituted record beats the queried one in score 18/20 times, surviving
position-detrending (`b7a86a5`). That is a real, replicated, score-space signal in this exact
failure mode — MMR's redundancy penalty is only useful when candidates are near-tied in relevance,
and score-space proximity is the one measurement already shown to track that near-tie condition
here. Building the similarity term on it, rather than on field content, is therefore evidence-
driven, not a default MMR choice made out of convenience.

## 2. Ceiling — computed from data already on disk, corrected-denominator convention

**Question**: of the contamination cases, how many involve two candidates close in the base
scorer's OWN ranking (not just adjacent in document position)? Computed directly from
`out/_step2_cases.json` (the 20 genuinely-separated contamination cases) and
`out/_step4_instance_data.json` (full per-instance record scores, already captured, no new GPU
work): for each case, `z_gap = (m_score - q_score) / SD(all record scores in that instance)` — a
score-space distance normalized to that instance's own scale, not position distance.

| \|z_gap\| threshold | fraction of the 20 cases qualifying |
|---|---|
| < 0.25 | 11/20 (55%) |
| < 0.5 | 17/20 (85%) |
| < 1.0 | 18/20 (90%) |

Only one case (`adakv_snapkv` inst52, z_gap=+5.90) is a genuine blowout, not a close competition —
the same case flagged earlier for a floor-region score artifact. **Restricted to the pilot's own
scope (SnapKV only)**: all 6 SnapKV contamination cases in this set have z_gap in [0.09, 0.36] —
**6/6 are close-score competitors.** This is the cleanest possible precondition for an MMR-style
fix: every SnapKV contamination case in this population is exactly the kind of near-tied
competition the mechanism targets.

**Accuracy ceiling**, computed the corrected way (query-level denominator — `score()` averages 4
variants per instance, so a fixed query moves accuracy by `1/(n_instances*4)`, not
`1/n_instances`; this convention was corrected mid-session tonight after the E3 fallback ceiling
was first mis-computed the other way, `7c0b98e`). Using the SAME 120-instance population
(`runs/nvidia/p4_spacing_extra_unconstrained_M2.jsonl`, instances 50-169) the 6 cases were drawn
from, computed directly rather than reusing a mismatched population's accuracy figure:

- `U-snapkv` accuracy in this population: **0.0792**. `floor_pos`: **0.1750**.
- 100%-conversion ceiling: `0.0792 + 6/480 = 0.0917` — **still 0.083 short of floor_pos**, even at
  the unreachable maximum.

**Lever 2, at its full ceiling, cannot beat floor_pos either** — same qualitative conclusion as
contamination's original ceiling and E3's fallback ceiling. It is a smaller, harder-to-detect
effect than either of those (implied delta = 0.0125, vs contamination's own ~0.10 and E3's ~0.05),
because so few cases (6, in a 120-instance/480-query sample) show the qualifying pattern for this
specific arm.

## 3. Pilot sizing — for the ceiling's OWN implied effect size, not the generic 0.04

Reusing `POWER_CALC_A3.md`'s formula (`n = (z_a/2+z_beta)^2 * sigma_d^2 / delta^2`, 80% power,
two-sided) with `sigma_d = 0.099` (SnapKV's own c=40, C=512 paired variance, already established)
and `delta = 6/480 = 0.0125` (this ceiling's own implied effect, not tonight's other tests'
generic 0.04 target, which would be a materially easier bar than what this mechanism can actually
deliver):

**n = 493 per arm.** This is roughly 2x the largest n run tonight (E2/E3's n=80) and nearly 25x the
original n=20 screens — a direct, honest consequence of how small this ceiling is, not a sizing
error.

Scope for the pilot, once run: **SnapKV only** (most contamination cases, no confound issues per
`PAPER4_COMPLETE_RECORD.md`), **one λ value** (not swept), **c=40, C=512** (the anchor cell for
everything else this session), **M2 only** (contamination is M2's dominant mode; M3 is 5%
contamination and was never this thread's target). Confound check (Δp_g, Δunits_complete) before
any scoring, same as every other test tonight.

**Projected GPU cost** (using the measured rate from tonight's E2 run, 23.4 rows/min, commit
`48f2b54` — the only directly-measured rate on record, not the earlier wall-clock proxy): 493
instances x 2 arms (floor_pos + MMR-snapkv, paired) = 986 rows -> **~42 minutes**. Not run.

## What this does and does not decide

- Does not pick λ. One value gets tested first per instruction; the design does not commit to
  a specific number here, since no data on file constrains it yet.
- Does not extend to expected_attn/adakv_snapkv/keydiff — those have their own confound histories
  (`DIAGNOSIS_EA.md`) and would need their own ceiling check before inclusion, not assumed to
  transfer from SnapKV's clean 6/6 close-competition result.
- Does not change the standing read that nothing in this program has yet beaten `floor_pos` at
  c=40 — Lever 2's ceiling, like contamination's and E3's, tops out below it. Its value, if REAL,
  would be as one ingredient in Phase 2's combined method, not a standalone winner.
