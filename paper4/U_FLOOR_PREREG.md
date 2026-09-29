# Item 3 — U-floor prereg, committed before any generation

**Mechanism**: keep whole records only, taken in recency order (most-recent-first, by document
position), until budget C is exhausted; skip a record that doesn't fit in the remaining budget
(same skip convention as every other allocator in this program); filler is never kept. No scorer,
no query — purely structural, deterministic given the document layout. Mandatory floor (sink +
window) always retained on top, same convention as every arm.

## Ceiling, computed from deterministic (no-model) completion first

U-floor's completion is 100% computable without a model (whole-record recency selection is pure
position arithmetic, same as `FloorPosPress` itself) — computed at n=50, c=40, C=512, both models:

| | floor_pos completion | U-floor completion | P1 bound |
|---|---|---|---|
| M2 | 0.2850 | **0.3300** | 0.3500 |
| M3 | 0.2950 | **0.3500** | 0.3500 |

U-floor recovers most (M2) or effectively all (M3, exact bound) of the filler-waste gap between
`floor_pos` and the query-agnostic bound established in Task 1 — expected, since it is a direct
implementation of that bound's own policy.

**Both ceilings, converting completion gain to accuracy** (M2 floor_pos accuracy 0.170, M3
0.105):

| | @ floor's conversion (0.60) | @ U-X's conversion (0.40) |
|---|---|---|
| M2 | 0.3300×0.60=0.1980 (**+0.028** vs floor_pos) | 0.3300×0.40=0.1320 (**−0.038** vs floor_pos) |
| M3 | 0.3500×0.60=0.2100 (**+0.105** vs floor_pos) | 0.3500×0.40=0.1400 (**+0.035** vs floor_pos) |

**Flagging a mismatch with the stated prediction before running anything**: the prediction below
expects "near zero at M3's anchor cell," but M3's own ceiling arithmetic (completion gain 0.055,
hitting the bound exactly) implies a ceiling AT LEAST as large as M2's at either conversion rate,
and larger at the floor-conversion rate. This is stated as an observation, not a revision — the
prediction is tested as given, not adjusted after seeing this.

## Prediction (fixed before generation)

1. Gain over `floor_pos` is positive at M2's anchor cell and near zero at M3's anchor cell.
2. Across all 13 cells from Task 1's bound table, gain correlates positively with the floor's
   own gap to its bound (bigger gap -> bigger U-floor gain).

## Decision rule

**REAL**: gain over `floor_pos` at M2's anchor has a 95% CI excluding zero, AND the gap-to-gain
correlation across cells is positive. **NOT-IT**: CI includes zero at M2's anchor, or the
correlation is null/negative.

**Scope decision, stated explicitly**: this pass runs the two ANCHOR cells only (M2 and M3, c=40,
C=512), sized properly for the ceiling's own implied effect. Prediction 2 (the 13-cell
correlation) needs U-floor GENERATION — not just deterministic completion — at all 13 cells x 2
models, an order of magnitude more GPU time than anything run tonight (13x the anchor-cell cost).
**Not run in this pass** — deferred, with its own cost estimate given below, rather than silently
narrowed to "REAL" on the anchor cell alone. The anchor-cell result is reported as answering
Prediction 1 only; the full REAL/NOT-IT verdict per the rule above needs the correlation test too.

## Sizing

Smaller of the two conversion-rate ceilings at M2 (the binding case): |−0.038| and |+0.028| —
using the smaller magnitude (0.028) as the target effect, conservative per this program's
convention (size for the harder-to-detect case). `n = (z_a/2+z_beta)^2 * sigma_d^2/delta^2`,
`sigma_d=0.099` (established SnapKV c=40 variance): **n ≈ 99, round to 100 per model.**

**Confound check before scoring**: Δp_g, Δunits_complete, same as every other test.

**Projected GPU cost**: 100 instances x 2 arms (floor_pos + U-floor, paired) x 2 models = 400
rows. At the measured ~23.4 rows/min rate: **~17 minutes.**

**Projected cost for the deferred 13-cell correlation test**, for the record: 13 cells x 100
instances x 2 arms x 2 models = 5200 rows -> **~3.7 hours** at the same rate. Not run.
