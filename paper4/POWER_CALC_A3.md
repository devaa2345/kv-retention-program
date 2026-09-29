# Power calculation for E2 (A3 floor-hybrid tie), sizing only — nothing run

Phase 0 item 3 of `PAPER4_STRATEGY_BIG_THREE.md`. Formula: two-sided paired-difference test,
alpha=0.05, 80% power: `n = (z_(a/2) + z_beta)^2 * sigma_d^2 / delta^2`, `z_(a/2)=1.960`,
`z_beta=0.842`, target `delta=0.04` (the effect size actually seen in this program). `sigma_d`
(the paired-difference standard deviation) is backed out from each already-reported 95% CI via
`sigma_d = ((hi-lo)/2 / 1.960) * sqrt(n_observed)`.

## sigma_d from two sources

**A3's own comparison** (M2, c=8, C=512, n=20, `hybrid-snapkv` vs `floor_pos`,
`out/probe_tier1.json`, delta=[0.0, -0.100, +0.100]): sigma_d = **0.228**.
-> **n_required = 255.4, round up to 256 per model.**

**c=40 pilot data** (n=50, `PILOT_C40_POSTFIX.md`, commit `68dc570`), per U-arm-vs-X pair —
a different cell (c=40 not c=8), included because it was asked for separately, not because it's
necessarily the right basis for an experiment run at A3's own cell:

| pair | CI | sigma_d | n_required |
|---|---|---|---|
| snapkv | [+0.010, +0.065] | 0.099 | 49 |
| adakv_snapkv | [+0.010, +0.080] | 0.126 | 79 |
| expected_attn | [+0.015, +0.060] | 0.081 | 33 |
| keydiff | [-0.020, +0.030] | 0.090 | 40 |

Range 33-79, snapkv (closest analog to A3's own arm) = 49. **This is 5-8x cheaper than the
c=8-derived estimate** — expected, since Bernoulli/paired variance is largest near p=0.5 (A3's
cell sits at accuracy ~0.20-0.24) and much smaller near the low accuracies of the c=40 cell
(~0.03-0.09). **Which of these applies depends on which cell E2 actually runs at — the strategy
doc does not yet say, and that choice needs to be made before sizing is final, not assumed here.**

## The three ρ candidates (ρ ∈ {0.10, 0.25, 0.50})

No data in this program measures how sigma_d changes with ρ — no pilot has run more than one
reserved-budget fraction. **Assumption stated explicitly: variance is held CONSTANT across ρ**,
since there is no basis to model it moving in either direction (a larger ρ could plausibly reduce
variance by relying more on the already-low-variance floor-heavy region, or increase it by mixing
two allocation regimes more evenly — nothing here decides between those). Under that assumption,
all three ρ values require the SAME n as the single-arm estimate above; this is a placeholder, not
a measurement, and should be revisited once even a small pilot at one ρ is run.

## GPU cost conversion

No directly-instrumented generation rate exists in this project's records. Inferred instead from
commit timestamps around the post-fix pilot run (`32b6635` at 21:43:55 -> `68dc570` at 22:27:36,
43.68 minutes elapsed for 1000 rows = 500/model x 2 models, `PILOT_C40_POSTFIX.md`): **~22.9
rows/min**. This is a session-wall-clock proxy (includes any incidental overhead in that window),
not a clean instrumented throughput figure — flagged as an approximation, not a rate log.

Each paired instance costs 2 rows (hybrid arm + floor_pos), both models needed:

| Design | n/model | Rows | Est. time |
|---|---|---|---|
| A3's existing design, single ρ, sized at A3's own cell (c=8) | 256 | 1024 | **44.7 min** |
| Same, x3 ρ values, constant-variance assumption | 256 each | 3072 | **134.2 min (~2.2 hours)** |
| If E2 instead runs at the c=40 cell, x3 ρ (range across pairs) | 33-79 each | 396-948 | **17.3-41.4 min** |

## Bottom line for deciding E2's cost

- If E2 stays at A3's own cell (c=8, C=512): budget **~2.2 hours** for all three ρ values, one
  model pair of arms each, both models — cheap enough to just run all three rather than
  pre-selecting one.
- If E2 instead targets c=40 (where the program's real effect sizes and the paper's actual claim
  cell live): **17-41 minutes** for all three ρ, dramatically cheaper — but this changes what A3's
  original n=20 result even means, since it was never measured at that cell. This mismatch (A3 run
  at c=8, but Paper 4's live claims are all at c=40) is worth resolving explicitly before E2 is
  scoped, not silently assumed either way.
- Not run. Sizing only, per instruction.
