# Stage 1 — Retrodiction against Paper 2

Scored against `P3_PREDICTIONS.md`, frozen at
`sha256 da09b14fde60cd22378e128296c647cfb648ae6de5288207906a17b6444c5b0a` and committed in
`bf754d9` before any of the numbers below were computed. `stage1_retrodiction.py` refuses to
run if that hash does not match, so the ordering is enforced by the code rather than by
intention.

No GPU. Everything runs on Paper 2 captures already on disk.

Each test is reported separately with its own numbers. **A failure is not pooled against a
pass.**

---

## Verdicts

| # | Test | Verdict |
|---|---|---|
| 1.1 | `p_g` and `ρ` measurable and stable per arm | **measurable everywhere; ρ is NOT stable — it drifts monotonically with budget** |
| 1.2 | Fit on one cell, predict the rest, four forms | **PASS** — F2 (linear) wins, 75.0% of held-out cells inside tolerance |
| 1.3 | Crossover `B*` matches observed winners | **FAIL** |
| 1.4 | Per-head vs union completeness | **PASS**, both registered parts, and robust |
| 1.5 | `Δ_head` vs per-head divergence | monotone and correctly signed on both models, on **n = 2 points per model** |

**Gate (1.3 or 1.4): PASS**, carried by 1.4 alone.

---

## Data audit — what Stage 1 needed and what was there

All four required capture classes are present, complete and parse without error.

| needed | file | rows | status |
|---|---|---|---|
| per-instance fragmentation, M2 | `frag_perinstance_M2.jsonl` | 6,000 | 200 instances × 6 budgets × 5 arms, no holes |
| per-instance fragmentation, M3 | `frag_perinstance_M3.jsonl` | 6,000 | same, no holes |
| slot-level completeness, M2 | `slotaware_M2.jsonl` | 6,000 | same, no holes |
| slot-level completeness, M3 | `slotaware_M3.jsonl` | 6,000 | same, no holes |
| M3 agnostic grid | `grid_M3_ledger_agnostic.jsonl` | 13,200 | 6 budgets × 11 arms × 200, complete |
| N9 decomposition, M2 | `n9_M2_ledger.jsonl` + `.analysis.json` | 4,353 | **447/4,800 absent** — see below |
| N9 decomposition, M3 | `n9_M3_ledger.jsonl` + `.analysis.json` | 4,796 | 4 absent |

Also used: `grid_M2_ledger_agnostic.jsonl` (10,000 rows, 5 budgets — M2 has **no C=16 accuracy
cell**, so the five M2 C=16 capture rows have no accuracy to join to and are excluded from
every accuracy-linked test).

**Three gaps, all found before deriving anything, none of them blocking:**

1. **Scorer scores are not on disk.** The plan defines `ρ` as the within-fact *score*
   correlation. Paper 2's captures store keep *sets*. The score correlation is therefore not
   recoverable without a fresh capture pass. Stage 0 substituted the within-fact correlation of
   the **keep indicator** — the quantity that actually converts `p_g` into `P(all c kept)` — and
   said so in the frozen file before scoring. Every `ρ` in this report is that.
2. **N9 on M2 is thin.** `delta_head_measured: false`, every cell flagged
   `structurally_degenerate: true` (M2 has 2 KV heads), and the C=16 `oracle_causal_perhead` and
   `oracle_prescient` arms were never run (400 of the 447 absent records). M2's C=16 `Δ_head`
   does not exist.
3. **`Δ_head` is non-zero at only two budgets per model** — M3 C=16 and C=32, M2 C=32 and C=64 —
   and is exactly 0.0 above, where all candidates fit. Test 1.5 therefore has at most two usable
   points per model. This was stated in the frozen file, in advance, as the reason 1.5 cannot
   carry the gate.

**Fact geometry** (`c`, `L`, floor coverage) is not recorded in the dumps but is a deterministic
function of the instance seeds. It was rebuilt on CPU and validated by reproducing the `n_ctx`
recorded in every Paper 2 grid row **exactly, for all 400 instances**:

| | M2 (Qwen2.5-3B) | M3 (Llama-3.2-3B) |
|---|---|---|
| `L` | 2067.8 | 2076.8 |
| `c` LINE unit | 18.900 | 12.908 |
| `c'` IDVAL unit | 10.000 | 4.000 |

---

## 1.0 — T1, the boundary condition (context, not evidence)

Reported because Stage 0 predicted it and because it is the reference against which "the
framework predicts nothing new" has to be judged. Both forms are **parameter-free**.

```
A_causal = ( n_free + min( floor(C/c), H - n_free ) ) / H
A_floor  = max(0, (n_window + C) - c + 1) / (L - n_sink - c + 1)
```

| model | C | A_causal | oracle_causal | err | A_floor | floor_pos | err |
|---|---|---|---|---|---|---|---|
| M2 | 32 | 0.2850 | 0.3113 | −0.0262 | 0.0382 | 0.0612 | −0.0230 |
| M2 | 64 | 0.7850 | 0.7837 | +0.0013 | 0.0539 | 0.0825 | −0.0286 |
| M2 | 128 | 1.0000 | 0.9850 | +0.0150 | 0.0853 | 0.1075 | −0.0222 |
| M2 | 256 | 1.0000 | 0.9875 | +0.0125 | 0.1480 | 0.1737 | −0.0258 |
| M2 | 512 | 1.0000 | 0.9925 | +0.0075 | 0.2733 | 0.2775 | −0.0042 |
| M3 | 16 | 0.2750 | 0.2800 | −0.0050 | 0.0331 | 0.0413 | −0.0081 |
| M3 | 32 | 0.5250 | 0.5312 | −0.0062 | 0.0409 | 0.0437 | −0.0029 |
| M3 | 64 | 1.0000 | 0.9962 | +0.0038 | 0.0564 | 0.0688 | −0.0123 |
| M3 | 128 | 1.0000 | 0.9975 | +0.0025 | 0.0876 | 0.0925 | −0.0049 |
| M3 | 256 | 1.0000 | 0.9900 | +0.0100 | 0.1498 | 0.1537 | −0.0040 |
| M3 | 512 | 1.0000 | 0.9938 | +0.0062 | 0.2743 | 0.2700 | +0.0043 |

Mean |error|: **0.0087** for `A_causal`, **0.0128** for `A_floor`. `I(C)` predicted 0.7498 at
M3 C=16 against 0.7507 measured.

This is a good fit and it is the easy case: both quantities are counting arguments about which
tokens are present, with no model of what a scorer does. T1 is the boundary condition T2 and T3
must reduce to, and nothing more.

---

## 1.1 — Measure `p_g` and `ρ` per arm per cell

**Both are measurable at all 48 arm-budget cells** (24 per model), on all four estimators, with
no failures.

**`ρ` is not stable within an arm, and the instability has a shape.** The implied
`c_eff* = ln q / ln p_g` rises monotonically with budget on every arm and both models:

| | C=16 | C=32 | C=64 | C=128 | C=256 | C=512 |
|---|---|---|---|---|---|---|
| M2 snapkv `c_eff*` | 1.100 | 1.104 | 1.154 | 1.257 | 1.470 | 1.913 |
| M2 expected_attn | 1.173 | 1.213 | 1.295 | 1.467 | 1.793 | 2.469 |
| M3 snapkv | 1.082 | 1.114 | 1.174 | 1.283 | 1.483 | 1.841 |
| M3 expected_attn | 1.108 | 1.162 | 1.280 | 1.521 | 2.027 | 3.184 |

Expressed as `ρ` under the working form F2 the within-arm spread looks small (sd 0.016–0.064 on
a [0,1] scale) only because F2 compresses a wide `c_eff` range into a narrow `ρ` range; the
underlying quantity moves by a factor of 1.7–2.9 across the ladder.

The four estimators disagree substantially in **level** but agree in **ordering and trend**:

| arm (M2) | ρ_F2 | ρ_F3 | ρ_BB | ρ_touched |
|---|---|---|---|---|
| snapkv | 0.981 ± 0.016 | 0.909 ± 0.067 | 0.833 ± 0.092 | ~0.79 |
| expected_attn | 0.968 ± 0.025 | 0.859 ± 0.089 | 0.734 ± 0.120 | ~0.61 |
| keydiff | 0.974 ± 0.026 | 0.882 ± 0.092 | 0.772 ± 0.129 | ~0.68 |
| adakv_snapkv | 0.981 ± 0.016 | 0.908 ± 0.068 | 0.831 ± 0.092 | ~0.79 |

`ρ_touched` is estimated from the all-40-record (complete, touched) pair and never touches the
queried-record target, so its agreement in ordering with the target-derived estimators is a
genuine consistency check. Its agreement in *level* is poor — 0.61 against 0.97, arm-averaged, for the
same arm — which means "ρ = 0.9" is not a well-defined number independent of the estimator,
and no downstream claim should be phrased as though it were.

**Answer to 1.1 as asked: both quantities are measurable in every arm and every cell. `p_g` is
stable and well behaved. `ρ` is not stable — it drifts monotonically with budget, and its level
depends on which estimator produced it.**

---

## 1.2 — Calibrate on one cell, predict the rest — **PASS**

Calibration: **M2, C=128, one cell per arm.** Each parametric form gets exactly one free
parameter, fixed by inverting that form on that cell. All other method cells are held out,
including **every M3 cell**, so the M3 predictions are cross-model extrapolations.

Scored as registered: a hit is `|log10(pred) − log10(obs)| ≤ 0.30` (a factor of 2), both floored
at 1/800.

| form | mean \|log10 err\| | median | hit rate | abs ≤ 0.05 | bias |
|---|---|---|---|---|---|
| F1 naive (`c_eff = c`) | 1.5196 | 1.4611 | **0.0%** | 72.7% | −1.5196 |
| **F2 linear** (`1+(c−1)(1−ρ)`) | **0.2536** | 0.1918 | **75.0%** | 75.0% | +0.0874 |
| F3 power (`c^(1−ρ)`) | 0.2629 | 0.1933 | 70.5% | 77.3% | +0.0523 |
| F4 non-parametric | 1.2952 | 1.3010 | **0.0%** | 77.3% | −1.2952 |

Per model: F2 scores 0.2542 / 70.0% on M2 and **0.2532 / 79.2% on M3** — the cross-model
extrapolation is not worse than the within-model one.

**Winner: F2_linear.** The registered gate (≥ 70% of held-out cells inside tolerance) is met by
F2 and F3. **PASS.**

Three things this result is not.

**(a) It is not a discrimination between F2 and F3.** `c_eff` depends only on `c` and `ρ`, both
held constant within a model because Paper 2 never varied fact cost. Once each form is
calibrated to the same `c_eff*` on the same cell, F2 and F3 are **identical at every M2 cell**
and differ on M3 only through `c` changing from 18.90 to 12.91. This was stated in the frozen
file before scoring. Separating them needs the Stage 4 `c` sweep; Stage 1 cannot do it.

**(b) The `abs ≤ 0.05` column makes F1 and F4 look competitive and it is an artefact.** Both
predict essentially zero everywhere, and most observed completions are small, so "within 0.05"
is satisfied by predicting nothing. On the log scale, which is the registered criterion, both
are wrong by more than a decade at every single held-out cell.

**(c) F2's residuals are structured, not noise.** This is the substantive caveat and it is
reported because the registered test would not have caught it:

```
r(residual, log10 C) = +0.879   (F2)      n = 44 held-out cells
r(residual, log10 C) = +0.892   (F3)
```

The residual is negative below the calibration budget and positive above it, **in all four arms
on both models**, rising to +0.91 log-decades (a factor of 8) for keydiff at M3 C=512. The one
fitted parameter is absorbing a budget dependence the functional form does not contain — the
data's implied `c_eff` rises from ~1.10 to ~3.18 across the ladder while F2 holds it fixed.

So the honest statement of 1.2 is: **the interpolating forms clear the registered bar and beat
both the naive bound and the non-parametric baseline by more than a decade, and they are
nonetheless the wrong shape in a way the ladder makes visible.**

---

## 1.3 — Crossover (P2.3) — **FAIL**

Scored under F2 (the section-4 winner, declared in advance as the scoring form) and under
comparator (a), the per-slot reading declared primary.

### M3 C=512 — the gate cell. `A_floor` = 0.2743, `floor_pos` accuracy = 0.2700

| arm | accuracy | beats floor? | predicted completion (a) | > A_floor? |
|---|---|---|---|---|
| snapkv | 0.1388 | no | 0.2555 | no |
| expected_attn | 0.3463 | **YES** | 0.2315 | no |
| keydiff | 0.2075 | no | 0.2917 | **YES** |
| adakv_snapkv | 0.3962 | **YES** | 0.2481 | no |

```
observed winner set   : {adakv_snapkv, expected_attn}
predicted set (a)     : {keydiff}                       MATCH = False
predicted set (b)     : {adakv_snapkv, keydiff, snapkv} MATCH = False
```

The prediction is not merely imprecise. It selects the **one arm that is not a winner** and
excludes **both arms that are**. Under the secondary union-lifted comparator it captures one of
the two winners but still admits keydiff and snapkv.

### M2 C=512 — the corroboration cell. `A_floor` = 0.2733, `floor_pos` accuracy = 0.2775

```
observed winner set   : (empty)
predicted set (a)     : (empty)                         MATCH = True
predicted set (b)     : {adakv_snapkv, snapkv}          MATCH = False
```

The M2 match is worth little: "no arm crosses" is what the model predicts almost everywhere, so
getting an empty set right is close to unfalsifiable at this cell.

**Why it fails, and it is not an independent reason.** 1.3 is evaluated at the top budget, which
is exactly where 1.2's residual analysis shows the over-prediction is largest. keydiff's
completion is over-predicted by a factor of ~8 there, which is what pushes it into the predicted
set. The two failures are one failure: the form's fixed `c_eff` is wrong at high budget.

**VERDICT 1.3: FAIL.** Paper 2's unexplained M3 C=512 exception is not converted into Paper 3's
first confirmation. It stays an open item.

---

## 1.4 — Per-head vs union completeness (P3.3) — **PASS**

Both registered parts pass, and the wider stress supports the conclusion more strongly than the
registered predicate did.

### Registered part 1 — accuracy tracks `q_mean`, not `q_any`

| subset | n | r(acc, q_mean) | r(acc, q_any) |
|---|---|---|---|
| all cells, both models (registered) | 55 | **+0.6690** | +0.6222 |
| M2 only | 25 | +0.8970 | +0.8339 |
| M3 only | 30 | +0.5944 | +0.5989 |
| method arms only | 44 | +0.6040 | +0.6478 |

**PASS on the registered statistic, by 0.047.** That is thin, and it inverts on two of three
subsets and on Spearman throughout. Part 1 alone would not be worth much.

### Registered part 2 — does the per-head measure resolve the M3 C=512 inversion?

`floor_pos` at M3 C=512: accuracy 0.2700, `q_mean` = `q_any` = 0.2700 (one global keep-set).

| arm | accuracy | vs floor | `q_any` | vs floor | `q_mean` | vs floor |
|---|---|---|---|---|---|---|
| snapkv | 0.1388 | below | **0.8962** | **ABOVE** | 0.1169 | below |
| expected_attn | 0.3463 | above | 0.1750 | below | 0.0286 | below |
| keydiff | 0.2075 | below | 0.1175 | below | 0.0361 | below |
| adakv_snapkv | 0.3962 | above | **0.9563** | **ABOVE** | 0.1331 | below |

The union measure says SnapKV makes the queried record available 3.3× more often than the floor
while it scores half the floor's accuracy. The per-head measure places it below the floor,
correctly. **PASS.**

### Unregistered stress — the part that makes the verdict worth something

The registered predicate is one-directional: a measure that never says "available" is never
wrong in the false-above direction. Scoring **both** directions across every cell:

| measure | cells where the verdict contradicts accuracy (of 44) |
|---|---|
| per-head `q_mean` | **2 (4.5%)** |
| union `q_any` | **17 (38.6%)** |

Under the IDVAL unit instead of LINE: 2 (4.5%) against 19 (43.2%). The result does not depend on
which completeness unit is used.

Within-cell rank agreement, which strips out the budget trend that dominates a pooled
correlation, ranking all five arms inside each (model, C):

```
q_mean wins 8 cells,  q_any wins 3,  0 ties     (LINE unit)
q_mean wins 8 cells,  q_any wins 1,  2 ties     (IDVAL unit)
```

**Two limitations, stated rather than buried.**

The two cells per-head gets wrong are **both at M3 C=512**, and both in the false-*below*
direction: `expected_attn` and `adakv_snapkv` beat the floor while `q_mean` says they should
not. And at that cell **neither measure ranks the four methods** — Spearman with accuracy is
+0.10 for LINE `q_mean` and +0.30 for the other three.

So: **per-head completeness resolves the specific contradiction the union measure creates at M3
C=512, and is dramatically better than the union measure across the grid as a whole. It does not
explain the ordering of the four arms at M3 C=512.** Stage 1 should not be written up as having
resolved that cell.

A correction to a reading I made mid-analysis and then checked: `expected_attn` at M3 C=512
appears to answer more often (0.3463) than the record is available (LINE union 0.1750), which
would suggest a leak. Under the IDVAL unit — id plus the 6-digit value, which is all the task
needs — its union availability is 0.9200. The apparent anomaly is an artefact of the LINE unit
being too strict, and no leak is implied.

**VERDICT 1.4: PASS.** This is something Paper 2's union accounting could not do: it explains
why a scorer that makes a record available more often can score worse, and it does so on Paper 2's
own captures at no compute cost.

---

## 1.5 — `Δ_head` vs per-head keep-set divergence (P3.2)

Divergence `D = recs_touched_any / recs_touched_mean`, averaged over the four method arms at
each budget. `Δ_head` comes from N9's oracle arms, so the pairing is per budget, not per arm.

### M3 (8 KV heads, `delta_head_measured: true`, 4/4800 records absent)

| C | D | Δ_head | 95% CI | usable |
|---|---|---|---|---|
| 16 | 10.857 | −0.2334 | [−0.2423, −0.2232] | yes |
| 32 | 9.259 | −0.1762 | [−0.2025, −0.1500] | yes |
| 64 | 6.413 | 0.0000 | — | no (all candidates fit) |
| 128 | 3.859 | 0.0000 | — | no |
| 256 | 2.245 | 0.0000 | — | no |
| 512 | 1.442 | 0.0000 | — | no |

### M2 (2 KV heads, `delta_head_measured: false`, all cells flagged degenerate, 447/4800 absent)

| C | D | Δ_head | 95% CI | usable |
|---|---|---|---|---|
| 16 | 5.675 | — | cell absent from N9 | no |
| 32 | 6.015 | −0.1536 | [−0.1732, −0.1340] | flagged degenerate |
| 64 | 5.230 | −0.1237 | [−0.1412, −0.1062] | flagged degenerate |
| 128–512 | 3.57–1.34 | 0.0000 | — | no |

Both models give **two** non-zero points, and on both the ordering is monotone with the
predicted sign: greater divergence goes with a more negative `Δ_head` (M3: D 10.86 → −0.2334,
D 9.26 → −0.1762; M2: D 6.02 → −0.1536, D 5.23 → −0.1237).

**This is a correct sign on n = 2 per model, which is a consistency check and not evidence.**
Two points are always monotone in *some* direction; the content is only that the direction is
the predicted one, twice. M2's contribution is weaker still, since its N9 run reports
`delta_head_measured: false` and flags every cell structurally degenerate. The frozen file said
all of this before scoring, which is why 1.5 was excluded from the gate in advance.

**VERDICT 1.5: consistent with P3.2, powerless to discriminate.**

---

## Errata against the frozen file

Recorded here rather than by editing `P3_PREDICTIONS.md`, which stays hashed as committed.

1. **Held-out cell count.** The frozen file states "all 60 (model, C, arm) method cells minus the
   4 calibration cells = 56". 60 counts `floor_pos`, which has no pointwise prediction. The
   method cells are 6 budgets × 4 arms × 2 models = 48, so the held-out set is **44**. The
   scoring *rule* — every method cell except the calibration cells — is unambiguous and was
   applied as written; only the arithmetic in the prose was wrong.
2. **Correlation type in the 1.4 predicate.** The frozen file writes `r(acc, q_mean) >
   r(acc, q_any)` without naming Pearson or Spearman. Pearson was used, as the natural reading
   and as implemented. Spearman is reported beside it and **goes the other way** (+0.8035 vs
   +0.8077 pooled). The 1.4 verdict is not left resting on that choice: the unregistered
   two-directional and within-cell analyses are what carry it, and both favour `q_mean` decisively.

---

## Gate

```
1.1  measurable and stable ............ measurable everywhere; rho NOT stable (no pass/fail attached)
1.2  out-of-sample c_eff forms ........ PASS   (F2 linear, 75.0%; residuals structured)
1.3  crossover ........................ FAIL
1.4  per-head vs union completeness ... PASS   (both registered parts; robust to unit and to
                                                within-cell ranking)
1.5  Delta_head vs divergence ......... correct sign on n=2 per model; inconclusive by design
```

**GATE (1.3 or 1.4): PASS**, on 1.4 alone.

1.4 is the test that explains something Paper 2 could not, and it is the one that passed. 1.3,
the other gate-eligible test, failed outright and failed in the informative direction: it named
the wrong arm and excluded both right ones, for a reason 1.2's residual structure identifies
precisely.

**What this licenses.** Stage 2's adversarial probes, per the plan. Nothing more. In particular
it does not license the claim that the co-retention framework predicts the M3 C=512 exception —
it does not — nor that `c_eff ≈ 1 + (c−1)(1−ρ)` is the right form, since Stage 1 cannot separate
it from `c^(1−ρ)` and shows both to be systematically wrong in budget.

**What Stage 2 should be told by this.** Probe 2.4 (`c` = 1) remains the sharpest test. The `c`
sweep (2.1) is now doing double duty: it is the only thing that can separate F2 from F3, and it
is the natural place to find out whether the budget-dependence in `c_eff` is really a `c`
dependence in disguise.

Stage 2 has not been started.
