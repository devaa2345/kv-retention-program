# PREREG_P3 — Paper 3, Stage 3

**Status: frozen.** This file records Stages 0–2 as they happened, restates T2 in the light of
what Stage 2 refuted, registers a fifth `c_eff` form derived from mechanism, and commits
numeric predictions for Stage 4 before any Stage 4 record exists.

Nothing here was written after seeing a Stage 4 number, because no Stage 4 number exists.
Stage 4 has not been started.

---

## 1. What this freezes, and what it does not

**Frozen:** the five `c_eff` forms and their code (`p3/theory.py`, `p3/theory5.py`); the single
calibration cell; the comparison procedure and its tolerance; the Stage 4 grid; the degenerate-
cell precondition; the harness invariants; and the prediction tables in §7.

**Not frozen, and deliberately so:** `p_g`. Stage 2 showed `p_g` is what carries the cost
effect, and no form here derives it. It is a measured input in Mode A (primary) and an
interpolated one in Mode B (secondary). §5.4 says exactly how each is scored.

**Superseded:** `P3_PREDICTIONS.md` (sha256 `da09b14f…`, commit `bf754d9`) remains valid as the
Stage 0/1 record and is not edited. Its two errata are in `STAGE1_REPORT.md`. Where this file
disagrees with it — specifically on the status of F2 — this file is later and governs.

---

## 2. Stage 1 — retrodiction against Paper 2 (record)

Scored against a hash frozen before any comparison ran; `stage1_retrodiction.py` refuses to run
against a changed file. Gate was 1.3 **or** 1.4.

| # | Test | Verdict |
|---|---|---|
| 1.1 | `p_g`, `ρ` measurable and stable per arm | measurable everywhere; **ρ NOT stable** — drifts monotonically with budget, and its level depends on the estimator (0.61 vs 0.97 arm-averaged) |
| 1.2 | Calibrate on one cell, predict the rest | **PASS** — F2 75.0 % of 44 held-out cells inside a factor of 2; F3 70.5 %; F1 and F4 **0.0 %**, wrong by over a log decade everywhere |
| 1.3 | Crossover `B*` matches observed winners | **FAIL** |
| 1.4 | Per-head vs union completeness | **PASS**, both registered parts |
| 1.5 | `Δ_head` vs per-head divergence | correct sign on **n = 2 points per model**; inconclusive by construction |

**Gate: PASS on 1.4 alone.**

**1.3's failure is recorded in full because it turned out to be diagnostic.** At M3 C=512 the
predicted crossover set was `{keydiff}` — the one arm that is not a winner — excluding both
that are (`adakv_snapkv`, `expected_attn`). It failed for the same reason 1.2's residual
identified: F2's residual correlates **+0.879 with log C** and flips sign at the calibration
budget, in every arm on both models, so the top budget is where over-prediction is worst
(keydiff by ~8×). 1.2 and 1.3 were one failure, not two.

**1.4 is what carried the gate and it is robust.** Per-head completeness contradicts accuracy in
2 of 44 cells (4.5 %) against the union measure's 17 (38.6 %); 2 vs 19 under the IDVAL unit;
per-head wins 8 of 11 within-cell rank contests. It resolves the SnapKV inversion at M3 C=512.
It does **not** explain the ordering of the four arms at that cell (Spearman +0.10 to +0.30 for
every measure), and Stage 1 is not written up as having resolved it.

**Named substitution, carried forward.** Paper 2 stored keep *sets*, not scorer scores, so every
`ρ` in this programme is the within-fact **keep-indicator** correlation, not the score
correlation the plan names. Recovering the latter needs a fresh capture pass and is not
scheduled.

---

## 3. Stage 2 — adversarial probes (record)

13,850 generation records, 3,000 capture records, **zero failures**, ~7 GPU-hours. Gate was 2.4
**and** at least two of 2.1–2.3.

| # | Probe | Breaking condition | Verdict |
|---|---|---|---|
| 2.1 | Cost sweep, `c` ∈ {8,19,40} × C ∈ {64,512} | deficit does not grow with `c` | **PASS** |
| 2.2 | Contiguity control | accuracy does not improve | **PASS** |
| 2.3 | Shuffled-fact control | floor's advantage does not collapse | **VACUOUS** |
| 2.4 | Single-token facts, `c` = 1 | methods still lose to the floor | **PASS** (required) |

**GATE: PASS at 2/3.** 2.3 is VACUOUS, not FAIL: it never reached a cell that could answer its
question in either direction, so it is neither support for nor evidence against the theory.

**2.4**, the sharpest test in the programme, passed with a ~3× margin: at C=512, 4/4 methods beat
the recency floor on both models (M2 floor 0.290 vs methods 0.775–0.825; M3 floor 0.280 vs
0.875–0.945), landing at the `full_cache` anchor. The mechanism was confirmed exactly, not just
the outcome: `|q_complete − p_g| = 0.0000` in every MARK-1 cell, and the 2.8× per-slot retention
ratio accounts for the 2.8–3.3× accuracy ratio.

**2.1** passed on the retention ratio (method ÷ floor), monotone across `c` = 1, 8, 19, 40 on both
models, crossing 1.0 between `c` = 1 and `c` = 19. C=64 was near-degenerate (floor 0.006–0.041)
and was excluded by a named precondition, now generalised in §9.

**2.2** passed with all four paired CIs excluding zero. The effect is understated: the matched arm
takes only whole facts, so it holds **fewer** gold tokens than its target (16.2 vs 26.4 on M2) and
still scores 2.6× higher, assembling 0.94 whole facts against the method's 0.49.

### 3.1 — Probe 2.3's three-layout history, recorded rather than tidied away

| layout | what it did | result |
|---|---|---|
| **1. disjoint halves** | every `/A` in the first half of the body, every `/B` in the second | recency tail held **0 `/A` and 12 `/B` lines in every instance**; `floor_pos` completed nothing *by construction*; every compressed arm exactly 0.0000 |
| **2. rejection-sampled separation** | shuffle-and-retry with a fallback pairing | fallback paired neighbouring slots; **crashed 3/50 M2 instances** on the separation assertion |
| **3. constructive pairing** (`scattered_uniform`) | slot `j` paired with slot `j+N`, orientation randomised per record | fixed the positional privilege (tail now holds 2–10 `/A` lines) but made separation **constant at exactly half the body** (61–63 lines of 127), so no window shorter than half the context can hold both halves — every budget-limited arm 0.000–0.010 at **C=512 and C=1024** |

A greedy variant was also tried and crashed 8/50. Only the oracles ever scored under scattering
(0.080–0.135), because only they can place both halves deliberately.

**Two of these were structural artifacts of my own construction, not properties of the theory,
and both barred the floor from winning by construction — which is precisely the thing the probe
existed to measure rather than assume.**

---

## 4. T2, restated

**Refuted (Stage 2.1b):** *pointwise scoring fragments and the loss is a power law in fact cost* —
`c_eff` rising with `c`.

Measured: the `c_eff` slope in `c` is **+0.000 to +0.014**, smaller than F2's own predicted
(1−ρ) ≈ 0.03–0.06, while the budget shift at fixed `c` is **+0.46 to +0.81**, positive at every
`c` on both models and both methods. Neither F2 nor F3 contains a budget term.

**Restated T2:** *pointwise scoring keeps a smaller **fraction** of a fact's tokens as the fact
gets longer, and the completion penalty follows from that.* The exponent is close to a constant
of the scorer-and-budget and is not what carries the cost effect. The load-bearing quantity moves
from `c_eff` to `p_g`, which fell 0.389 → 0.348 → 0.257 across `c` = 8 → 40 on M2 at C=512.

### 4.1 — What survives unchanged

The **co-retention structure itself**, validated at both boundaries independently of any
parametric form:

- `c_eff` = 1.00 for a contiguous policy in every cell (M2 0.995–1.057, M3 1.002–1.092). A
  recency block holds a record whole or not at all.
- `q_complete` = `p_g` exactly at `c` = 1 (difference 0.0000, every MARK-1 cell, both models).
- Probe 2.2's construction result: same budget, fewer gold tokens, 2.6× accuracy, 0.94 whole
  facts against 0.49.

"Facts must survive together" stands. "Longer facts are exponentially harder to keep whole" does
not.

### 4.2 — `p_g`'s decline is a SEPARATE mechanism, and this is forced

`ρ` describes how one fact's tokens co-vary about that fact's own level; conditioning on the
shared component removes `ρ` from the marginal entirely. So `q(c, ρ, p_g)` is derived and
`p_g(c)` is an input. This is a structural fact about the model, not an empirical finding.

Measured decomposition (`p3/pg_decomposition.py`) confirms a real second effect. A recency floor
performs no content selection, so its decline with `c` is pure geometry; dividing it out isolates
the scorer:

| scorer gold advantage over floor | `c`≈8 | `c`≈19 | `c`≈40 |
|---|---|---|---|
| M2 snapkv C=512 | 1.115 | 1.063 | 0.858 |
| M2 adakv C=512 | 1.110 | 1.066 | 0.871 |
| M3 snapkv C=512 | 1.343 | 0.974 | 0.811 |
| M3 adakv C=512 | 1.274 | 0.999 | 0.879 |

Falls in **all 8** arm × budget × model combinations, by −14 % to −40 %. The scorer's information
advantage over blind position **inverts** as facts lengthen: above the floor at `c`≈8, below it by
`c`≈40. That inversion drives 2.1, and it is a marginal-distribution property no correlation model
reaches.

**Registered consequence:** any Stage 4 claim about the cost axis must be stated in terms of
`p_g(c)` and reported with the geometric part divided out.

---

## 5. The five forms and the comparison procedure

### 5.1 — The roster

```
F1  naive        c_eff = c                              no free parameter
F2  linear       c_eff = 1 + (c-1)(1-rho)               one, calibrated
F3  power        c_eff = c^(1-rho)                      one, calibrated
F4  nonparam     q(c) = q(c_min)^(c/c_min)              no free parameter
F5  threshold    q = INT phi(u) Phi_bar((z-sqrt(rho)u)/sqrt(1-rho))^c du,  z = Phi_bar^-1(p_g)
                                                        one, calibrated
```

All completion predictions are `q`, the per-fact completion rate averaged over (layer, KV-head)
slots, LINE unit — the same target as Stages 1 and 2.

### 5.2 — F5, derived from mechanism

Every admitted method is a **scorer press**: `kvpress` computes `n_kept = int(n(1−ratio))` and
takes `scores.topk(n_kept)`. Keeping the top `B` of `L` is thresholding the score at the
(1−B/L) quantile. So for token `i` of fact `f`, write `S_i = μ_f + ε_i` with `Var(μ_f) = ρ`,
`Var(ε_i) = 1−ρ`. `μ_f` is what the tokens of one fact share; `ε_i` is what differs between them.
Keep iff `S_i > z`. Conditioning on `μ_f` makes the tokens independent, giving the integral above.

**Boundary conditions, forced by construction and verified numerically:**

| corner | requires | verified |
|---|---|---|
| ρ → 1 | `c_eff` = 1 (contiguous policy) | exact, all `c`, all `p_g` |
| `c` = 1 | `c_eff` = 1 | exact, all ρ, all `p_g` |
| ρ → 0 | `c_eff` = `c` (recovers F1 as a corner, not a rival) | exact to 1e-4 |

**Budget limits, derived analytically.** Minimising `xᵀΣ⁻¹x/2` subject to every `x_i ≥ u` for
equicorrelated Σ gives quadratic form `c u²/(1+(c−1)ρ)` while `ln p_g ~ −u²/2`, so
`c_eff → c/(1+(c−1)ρ)` as `p_g → 0`, saturating at `1/ρ` — **near-flat in `c` by construction**.
As `p_g → 1` the drops become a union of rare events and `c_eff → c`. So the model predicts
`c_eff` rising monotonically with budget, which is what Stage 2 measured and what F2 and F3
cannot express.

**The tight closed form is a LIMIT, not an approximation.** Prefactors decay like
`1/(2 ln(1/p_g))`, so convergence is logarithmic: at ρ = 0.5, `c` = 8 the limit is 1.778 while the
integral gives 2.371 at `p_g` = 1e−2 and is still 1.990 at `p_g` = 1e−14. **F5 always evaluates
the integral**; the closed form is quoted only because it makes the flatness intelligible.

### 5.3 — F5's known deficiency, registered now

Calibrated on one cell (M2, `c`≈19, C=64) and scored on the other 22 Stage 2 cells:

| | observed | F5 predicted |
|---|---|---|
| cost slope d(`c_eff`)/d`c` | +0.000 to +0.014 | +0.001 to +0.007 |
| budget shift at fixed `c` | **+0.46 to +0.81** | **+0.05 to +0.21** |

Mean |error| 0.329 on held-out cells, with residuals still structured (≈0 at C=64, −0.38 to −0.69
at C=512).

**F5 is registered as a partial success.** It reproduces the near-flatness in `c` and gets the
budget effect's direction from mechanism — neither of which F2 or F3 can express — and it
under-predicts the budget magnitude by 4–10×. It has **not** been adjusted to close that gap.

**The residual budget dependence is registered as a stated empirical regularity without a
mechanism.** Concretely: `c_eff` rises with budget by roughly +0.5 to +0.8 per decade of `B`
beyond what F5 predicts. No form here derives that. Whether it is heavier-than-Gaussian score
tails, heterogeneous token means within a fact, or budget-dependent `ρ` is **not decided here**,
because deciding it by matching this residual would be fitting.

### 5.4 — Comparison procedure, fixed in advance

- **Mode A (primary).** `p_g` is measured per Stage 4 cell and fed to every form. This scores the
  form and nothing else.
- **Mode B (secondary).** `p_g` is predicted by the §7 interpolation. Scores form + `p_g` model
  jointly.
- **Hit:** `|log10(pred) − log10(obs)| ≤ 0.30` (a factor of 2), both floored at `1/(200·4)` =
  0.00125 so exact-zero cells remain scorable.
- **Winner:** lowest mean |log10 error| over held-out cells, ties broken by hit rate. **Reported
  whichever it is, including F1 or F4** — in which case the interpolations are dropped and the
  theory is stated more weakly.
- **Held-out set:** every method cell on the Stage 4 grid except the single calibration cell.
- A form is *not* credited for a cell excluded by the §9 degeneracy rule.
- Secondary diagnostic, reported always: correlation of each form's log10 residual with `log B`
  and with `c`. Stage 1 passed its hit-rate threshold while carrying an r = +0.879 residual
  trend; a hit rate alone will not be reported as success again.

---

## 6. Stage 4 grid

```
Axis 1  fact cost c   {1, 8, 19, 40}     c=1 via MARK-1; 8/19/40 via LEDGER-C
Axis 2  spans k       {1, 2}             at matched c~19, ADJACENT placement
Axis 3  budget C      {32, 64, 128, 256, 512}
Axis 4  arms          ladder (6) + snapkv, adakv_snapkv, expected_attn, keydiff
Axis 5  models        M2, M3             instances 200, H = 4, N = 40, L ~ 2048
```

Not fully crossed: `c × C` is the load-bearing plane and gets full coverage on both models; `k`
runs at two budgets on M3 only, where the head axis is live.

**`k` uses adjacent placement, not scattered.** Stage 2.3 established that scattered placement
drives every budget-limited arm to zero and takes `full_cache` with it.

**`k` = 4 was dropped at calibration, before any anchor was run.** Each part must carry the
record id and a part label to be identifiable, costing ~6 tokens on M2 and ~4 on M3 before any
field content, so `k` = 4 forces `c` >= 42.5 (M2) / 30.5 (M3) against a target of 19. Holding
`c` fixed while varying `k` is therefore impossible at `k` = 4, and an unmatched axis would
confound span count with fact cost. `k` runs on **both** models rather than M3 only, since with
only two levels the cost is small and the M2 replicate is worth having.

Achieved `c` across the `k` axis: M2 20.98 (k=1) and 18.98 (k=2); M3 18.46 and 18.98. Matched
to within ~2 tokens, which is small against a cost axis spanning 8 to 40, and reported rather
than rounded away.

---

## 7. Committed predictions

### Calibration, one cell for every parametric form: M2, c = 18.92, C = 64

| arm | p_g (cal) | q (cal) | rho F2 | rho F3 | rho F5 |
|---|---|---|---|---|---|
| snapkv | 0.0825 | 0.0456 | 0.9867 | 0.9275 | 0.9696 |
| adakv_snapkv | 0.0822 | 0.0469 | 0.9874 | 0.9310 | 0.9713 |

F1 and F4 have no free parameter. F4 extrapolates the SAME cell's measured completion
at the smallest cost on the plane, `q(c) = q(c_min)^(c/c_min)`, so it is available only
once c_min is measured at Stage 4 and is scored in Mode A only.

### Registered `p_g` interpolation for Mode B

`ln p_g` linear in `ln B` through the two Stage 2 budgets, per (model, arm, c).
C = 128, 256 interpolate; **C = 32 extrapolates below the measured range** and its
predictions are flagged accordingly.

| model | arm | c | slope | p_g C=32* | C=64 | C=128 | C=256 | C=512 |
|---|---|---|---|---|---|---|---|---|
| M2 | snapkv | 8.24 | +1.0323 | 0.0655 | 0.0865 | 0.1287 | 0.2145 | 0.3892 |
| M2 | snapkv | 18.92 | +0.9882 | 0.0633 | 0.0825 | 0.1208 | 0.1970 | 0.3484 |
| M2 | snapkv | 39.51 | +0.9538 | 0.0496 | 0.0641 | 0.0926 | 0.1484 | 0.2572 |
| M2 | adakv_snapkv | 8.24 | +1.0301 | 0.0655 | 0.0863 | 0.1285 | 0.2138 | 0.3874 |
| M2 | adakv_snapkv | 18.92 | +0.9926 | 0.0630 | 0.0822 | 0.1206 | 0.1970 | 0.3493 |
| M2 | adakv_snapkv | 39.51 | +0.9668 | 0.0492 | 0.0638 | 0.0926 | 0.1494 | 0.2610 |
| M3 | snapkv | 8.93 | +1.0215 | 0.0679 | 0.0894 | 0.1325 | 0.2196 | 0.3960 |
| M3 | snapkv | 18.61 | +1.0162 | 0.0465 | 0.0611 | 0.0904 | 0.1494 | 0.2685 |
| M3 | snapkv | 39.28 | +1.1492 | 0.0340 | 0.0463 | 0.0721 | 0.1273 | 0.2470 |
| M3 | adakv_snapkv | 8.93 | +1.0193 | 0.0647 | 0.0851 | 0.1260 | 0.2086 | 0.3757 |
| M3 | adakv_snapkv | 18.61 | +1.0530 | 0.0447 | 0.0594 | 0.0891 | 0.1500 | 0.2753 |
| M3 | adakv_snapkv | 39.28 | +1.1622 | 0.0360 | 0.0492 | 0.0771 | 0.1369 | 0.2677 |

### Predicted per-fact completion `q` on the c x C plane (Mode B)

Prediction target is `qcpl` measured per (layer, KV-head) slot, the LINE unit, exactly
as in Stage 1 and Stage 2. `A_floor` is the parameter-free contiguous branch and is
tabulated as the comparator the crossover test uses.

#### M2   (L = 2071)

| arm | c | C | p_g pred | F1 | F2 | F3 | F5 | A_floor |
|---|---|---|---|---|---|---|---|---|
| snapkv | 8.2 | 32* | 0.0655 | 1.75e-10 | 0.050449 | 0.041779 | 0.038542 | 0.0432 |
| snapkv | 8.2 | 64 | 0.0865 | 1.72e-09 | 0.068341 | 0.057690 | 0.053350 | 0.0587 |
| snapkv | 8.2 | 128 | 0.1287 | 4.58e-08 | 0.105729 | 0.091744 | 0.081526 | 0.0899 |
| snapkv | 8.2 | 256 | 0.2145 | 3.08e-06 | 0.185049 | 0.166349 | 0.141032 | 0.1521 |
| snapkv | 8.2 | 512 | 0.3892 | 0.000418 | 0.355440 | 0.332967 | 0.305644 | 0.2767 |
| snapkv | 18.9 | 32* | 0.0633 | 2.14e-23 | 0.032863 | 0.032863 | 0.027409 | 0.0382 |
| snapkv | 18.9 | 64 | 0.0825 | 3.22e-21 | 0.045625 | 0.045625 | 0.045625 | 0.0538 |
| snapkv | 18.9 | 128 | 0.1208 | 4.35e-18 | 0.073123 | 0.073123 | 0.059668 | 0.0851 |
| snapkv | 18.9 | 256 | 0.1970 | 4.51e-14 | 0.133914 | 0.133914 | 0.116415 | 0.1477 |
| snapkv | 18.9 | 512 | 0.3484 | 2.18e-09 | 0.271177 | 0.271177 | 0.222064 | 0.2729 |
| snapkv | 39.5 | 32* | 0.0496 | 2.87e-52 | 0.010696 | 0.019815 | 0.022414 | 0.0284 |
| snapkv | 39.5 | 64 | 0.0641 | 7.05e-48 | 0.015743 | 0.027673 | 0.024412 | 0.0442 |
| snapkv | 39.5 | 128 | 0.0926 | 1.45e-41 | 0.027443 | 0.044731 | 0.047275 | 0.0758 |
| snapkv | 39.5 | 256 | 0.1484 | 1.80e-33 | 0.055978 | 0.082818 | 0.069738 | 0.1390 |
| snapkv | 39.5 | 512 | 0.2572 | 4.99e-24 | 0.128537 | 0.169858 | 0.140296 | 0.2655 |
| adakv_snapkv | 8.2 | 32* | 0.0655 | 1.74e-10 | 0.051115 | 0.042729 | 0.039583 | 0.0432 |
| adakv_snapkv | 8.2 | 64 | 0.0863 | 1.70e-09 | 0.069099 | 0.058822 | 0.053984 | 0.0587 |
| adakv_snapkv | 8.2 | 128 | 0.1285 | 4.49e-08 | 0.106585 | 0.093133 | 0.083059 | 0.0899 |
| adakv_snapkv | 8.2 | 256 | 0.2138 | 3.00e-06 | 0.185838 | 0.167916 | 0.142258 | 0.1521 |
| adakv_snapkv | 8.2 | 512 | 0.3874 | 0.000402 | 0.355374 | 0.333895 | 0.307592 | 0.2767 |
| adakv_snapkv | 18.9 | 32* | 0.0630 | 1.94e-23 | 0.033829 | 0.033829 | 0.027867 | 0.0382 |
| adakv_snapkv | 18.9 | 64 | 0.0822 | 2.99e-21 | 0.046875 | 0.046875 | 0.046875 | 0.0538 |
| adakv_snapkv | 18.9 | 128 | 0.1206 | 4.17e-18 | 0.074919 | 0.074919 | 0.060233 | 0.0851 |
| adakv_snapkv | 18.9 | 256 | 0.1970 | 4.51e-14 | 0.136714 | 0.136714 | 0.117111 | 0.1477 |
| adakv_snapkv | 18.9 | 512 | 0.3493 | 2.28e-09 | 0.275700 | 0.275700 | 0.224393 | 0.2729 |
| adakv_snapkv | 39.5 | 32* | 0.0492 | 2.11e-52 | 0.011480 | 0.020633 | 0.022754 | 0.0284 |
| adakv_snapkv | 39.5 | 64 | 0.0638 | 5.96e-48 | 0.016867 | 0.028822 | 0.024575 | 0.0442 |
| adakv_snapkv | 39.5 | 128 | 0.0926 | 1.49e-41 | 0.029327 | 0.046603 | 0.049051 | 0.0758 |
| adakv_snapkv | 39.5 | 256 | 0.1494 | 2.40e-33 | 0.059619 | 0.086317 | 0.073852 | 0.1390 |
| adakv_snapkv | 39.5 | 512 | 0.2610 | 8.95e-24 | 0.136365 | 0.177117 | 0.150419 | 0.2655 |

#### M3   (L = 2075)

| arm | c | C | p_g pred | F1 | F2 | F3 | F5 | A_floor |
|---|---|---|---|---|---|---|---|---|
| snapkv | 8.9 | 32* | 0.0679 | 3.77e-11 | 0.051212 | 0.042780 | 0.040000 | 0.0428 |
| snapkv | 8.9 | 64 | 0.0894 | 4.35e-10 | 0.069327 | 0.058984 | 0.054052 | 0.0583 |
| snapkv | 8.9 | 128 | 0.1325 | 1.47e-08 | 0.107148 | 0.093597 | 0.083999 | 0.0894 |
| snapkv | 8.9 | 256 | 0.2196 | 1.33e-06 | 0.187293 | 0.169232 | 0.144095 | 0.1515 |
| snapkv | 8.9 | 512 | 0.3960 | 0.000256 | 0.359218 | 0.337629 | 0.310298 | 0.2759 |
| snapkv | 18.6 | 32* | 0.0465 | 1.58e-25 | 0.022708 | 0.022530 | 0.022766 | 0.0382 |
| snapkv | 18.6 | 64 | 0.0611 | 2.52e-23 | 0.031786 | 0.031558 | 0.026237 | 0.0539 |
| snapkv | 18.6 | 128 | 0.0904 | 3.71e-20 | 0.051546 | 0.051228 | 0.051098 | 0.0851 |
| snapkv | 18.6 | 256 | 0.1494 | 4.30e-16 | 0.095834 | 0.095366 | 0.087461 | 0.1475 |
| snapkv | 18.6 | 512 | 0.2685 | 2.35e-11 | 0.197510 | 0.196843 | 0.180704 | 0.2724 |
| snapkv | 39.3 | 32* | 0.0340 | 2.07e-58 | 0.006108 | 0.012125 | 0.011620 | 0.0284 |
| snapkv | 39.3 | 64 | 0.0463 | 3.76e-53 | 0.009723 | 0.018130 | 0.021202 | 0.0442 |
| snapkv | 39.3 | 128 | 0.0721 | 1.37e-45 | 0.018968 | 0.032329 | 0.027552 | 0.0758 |
| snapkv | 39.3 | 256 | 0.1273 | 6.85e-36 | 0.044698 | 0.067890 | 0.057497 | 0.1388 |
| snapkv | 39.3 | 512 | 0.2470 | 1.40e-24 | 0.121455 | 0.161267 | 0.129162 | 0.2650 |
| adakv_snapkv | 8.9 | 32* | 0.0647 | 2.44e-11 | 0.049281 | 0.041412 | 0.037806 | 0.0428 |
| adakv_snapkv | 8.9 | 64 | 0.0851 | 2.80e-10 | 0.066565 | 0.056916 | 0.053178 | 0.0583 |
| adakv_snapkv | 8.9 | 128 | 0.1260 | 9.36e-09 | 0.102553 | 0.089905 | 0.078425 | 0.0894 |
| adakv_snapkv | 8.9 | 256 | 0.2086 | 8.43e-07 | 0.178532 | 0.161610 | 0.134202 | 0.1515 |
| adakv_snapkv | 8.9 | 512 | 0.3757 | 0.000160 | 0.340791 | 0.320235 | 0.290181 | 0.2759 |
| adakv_snapkv | 18.6 | 32* | 0.0447 | 7.71e-26 | 0.022515 | 0.022345 | 0.022683 | 0.0382 |
| adakv_snapkv | 18.6 | 64 | 0.0594 | 1.48e-23 | 0.031789 | 0.031570 | 0.025846 | 0.0539 |
| adakv_snapkv | 18.6 | 128 | 0.0891 | 2.84e-20 | 0.052194 | 0.051887 | 0.051550 | 0.0851 |
| adakv_snapkv | 18.6 | 256 | 0.1500 | 4.61e-16 | 0.098596 | 0.098141 | 0.091003 | 0.1475 |
| adakv_snapkv | 18.6 | 512 | 0.2753 | 3.75e-11 | 0.207009 | 0.206359 | 0.191745 | 0.2724 |
| adakv_snapkv | 39.3 | 32* | 0.0360 | 2.02e-57 | 0.007298 | 0.013827 | 0.014255 | 0.0284 |
| adakv_snapkv | 39.3 | 64 | 0.0492 | 4.22e-52 | 0.011579 | 0.020661 | 0.022760 | 0.0442 |
| adakv_snapkv | 39.3 | 128 | 0.0771 | 1.87e-44 | 0.022484 | 0.036807 | 0.033384 | 0.0758 |
| adakv_snapkv | 39.3 | 256 | 0.1369 | 1.20e-34 | 0.052669 | 0.077198 | 0.061693 | 0.1388 |
| adakv_snapkv | 39.3 | 512 | 0.2677 | 3.30e-23 | 0.142123 | 0.183113 | 0.160841 | 0.2650 |

### The `k` axis — registered as BOUNDS, not as a point prediction

A fact of `k` spans introduces a span-level component between the fact level and the
token level. Fitting its variance would be a second free parameter, so instead the two
corners are registered and Stage 4 measures where the truth sits between them:

* **UPPER bound on completion -- `k` inert.** Adjacent spans behave as one span: the
  between-span correlation equals the within-span correlation, so `c_eff(k) = c_eff(1)`
  and completion is unchanged by `k` at fixed `c`. This is the most favourable corner.
* **LOWER bound on completion -- spans independent given the fact.** Each of the `k`
  spans must survive on its own: `q_k = q(c/k, rho, p_g)^k`. Least favourable corner.

**Registered prediction P-k: completion is non-increasing in `k` at fixed `c` and `C`,
and lies between these two bounds.** Falling below the LOWER bound would mean `k` costs
something the correlation model cannot express at all; exceeding the UPPER bound would
mean extra spans help, which no form here allows. The bounds are wide, so this is a
weak test by design; it is registered as a containment check, not as a discriminating
prediction.

**`k` = 4 is DROPPED as infeasible, and the arithmetic is why.** Every part must carry
the record id and a part label to be identifiable, costing ~6 tokens on M2 and ~4 on M3
BEFORE any field content. At `k` = 4 that overhead alone forces `c` >= 42.5 on M2 and
>= 30.5 on M3, against a target of 19 -- so `c` cannot be held fixed while `k` varies,
and an unmatched `k` axis would confound span count with fact cost, which is the exact
confound Stage 1 spent a stage disentangling for `c` and `C`. Measured during
calibration, before any anchor was run. Reaching `k` > 2 at matched cost needs
unlabelled continuation lines (`+ field`), which makes a part unidentifiable on its own
-- a different construction, deferred rather than improvised here.

| model | arm | c | C | k | UPPER (k inert) | LOWER (independent spans) |
|---|---|---|---|---|---|---|
| M3 | snapkv | 18.6 | 64 | 1 | 0.026237 | 0.026237 |
| M3 | snapkv | 18.6 | 64 | 2 | 0.026237 | 0.001062 |
| M3 | snapkv | 18.6 | 512 | 1 | 0.180704 | 0.180704 |
| M3 | snapkv | 18.6 | 512 | 2 | 0.180704 | 0.038489 |
| M3 | adakv_snapkv | 18.6 | 64 | 1 | 0.025846 | 0.025846 |
| M3 | adakv_snapkv | 18.6 | 64 | 2 | 0.025846 | 0.001009 |
| M3 | adakv_snapkv | 18.6 | 512 | 1 | 0.191745 | 0.191745 |
| M3 | adakv_snapkv | 18.6 | 512 | 2 | 0.191745 | 0.041149 |

---

## 8. Probe 2.3 is DROPPED, not redesigned

Three reasons, in order of weight:

1. **`MIN_SEP_LINES` is the wrong constraint.** Enforcing a large *minimum* separation is exactly
   what makes "both halves inside one recency window" impossible, which removes the floor
   advantage the probe exists to watch collapse. A working control needs the separation drawn
   from a *distribution* spanning short to long. That is a task redesign.
2. **The anchors are out of band on M2** — 0.542 for `scattered`, **0.400** for
   `scattered_uniform`, against a required [0.55, 0.97]. The task would need redesigning *and*
   re-anchoring before it could be trusted, and re-anchoring is itself GPU time.
3. **The confound is not retention at all.** `full_cache` falls from **0.910 to 0.400** on M2
   (0.925 → 0.680 on M3) purely by scattering the two halves, with **no compression whatsoever**.

### 8.1 — That third observation is a finding, and it is carried forward

A 3B model given the *entire* context, uncompressed, loses more than half its accuracy when one
fact's two halves are moved apart. The information is all present; only its arrangement changed.
**The model is much worse at assembling distributed evidence, independently of any retention
question.**

This connects directly to Paper 2's blocker B8, where the two-hop LEDGER could not be brought
into the competence band on any of three models at any distractor count (anchors 0.100/0.200/0.150
against a required [0.55, 0.97]), and the diagnosis was that "the two-hop structure is the binding
constraint, not distractor count".

Both observations say the same thing from different directions: **at this model scale, composing
evidence across separated context positions is a capability limit that sits above and independent
of KV retention.** A retention policy cannot fix it and a retention benchmark cannot measure
through it.

**Carried into Paper 4 explicitly.** A set-valued retention policy is worth building only for
facts the model could use if they were retained. If distributed facts are unusable at full cache,
retaining them coherently buys nothing, and Paper 4's method should target contiguous or
near-contiguous units. This bounds Paper 4's claim before it is made.

---

## 9. Carried-forward invariants — asserted, not assumed

From Paper 2's defects and Stage 2's:

1. **`floor_pos` has ONE definition.** `kept = first n_sink ∪ last (n_window + C)`, with
   `B = C + n_sink + n_window = C + 72`, `n_sink = 8`, `n_window = 64`. The total is **B**, not C.
   The chat-template prefix **is** compressible and is included in `L`; `L` is measured over the
   templated prefix in the model's own tokenizer. Paper 2's blind reimplementation found this
   defined twice, incompatibly, readings 0.034 apart — 3× the agreement criterion — and it is the
   denominator of every ratio.
2. **Instance seed convention.** CRC32 over the canonical key with `seed` at a sentinel, and with
   `arm`, `B`, `C`, `max_new`, `matched_to` nulled, so every arm at every budget sees the *same*
   instance. `p3/keys3.py`.
3. **Prompt assembly.** Chat template applied to the context, cut at a marker; the query is
   appended **after** compression (agnostic protocol). `runner.templated_parts`.
4. **Decoding parameters.** Greedy (`argmax`), no sampling, no repetition penalty — Qwen2.5 ships
   `repetition_penalty = 1.05`, which on a copy task penalises exactly the answer.
   `max_new_tokens` **scales with the answer**: `len(answer_tokens) + 16`, and MARK-1 uses 8.
5. **Per-record wall-time and an EOS-vs-token-cap flag** are recorded for every generation.
6. **`position_ids` continue from the UNCOMPRESSED context length**, asserted in
   `runner.generate_with`, not assumed.
7. **Every press check generates tokens.** Compression happens in a forward hook after attention,
   so prefill logits are identical at every ratio and `get_seq_length()` does not shrink for
   head-wise presses.
8. **Method arms honour the mandatory sink and window floors.** The sink alone is worth ~0.27.
9. **Realised budget parity asserted per instance from captured keep-sets**, not from the
   requested ratio, with AdaKV's per-layer-total exception handled separately. Stage 2: `B = C+72`
   on all 3,000 captured cells.
10. **Single process on Machine N.** No concurrent runners.
11. **No piped runners.** Stage 2's driver piped through `grep` and the pipeline exit code masked a
    fatal assertion, losing an entire anchor run silently.
12. **Generations are stored on every record**, so scoring can be revised without re-running the
    GPU. Stage 2's first scorer was wrong and this is what made the fix free.

### 9.1 — Degenerate-cell precondition, registered as a RULE

> **A cell whose `floor_pos` accuracy is below 0.05 cannot express a deficit in either direction
> and is EXCLUDED from every verdict.** Exclusion is by this threshold, applied mechanically, not
> by judgement after seeing the result. Excluded cells are still tabulated and still labelled.

Companion rule, from the same failure mode: **a cell in which every compressed arm scores at or
below 0.02 is scored VACUOUS** — neither pass nor fail — and the probe reports why. Paper 2's own
Stage 4 ablation reported PASS at anchor 0.000, where "bindings deleted ≤ chance" is satisfied by
there being nothing to answer; both rules exist to make that impossible.

---

## 10. Kill criteria, set in advance

- **Stage 3 gate (§11):** anchors out of band at any planned `(c, k)` → **report and stop, do not
  tune**. The task family cannot express the axis.
- **Stage 4:** no form beats F4, the calibration-free non-parametric baseline, out of sample →
  drop the interpolations, state the theory as the boundary conditions plus a measured `p_g(c)`.
- **Stage 4:** if F5's residual budget trend is as large as F2's was (|r| with `log B` above 0.5),
  the mechanism is not the operative one and §5.3's regularity is the honest stopping point.
- **Any stage:** the framework explains everything observed and predicts nothing unobserved →
  publish the empirics without it.

---

## 11. Stage 3 gate

Run before freezing is final, and reported in §12:

1. **Competence anchors in band [0.55, 0.97] at every planned `(c, k)`**, on both models,
   `full_cache`, n ≥ 24.
2. **Shortcut probes pass at every `(c, k)`**: BM25 with the query hidden, regex value-extractor,
   and position prior, each ≤ chance + 0.02 where chance = 1/N = 0.025, so the threshold is 0.045.

Both are reported whatever they show. An out-of-band anchor stops the programme at this gate.

### 11.1 — Gate results: **PASS on both requirements**

**Competence anchors** (`full_cache`; `c` cells and MARK-1 from the Stage 2 pass at n=50, the
`k` cells run here at n=24):

| model | cell | achieved `c` | anchor | n | band |
|---|---|---|---|---|---|
| M2 | c=8, k=1 | 8.24 | 0.8438 | 50 | in |
| M2 | c=19, k=1 | 18.92 | 0.9271 | 50 | in |
| M2 | c=40, k=1 | 39.51 | 0.8646 | 50 | in |
| M2 | c=1 (MARK-1) | 1.00 | 0.7917 | 50 | in |
| M2 | c~19, k=1 | 20.98 | 0.8021 | 24 | in |
| M2 | c~19, k=2 | 18.98 | 0.8229 | 24 | in |
| M3 | c=8, k=1 | 8.93 | 0.9375 | 50 | in |
| M3 | c=19, k=1 | 18.61 | 0.9583 | 50 | in |
| M3 | c=40, k=1 | 39.28 | 0.7708 | 50 | in |
| M3 | c=1 (MARK-1) | 1.00 | 0.9167 | 50 | in |
| M3 | c~19, k=1 | 18.46 | 0.8438 | 24 | in |
| M3 | c~19, k=2 | 18.98 | 0.8438 | 24 | in |

**Every planned `(c, k)` is in band on both models.** Nothing was tuned to achieve this; the
one design change made at this stage — dropping `k` = 4 — was forced by cost arithmetic during
calibration, before any anchor was run (§6).

**Shortcut probes**, n = 200 per cell, chance 1/N = 0.0250, threshold 0.0450:

| model | cell | bm25 hidden | regex | position |
|---|---|---|---|---|
| M2 | c=8, k=1 | 0.0000 | 0.0000 | 0.0288 |
| M2 | c=19, k=1 | 0.0163 | 0.0000 | 0.0262 |
| M2 | c=40, k=1 | 0.0312 | 0.0000 | 0.0262 |
| M2 | c=1 (MARK-1) | 0.0000 | 0.0000 | 0.0163 |
| M2 | c~19, k=1 | 0.0175 | 0.0000 | 0.0200 |
| M2 | c~19, k=2 | 0.0288 | 0.0000 | 0.0300 |
| M3 | c=8, k=1 | 0.0225 | 0.0000 | 0.0350 |
| M3 | c=19, k=1 | 0.0225 | 0.0000 | 0.0312 |
| M3 | c=40, k=1 | 0.0213 | 0.0000 | 0.0187 |
| M3 | c=1 (MARK-1) | 0.0000 | 0.0000 | 0.0262 |
| M3 | c~19, k=1 | 0.0200 | 0.0000 | 0.0225 |
| M3 | c~19, k=2 | 0.0200 | 0.0000 | 0.0288 |

**All 36 probe readings are below threshold.** The highest is 0.0350 (M3, c=8, position),
against 0.0450.

**A scoring-convention error in the first probe run, recorded because it inverted the verdict.**
The first version scored "the query-blind rule landed on *any* of the H = 4 queried units",
which puts chance at H/N = 0.10 rather than 1/N = 0.025, and reported FAIL on 5 of 6 M2 cells.
A query-blind rule commits to one unit for all H queries, so it can be right for at most one of
them and its instance score is `hit / H` — the same mean-over-variants convention the task
itself uses, and the one Paper 2's probes used. Corrected, and re-run at n = 200 rather than 24
because at n = 24 the standard error on these readings is comparable to the threshold.

---

## 12. Hash

One procedure, applied to the committed text, verified by round-trip before freezing:
`hash_file.py` takes sha256 over the LF-normalised bytes, asserts the normalisation is
idempotent, re-reads and re-derives. The result is in `PREREG_P3.md.sha256` and is verified
against `git show HEAD:PREREG_P3.md`.
