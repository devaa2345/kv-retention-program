# P3_PREDICTIONS — Paper 3, Stage 0

Written 2026-09-08 05:34:48Z. Committed and hashed **before** any comparison against the held-out cells.

Everything below is a number derived from closed forms in `p3/theory.py` plus measured inputs listed explicitly per section. No quantity that this file predicts was consulted while writing it, with the single declared exception of the calibration cell.

## 0. Constants, measured once, not fitted

Fact geometry is a deterministic function of the Paper 2 instance seeds. It was rebuilt on CPU by `p3/facts.py` and validated by reproducing the `n_ctx` recorded in every Paper 2 grid row **exactly**, for all 400 instances (200 per model). A mismatch would mean the rebuilt instance is not the one that was run; the cache refuses to write in that case.

| symbol | M2 (Qwen2.5-3B-Instruct) | M3 (Llama-3.2-3B-Instruct) | source |
|---|---|---|---|
| `L` context tokens | 2067.76 | 2076.76 | rebuilt == recorded `n_ctx` |
| `c` fact cost, LINE unit | 18.9000 | 12.9075 | rebuilt token spans |
| `c'` fact cost, IDVAL unit | 10.0000 | 4.0000 | rebuilt token spans |
| `n_free` candidates inside the floors | 0.140 | 0.100 | rebuilt |
| `S` (layer, KV-head) slots | 72 | 224 | capture dumps |
| `N` records / `H` queried | 40 / 4 | 40 / 4 | task definition |
| `n_sink` / `n_window` | 8 / 64 | 8 / 64 | harness floors, mandatory for every arm |
| `B` total retained | `C + 72` | `C + 72` | harness budget arithmetic |

## 1. T1 — the causal ceiling and the information share

```
A_causal(C, c, H) = ( n_free + min( floor(C/c), H - n_free ) ) / H
A_floor (C, c, L) = max(0, (n_window + C) - c + 1) / (L - n_sink - c + 1)
A_presc           = 1                       (a single fact always fits)
I(C)              = (A_presc - A_causal) / (A_presc - A_floor)
```

`A_floor` is the exact discrete form; the plan's approximation `(B-c)/L` is tabulated beside it so the difference is visible rather than absorbed. Both are parameter-free.

| model | C | A_causal | A_floor (exact) | A_floor (B-c)/L | I(C) |
|---|---|---|---|---|---|
| M2 | 16 | 0.0350 | 0.0304 | 0.0334 | 0.9953 |
| M2 | 32 | 0.2850 | 0.0382 | 0.0412 | 0.7434 |
| M2 | 64 | 0.7850 | 0.0539 | 0.0566 | 0.2273 |
| M2 | 128 | 1.0000 | 0.0853 | 0.0876 | 0.0000 |
| M2 | 256 | 1.0000 | 0.1480 | 0.1495 | 0.0000 |
| M2 | 512 | 1.0000 | 0.2733 | 0.2733 | 0.0000 |
| M3 | 16 | 0.2750 | 0.0331 | 0.0362 | 0.7498 |
| M3 | 32 | 0.5250 | 0.0409 | 0.0439 | 0.4952 |
| M3 | 64 | 1.0000 | 0.0564 | 0.0593 | 0.0000 |
| M3 | 128 | 1.0000 | 0.0876 | 0.0901 | 0.0000 |
| M3 | 256 | 1.0000 | 0.1498 | 0.1517 | 0.0000 |
| M3 | 512 | 1.0000 | 0.2743 | 0.2750 | 0.0000 |

**Prediction T1.** `oracle_causal` accuracy tracks `A_causal` and `floor_pos` accuracy tracks `A_floor`, at every cell, with no fitted parameter. T1 is the boundary condition, not evidence: it is the easy case and is reported as such.

## 2. Calibration — declared before scoring

**Calibration cell: `M2`, `C = 128`, one cell per arm.** Mid-ladder, away from both budget extremes, on the model whose fact cost (~18.9 tokens) is the one the plan's `rho` discussion is written around. Every other cell — all other budgets on M2, and **every cell on M3** — is held out. M3 predictions are therefore a cross-model extrapolation, which is the harder and more informative test.

Each parametric form gets **exactly one** free parameter, fixed by inverting that form on the calibration cell:

```
c_eff*  = ln(q_measured) / ln(p_g_measured)          on the calibration cell only
F1  c_eff = c                       -- no free parameter at all
F2  c_eff = 1 + (c-1)(1-rho)        -- rho = 1 - (c_eff* - 1)/(c - 1)
F3  c_eff = c^(1-rho)               -- rho = 1 - ln(c_eff*)/ln(c)
F4  q(c)  = q(c')^(c/c')            -- no free parameter; c' is the IDVAL unit,
                                       measured in the SAME cell being predicted
```

`rho_BB` below is an independent estimate of the same quantity by moment-matching a BetaBinomial to (p_g, q); `rho_touched` estimates it from the all-40-record touched/complete pair, which never touches the prediction target. Their agreement is test 1.1's stability question and is not used to make any prediction here.

| arm | p_g (cal) | q (cal) | c_eff* | rho F2 | rho F3 | rho_BB | rho_touched |
|---|---|---|---|---|---|---|---|
| snapkv | 0.1094 | 0.0620 | 1.2566 | 0.9857 | 0.9223 | 0.8398 | 0.7779 |
| expected_attn | 0.1020 | 0.0351 | 1.4672 | 0.9739 | 0.8696 | 0.7306 | 0.5772 |
| keydiff | 0.0754 | 0.0353 | 1.2936 | 0.9836 | 0.9124 | 0.8011 | 0.6565 |
| adakv_snapkv | 0.1106 | 0.0617 | 1.2646 | 0.9852 | 0.9201 | 0.8359 | 0.7736 |

**A named substitution.** The plan defines `rho` as the within-fact *score* correlation. Paper 2's captures store keep SETS, not scorer scores, so the score correlation is not recoverable from anything on disk. What is recoverable — and what the theory actually consumes — is the within-fact correlation of the KEEP INDICATOR, the quantity that converts `p_g` into `P(all c kept)`. Every `rho` in this file is that. Recovering the score correlation would need a fresh capture pass and is out of scope for a no-GPU block.

## 3. T2 — predicted per-fact completion at every Paper 2 cell

```
E[complete | pointwise] = p_g ** c_eff        per fact, per (layer, KV-head) slot
E[complete | contiguous] = A_floor(C, c, L)   parameter-free
```

The prediction target is `qcpl_line` — the per-fact completion rate averaged over slots, the LINE unit. `p_g` is the measured per-gold-token keep rate of that same cell.

Cells marked **CAL** are the calibration cells; every other row is out of sample. F4 is calibrated by nothing.

**A degeneracy in the registered comparison, found while deriving and stated here rather than discovered at scoring.** `c_eff` depends only on `c` and `rho`, both of which are constant within a model on Paper 2 data — the fact cost was never varied. Once F2 and F3 are each calibrated to reproduce the same `c_eff*` on the same calibration cell, they produce **identical** predictions at every M2 cell, and differ on M3 only through the change in `c` (18.90 -> 12.91). So Paper 2 can separate F2 from F3 only by cross-model extrapolation, and weakly. Separating them properly is exactly what the Stage 4 `c` sweep is for; Stage 1 cannot do it and will not be reported as if it could.

### M2

| C | arm | p_g | q'(IDVAL) meas | F1 naive | F2 linear | F3 power | F4 nonparam | A_floor | held out |
|---|---|---|---|---|---|---|---|---|---|
| 16 | snapkv | 0.0612 | 0.0482 | 1.17e-23 | 0.029882 | 0.029882 | 0.003249 | 0.0304 | yes |
| 16 | expected_attn | 0.0574 | 0.0350 | 3.49e-24 | 0.015102 | 0.015102 | 0.001771 | 0.0304 | yes |
| 16 | keydiff | 0.0534 | 0.0350 | 8.90e-25 | 0.022588 | 0.022588 | 0.001771 | 0.0304 | yes |
| 16 | adakv_snapkv | 0.0612 | 0.0480 | 1.17e-23 | 0.029220 | 0.029220 | 0.003216 | 0.0304 | yes |
| 32 | snapkv | 0.0670 | 0.0522 | 6.43e-23 | 0.033462 | 0.033462 | 0.003766 | 0.0382 | yes |
| 32 | expected_attn | 0.0631 | 0.0350 | 2.08e-23 | 0.017348 | 0.017348 | 0.001773 | 0.0382 | yes |
| 32 | keydiff | 0.0549 | 0.0350 | 1.50e-24 | 0.023409 | 0.023409 | 0.001771 | 0.0382 | yes |
| 32 | adakv_snapkv | 0.0673 | 0.0519 | 7.01e-23 | 0.032934 | 0.032934 | 0.003726 | 0.0382 | yes |
| 64 | snapkv | 0.0804 | 0.0559 | 2.04e-21 | 0.042116 | 0.042116 | 0.004292 | 0.0539 | yes |
| 64 | expected_attn | 0.0752 | 0.0350 | 5.70e-22 | 0.022432 | 0.022432 | 0.001773 | 0.0539 | yes |
| 64 | keydiff | 0.0594 | 0.0352 | 6.61e-24 | 0.025910 | 0.025910 | 0.001795 | 0.0539 | yes |
| 64 | adakv_snapkv | 0.0810 | 0.0565 | 2.34e-21 | 0.041648 | 0.041648 | 0.004375 | 0.0539 | yes |
| 128 | snapkv | 0.1094 | 0.0634 | 6.91e-19 | 0.062031 | 0.062031 | 0.005436 | 0.0853 | CAL |
| 128 | expected_attn | 0.1020 | 0.0352 | 1.83e-19 | 0.035104 | 0.035104 | 0.001786 | 0.0853 | CAL |
| 128 | keydiff | 0.0754 | 0.0355 | 6.01e-22 | 0.035278 | 0.035278 | 0.001823 | 0.0853 | CAL |
| 128 | adakv_snapkv | 0.1106 | 0.0630 | 8.40e-19 | 0.061736 | 0.061736 | 0.005372 | 0.0853 | CAL |
| 256 | snapkv | 0.1736 | 0.0779 | 4.24e-15 | 0.110780 | 0.110780 | 0.008032 | 0.1480 | yes |
| 256 | expected_attn | 0.1585 | 0.0374 | 7.63e-16 | 0.067062 | 0.067062 | 0.002007 | 0.1480 | yes |
| 256 | keydiff | 0.1298 | 0.0368 | 1.75e-17 | 0.071303 | 0.071303 | 0.001944 | 0.1480 | yes |
| 256 | adakv_snapkv | 0.1765 | 0.0783 | 5.79e-15 | 0.111521 | 0.111521 | 0.008106 | 0.1480 | yes |
| 512 | snapkv | 0.3112 | 0.1101 | 2.63e-10 | 0.230702 | 0.230702 | 0.015453 | 0.2733 | yes |
| 512 | expected_attn | 0.2818 | 0.0458 | 4.01e-11 | 0.155929 | 0.155929 | 0.002944 | 0.2733 | yes |
| 512 | keydiff | 0.2714 | 0.0434 | 1.98e-11 | 0.185088 | 0.185088 | 0.002664 | 0.2733 | yes |
| 512 | adakv_snapkv | 0.3151 | 0.1119 | 3.33e-10 | 0.232174 | 0.232174 | 0.015935 | 0.2733 | yes |

### M3

| C | arm | p_g | q'(IDVAL) meas | F1 naive | F2 linear | F3 power | F4 nonparam | A_floor | held out |
|---|---|---|---|---|---|---|---|---|---|
| 16 | snapkv | 0.0396 | 0.0349 | 7.86e-19 | 0.022802 | 0.019451 | 1.99e-05 | 0.0331 | yes |
| 16 | expected_attn | 0.0358 | 0.0250 | 2.16e-19 | 0.012721 | 0.009579 | 6.79e-06 | 0.0331 | yes |
| 16 | keydiff | 0.0323 | 0.0250 | 5.62e-20 | 0.016494 | 0.013618 | 6.77e-06 | 0.0331 | yes |
| 16 | adakv_snapkv | 0.0390 | 0.0339 | 6.50e-19 | 0.022025 | 0.018687 | 1.80e-05 | 0.0331 | yes |
| 32 | snapkv | 0.0475 | 0.0396 | 8.36e-18 | 0.028255 | 0.024322 | 2.99e-05 | 0.0409 | yes |
| 32 | expected_attn | 0.0418 | 0.0251 | 1.62e-18 | 0.015606 | 0.011908 | 6.82e-06 | 0.0409 | yes |
| 32 | keydiff | 0.0349 | 0.0250 | 1.58e-19 | 0.018150 | 0.015052 | 6.77e-06 | 0.0409 | yes |
| 32 | adakv_snapkv | 0.0463 | 0.0384 | 5.90e-18 | 0.026927 | 0.023045 | 2.70e-05 | 0.0409 | yes |
| 64 | snapkv | 0.0634 | 0.0496 | 3.45e-16 | 0.039595 | 0.034570 | 6.17e-05 | 0.0564 | yes |
| 64 | expected_attn | 0.0560 | 0.0252 | 6.96e-17 | 0.022867 | 0.017888 | 6.92e-06 | 0.0564 | yes |
| 64 | keydiff | 0.0434 | 0.0250 | 2.57e-18 | 0.023502 | 0.019727 | 6.77e-06 | 0.0564 | yes |
| 64 | adakv_snapkv | 0.0608 | 0.0464 | 2.00e-16 | 0.037124 | 0.032215 | 4.97e-05 | 0.0564 | yes |
| 128 | snapkv | 0.0952 | 0.0655 | 6.57e-14 | 0.063739 | 0.056775 | 0.000151 | 0.0876 | yes |
| 128 | expected_attn | 0.0888 | 0.0259 | 2.69e-14 | 0.041868 | 0.034064 | 7.55e-06 | 0.0876 | yes |
| 128 | keydiff | 0.0717 | 0.0250 | 1.67e-15 | 0.042819 | 0.036963 | 6.81e-06 | 0.0876 | yes |
| 128 | adakv_snapkv | 0.0916 | 0.0630 | 4.01e-14 | 0.060173 | 0.053314 | 0.000134 | 0.0876 | yes |
| 256 | snapkv | 0.1630 | 0.0991 | 6.77e-11 | 0.119581 | 0.109370 | 0.000576 | 0.1498 | yes |
| 256 | expected_attn | 0.1642 | 0.0311 | 7.48e-11 | 0.093686 | 0.080321 | 1.38e-05 | 0.1498 | yes |
| 256 | keydiff | 0.1579 | 0.0288 | 4.51e-11 | 0.110126 | 0.099350 | 1.07e-05 | 0.1498 | yes |
| 256 | adakv_snapkv | 0.1601 | 0.1007 | 5.36e-11 | 0.115929 | 0.105655 | 0.000607 | 0.1498 | yes |
| 512 | snapkv | 0.3117 | 0.1838 | 2.92e-07 | 0.255477 | 0.241236 | 0.004225 | 0.2743 | yes |
| 512 | expected_attn | 0.3275 | 0.0780 | 5.52e-07 | 0.231471 | 0.210468 | 0.000266 | 0.2743 | yes |
| 512 | keydiff | 0.3568 | 0.0462 | 1.67e-06 | 0.291728 | 0.275425 | 4.92e-05 | 0.2743 | yes |
| 512 | adakv_snapkv | 0.3057 | 0.2007 | 2.27e-07 | 0.248148 | 0.233692 | 0.005620 | 0.2743 | yes |

## 4. Registered scoring rule for T2 (fixed here, applied in Stage 1)

- A held-out cell counts as **predicted** if `|log10(pred) - log10(obs)| <= 0.3` (a factor of 2). Observed completion is floored at `1/(200*4) = 0.001250` so that exact-zero cells are scorable rather than dropped; predictions are floored the same way.
- A secondary absolute band `|pred - obs| <= 0.05` is reported alongside, because a log band is generous where both numbers are tiny.
- **The winner is the form with the lowest mean |log10 error| over the held-out cells.** Ties broken by hit rate inside the tolerance. The winner is reported whichever it is, including F1 (the naive bound) or F4 (the non-parametric baseline) — in which case the interpolation is dropped and the theory is stated more weakly, per plan section 1.4.
- Held-out set = all 60 (model, C, arm) method cells minus the 4 calibration cells = 56.
- Test 1.2 passes if at least one form predicts >= 70% of held-out cells within the log tolerance.

## 5. T2.3 — the crossover budget B*

```
B* solves   p_g(B) ** c_eff  =  A_floor(C = B - 72)
```

`p_g(B)` is the measured per-gold-token keep rate interpolated log-linearly in `B` over the measured ladder; the fit quality is reported so a bad fit cannot hide inside a confident-looking `B*`. `B*` is scanned over integer `B` up to `L`, above which there is no compression and the question is empty; `inf` means the pointwise branch never overtakes the contiguous one at any compressing budget, which is itself a prediction. Every `B*` beyond the ladder top (B = 584) is an extrapolation past the measured range and is quoted as such, not as an observation.

Two comparators are registered, because the plan's inequality is ambiguous about which completeness scale a head-wise method should be compared on, and choosing after the fact would be the exact vice this ordering exists to prevent:

- **(a) per-slot, PRIMARY** — the literal reading. `p_g ** c_eff` is compared directly against `A_floor`. `floor_pos` has one global keep-set, so its per-slot and union rates coincide and the comparison is well posed.
- **(b) union-lifted, SECONDARY** — a method spread across `S` slots may make a fact reachable somewhere even when a typical slot lacks it. The per-slot prediction is lifted by `q_any = 1 - (1 - q_mean) ** s_eff`, with `s_eff` read off the SAME single calibration cell and carried unchanged. This adds no second fitted parameter to the theory.

| arm | s_eff (from cal cell) |
|---|---|
| snapkv | 1.5372 |
| expected_attn | 1.1423 |
| keydiff | 1.3185 |
| adakv_snapkv | 1.7847 |

### M2   (ladder tops out at C = 512, i.e. B = 584; L = 2068, above which there is no compression)

| arm | p_g(B) slope | R^2 | B* F1 (a) | B* F2 (a) | B* F3 (a) | B* F1 (b) | B* F2 (b) | B* F3 (b) |
|---|---|---|---|---|---|---|---|---|
| snapkv | +0.8642 | 0.9930 | inf | inf | inf | inf | 73 | 73 |
| expected_attn | +0.8429 | 0.9933 | inf | inf | inf | inf | inf | inf |
| keydiff | +0.8634 | 0.9461 | inf | inf | inf | inf | inf | inf |
| adakv_snapkv | +0.8711 | 0.9938 | inf | inf | inf | inf | 73 | 73 |

### M3   (ladder tops out at C = 512, i.e. B = 584; L = 2077, above which there is no compression)

| arm | p_g(B) slope | R^2 | B* F1 (a) | B* F2 (a) | B* F3 (a) | B* F1 (b) | B* F2 (b) | B* F3 (b) |
|---|---|---|---|---|---|---|---|---|
| snapkv | +1.0865 | 0.9999 | 1701 | 826 | 941 | 1674 | 73 | 207 |
| expected_attn | +1.1823 | 0.9992 | 1486 | 838 | 922 | 1478 | 678 | 778 |
| keydiff | +1.3116 | 0.9869 | 1318 | 613 | 674 | 1304 | 377 | 446 |
| adakv_snapkv | +1.0897 | 0.9996 | 1734 | 928 | 1038 | 1696 | 73 | 116 |

**Registered decision rule for test 1.3.** The *predicted crossover set* at a cell is

```
{ arm : predicted completion at that cell  >  A_floor at that cell }
```

evaluated under the form that wins section 4 — declared now, so the winner cannot be chosen to suit — and under comparator (a), with (b) reported beside it. Test 1.3 passes iff, at **M3 C=512**, the predicted set equals the observed winner set `{adakv_snapkv, expected_attn}`: it must contain both and exclude `snapkv` and `keydiff`. **M2 C=512** is scored as a second, independent instance of the same rule — no method beats the floor there, so the predicted set must be **empty**. M3 is the gate; M2 is corroboration or contradiction and is reported either way.

Observed accuracies at those two cells are already public in Paper 2 and restated in the Paper 3 plan, so no claim is made that the *winner set* was unknown while this was written. What was not consulted is every predicted quantity above.

**The predicted sets, written out now so scoring is a lookup and not a judgement:**

| model | C | comparator | F1 | F2 | F3 | F4 |
|---|---|---|---|---|---|---|
| M2 | 512 | (a) per-slot | (empty) | (empty) | (empty) | (empty) |
| M2 | 512 | (b) union-lifted | (empty) | snapkv, adakv_snapkv | snapkv, adakv_snapkv | (empty) |
| M3 | 512 | (a) per-slot | (empty) | keydiff | keydiff | (empty) |
| M3 | 512 | (b) union-lifted | (empty) | snapkv, keydiff, adakv_snapkv | snapkv, keydiff, adakv_snapkv | (empty) |

The observed winner set is `adakv_snapkv, expected_attn` at M3 C=512 and `(empty)` at M2 C=512. Whether any cell of the table above matches is settled in Stage 1, not here.

## 6. T3 — the per-head coherence term

```
E[usable | per-head] = p_g ** c_eff  *  P(coherent in the reading heads)
P(coherent)          = q_mean / q_any        measured on the slot dumps
```

`q_mean` is completion in a typical (layer, KV-head) slot; `q_any` is completion in at least one slot — the union. For `floor_pos` the two coincide by construction (one global keep-set), which is exactly the asymmetry under test.

**Registered prediction P3.3.** Accuracy tracks `q_mean`, not `q_any`. Formally, across all 60 (model, C, arm) cells:

1. `r(acc, q_mean) > r(acc, q_any)`, and
2. at **M3 C=512** the `q_mean` ordering places `snapkv` **below** `floor_pos` while the `q_any` ordering places it **above** — i.e. the per-head measure resolves the inversion and the union measure creates it.

Test 1.4 passes iff both hold. Failing (2) while passing (1) is reported as a partial result and does **not** satisfy the Stage 1 gate.

**Registered prediction P3.2 (test 1.5).** Per-head keep-set divergence

```
D = recs_touched_any / recs_touched_mean      (union spread over slots, >= 1)
```

is monotonically associated with `|Delta_head|` from N9, with the sign such that greater divergence goes with a more negative `Delta_head`. Test 1.5 passes iff the association is monotone and correctly signed on **both** models.

**Stated before scoring, because it bounds what 1.5 can show:** `Delta_head` is non-zero at only two budgets per model — M3 C=16 and C=32, M2 C=32 and C=64 — and is exactly 0 everywhere above, where all candidates fit and the contrast is structurally degenerate. M2's N9 analysis additionally reports `delta_head_measured: false` and flags every cell `structurally_degenerate: true` (2 KV heads), and its C=16 `oracle_causal_perhead` arm was never run (447 of 4800 records absent). So 1.5 has at most **2 usable points per model**. A correlation on n=2 is not evidence, and 1.5 is reported as a sign-and-ordering check only. This is one reason the Stage 1 gate rests on 1.3 and 1.4.

## 7. Gate

Stage 1 proceeds past its gate only if **1.3 or 1.4 passes**. 1.1, 1.2 and 1.5 are consistency checks a merely descriptive framework would also pass, and are reported separately with their own numbers. A failure is not pooled against a pass.

