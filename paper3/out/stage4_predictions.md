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

