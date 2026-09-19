# Floor sensitivity (Paper 2, TODO-T1): both readings of `floor_pos`, both models

Re-measured on Machine N; nothing taken from the missing A6 report.
PREREG_P2_v2 defines `floor_pos` twice: **Reading A** (§3.12(b), as run) keeps sink + last B−8 =
**B = C + 72** tokens; **Reading B** (§4.1 literal) keeps **C** tokens. Only A_floor changes: the
oracle and method arms retain B regardless (verified in `harness/ladder.py`, `methods.py`), so
A_m, A_causal, A_presc are read unchanged from the main grid and only `floor_pos` was re-run
(2,200 records, n = 200 instances per cell, `runs/nvidia/floor_readingC_M{2,3}.jsonl`, zero holes).
G_m and I recomputed under each reading; paired bootstrap, 50,000 draws, propagated through the
ratio; refusal rule applied to each reading separately.
Script: `paper2/bench/floor_sensitivity_analysis.py`; output `paper2/gates/nvidia/floor_sensitivity.json`.

## Verdict

**The sign of the headline survives Reading B wherever Reading B is a valid budget; it does not
survive where Reading B is not.** G_m ≤ 0 in 48 of 50 method-cells under A (as published), 24 of 50
under B, with 24 sign flips. The claim "G_m ≤ 0 almost everywhere" therefore depends on the reading
**at low budgets**, and this must be stated in the paper, not left to a reviewer to find.

| budget range | method-cells | flips A → B | what happens |
|---|---|---|---|
| **C ≤ 64** (M2 C = 32, 64; M3 C = 16, 32, 64) | 23 | **23 of 23** (one M2 C=32 pair is exactly 0.000 under A) | G_m becomes positive under B (+0.00 to +0.13) |
| **C ≥ 128** | 27 | **1 of 27**, and not significant | M3 C = 256 AdaKV: A −0.028 [−0.058, +0.000], B +0.020 [−0.006, +0.045] |

At C ≥ 128, G_m ≤ 0 holds under A in 25 of 27 cells and under B in 24 of 27; the positive cells
under B are M3 C = 512 AdaKV (+0.21) and ExpectedAttention (+0.14), which are also positive under A
(+0.17, +0.11) — the two exceptions the paper already reports — plus the non-significant M3 C = 256
AdaKV cell above.

## Why the low-budget flips are a property of Reading B, not evidence for the methods

Under Reading B the floor arm retains **C** tokens while every other arm retains **B = C + 72**.
That breaks the equal-budget parity every other arm honours, and below C = 72 it cannot even contain
the mandatory 8-sink + 64-window floor that §4.1's own accounting requires of every arm:

| C | Reading A keeps | Reading B keeps | mandatory 72 present under B? |
|---|---|---|---|
| 16 | 88 | 16 | no (56 missing) |
| 32 | 104 | 32 | no (40 missing) |
| 64 | 136 | 64 | no (8 missing) |
| ≥ 128 | B | C | yes |

So at C ≤ 64 the "floor" of Reading B is a *smaller-budget* arm than the methods it is compared
with, and G_m > 0 there measures a 72-token handicap on the floor, not a method advantage. That is
measured (`n_mandatory_missing_in_readingC` per row), not asserted.

## I survives both readings

I(A) vs I(B) at the binding budgets: M2 C = 32 0.732 vs 0.712; M2 C = 64 0.219 vs 0.209; M3 C = 16
0.751 vs 0.720; M3 C = 32 0.489 vs 0.472. The intervals are narrow and the values differ by ≤ 0.03.
I = 0.000 [0.000, 0.000] at every budget ≥ 64 (M3) / ≥ 128 (M2) under both readings, so the
structural transition (I falls to exactly zero once the budget exceeds the payable cost) is
reading-independent.

## Floor level shift

| model | C | A_floor (A) | A_floor (B) |
|---|---|---|---|
| M2 | 512 | 0.2775 | 0.2587 |
| M3 | 512 | 0.2700 | 0.2375 |
| M2 | 256 | 0.1737 | 0.1275 |
| M3 | 256 | 0.1537 | 0.1125 |

The brief's "0.0338 apart" is not a number this analysis reproduces or contradicts at any single
cell reported here; it remains unsourced (TODO-T1 asked for exactly this re-measurement, which is
now on disk).

## What to write in Paper 2

State that the primary uses Reading A (established from `press.py`), that Reading B is not a valid
equal-budget comparison below C = 72 and breaks parity everywhere, and that under Reading B the
G_m ≤ 0 conclusion holds at C ≥ 128 (24 of 27 cells) but the low-budget cells flip. Do not claim
the sign result is reading-invariant.
