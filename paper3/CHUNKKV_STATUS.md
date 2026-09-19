# ChunkKV on the LEDGER grid — status and results (2026-09-19/20)

**Which case applied:** `kvpress.ChunkKVPress` **exists** in the pinned kvpress 0.5.4 (NVIDIA's
implementation of arXiv:2502.00299; SnapKV scorer, chunk length 20, layer-wise chunk selection).
No eviction algorithm was written. `p3/chunkkv.py` is an adapter only.

**Admission status: PROVISIONAL.** Because this is third-party code, the in-house-reimplementation
reproduction gate does not apply. But the roster gates G1 (published-number reproduction), G2
(permutation) and G3 (oracle overlap) were **not run** for ChunkKV, and G1 is not on disk for any
method in this project (Paper 2 TODO-T5). ChunkKV numbers must not enter a comparison table as an
admitted roster member until at least G2/G3 are run and G1 is either run or its absence disclosed.

## Two departures from the harness invariants (both measured, both in the rows)

1. **Budget parity.** ChunkKV keeps whole chunks, so it cannot retain an arbitrary B. It spends the
   largest chunk-multiple ≤ B: **under** budget by 0–19 tokens (mean ≈ 10 of B = 584, ≤ 3.3%).
   Realised keep-count was measured from the compressed cache on the first three instances per cell
   and equals the expected value exactly (asserted). This handicaps ChunkKV slightly; it cannot
   advantage it.
2. **Floors at chunk granularity.** The 8-sink and 64-window floors are pinned as for every other
   method (via the SnapKV scorer), but at chunk granularity whole chunks overlapping them are
   forced, which can spend up to ~19 more tokens on floors than the nominal 72.

## Result (LEDGER, C = 512, n = 100 per cell, same instances/seeds as the plane; paired bootstrap 20,000)

R = A_method / A_floor. The four existing methods reproduce LEDGER Table 1 exactly (e.g. M2 SnapKV
c = 1: 2.87 [2.53, 3.30]).

| model | c | SnapKV | AdaKV | ExpAttn | KeyDiff | **ChunkKV** | ChunkKV G_m |
|---|---|---|---|---|---|---|---|
| M2 | 1 | 2.87 | 2.87 | 2.72 | 2.77 | **1.58** [1.36, 1.83] | 0.29 [0.20, 0.37] |
| M2 | 8 | 1.07 | 1.43 | 0.52 | 0.56 | **0.81** [0.63, 1.03] | −0.05 [−0.11, 0.01] |
| M2 | 19 | 0.77 | 0.77 | 0.32 | 0.47 | **0.60** [0.42, 0.83] | −0.08 [−0.13, −0.03] |
| M2 | 40 | 0.29 | 0.31 | 0.01 | 0.10 | **0.32** [0.21, 0.46] | −0.23 [−0.33, −0.15] |
| M3 | 1 | 2.94 | 2.97 | 2.92 | 3.10 | **1.77** [1.57, 2.03] | 0.34 [0.27, 0.41] |
| M3 | 8 | 0.47 | 1.52 | 0.47 | 0.77 | **0.27** [0.18, 0.37] | −0.29 [−0.37, −0.22] |
| M3 | 19 | 0.17 | 0.47 | 0.24 | 0.35 | **0.41** [0.31, 0.52] | −0.33 [−0.44, −0.24] |
| M3 | 40 | 0.19 | 0.43 | 0.24 | 0.16 | **0.51** [0.31, 0.74] | n/a (headroom 0.120 < 0.15) |

What it shows, and what it does not:

- **ChunkKV follows the same cost reversal.** It beats the floor at c = 1 (1.58 / 1.77) but is the
  *lowest* of the five methods there (others 2.7–3.1). It falls below the floor as cost grows:
  M2 0.81 (interval includes 1) at c = 8, then 0.60 and 0.32; M3 0.27, 0.41, 0.51 at c = 8, 19, 40
  (all intervals below 1; the c = 40 cell is refused for headroom).
- **Its relative position improves with cost.** At c = 40 its point estimate is the highest of the
  five on both models (M2 0.32 vs 0.31; M3 0.51 vs 0.43), which is the direction a
  coherence-preserving method should move. **The intervals overlap those of SnapKV/AdaKV in every
  such cell, so this is a direction, not an established advantage.** At c = 8 and c = 19 it is not
  consistently ahead (M3 c = 8: 0.27 vs AdaKV 1.52).
- The M3 c = 40 cell has A_causal − A_floor = 0.12, below Paper 2's 0.15 threshold: G_m is refused
  and R is reported raw.
