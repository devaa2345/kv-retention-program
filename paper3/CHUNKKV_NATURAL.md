# ChunkKV on the natural-text dataset (Experiment C cells: C = 256/512, levels 1/3/5, n = 100)

Same instances and scorer v2 as Experiment C; ChunkKV via kvpress 0.5.4 (`p3/chunkkv.py`, see
`CHUNKKV_STATUS.md`): **provisional** (roster gates G1-G3 not run), under-budget by 0-19 tokens.
Full table with 95% paired-bootstrap CIs: `out/natCK_report.md`. `†` = marginal cell, `V` = all
compressed arms <= 0.02 (vacuous).

R = A_method / A_floor_pos at C = 512 (non-vacuous cells):

| model | level | SnapKV | AdaKV | ExpAttn | KeyDiff | **ChunkKV** |
|---|---|---|---|---|---|---|
| M2 | 1 | 0.13 | 0.18 | 0.17 | 0.03 | **0.33** [0.21, 0.47] |
| M2 | 3 | 0.09 | 0.09 | 0.15 | 0.05 | **0.29** [0.17, 0.42] |
| M2 | 5 | 0.10 | 0.10 | 0.08 | 0.04 | **0.31** [0.18, 0.46] |
| M3 † | 1 | 0.32 | 0.60 | 0.08 | 0.15 | **0.61** [0.47, 0.76] |
| M3 | 3 | 0.27 | 0.44 | 0.06 | 0.10 | **0.56** [0.44, 0.69] |
| M3 | 5 | 0.24 | 0.39 | 0.08 | 0.06 | **0.55** [0.42, 0.67] |

- **ChunkKV is the highest of the five methods in every C = 512 cell (point estimates), and still
  below `floor_pos` in every cell** (upper CI <= 0.79). The natural-text conclusion that methods
  lose to the positional baseline therefore holds for ChunkKV as well.
- Its margin over the other methods is not established: at M3 level 1 it ties AdaKV (0.61 vs 0.60);
  elsewhere its interval overlaps AdaKV's (M3 level 5: [0.42, 0.67] vs [0.23, 0.56]). On M2 the gap
  to the best other method is larger (0.33 vs 0.18 at level 1) but intervals still touch.
- Consistent with the LEDGER result (`CHUNKKV_STATUS.md`): the coherence-preserving method moves
  toward the floor as cost grows relative to the fragmentary ones, but does not reach it.
- C = 256 cells are marginal (perfect-reader floor ceiling 0.089) and vacuous on M2.
