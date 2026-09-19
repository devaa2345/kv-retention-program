# Natural text vs LEDGER: method / floor_pos accuracy ratio at matched cost

R = A_method / A_floor_pos; R < 1 means the method loses to the positional baseline. Pointwise 95% paired-bootstrap intervals over instances. `X` = floor < 0.05, cell excluded by the frozen admissibility rule; `V` = all compressed arms <= 0.02 (vacuous, reported raw). Natural-text cells marked `†` are marginal (M3 level 1 anchor 0.968 vs 0.97 ceiling; all C=256 cells).


## c ~ 17 (natural L1) vs LEDGER c ~ 19; same absolute C = 512

| model | method | natural R (c) | LEDGER R (c) | natural floor | LEDGER floor |
|---|---|---|---|---|---|
| M2 | snapkv | 0.13 [0.05, 0.22] (16.6) | 0.77 [0.56, 1.05] (18.9) | 0.150 | 0.133 |
| M2 | adakv_snapkv | 0.18 [0.09, 0.29] (16.6) | 0.77 [0.55, 1.05] (18.9) | 0.150 | 0.133 |
| M2 | expected_attn | 0.17 [0.08, 0.27] (16.6) | 0.32 [0.18, 0.49] (18.9) | 0.150 | 0.133 |
| M2 | keydiff | 0.03 [0.00, 0.09] (16.6) | 0.47 [0.30, 0.72] (18.9) | 0.150 | 0.133 |
| M3 | snapkv | 0.32 [0.19, 0.47] (16.6)† | 0.17 [0.09, 0.25] (18.6) | 0.155 | 0.300 |
| M3 | adakv_snapkv | 0.60 [0.37, 0.87] (16.6)† | 0.47 [0.36, 0.60] (18.6) | 0.155 | 0.300 |
| M3 | expected_attn | 0.08 [0.02, 0.15] (16.6)† | 0.24 [0.16, 0.33] (18.6) | 0.155 | 0.300 |
| M3 | keydiff | 0.15 [0.06, 0.24] (16.6)† | 0.35 [0.25, 0.47] (18.6) | 0.155 | 0.300 |

## c ~ 17 (natural L1) vs LEDGER c ~ 19; same fraction of context (natural 512 / LEDGER 256)

| model | method | natural R (c) | LEDGER R (c) | natural floor | LEDGER floor |
|---|---|---|---|---|---|
| M2 | snapkv | 0.13 [0.05, 0.22] (16.6) | 0.43 [0.18, 0.77] (18.9) | 0.150 | 0.070 |
| M2 | adakv_snapkv | 0.18 [0.09, 0.29] (16.6) | 0.61 [0.32, 1.04] (18.9) | 0.150 | 0.070 |
| M2 | expected_attn | 0.17 [0.08, 0.27] (16.6) | 0.25 [0.09, 0.48] (18.9) | 0.150 | 0.070 |
| M2 | keydiff | 0.03 [0.00, 0.09] (16.6) | 0.50 [0.26, 0.86] (18.9) | 0.150 | 0.070 |
| M3 | snapkv | 0.32 [0.19, 0.47] (16.6)† | 0.13 [0.05, 0.21] (18.6) | 0.155 | 0.158 |
| M3 | adakv_snapkv | 0.60 [0.37, 0.87] (16.6)† | 0.30 [0.18, 0.44] (18.6) | 0.155 | 0.158 |
| M3 | expected_attn | 0.08 [0.02, 0.15] (16.6)† | 0.10 [0.03, 0.18] (18.6) | 0.155 | 0.158 |
| M3 | keydiff | 0.15 [0.06, 0.24] (16.6)† | 0.22 [0.12, 0.34] (18.6) | 0.155 | 0.158 |

## c ~ 41-46 (natural L5) vs LEDGER c ~ 40; same absolute C = 512

| model | method | natural R (c) | LEDGER R (c) | natural floor | LEDGER floor |
|---|---|---|---|---|---|
| M2 | snapkv | 0.10 [0.02, 0.20] (46.1) V | 0.29 [0.18, 0.43] (39.5) | 0.120 | 0.180 |
| M2 | adakv_snapkv | 0.10 [0.02, 0.20] (46.1) V | 0.31 [0.19, 0.45] (39.5) | 0.120 | 0.180 |
| M2 | expected_attn | 0.08 [0.02, 0.17] (46.1) V | 0.01 [0.00, 0.05] (39.5) | 0.120 | 0.180 |
| M2 | keydiff | 0.04 [0.00, 0.11] (46.1) V | 0.10 [0.03, 0.18] (39.5) | 0.120 | 0.180 |
| M3 | snapkv | 0.24 [0.13, 0.37] (41.1) | 0.19 [0.05, 0.37] (39.3) | 0.155 | 0.092 |
| M3 | adakv_snapkv | 0.39 [0.23, 0.56] (41.1) | 0.43 [0.23, 0.71] (39.3) | 0.155 | 0.092 |
| M3 | expected_attn | 0.08 [0.02, 0.15] (41.1) | 0.24 [0.11, 0.42] (39.3) | 0.155 | 0.092 |
| M3 | keydiff | 0.06 [0.02, 0.13] (41.1) | 0.16 [0.05, 0.31] (39.3) | 0.155 | 0.092 |

## c ~ 41-46 (natural L5) vs LEDGER c ~ 40; same fraction of context (natural 512 / LEDGER 256)

| model | method | natural R (c) | LEDGER R (c) | natural floor | LEDGER floor |
|---|---|---|---|---|---|
| M2 | snapkv | 0.10 [0.02, 0.20] (46.1) V | X (floor<0.05) (39.5) | 0.120 | 0.035 |
| M2 | adakv_snapkv | 0.10 [0.02, 0.20] (46.1) V | X (floor<0.05) (39.5) | 0.120 | 0.035 |
| M2 | expected_attn | 0.08 [0.02, 0.17] (46.1) V | X (floor<0.05) (39.5) | 0.120 | 0.035 |
| M2 | keydiff | 0.04 [0.00, 0.11] (46.1) V | X (floor<0.05) (39.5) | 0.120 | 0.035 |
| M3 | snapkv | 0.24 [0.13, 0.37] (41.1) | X (floor<0.05) (39.3) | 0.155 | 0.043 |
| M3 | adakv_snapkv | 0.39 [0.23, 0.56] (41.1) | X (floor<0.05) (39.3) | 0.155 | 0.043 |
| M3 | expected_attn | 0.08 [0.02, 0.15] (41.1) | X (floor<0.05) (39.3) | 0.155 | 0.043 |
| M3 | keydiff | 0.06 [0.02, 0.13] (41.1) | X (floor<0.05) (39.3) | 0.155 | 0.043 |