# AGREEMENT REPORT

Deliverable 6. This blind reimplementation against every reference value stated in
`SPEC_REIMPL_v1.md`. Disagreements greater than 0.05 are traced to a named gap in
`SPEC_QUESTIONS.md`.

Per the spec's own §10: "Agreement on the direction and rough magnitude is what matters.
Exact numeric identity is not expected and would be suspicious."

| # | quantity | reference | measured | delta | agree | traced to |
|---|---|---|---|---|---|---|
| 1 | quantizer error 8-bit keys | 1.17% | 1.18% | +0.0100 | yes | [GAP-Q] resolved |
| 2 | quantizer error 8-bit values | 0.88% | 0.91% | +0.0300 | yes | [GAP-Q] resolved |
| 3 | quantizer error 4-bit keys | 20.06% | 19.98% | -0.0800 | yes | [GAP-Q] resolved |
| 4 | quantizer error 4-bit values | 14.99% | 15.20% | +0.2100 | yes | [GAP-Q] resolved |
| 5 | quantizer error 2-bit keys | 101.11% | 103.35% | +2.2400 | yes | [GAP-Q] resolved |
| 6 | quantizer error 2-bit values | 74.68% | 75.55% | +0.8700 | yes | [GAP-Q] resolved |
| 7 | full_cache_ref | 0.9470 | 0.7989 | -0.1481 | **NO** | [GAP-U] copy fidelity — unresolved |
| 8 | interaction @ budget 257, **4-bit** | 0.0020 | 0.0133 | +0.0113 | yes | [GAP-W] Phase 1 bit-width unspecified |
| 9 | interaction @ budget 257, **8-bit** | 0.0020 | -0.0011 | -0.0031 | yes | [GAP-W] test of the traced cause |
| 10 | interaction @ budget 514, **4-bit** | -0.0020 | -0.1156 | -0.1136 | **NO** | [GAP-W] Phase 1 bit-width unspecified |
| 11 | interaction @ budget 514, **8-bit** | -0.0020 | -0.0056 | -0.0036 | yes | [GAP-W] test of the traced cause |
| 12 | Phase 2 attention | 0.2030 | 0.1686 | -0.0344 | yes | [GAP-U] ceiling / [SPEC-GAP 7] for epiphany |
| 13 | Phase 2 epiphany | 0.2160 | 0.1411 | -0.0749 | **NO** | [GAP-U] ceiling / [SPEC-GAP 7] for epiphany |
| 14 | Phase 2 random | 0.1540 | 0.1133 | -0.0407 | yes | [GAP-U] ceiling / [SPEC-GAP 7] for epiphany |
| 15 | Phase 2 roundrobin | 0.1840 | 0.1167 | -0.0673 | **NO** | [GAP-U] ceiling / [SPEC-GAP 7] for epiphany |
| 16 | Phase 2 oracle | 0.1880 | 0.1378 | -0.0502 | **NO** | [GAP-U] ceiling / [SPEC-GAP 7] for epiphany |
| 17 | FULL−QUANT band @ 257, 4-bit | 0.2780 | 0.1600 | -0.1180 | **NO** | [GAP-C] tier assignment rule |
| 18 | Phase 2 attention — **corrected signals** | 0.2030 | 0.1189 | -0.0841 | **NO** | [GAP-U] ceiling; manipulation now passes |
| 19 | Phase 2 epiphany — **corrected signals** | 0.2160 | 0.1411 | -0.0749 | **NO** | [GAP-U] ceiling; manipulation now passes |
| 20 | Phase 2 random — **corrected signals** | 0.1540 | 0.1133 | -0.0407 | yes | [GAP-U] ceiling; manipulation now passes |
| 21 | Phase 2 roundrobin — **corrected signals** | 0.1840 | 0.1122 | -0.0718 | **NO** | [GAP-U] ceiling; manipulation now passes |
| 22 | Phase 2 oracle — **corrected signals** | 0.1880 | 0.1344 | -0.0536 | **NO** | [GAP-U] ceiling; manipulation now passes |
| 23 | rho P1 (attention) — original | 1.0000 | 1.0000 | -0.0000 | yes | manipulation check, n=100 |
| 24 | rho P1 (attention) — corrected | 1.0000 | 1.0000 | -0.0000 | yes | manipulation check, n=100 |
| 25 | rho P2 (epiphany) — original | 0.0000 | 0.0964 | +0.0964 | yes | manipulation check, n=100 |
| 26 | rho P2 (epiphany) — corrected | 0.0000 | 0.0964 | +0.0964 | yes | manipulation check, n=100 |
| 27 | rho P3 (random) — original | 0.0000 | 0.0958 | +0.0958 | yes | manipulation check, n=100 |
| 28 | rho P3 (random) — corrected | 0.0000 | -0.0560 | -0.0560 | yes | manipulation check, n=100 |
| 29 | rho P4 (roundrobin) — original | 0.0000 | 0.4974 | +0.4974 | **NO** | manipulation check, n=100 |
| 30 | rho P4 (roundrobin) — corrected | 0.0000 | -0.0218 | -0.0218 | yes | manipulation check, n=100 |
| 31 | rho P5 (oracle) — original | 0.0000 | 0.8489 | +0.8489 | **NO** | manipulation check, n=100 |
| 32 | rho P5 (oracle) — corrected | 0.0000 | -0.0781 | -0.0781 | yes | manipulation check, n=100 |
| — | interaction @ 257, iso-**memory** 4-bit | (none stated) | +0.0133 | – | – | §6 condition, no reference given |
| — | interaction @ 514, iso-**memory** 4-bit | (none stated) | -0.0533 | – | – | §6 condition, no reference given |

## Interaction detail

| budget | bit-width | interaction | 95% CI | n differ | n | reference |
|---|---|---|---|---|---|---|
| 154 | 4-bit | -0.0067 | [-0.0122, -0.0022] | 6 | 150 | — |
| 154 | 8-bit | +0.0011 | [+0.0000, +0.0033] | 1 | 150 | — |
| 257 | 4-bit | +0.0133 | [-0.0089, +0.0356] | 69 | 150 | +0.002 |
| 257 | 8-bit | -0.0011 | [-0.0056, +0.0033] | 5 | 150 | +0.002 |
| 514 | 4-bit | -0.1156 | [-0.1444, -0.0856] | 112 | 150 | −0.002 |
| 514 | 8-bit | -0.0056 | [-0.0189, +0.0067] | 32 | 150 | −0.002 |

