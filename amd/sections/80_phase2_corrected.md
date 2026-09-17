## Phase 2 with corrected promotion signals (primary)

The originally-run Phase 2 failed its §7 manipulation check on two of five signals
(P4 ρ = +0.4974, P5 ρ = +0.8489 at n=100). Those arms do not test orthogonal promotion, so
Phase 2 was re-run in full with corrected signals. Both runs are shown; the corrected run
is primary.

| id | signal | original | corrected | Δ | reference | n |
|---|---|---|---|---|---|---|
| P1 | attention | 0.1189 | **0.1189** | +0.0000 | 0.203 | 150 |
| P2 | epiphany | 0.1411 | **0.1411** | +0.0000 | 0.216 | 150 |
| P3 | random | 0.1133 | **0.1133** | +0.0000 | 0.154 | 150 |
| P4 | roundrobin | 0.1167 | **0.1122** | -0.0044 | 0.184 | 150 |
| P5 | oracle | 0.1378 | **0.1344** | -0.0033 | 0.188 | 150 |

### Corrected-signal contrasts vs P3 (random control), BH-corrected

| contrast | diff | 95% CI | n differ | p_BH |
|---|---|---|---|---|
| P1 vs P3 | +0.0056 | [-0.0067, +0.0178] | 29 | 0.6169 |
| P2 vs P3 | +0.0278 | [+0.0156, +0.0411] | 34 | 0.0008 |
| P4 vs P3 | -0.0011 | [-0.0089, +0.0067] | 10 | 1.0000 |
| P5 vs P3 | +0.0211 | [+0.0089, +0.0333] | 33 | 0.0030 |

Retention is held fixed and attention-ranked in every Phase 2 arm, with protection ON
and tiered eviction ON, so these contrasts isolate the promotion decision from retention.

