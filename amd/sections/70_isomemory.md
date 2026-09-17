## iso-memory condition (§6) and the [SPEC-GAP 3] byte-cost sensitivity

Under iso-memory the tiered arms receive more raw positions, funded by the cold tier's
lower byte cost, so total bytes match the permanent arms:
`T = budget / (f + (1-f)·cost)`. With `f = 0.5` and `cost = 0.25` this is `T = 1.6 ×
budget`, reproducing §6's worked example exactly (257 → 411). Permanent arms are unaffected
by the condition, so their iso-token rows are reused rather than recomputed.

| budget | T | arm3 iso-token | arm3 iso-memory | arm4 iso-token | arm4 iso-memory | arm4 diff | 95% CI |
|---|---|---|---|---|---|---|---|
| 154 | 246 | 0.0000 | 0.0000 | 0.0200 | 0.1000 | **+0.0800** | [+0.0633, +0.0967] |
| 257 | 411 | 0.0000 | 0.0044 | 0.1189 | 0.1822 | **+0.0633** | [+0.0411, +0.0867] |
| 514 | 822 | 0.0400 | 0.0367 | 0.2400 | 0.2989 | **+0.0589** | [+0.0233, +0.0944] |

### Interaction under each iso-condition

§8 requires BH correction within each iso-condition separately, which presupposes both
conditions exist. Both are now reported.

| iso-condition | budget | interaction | 95% CI | n differ |
|---|---|---|---|---|
| iso_token | 154 | -0.0067 | [-0.0122, -0.0022] | 6 |
| iso_token | 257 | -0.0456 | [-0.0611, -0.0311] | 47 |
| iso_token | 514 | -0.1156 | [-0.1444, -0.0856] | 112 |
| iso_memory | 154 | +0.0733 | [+0.0556, +0.0900] | 77 |
| iso_memory | 257 | +0.0133 | [-0.0111, +0.0378] | 75 |
| iso_memory | 514 | -0.0533 | [-0.0978, -0.0078] | 122 |

### [SPEC-GAP 3] sensitivity to `quant_byte_cost`

§3 registers 0.25 but notes it ignores per-group scales and zero-points, and that a real
int8 scheme with bf16 scales is nearer 0.28–0.31. Changing the constant changes only the
iso-memory token grant, so this is the condition where it can bite.

| cost | T @ budget 257 | arm3 | arm4 |
|---|---|---|---|
| 0.25 | 411 | 0.0044 | 0.1822 |
| 0.28 | 402 | 0.0033 | 0.1922 |
| 0.31 | 392 | 0.0033 | 0.1811 |

Iso-token results are unaffected by this constant by construction, and Phase 1 and
Phase 2 both run iso-token — so the headline results carry no exposure to it. Every output
row records `quant_byte_cost` and a derived `physical_byte_cost = bits/16` ([GAP-N]) so the
grant is re-derivable.

