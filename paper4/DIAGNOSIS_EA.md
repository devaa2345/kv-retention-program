# ExpectedAttention confound — diagnosis (CPU only, grid halted, nothing resumed)

Data: the 15 complete single-session cells on disk (M2 c=1 and c=8; M3 c=1). No GPU used, no code
changed, no cell re-run. Grid stopped after M2 c=8 C=512; M3 c≥8 and all c=19/c=40 unrun.

## 1. Bug check — NEGATIVE

- **Same code path.** `ExpectedAttentionPress` is a plain `ScorerPress`, so `make_unit_aware`
  takes the identical `_U` branch used for SnapKV and KeyDiff (AdaKV is the only special case).
  `common.build_press` constructs all four identically: `build_method` → `make_floor_constrained`
  → wrapper, with the same `compression_ratio`.
- **Decisive evidence already on disk:** the singleton-unit identity passed **40/40**, including
  ExpectedAttention at all five budgets on both models. With units forced to singletons, the
  wrapper reproduces this press's own top-k keep-set exactly — so score call, floor pinning,
  budget solve and gather are all correct *for this press*.
- **Does unit-packing change what "future" means for it?** No. ExpectedAttention estimates future
  query statistics from `hidden_states` and RoPE at future positions, all computed inside
  `score()` during prefill, before any allocation. Allocation consumes the score vector and never
  feeds back into it; the wrapper does not alter hidden states, keys, or positions.
- **One real asymmetry, not a bug:** ExpectedAttention's score is `softmax(·) * ||V||` — strictly
  positive, heavy-tailed, and **not position-smoothed**, whereas SnapKV's is pooled attention
  (`avg_pool1d`, kernel 5), which makes neighbouring tokens similar and within-unit variance low.
  Mean-vs-max aggregation diverges most for a heavy-tailed unsmoothed score. Measuring within-unit
  score dispersion directly needs a prefill (GPU) and was **not** run.

**Conclusion: no implementation bug found. Earlier cells are not retroactively suspect.**

## 2. What actually changed in the keep-sets (instances 0–4, all slots, region tokens)

Mean tokens per slot, by destination:

| cell | pair | overlap | gold X→U | record-line X→U | other X→U | units complete X→U |
|---|---|---|---|---|---|---|
| c=1 C=512 | expected_attn | 0.97 | 3.2 → 3.2 | 81.0 → 86.8 | 431.0 → 425.2 | 11.5 → 16.3 |
| c=1 C=512 | snapkv | 0.99 | 3.1 → 3.1 | 83.1 → 83.2 | 428.9 → 428.8 | 18.1 → 19.3 |
| c=8 C=256 | expected_attn | 0.87 | 5.0 → 5.9 | 51.7 → 54.7 | 204.3 → 201.3 | 1.6 → 8.2 |
| c=8 C=256 | snapkv | 0.93 | 6.6 → 6.7 | 55.6 → 50.7 | 200.4 → 205.3 | 4.0 → 7.8 |
| **c=8 C=512** | **expected_attn** | **0.88** | **9.9 → 12.7** | **104.0 → 131.9** | **408.0 → 380.1** | **2.3 → 17.6** |
| c=8 C=512 | snapkv | 0.94 | 11.5 → 11.4 | 110.5 → 109.4 | 401.5 → 402.6 | 7.0 → 15.0 |

**The two methods differ in kind, not degree:**

- **SnapKV — same content, packed differently.** Gold tokens unchanged (11.5 → 11.4), record-line
  tokens unchanged (110.5 → 109.4), non-record tokens unchanged — yet complete units more than
  double (7.0 → 15.0). This is fragmentation removal, exactly as the wrapper claims.
- **ExpectedAttention — different content *and* packed differently.** At c=8 C=512 the wrapper
  moves ~28 tokens per slot out of non-record text into record lines (408 → 380 and 104 → 132)
  and picks up **+2.8 gold tokens**, alongside the packing gain (2.3 → 17.6 complete units).
  12% of the keep-set is replaced, and the replacement is systematically biased toward records.

## 3. What the effect scales with (admissible cells only, n = 12)

| Δ_U vs | r | slope |
|---|---|---|
| **Δ p_g** | **+0.944** | +2.11 per unit |
| log2 C × c | +0.772 | +0.0017 |
| c | +0.730 | +0.0140 |
| log2 C | +0.378 | +0.0133 |
| oracle_causal accuracy | −0.034 | −0.025 |

ExpectedAttention's U-gain tracks **the gold-retention shift almost perfectly** (r = 0.944), grows
with fact cost, and is unrelated to the cell's ceiling. Where Δ p_g is ≈ 0 or negative (M2 c=1
C≥256; the three degenerate c=8 cells), Δ_U is ≈ 0 or negative too. The largest point, M2 c=8
C=512, has the largest Δ p_g (+0.084) of any cell.

**Reading:** for ExpectedAttention the wrapper's accuracy gain is mostly *not* fragmentation
removal — it is predicted by how much additional gold the re-allocation happens to retain. That is
A3.3's confound, and it is larger at c=8 than at c=1, so it is **not confined to c=1** as A3.3
assumed. The correlation is observational: Δ p_g is not randomised, and both quantities are
downstream of the same allocation change, so this identifies the channel, not a causal share.

## 4. Status

Reported, not resolved. No wrapper change, no prereg change, no resumption. The open decision is
whether c=19/c=40 run as frozen (10 arms) or with the ExpectedAttention/KeyDiff arms treated
differently, before spending the remaining ~8 GPU-hours.
