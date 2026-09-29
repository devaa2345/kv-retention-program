# Post-fix pilot cell (c=40, C=512, n=50, both models) — DECISION_C40 rule says HOLD

Wrapper fixed per A4 (`gather: "score"` on every row); prereg `843c788c…`. Allocation-only by
construction: X and U-X now differ in allocation alone. 500 rows per model, 0 failures.

## Results

**M2 — floor_pos 0.170, oracle_causal 0.695**

| pair | X → U-X | Δ_U [95% CI] | Δ p_g | Δ units_complete |
|---|---|---|---|---|
| snapkv | 0.045 → 0.080 | +0.035 [+0.010, +0.065] | −0.0220 | +7.55 |
| adakv_snapkv | 0.045 → 0.090 | +0.045 [+0.010, +0.080] | −0.0120 | +7.98 |
| expected_attn | 0.000 → 0.035 | +0.035 [+0.015, +0.060] | −0.0204 | +9.39 |
| keydiff | 0.020 → 0.025 | +0.005 [−0.020, +0.030] | −0.0548 | +5.25 |

**M3 — floor_pos 0.105, oracle_causal 0.240**

| pair | X → U-X | Δ_U [95% CI] | Δ p_g | Δ units_complete |
|---|---|---|---|---|
| snapkv | 0.025 → 0.025 | +0.000 [−0.020, +0.020] | −0.0160 | +8.01 |
| adakv_snapkv | 0.025 → 0.035 | +0.010 [−0.020, +0.040] | +0.0016 | +8.66 |
| expected_attn | 0.020 → 0.015 | −0.005 [−0.020, +0.010] | −0.0229 | +8.97 |
| keydiff | 0.020 → 0.045 | +0.025 [+0.005, +0.045] | −0.0567 | +5.13 |

## Verdict under DECISION_C40.md (committed 6cb4b11, before the result)

| pair | M2 | M3 | verdict |
|---|---|---|---|
| snapkv | PASS (+0.035) | fail (+0.000) | **MODELS DISAGREE — hold** |
| adakv_snapkv | PASS (+0.045) | fail (+0.010) | **MODELS DISAGREE — hold** |
| expected_attn | PASS (+0.035) | fail (−0.005) | **MODELS DISAGREE — hold** |
| keydiff | fail (+0.005) | PASS (+0.025) | **MODELS DISAGREE — hold** |

Not averaged, not resolved in favour of either model. Per the rule, the remaining claim-bearing
cells are NOT run and the grid stays paused.

## Three things the numbers say beyond the verdict

1. **The headline reproduces under the fixed wrapper on M2**: AdaKV +0.045 against Stage 2's
   +0.045, SnapKV +0.035 against +0.045. The gather-order confound did not manufacture it.
2. **The fragmentation claim is now CLEANER than before, because Δ p_g went negative.** On M2 every
   U arm retains *fewer* gold tokens than its X arm (−0.012 to −0.055) and scores *higher*. Accuracy
   rises while gold retention falls — the gain can only come from arrangement. The Stage 2 worry
   that U-X wins by quietly retaining more gold is refuted at this cell, on both models
   (M3 also negative except AdaKV at +0.0016).
3. **U-X still loses badly to `floor_pos` everywhere** (M2 0.080–0.090 vs 0.170; M3 0.025–0.045 vs
   0.105), so Stage 2's "removing fragmentation is not sufficient" survives intact, and K3 is
   nowhere near firing.

## The model disagreement is itself a finding

M3's ceiling is far lower at this cell (`oracle_causal` 0.240 vs M2's 0.695) while its floor is
similar (0.105 vs 0.170). There is roughly 0.13 of headroom between floor and oracle on M3 against
0.53 on M2, so a +0.022 effect is a much larger share of the available room on M3 — and none of
M3's attention-based pairs finds it. Whether unit-awareness is model-dependent or headroom-dependent
is not decidable from one cell.

## Summary (Track 1, closed)

At c=40, C=512 (n=50, post-A4 wrapper), the fragmentation effect — arrangement, not retention,
since Δp_g is negative on every pair — was confirmed on M2 with CIs excluding zero on three of four
pairs (+0.035 to +0.045). On M3, SnapKV and ExpectedAttention gave adequately-powered nulls (power
0.93 and 0.95 against an M2-sized effect), AdaKV's null was underpowered (0.68), and KeyDiff instead
showed a positive effect M2 did not (+0.025 [+0.005, +0.045]). A direct cross-model test found the
M2−M3 gap itself not significant (+0.035 [+0.000, +0.070]), so the data do not separate a single
shared effect measured at differing precision from genuine model-dependence; resolving it would need
roughly n=100 per model at this cell.

**ExpectedAttention carries its A3.3 confound caveat here too.** Its Δp_g is *negative* at this cell
(−0.0204), so its gain here is arrangement like the others — but Δp_g was **+0.084 at c=8, C=512**,
where its gain rode on extra gold retention (`DIAGNOSIS_EA.md`). **Its status is cell-dependent, and
that must be stated wherever this pair is cited**: a clean ExpectedAttention result at one cell does
not license quoting the pair without the caveat elsewhere.

**Track 1 is closed.** No further Track 1 GPU work unless the n=100 resolution run is explicitly
requested.
