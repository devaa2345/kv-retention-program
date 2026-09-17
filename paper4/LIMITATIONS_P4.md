# Paper 4 — Limitations and failed hypotheses (running record)

All numbers produced on the RTX 5070 (Machine N).

## L1 — Dispersion does not explain the conversion gap (Diagnostic D, NOT SUPPORTED)

Rule committed before recapture: `DISPERSION_RULE.md` (commit `ab0b92f`). Keep-sets recaptured by
prefill and verified to reproduce all 500 c≈40 pilot rows exactly (max |Δ| 0).

**Setting.** At c ≈ 40, C = 512, U-X reaches queried-fact completion near `floor_pos` but converts
worse (acc / q_complete: M2 0.40, 0.38 vs 0.60; M3 0.12, 0.20 vs 0.33). Hypothesis: the floor keeps
one contiguous block, U-X keeps scattered whole units, and dispersion lowers conversion.

**Within-arm test** (method arms pooled, arm fixed effects, control q_fact, per-SD coefficient,
instance bootstrap):

| model | measure | correct / queries | β per SD [95% CI] | r(D, q_fact) | verdict |
|---|---|---|---|---|---|
| M2 | D_runs (primary) | 54 / 800 | −0.099 [−0.253, +0.030] | −0.30 | NOT SUPPORTED |
| M2 | D_iso | 54 / 447 | +0.052 [+0.016, +0.090] | −0.53 | CONTRADICTED |
| M3 | D_runs (primary) | 27 / 800 | +0.013 [−0.272, +0.300] | −0.38 | NOT SUPPORTED |
| M3 | D_iso | 26 / 633 | +0.010 [−0.014, +0.033] | −0.28 | NOT SUPPORTED |

Mechanism claim requires D_runs SUPPORTED on both models: **NOT SUPPORTED.** On M2 the secondary
measure points the wrong way: at fixed completion a more isolated fact is answered MORE often.
Power is low (27–54 correct queries), so "not supported" is weak evidence of absence, not a null.

**The cross-arm pattern also runs against the hypothesis** (descriptive, confounded): X arms are the
MOST dispersed (D_runs 65–78 vs U-X 20–29 vs floor 1) yet convert as well as or better than the
floor (M2 0.67/0.68 vs 0.60; M3 snapkv 0.45 vs 0.33). The conversion deficit is specific to U-X,
not to scattered retention in general.

**Observation this exposes (not tested, not chased).** "Conversion" is accuracy over PER-SLOT
completion. X keeps fragments whose union across heads covers the fact more often than any one
slot does (q_any 0.19–0.72) and converts well per slot-complete fact; U-X makes the fact complete in
~23–28% of slots with union ≈ 0.9–1.0 and converts worse. So the per-slot denominator is not a
neutral measure of "a kept fact": which slots (layers, heads) hold the fact may matter more than how
many. Recorded so the Stage 3 plan does not treat per-slot completion as the usability-relevant
quantity by default.

This joins Paper 3's ruled-out mechanisms: interference from other whole records (not supported)
and completeness as an index of usability (Evidence 1–4, LIMITATIONS_P3.md).

## L2 — Stage 2 statements carried forward

- Missing `PAPER4_CONTEXT_AND_PLAN.md` at build time (STAGE12_RULES.md §0).
- Verify condition (a) amended for M3 (Amendment 1): pre-power-loss captures, session effect.
- Pilot gate: 4 comparisons at c = 40 without multiplicity correction; M3 adakv lower bound +0.005.
