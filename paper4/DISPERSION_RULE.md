# Diagnostic D — does positional dispersion of the retained set predict conversion?

**Committed before any keep-set is recaptured or any dispersion value is computed.**
Produced on RTX 5070 (Machine N). Not a Stage 3 experiment; a diagnostic on the Stage 2 pilot.

## Why

At c ≈ 40, C = 512 the pilot's U-X arms reach queried-fact completion close to `floor_pos`
(M2 0.225/0.239 vs 0.285; M3 0.245/0.281 vs 0.305) but convert kept facts into correct answers
far worse (acc/q_complete M2 0.40/0.38 vs 0.60; M3 0.12/0.20 vs 0.33). The obvious remaining
difference: `floor_pos` keeps one contiguous block, U-X keeps whole units scattered across the
context. Hypothesis H_D: **the more dispersed the retained set, the lower the conversion of kept
facts into correct answers.**

## Data (deviation from "no GPU", stated)

The pilot rows store summaries, not keep-sets. Keep-sets are recaptured by a prefill-only pass
(no generation) over the c = 40 pilot cells: both models × instances 0–49 × arms `floor_pos`,
`snapkv`, `adakv_snapkv`, `U-snapkv`, `U-adakv_snapkv`. **Validity condition:** every recaptured
row must reproduce the pilot row's p_g, q_complete, q_any, units_complete, units_touched exactly
(|Δ| ≤ 1e-9) — i.e. the keep-sets are those that generated. Any mismatch: stop, not scored.
Correctness per query is the pilot's stored `per_variant` score. c = 1 is not analysed (conversion
≈ 1 there, no gap to explain).

## Measures (per arm, instance; region = [8, n_ctx − 64), mean over (layer, KV-head) slots)

- **D_runs (PRIMARY, instance-level):** number of maximal contiguous runs of retained region tokens.
- **D_iso (SECONDARY, query-level):** isolation of the queried fact = 1 − fraction of the w = 32
  region tokens immediately before and after the fact's span that are retained, averaged over the
  slots in which the fact is complete. Undefined (query dropped from this measure) if the fact is
  complete in no slot.
- Control **q_fact (query-level):** fraction of slots in which the queried fact is complete.

## Test (within-arm variation; per model)

Group = the four method arms pooled (`snapkv`, `adakv_snapkv`, `U-snapkv`, `U-adakv_snapkv`),
unit = query (50 instances × 4 queries × 4 arms). OLS: `correct ~ arm fixed effects + q_fact + z(D)`,
D standardised within group. 95% percentile bootstrap resampling INSTANCES (all their queries and
arms together), 10,000 resamples, CRC32-seeded. `floor_pos` is excluded from the regression because
its D_runs is constant (one block) and its D_iso is ≈ 0 by construction, so it carries no
within-arm variation; it is reported descriptively only.

**Preconditions (checked before scoring; failure = "not scored", reported as such):**
- power: at least 20 correct queries in the group for that model;
- separability: |r(z(D), q_fact)| ≤ 0.8 within the group (Paper 3's D-vs-completion collinearity
  lesson); otherwise "not separable".

**Decision rule, per measure per model:** SUPPORTED if the coefficient on z(D) has its CI entirely
below 0; CONTRADICTED if entirely above 0; otherwise NOT SUPPORTED.
**Mechanism claim ("dispersion predicts conversion") requires D_runs SUPPORTED on both models.**
Anything else is reported as a failed hypothesis in the limitations, alongside Paper 3's
interference test. D_iso is secondary: reported, cannot establish the claim on its own. Two
measures × two models, no multiplicity correction — stated.

## Descriptive, not a test

Per arm: mean D_runs, mean D_iso, q_complete, accuracy, conversion — including `floor_pos`. The
cross-arm association is confounded by scorer and allocation rule and is not evidence for H_D in
either direction (Paper 3's cross-cell vs within-cell sign reversal is the reason).
