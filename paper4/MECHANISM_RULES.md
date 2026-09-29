# Decision rules for the mechanism tests (Part 4) — committed BEFORE any test runs

Candidates and evidence: `MECHANISM_SURVEY.md`. Tests are CPU-only, on data already on disk
(Stage 2 `p4_keepsets_c40_*.npz` + pilot rows; Stage 3 `p4_s3_*.jsonl` + `p4_s3_keepsets_*`).
No test below spends GPU. Nothing here is adjusted after seeing a result; a rule that turns out to
be badly specified is reported as such, not rewritten.

## Shared statistical protocol

- Unit of analysis: **instance** (H=4 queries, accuracy `a_i` in {0, .25, .5, .75, 1}).
- Within-arm regression pooled over the four method arms with **arm fixed effects**, controlling
  `q_slot`: `a_i ~ arm + q_slot + z(M)`, where `M` is the candidate's measure, standardised within
  the pooled group.
- CI: 95% percentile bootstrap, **10,000 resamples of instances**, seeded
  `CRC32("p4|mech|<candidate>|<model>|<cells>")`.
- **Preconditions** (failure ⇒ "NOT SCORED", reported as such): ≥ 20 correct queries in the group;
  |r(z(M), q_slot)| ≤ 0.8 (Paper 3's collinearity lesson).
- Cells used: every **admissible** (floor ≥ 0.05) cell available at test time, both models,
  reported per model. Degenerate cells are excluded, as everywhere else.
- `floor_pos` and `oracle_causal` are excluded from the regressions (no within-arm variation in the
  candidate measures for the floor: it is head-agnostic, fixed-position, and exemplar-free) and are
  reported descriptively.

## C2 — denominator artifact

- **Measure.** Conversion `V = A / q` under `q ∈ {q_slot, q_any, q_maj}`; gap
  `G_q = V_q(floor_pos) − V_q(U-X)` per admissible cell, per model, for both U pairs.
- **REAL** ("per-slot completion was the wrong denominator"): under `q_maj`, the mean gap `G_qmaj`
  **falls below 0.05 in absolute value, or reverses sign, on both models**.
- **NOT IT**: `U-X` remains below `floor_pos` by more than 0.05 under **all three** denominators on
  both models.
- **Ambiguous** (report, do not resolve): shrinks on one model only.
- **GPU cost if pursued:** 0 — this changes the reported measure, not the method. A method-building
  consequence would only follow via C1.

## C1 — head/slot disagreement

- **Measures.** (a) `agree_jaccard`: mean pairwise Jaccard of keep-sets across slots;
  (b) `agree_fact`: mean over the H queried facts of the fraction of slots holding that fact whole
  (this is `q_slot` restricted to the fact, so if it collides with the control, (a) is primary).
  **Primary = (a).**
- **REAL** ("head consensus is a real mechanism worth building a method around"): coefficient on
  z(agree_jaccard) is **positive with CI excluding zero on both models**, and the across-arm
  descriptive ordering is consistent (`floor_pos` agreement 1.0 with the highest conversion;
  `U-X` lowest agreement and lowest conversion).
- **NOT IT**: CI includes zero on both models, or the sign is negative on either.
- **Ambiguous**: one model positive-significant, the other null → report, do not build on it.
- **GPU cost if pursued into a method:** a head-consensus wrapper (all heads forced to one unit
  set, budget held at `B` per slot) — one-cell pilot ≈ 0.5 GPU-h; two-cost × two-model
  confirmation ≈ 2 GPU-h; full grid ≈ 5 GPU-h.

## C3 — query-adjacency

- **Measure.** `dist_gold`: mean over slots of the mean distance (in tokens) from each retained
  gold token to the end of the context, normalised by `n_ctx`. Lower = closer to the query.
  **Stated explicitly: this is distance-to-query, NOT the contiguity/isolation measure that
  `DISPERSION_RULE.md` already closed.**
- **REAL**: coefficient on z(dist_gold) is **negative with CI excluding zero on both models**
  (closer content converts better) **and** the effect survives with `p_g` added to the controls
  (so it is not just "the floor's gold is late because the floor keeps late tokens").
- **NOT IT**: CI includes zero on both models, or positive sign, or the effect vanishes when `p_g`
  is controlled.
- **GPU cost if pursued:** a recency-prior unit wrapper — one-cell pilot ≈ 0.5 GPU-h.

## C4 — scaffolding (exemplar) retention

- **Measure.** `exemplar_frac`: mean over slots of the fraction of the two worked-example record
  units' tokens retained.
- **REAL**: coefficient on z(exemplar_frac) **positive with CI excluding zero on both models**.
- **NOT IT**: CI includes zero on both models, or negative sign.
- **Degenerate-measure escape hatch:** if `exemplar_frac` has ~zero variance within the method arms
  (e.g. every arm always drops or always keeps the exemplars), the test is **NOT SCORED** for lack
  of variation, and that fact is reported — it would itself show the exemplars are not a live
  difference between arms.
- **GPU cost if pursued:** pin exemplar units in the wrapper — one-cell pilot ≈ 0.3 GPU-h.

## C5 / C6 — not run now

- **C5 (gather order)** needs generation; **not run** under the CPU/prefill-only constraint.
  If run later: 20 instances × 2 orders × 1 arm × 1 cell ≈ 10 GPU-min. **REAL (i.e. a hygiene
  violation)** iff the paired accuracy difference between two gather orders over the *identical*
  keep-set has a CI excluding zero; expected inert.
- **C6 (surface-form integrity)** cannot be tested on Stage 3 data (no within-stage variation:
  every unit is a whole line). Requires new task construction; ≈ 1–2 GPU-h for a pilot.

## What "survives" means for Part 6

A candidate is reported as **surviving** only if it meets its REAL rule exactly as written above.
Anything else is reported as ruled out, ambiguous, or not scored — including tests that come back
negative, which are reported individually and never dropped.
