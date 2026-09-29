# Slot-vs-identity probe — hypothesis and threshold, committed before any generation

Grounded in `out/_step5_result.json` (commit `b7a86a5`): position-detrended residual stickiness
(Check 5) stayed significant (Pearson r=+0.668, p=0.0013; Spearman rho=+0.625, p=0.0032) across
20 contamination cases, but only **9 distinct record-content indices** back those 20 cases, and
all 9 sit at record index i in {26,29,30,31,32,33,34,36,38} (R027,R030-R035,R037,R039) — the last
14 of 40 record slots.

**The confound this probe exists to break.** In `ledger_c.py:build`'s `assemble()`
(`paper3/p3/tasks/ledger_c.py:170-187`), record i's document slot is
`pos = round((i+0.5) * total / n_records)` — **a deterministic function of record index alone**,
constant in every instance built from this task (the same formula runs every time, only the
filler count `total` varies slightly to hit target token length). This means record index and
document slot are perfectly confounded in every instance generated so far: the 9 flagged indices
have NEVER appeared anywhere but the last third of the document, in any instance in this dataset.
The per-instance linear detrend removes a straight-line position trend, but if the true
recency/tail effect is nonlinear (e.g. sharper very close to the sink/window boundary) or
interacts with anything else index-invariant about "being near the end," a linear fit will
under-remove it near the tail specifically — and Check 5's residual correlation, computed only
ever on these same 9 always-late indices, cannot distinguish "these indices have some content
property the scorer prefers" from "linear detrending doesn't fully correct the tail, and these
are the only indices ever observed in the tail."

## The design

Modify `assemble()` for a fresh batch of instances so that **which record content occupies which
target slot is a per-instance random permutation**, decoupled from record index. Concretely: keep
the existing target-slot list `[pos(i) for i in range(n_records)]` (same formula, same physical
slot positions used throughout this project) unchanged, but assign the 40 pieces of record content
to those 40 target slots via a fresh random permutation per instance, instead of the identity
mapping (content i -> slot i) used everywhere else. Record content generation (surname draw,
field values) stays tied to index i exactly as before, via the same seeded RNG — only the
slot each content bundle lands in changes, instance to instance.

This is a small, local patch: one new build function reusing `ledger_c.py`'s field/exemplar/
filler generation unchanged, with only the final content->slot assignment line replaced by a
permutation draw. No change to token budgets, task, model, or scoring.

## Threshold (fixed before generating anything)

**Precondition (checked before any correlation is computed):** across the fresh batch, each of
the 9 flagged indices must appear across a slot range spanning **>=15 document-position
difference** between its earliest and latest occurrence (i.e., the permutation must have actually
moved it well outside the tail region at least once). Any index that stays clustered in the tail
across the whole batch by chance is dropped from the analysis and the drop is reported by name,
not silently absorbed. If fewer than 6 of the 9 indices clear this bar, the whole probe is
reported as **NOT SCORED** (underpowered permutation coverage), not interpreted either way.

**Test:** single arm only (`snapkv` — cheapest, and the arm with the most flagged occurrences in
the original 20 cases). Capture raw base-scorer score and assigned slot for all 40 records in
every fresh instance (same floor-constrained, payable-region-only capture as
`out/_step4_fulldata.py`). Per instance, fit the same linear position->score detrend used
throughout (`out/_step5_stickiness_residual.py`'s method, unchanged) and compute every record's
residual. For the indices clearing the precondition, compute the same cross-instance bystander
correlation as Check 5 / Step 5 (residual in instance A vs mean residual for the same index in
other instances), using ONLY instance-pairs whose slot for that index differs by >=15 positions
(so the correlation is measured specifically where slot has actually been decoupled from
identity — pairs where the index happens to land in nearby slots again are excluded, since they
don't test the confound).

- **REAL** (identity effect, independent of slot): Pearson r >= +0.5 and p < 0.05 on the
  slot-decoupled pairs — a genuine per-record-content scorer preference survives moving the
  content to different document positions.
- **NOT-IT** (slot/tail-detrending artifact, not identity): Pearson r < +0.3, or the 95% bootstrap
  CI (10,000 resamples, CRC32-seeded) includes zero — the original stickiness collapses once slot
  is no longer confounded with which 9 indices get observed, confirming it was an imperfect-linear-
  detrend artifact specific to the document tail.
- Anything between +0.3 and +0.5, or a significant-but-weaker result, is reported as **inconclusive
  at this n** and not rounded to either side — a larger batch would be the next step, not a
  re-drawn threshold.

## What this does and does not require

- Same model/task/cell as every prior step in this pilot (M2 = Qwen2.5-3B-Instruct, c=40, C=512,
  LEDGER-C, floor-constrained snapkv). No new model load, no new task design beyond the slot
  permutation.
- Single arm only, by design, to keep this the smallest version that can still be informative;
  if the result is REAL or a clean NOT-IT on snapkv alone, that is treated as answering the
  question for this probe. Extending to the other flagged arms (adakv_snapkv, expected_attn) is a
  separate follow-up only if snapkv alone comes back inconclusive.
- Does not yet propose any fix to the unit-aware wrapper — this is still diagnosis of what Check
  5 was actually measuring.

## GPU cost of the smallest informative version (reported before running anything)

- **N = 20 fresh instances**, single arm (`snapkv`), same cell as every prior step
  (M2/c=40/C=512/LEDGER-C, ~2048-token context, floor-constrained region capture as in
  `out/_step4_fulldata.py`).
- Why 20: matches the instance count already used in every prior step of this pilot (`_step2`
  through `_step5`), so it is a known, already-paid-for cost class, not a new budget request. A
  uniform-random permutation of 40 slots across 20 draws makes each of the 9 flagged indices
  spanning a >=15-slot range across the batch a near-certainty (each index's slot is drawn
  independently and uniformly from 40 positions per instance; getting confined to a <15-wide
  band in all 20 draws has negligible probability for any single index), so 20 should clear the
  precondition without needing a second batch.
- **Per-instance cost**: one instance build (CPU, cheap) + one single-arm floor-constrained
  prefill of a ~2048-token context on Qwen2.5-3B-Instruct (bf16, sdpa) — the same operation
  `out/_step4_fulldata.py` and `out/_step2_basescore.py` already performed 20 times each this
  session, at a few seconds per prefill after model load.
- **Total**: 20 prefills, one arm, one model load — roughly a **quarter to a third** of the GPU
  work already spent on `_step4_fulldata.py` (which did 20 prefills split across 3 different
  arms/methods, each with its own press construction) and comparable to `_step2_basescore.py`'s
  cost. Expect on the order of a few minutes wall-clock including model load, not a new order of
  magnitude of spend.
- Not yet run. Awaiting go-ahead before generating the fresh batch or touching the GPU.
