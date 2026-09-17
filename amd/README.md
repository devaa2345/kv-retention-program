# Protection × Recoverability Harness — blind reimplementation

An independent implementation of `SPEC_REIMPL_v1.md`, written **without reading the original
`kvcache_harness/`**, so that the two can be compared. No file of the original was opened,
imported, listed, or searched at any point.

## Layout

```
kvre/
  task.py          task generation + scoring (spec §2)
  cache_engine.py  quantizer, three tiers, byte accounting (spec §3)
  policy.py        retention floors, protection, tiering (spec §4, §5, §7)
  engine.py        chunked prefill, per-step policy application, multi-turn loop
  arms.py          arm table (spec §5) and promotion signals (spec §7)
  gates.py         calibration gates G1-G5
  runner.py        append-only JSONL checkpointing, resume, failure logging
  analysis.py      paired bootstrap, permutation test, BH, CSV export
  model.py         model loading (bf16, eager attention)

run_gates.py           calibration gates G1-G5 -> results/calibration_gates.json
check_determinism.py   two-process retained-set determinism assertion
run_phase1.py          Phase 1: iso-token, budgets 154/257/514, arms 1-6, n=150 (4-bit)
run_phase1_8bit.py     Phase 1 tiered arms at 8-bit -> tests [GAP-W]
run_phase2.py          Phase 2: five promotion signals, n=150 (original signals)
run_phase2_fixed.py    Phase 2 re-run with corrected P4/P5 signals (PRIMARY)
run_phase2_8bit.py     Phase 2 at 8-bit, residual test for [GAP-W]
run_sweep.py           bit-width sweep 8/7/6/5/4/3, n=50
calibrate_context.py   context calibration + gate RE-RUN at 2048 / 4096
run_context.py         2x2 factorial at 2048 / 4096, retention held at ~13%
run_isomemory.py       iso-memory condition (§6) + [SPEC-GAP 3] byte-cost sensitivity
run_band.py            §7 FULL-minus-QUANT band check at budget 257
run_rho.py             §7 Spearman rho manipulation check
rho_power.py           rho at n=100, both original and corrected signal definitions
run_posthoc.py         P5 ceiling check, level occupancy, intercepted-error monotonicity,
                       credential-survival trajectories
run_ablation_distractor.py  [GAP-V] distractor surface-shape ablation

analyze_isomemory.py   iso-memory + sensitivity analysis
analyze_phase2fix.py   Phase 2 original vs corrected
analyze_3factor.py     protection x tiering x bit-width
analyze_sweep.py       full sweep table on a common seed set
analyze_scoring.py     [GAP-U] scoring-convention ladder
regen_sections.py      regenerates sections/*.md from stored JSON (idempotent)
make_report.py         FINDINGS.md + CSV exports
make_agreement.py      AGREEMENT.md - every reference value compared
make_summary.py        SUMMARY.md - executive summary
supervise.sh           restarts a job if the GPU faults; all work is checkpointed

SUMMARY.md           executive summary (start here)
FINDINGS.md          full results; narrative sections assembled from sections/*.md
AGREEMENT.md         deliverable 6: every reference value, disagreements traced to gaps
sections/            narrative fragments, concatenated into FINDINGS.md by make_report.py
SPEC_QUESTIONS.md    every ambiguity found and the default chosen (deliverable 2)
FINDINGS.md          results (deliverable 3-6)
results/
  bits<N>/*.jsonl            raw rows, one directory per bit-width (spec §6)
  p2fix/*.jsonl              Phase 2 with corrected promotion signals
  */*.failures.jsonl         failed runs, never silently dropped
  csv/*.csv                  CSV exports
  *.json                     per-analysis outputs
```

## Operational notes

Running four concurrent ROCm processes on this GPU triggered a `hipErrorLaunchFailure` that
killed all of them mid-run. Checkpointing is append-only with per-row `fsync`, so no data was
lost or corrupted (all 6,808 rows written at that point parsed cleanly) and the jobs resumed from
their last committed row. Concurrency is capped at two, and `supervise.sh` restarts a job that
faults. Two concurrent processes measured ~1.8x the throughput of one; four was unstable.

## Reproducing

```bash
export PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=0
PY=/home/kxrx26/quant-rocm/bin/python
$PY run_gates.py          # must pass before measuring
$PY check_determinism.py
./run_all.sh
```

Everything checkpoints. Re-running skips completed
`(nominal_budget, iso_condition, arm, seed, quant_bits, promotion, context_target)` keys, so an
interrupted run resumes without redoing work and without duplicating rows.

## Invariants enforced at runtime

- `total_budget − recency_window − sink > 0` asserted before **every** cell; violating cells are
  refused, not silently run (spec §4 records that the original swept two arithmetically void
  budgets).
- Quantizer level counts ≤ 2^bits, asserted per group on tensors **actually written during
  generation**, recorded per row as `quant_violations`.
- Reconstruction error monotone in bit-width, checked both on cached prefill K/V and on the
  intercepted generation tensors themselves.
- Two processes with `PYTHONHASHSEED=0` produce identical retained-token index sets.
- No selection over an unordered Python `set` anywhere that affects which tokens survive; every
  ranking sorts explicitly with a deterministic tie-break on ascending position index.
- Failed runs are appended to a separate failures file so paired seed sets stay identical across
  arms — silent dropout would break the paired analysis without complaining.

## Row schema

Every output row carries `nominal_budget`, `effective_budget`, `effective_tokens`,
`iso_condition`, `quant_bits`, `quant_byte_cost`, `physical_byte_cost`, `model`,
`context_length`, plus arm/protection/eviction/promotion identity and the realised
`n_full` / `n_quant` / `bytes` tier accounting. Different bit-widths are written to different
directories, per spec §6.
