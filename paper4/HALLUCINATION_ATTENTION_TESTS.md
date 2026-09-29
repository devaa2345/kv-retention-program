# Hallucination attention tests — ceiling arithmetic and three thresholds, committed before any extraction

Three independently-scored tests (A, B, C) on M2's plausible-fabrication cases, sharing ONE GPU
extraction pass (attention weights captured instead of discarded, at the same forward call
Task 3's margin diagnostic already uses). Scored and reported separately, per instruction; not
combined into one verdict.

## Ceiling arithmetic (reported before any test runs)

- **Hallucination's share of M2's disagreement population**: 35/141 ≈ **24.8%** (`fe5fbf7`,
  "Task 1 results", full M2 disagreement population read-through).
- **Plausible-fabrication's share**: 25/35 of hallucination cases ≈ **71.4% of hallucination**,
  i.e. **25/141 ≈ 17.7% of M2's total disagreement population** — the single largest closed-form
  share found in this project so far (contamination was 14/82 ≈ 17%, a comparable population
  share, but concentrated differently across arms; see below).
- **Arm breakdown of the 25 cases** (`out/_task1_tagged.json`): expected_attn 10, snapkv 8,
  adakv_snapkv 5, keydiff 2.
- **Current U-X accuracy at this cell** (`PILOT_C40_POSTFIX.md`, commit `68dc570`), n=50/arm:
  snapkv 0.080, adakv_snapkv 0.090, expected_attn 0.035, keydiff 0.025. `floor_pos` = 0.170.
- **100%-effective-fix ceiling, per arm** (case count / 50): snapkv +8/50=+0.160 -> **0.240**;
  adakv_snapkv +5/50=+0.100 -> **0.190**; expected_attn +10/50=+0.200 -> **0.235**; keydiff
  +2/50=+0.040 -> **0.065**.
- **Read against `floor_pos` (0.170)**: at the theoretical, never-fully-achievable ceiling, THREE
  of the four arms (snapkv, adakv_snapkv, expected_attn) would clear `floor_pos` outright; only
  keydiff would stay short. This is a materially larger lever than contamination's (which topped
  out near-parity on 2/3 arms and short on the third) — hallucination/plausible-fabrication is
  the single highest-ceiling mechanism identified in this pilot so far, which is exactly why it
  is worth three independently-scored attention tests rather than one.

## Shared extraction (one GPU pass, three scorings)

For each of the 24 usable case/match pairs in `out/_task3_m2_pairs.json` (25 built, 1 has no
completion-matched partner and is excluded, same population Task 3's calibration test already
used), run the SAME forward call `out/_task3_margins.py:margin_for` already performs — U-X
compressed cache, forced-gold prefix up to `div_token_idx` (0 for 23/24 pairs, per the file), one
more forward step producing the logits that diverge — but with `output_attentions=True` on that
final call instead of discarding it, for both the fabrication case and its completion-matched
correct-case partner. Per-record token spans and the record-ID token position come from the same
`line_units` accounting used throughout this project; non-headwise arms (snapkv, expected_attn,
keydiff) require inverting the unit-aware wrapper's per-head gather order back to original context
indices (`p4/unitwrap.py:247-274` gathers per head in descending-score order; reconstructed via
`stats["store_scores"]` plus each head's captured retained set, sorted the same way the actual
gather sorted them); headwise AdaKV needs no remapping (cache stays full length, masked positions
read near-zero attention directly).

Attention is averaged over all layers and heads, mirroring the base-score averaging already used
in every base-scorer analysis this session (`out/_step4_fulldata.py`, `out/_step6_...`).

A 2-pair smoke test runs first to confirm `output_attentions=True` is compatible with the
kvpress-patched attention path (both the gather-based and masked-AdaKV mechanisms) before the full
24-pair batch runs.

## Test A — attention mass on the queried record's span

**Measure**: at the divergence-token forward call, total attention mass (averaged over layers/
heads) on the queried record's full token span (all fields, all "payable" i.e. non-floor tokens),
for both the fabrication case and its completion-matched correct case.

**REAL**: paired difference (fabrication minus correct) is measurably LOWER, 95% bootstrap CI
(10,000 resamples, CRC32-seeded) excludes zero, in the predicted direction (fabrication < correct).
**NOT-IT**: CI includes zero, or the direction is reversed.

## Test B — attention entropy / diffuseness

**Measure**: Shannon entropy of the full attention distribution over the entire retained context
at the divergence-token forward call (not restricted to any one record), same pairs.

**REAL**: fabrication cases show measurably HIGHER entropy (more diffuse, no clear focus), 95% CI
excludes zero in the predicted direction. **NOT-IT**: CI includes zero, or reversed.

## Test C — record-boundary (record-ID token) attention

**Measure**: attention mass specifically on the queried record's ID token (e.g. the "R031" token
itself, distinct from its field values), same pairs.

**REAL**: correct cases show measurably HIGHER record-ID-token attention than fabrication cases,
95% CI excludes zero in the predicted direction. **NOT-IT**: CI includes zero, or reversed.

Each test scored and reported on its own. If two or more agree in direction, that is noted as
corroboration, not merged into a single combined statistic; if they disagree, that is reported
plainly too.

## GPU cost (reported before running)

- Same n=24 case/match pairs already used by Task 3 (`out/_task3_m2_pairs.json`), same model
  (Qwen2.5-3B-Instruct), same cell (c=40, C=512). No new instance generation.
- Per pair: 2 forward passes (case + match), each a prefill (already-paid-for cost class, same as
  every prior GPU step this session) plus at most a handful of forced-token steps (`div_token_idx`
  is 0 for 23/24, 1 for one, 2 for one — negligible extra compute) and one attention-capturing
  final step.
- **Total: 48 forward passes** (24 pairs x 2), one model load, plus a 2-pair (4-pass) smoke test
  first. This is the SAME order of magnitude as `out/_task3_margins.py`'s already-run cost (which
  did exactly this many prefills+forced-steps without attention capture); enabling
  `output_attentions=True` adds compute/memory overhead per call but no new passes. Expect low
  single-digit minutes wall-clock including model load.
- All three tests (A/B/C) are scored from this ONE extraction pass — not three separate re-runs.
