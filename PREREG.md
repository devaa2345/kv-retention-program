# Pre-registration — Protection × Recoverability Test Program

Written and hashed before any Phase 1 GPU time is spent, per the test-program design (v1).
Nothing below is a result. This document fixes the analysis so Phase 1's numbers can't be
read selectively after the fact.

## 1. Calibrated instrument (Phase 0-R)

Task: `kvcache_harness/tasks/multi_credential.py`, multi-credential retrieval with induced
dormancy. Config that passed all five calibration gates at n=10
(`kvcache_harness/configs/phase0r_calibration.yaml`, results in
`results/phase0r_calibration/calibration_summary.json`):

| parameter | value |
|---|---|
| model | Qwen/Qwen2.5-1.5B-Instruct, bfloat16, eager attention |
| n_credentials (N) | 6 |
| credential value length | 14 hex chars (17 chars incl. `sk-` prefix) |
| n_distractors | 20 |
| words_per_paragraph (filler) | 50 |
| total_budget (calibration) | 280 |
| full_fraction | 0.5 |
| recency_window | 64 |
| keep_sink | true |
| max_answer_tokens | 22 |
| protect_after_chars | 2 |
| mean context length (dump + first question) | 1029 tokens (n=10, range 1011–1042) |

Calibration result (n=10, `results/phase0r_calibration/calibration_summary.json`):

| gate | criterion | value | result |
|---|---|---|---|
| G1 ceiling (oracle_static) | ≥0.95 | 0.983 | PASS |
| G2 floor (no protection, permanent) | ≤0.10 | 0.033 | PASS |
| G3 headroom (protection alone) | 0.45–0.75 | 0.567 | PASS |
| G4 no-eviction reference | ≥0.95 | 0.983 | PASS |
| G5 dormancy check | ≥0.80 | 1.000 | PASS |

Two mechanism bugs were fixed to get here (not just parameter tuning), both load-bearing for
Phase 1 and documented in code: `cache/structural_protection.py` (protection made atomic per
structurally-matched line, not per-token — token-level competition was fragmenting individual
credential values) and `cache/oracle_static.py` (budget-trimming was slicing an unordered
Python `set`, silently dropping an arbitrary, non-reproducible subset of credentials when the
oracle's must-keep set exceeded budget; oracle also needed each credential's label, not just
its bare value, to disambiguate which credential a floating value belongs to).

N was reduced from an initial 8 to 6 during calibration: at N=8, G1/G4 landed at 0.90/0.925 —
short of 0.95 but tracking each other almost exactly, which is the signature of compounding
per-turn copy error over 8 sequential turns (a base-model capability ceiling), not a cache
artifact — oracle_static (perfect retention) and full_cache_ref (no eviction mechanism at all)
have no reason to track each other unless both are hitting the same non-cache ceiling. Reducing
sequential turns was the direct fix; it was chosen over switching to a larger model (would force
re-calibrating budget/recency from scratch at much higher compute cost, and is better spent
later as a deliberate Phase 5 generalization check) or relaxing the gate (would eat directly into
the power to detect Phase 1's minimum effect of interest).

## 2. Budget sweep (Phase 1)

Budgets as fractions of mean context length (1029 tokens), rounded:

| fraction | total_budget (tokens) |
|---|---|
| 0.05 | 51 |
| 0.10 | 103 |
| 0.15 | 154 |
| 0.25 | 257 |
| 0.50 | 514 |

Run at both:
- **1a — iso-retained-tokens**: total_budget as above, full_fraction=0.5 for tiered arms (matches how QEvict reports).
- **1b — iso-memory-bytes**: tiered arms' *hot* (FULL) allocation reduced so total bytes (FULL tokens costing 1x, QUANT tokens costing 1/4x under the 8-bit quant scheme in `engine.py`) match the permanent-eviction arms' bytes at the same total_budget — i.e. tiered arms get proportionally more QUANT slots to pay for their cold tier within the same memory envelope, not the same raw token count.

Design: 2 (protection: none / structural) × 2 (eviction: permanent / recoverable tiered) × 5 budgets × 2 (iso-token / iso-memory) = 40 cells, plus `oracle_static`, `full_cache_ref` at each budget (iso-token only, since they aren't tiered). n≥150 prompts per cell, seeds disjoint from calibration (seed_start=2000..2009 reserved for calibration; Phase 1 uses seed_start≥3000).

## 3. Primary analysis — fixed in advance

- **Primary metric**: fraction retrieved (k/N), per prompt.
- **Primary estimand**: the interaction term — (recoverability effect | protection on) −
  (recoverability effect | protection off) — computed independently per budget, per
  iso-token/iso-memory condition. No parametric functional form is assumed across budgets
  (i.e., no curve is fit through the 5 budget points); each budget's interaction is reported
  as its own estimate. This is the "commit to the form" choice from the design doc — treating
  the 5 points non-parametrically rather than CV-selecting a curve, because there's no prior
  reason to expect a specific functional shape and fitting one would be a second, unregistered
  degree of freedom.
- **Estimator**: bootstrapped **median** of paired per-prompt differences (not the mean),
  10,000 resamples, resampling unit = prompt (all N within-prompt credential outcomes travel
  together with their prompt in each resample — this is the "prompt-clustered" part; individual
  credential outcomes within a prompt are not treated as independent draws).
- **Correction**: Benjamini-Hochberg across the 5-budget family, within each of iso-token/iso-memory separately (10 tests total per protection-recoverability contrast).
- **Minimum effect of interest**: 0.05 on fraction-retrieved. Equivalence (null) is declared
  when the 95% bootstrap CI is fully contained in ±0.05.
- **Nuisance parameters fixed**: `recency_window`=64, `keep_sink`=true, `protect_after_chars`=2,
  `full_fraction`=0.5 (1a) / memory-matched (1b), `rebalance_every`=1, greedy decoding
  (no sampling) throughout. None of these are swept or selected post hoc in Phase 1; any
  deviation from these values for a specific cell is called out explicitly in the results, not
  silently varied.

## 4. Kill gates

- **This document's hash** (below) is the committed baseline. A Phase 1 run against a modified
  instrument (different N, budget set, scorer, or nuisance parameters) is a new pre-registration,
  not an amendment to this one.
- If the Phase 1 interaction CI is inside ±0.05 at every budget under both iso-token and
  iso-memory conditions: H-SUB is confirmed for this task/model. Proceed to Phase 2 (H-ORTH is
  independent of the interaction result) but the program's output is a decomposition paper, not
  a method paper.
- Phase 2/3/4/5 kill gates as specified in the test-program design v1, §4, unchanged here.

## 5. Hash

SHA-256 of this file (computed and recorded after writing, before any Phase 1 run):
`<computed below, appended by the harness — see PREREG.sha256>`
