# Phase 2, Stage 2.0 — CPU-only findings

All three tasks below used only the tokenizer (no model weights, no GPU) and generation data
already on disk (`runs/nvidia/p4_s3_M2.jsonl`, Stage 3's frozen grid, n=50 each for `floor_pos`
and `oracle_causal` at c=40, C=512). One sub-question (is the max-attention record the queried
record, for tasks 2/3) needs GPU and is flagged, not run, at the end.

## Task 1 — P1 bound, verified

**Method.** `FloorPosPress` (`harness/press.py:118-126`) scores by ascending position with sinks
forced to KEEP, so its kept set is exactly `{0..n_sink-1} ∪ {last (C+n_window) tokens}` — a pure
position computation, independent of the model. Bound = `min(N, floor((C+window)/c)) / N`, using
the calibrated per-record token cost `c` (L-invariant: record field generation doesn't depend on
target length, only filler count does, so the c=2048-calibrated cost applies unchanged at longer
L). Measured completion = fraction of queries where the SPECIFIC queried record's full token span
lies inside that same deterministic kept set — computed directly from `line_units`, no generation
needed.

**Headline cell, cross-checked at two sample sizes and two instance-id conventions** (M2, c=40,
C=512, L=2048): bound = 14/40 = **0.350** (not the rough 0.30 guess, which omitted the window).
Measured completion: 0.2375 (n=20, fresh ids) / **0.2600 (n=50, standard `s4_` ids)**. Ratio to
bound: **0.68–0.74**, not the ~0.95 the rough arithmetic implied. M3 at the same cell: 0.3375/0.35
= **0.964** — much closer to its bound.

**This is genuinely mixed, model-dependent, and NOT simply "near the bound everywhere"** — full
table (`out/_p1_bound_results.json`, n=20/cell across 13 cells x 2 models):

| cell (c_tag, C, L) | bound | M2 measured/bound | M3 measured/bound |
|---|---|---|---|
| 40, 512, 2048 | 0.350 | 0.679 | 0.964 |
| 8, 512, 2048 | 1.000 | 0.312 | 0.250 |
| 19, 512, 2048 | 0.750 | 0.350 | 0.417 |
| 8, 1024, 4096 | 1.000 | 0.625 | 0.575 |
| 19, 1024, 4096 | 1.000 | 0.575 | 0.588 |
| 8, 512, 4096 | 1.000 | 0.312 | 0.250 |
| 19, 512, 4096 | 0.750 | 0.350 | 0.417 |
| 8, 256, 4096 | 0.950 | 0.184 | 0.086 |
| 19, 256, 4096 | 0.400 | 0.281 | 0.471 |
| 8, 128, 4096 | 0.575 | 0.174 | 0.119 |
| 19, 128, 4096 | 0.250 | 0.200 | 0.600 |
| 8, 2048, 8192 | 1.000 | **1.000** | **1.000** |
| 19, 2048, 8192 | 1.000 | **1.000** | **1.000** |

**Pattern**: the bound is only a tight predictor of measured completion at the two extremes —
when the budget is so generous relative to record cost that packing is trivial (8192/2048 cells,
ratio exactly 1.000 both models), or, less cleanly, at the standard c=40 anchor cell for M3
specifically. At every intermediate cell (bound<1 but not tiny), measured completion sits well
below the bound (0.17-0.63) — the naive "floor(block/c)" arithmetic assumes perfect packing with
no boundary waste, and real documents don't pack that cleanly (a record straddling the exact
block boundary is entirely lost, not partially credited, and filler between records isn't free).

**Cross-check flagged, not resolved**: `PAPER4_STRATEGY_BIG_THREE.md`/`PAPER4_COMPLETE_RECORD.md`
cite floor's completion at this cell as **0.285**; this measurement gives 0.26 at matched n=50.
The gap is modest and could be sampling noise, a different completion definition (e.g. mean
fractional `q_slot` vs strict binary whole-record retention), or a different query population —
not chased further here, stated as unresolved per instruction to verify rather than assume.

**Implication for P1**: NOT confirmed as stated. The floor is close to the query-agnostic bound
only in the regimes where the bound is nearly vacuous (very generous budgets) or, for M3
specifically, at the one cell every live claim rests on. It is NOT uniformly near-bound — meaning
retention-side headroom below the bound genuinely exists at most cells (measured/bound as low as
0.09-0.20), which argues AGAINST "every retention lever was capped before it started" as a
blanket conclusion, and instead argues Levers 1-2's failure needs its own explanation beyond
"the bound made it impossible" — they may indeed have "failed on execution," per Part 3's own
fallback row in the decision table.

## Task 2 — Floor's own failures (never read before)

**Population**: floor_pos, c=40, C=512, M2, n=50 instances (200 queries), restricted to cases
where the queried record IS held whole (per Task 1's deterministic completion check) but the
answer is wrong. **n = 23/200 (11.5%)** — closely matching the Phase-2-plan's own back-of-envelope
11.4% ("held-but-wrong") estimate, a good independent cross-check despite the 0.26-vs-0.285
completion discrepancy above.

**Taxonomy** (Track 2's existing rule, reused verbatim — record-distance<=2/overlap>=6 for
contamination, overlap>=4 for distant misattribution, refusal regex, fabricated-number check):

| tag | n | share |
|---|---|---|
| **other** (no record match, no fabrication signature, not refusal) | **20** | **87%** |
| misattribution (distant) | 2 | 9% |
| contamination | 1 | 4% |
| refusal | 0 | 0% |

**This population looks nothing like the U-X disagreement population Track 2 originally read**
(which was 17% contamination on M2). Floor's own failures are overwhelmingly "other" — reading
the actual answers (`out/_p2p3_results.json`), many are schema-conforming 8-field lines whose
individual field values don't fabricate new numbers and don't match any single other record
closely, suggesting a **blending/recombination pattern** (values drawn from more than one nearby
record, diluted enough to miss both the contamination and fabrication thresholds) rather than a
single clean failure mode. This is a genuinely new observation the existing 5-way taxonomy
doesn't cleanly capture, not previously visible because this population was never read.

## Task 3 — Oracle's failures

**Population**: `oracle_causal`, same cell, same n=50/200. Oracle holds every queried fact
complete by construction, so no completion pre-check is needed — every miss qualifies. **n =
61/200 (30.5%)**, matching oracle's cited accuracy (0.695, so miss rate 0.305) exactly.

| tag | n | share |
|---|---|---|
| **other** | **54** | **89%** |
| contamination | 4 | 7% |
| misattribution (distant) | 3 | 5% |
| refusal | 0 | 0% |

**Structurally identical share pattern to floor's own failures** (89% vs 87% "other"). This is the
key finding for P3: oracle's failures are **not** dominated by contamination or refusal — they are
overwhelmingly the SAME unclassified "other" pattern as the floor's, at nearly the same rate. This
is consistent with (not proof of) an attention/grounding failure at decode time, present regardless
of whether retention itself is perfect (oracle) or merely adequate (floor) — exactly the profile
Part 3 hoped for ("if largely attention failures, grounding could move the usability ceiling
itself"). But this reading is CPU-only pattern-matching on answer TEXT; it does not yet confirm an
attention mechanism — that requires the GPU step below.

## What's not done: the max-attention sub-question

Both tasks 2 and 3 asked, per case, whether the max-attention record is the queried record —
this needs the same eager-attention GPU extraction used for Test A/E3, applied to a NEW
population (84 total cases: 23 floor + 61 oracle, overlapping instances but different arms/
presses) that has never been profiled this way. This is GPU work, and the instruction for this
stage was explicitly CPU-only — flagging it here rather than silently running it. **Not run.
Awaiting confirmation before any GPU touches this stage.**

## Bottom line for P1 and P3

- **P1 is NOT verified as originally framed.** The floor sits near its query-agnostic bound only
  at the extremes (trivially generous budgets, or M3's specific anchor cell) — measured/bound
  ranges from 0.09 to 1.00 across cells actually used in this program. Retention-side headroom
  below the bound is real at most cells; Levers 1-2's failure cannot be fully attributed to "the
  bound made it impossible" without more care about which cell is being discussed.
- **P3 has something to act on, tentatively.** Oracle's failures (89% "other") look structurally
  identical to the floor's own failures (87% "other") — both dominated by an unclassified pattern
  distinct from contamination/refusal, consistent with (but not proof of) grounding being the
  shared bottleneck. This needs the GPU max-attention check to become more than a text-pattern
  observation.
