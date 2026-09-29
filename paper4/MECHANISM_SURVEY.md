# Why do completed units still convert poorly? — survey across Papers 1–4 (Parts 2–3)

Question: at matched completion, `U-X` converts retained facts into correct answers far worse than
`floor_pos` (M2 0.40/0.38 vs 0.60; M3 0.12/0.20 vs 0.33), while raw `X` converts **as well as or
better than** the floor despite being the most fragmented and most dispersed arm (M2 0.67/0.68 vs
0.60; M3 SnapKV 0.45 vs 0.33). Every candidate below is drawn from data already collected.

**Closed, not re-opened** (PART 5 item 5): positional dispersion (`LIMITATIONS_P4.md` L1 — D_runs
null on both models, D_iso contradicted on M2) and distractor load / interference from other whole
records (Paper 3, within-cell null on the floor). A candidate is only listed if it is distinct
from both.

---

## C1 — Head/slot disagreement: the heads hold *different* units

**Evidence for.**
- Paper 2, `Δ_head`: letting the causal oracle hold **different** candidates in different heads at
  equal total budget is **negative at every squeezed budget on all three models** (M2 2 KV heads,
  M3 8, 14B 8). Head-wise divergence, in isolation, *hurts* — measured, not argued.
- Paper 4 pilot: `U-X` has `q_any` ≈ 0.90–1.00 against `q_slot` ≈ 0.225–0.281 — the largest
  union-vs-per-slot gap of any arm. `X` sits at `q_any` 0.19–0.72. **`floor_pos` is head-agnostic
  by construction** (one keep-set replicated to every slot, `press.FloorPosPress`), i.e. agreement
  = 1.0 — and it is the arm with the best conversion.
- Paper 3, Evidence 2 (c=1, KeyDiff): gold token present in ≥1 slot on 98.5–100% of instances,
  per-slot retention ~2× the floor's, and accuracy still **below** the floor. Same signature:
  union availability high, usability low.
- Mechanistically `U-X` maximises this by construction: whole-unit greedy is a step function of
  per-head scores, so small score differences flip an entire unit in one head and not another.

**Against / limits.** Never measured directly in Paper 4. `U-X` and `floor_pos` differ in more than
agreement, so the across-arm ordering alone is confounded; the within-arm test is the real one.

**Cheapest test (CPU, saved keep-sets).** Per instance and arm: slot agreement (mean pairwise
Jaccard over slots; and the fraction of slots holding each queried fact). Regress accuracy on
`q_slot` + z(agreement) within the method arms, instance-bootstrap. Data already on disk.

## C2 — Denominator artifact: per-slot completion is not "availability"

**Evidence for.**
- The recorded lead in `LIMITATIONS_P4.md`: "which layers/heads hold the fact may matter more than
  how many do" — conversion's denominator (`q_slot`) may simply be mis-measuring what is available.
- Paper 3, Evidence 1: three crossover cells where a method's **measured completion is below the
  floor's and its accuracy is above** (M2 c=8.2 C=256: floor q 0.181/acc 0.072 vs SnapKV q
  0.110/acc 0.102). A completion measure that mis-orders arms cannot be the right denominator.
- `q_maj` (complete in >50% of slots) is recorded for every Stage 3 row but has never been used.

**Against / limits.** This is a measurement hypothesis, not a mechanism; if it survives it reframes
the deficit rather than explaining it.

**Cheapest test (CPU, rows only).** Recompute the floor-vs-`U-X` conversion gap under all three
denominators in every admissible cell.

## C3 — Query-adjacency: distance from retained content to the query, not its contiguity

**Distinct from the closed dispersion null**, which measured run-structure (`D_runs`) and
neighbourhood isolation (`D_iso`) — both *local shape* measures. This is *absolute distance to the
query position*, which neither measured.

**Evidence for.**
- `floor_pos` is recency by construction (first 8 + last C+64) and is the best converter in every
  admissible cell of Papers 2, 3 and 4.
- Paper 3 Stage 1: the method/floor ratio falls with `c` in 24/24 sequences — the arm that wins at
  high cost is exactly the one whose content sits adjacent to the query.
- Paper 2: `G_m ≤ 0` in 48/50 method-cells against that same recency floor.

**Against / limits.** Adjacent in spirit to the closed hypothesis; must be reported as
distance-to-query and not conflated with contiguity. Also partly collinear with `p_g` in the floor
arm (its gold is retained *because* it is late).

**Cheapest test (CPU, saved keep-sets).** Mean distance from retained gold tokens to the context
end; within-arm regression on accuracy controlling `q_slot`.

## C4 — Scaffolding retention: the 2-shot exemplar frame

**Evidence for.**
- Paper 2's blind reimplementation: **dropping the BOS/sink token collapses every arm, including
  both oracles, to 0.0000.** A handful of structural tokens can be entirely load-bearing — a
  measured, not hypothetical, precedent for scaffolding effects.
- Both tasks carry a 2-shot worked-example frame at the **start** of the context (`_exemplars` in
  `ledger_c.py`, `EXEMPLARS` in `mark1.py`), which teaches the output format. `floor_pos` keeps
  only sink+tail and therefore drops it; `U-X` may retain the exemplar record lines (they are
  units: R900/R901, part of Paper 3's 42-unit accounting) depending on score.
- Stage 2 generations show format failures exactly of this kind (`R021|Missing|Missing|…`).

**Against / limits.** If the floor drops the exemplars and still converts best, scaffolding cannot
be the whole story; the test is whether retention *within* an arm predicts accuracy.

**Cheapest test (CPU, saved keep-sets).** Exemplar-unit retention per instance/arm; within-arm
regression as above.

## C5 — Cache gather **order** (hygiene, not a mechanism)

`ScorerPress.compress` gathers keys/values in **top-k score order**; `p4/unitwrap.py` gathers in
**ascending position order**; `floor_pos` (an `_ExactSetPress` with KEEP/DROP ties) gathers in yet
another order. Attention is permutation-invariant over key positions once RoPE is baked in, so this
*should* be inert — but it is an uncontrolled difference between the arms being compared, and it is
cheap to falsify. Listed so it is not assumed away.

**Cheapest test.** Small-n GPU: one cell, one arm, identical keep-set, two gather orders, compare
generations. **Needs generation — not run under the current CPU/prefill-only constraint.**

## C6 — Surface-form integrity of the fact (recorded, not cheaply testable)

Paper 3, Evidence 4: the causal oracle keeps **every** queried record whole at k=1 and k=2, and M3
accuracy still halves (0.730 → 0.370) when the record is split into two adjacent labelled parts —
against the direction the answer-length confound predicts. Also Stage 2.3: `full_cache` falls
0.910 → 0.400 under scattering with **no compression at all**. Strong evidence that *form*, not
retention, can carry the loss. In Stage 3 every unit is a whole line, so there is **no variation to
regress on** — testing this needs new task construction (a full pilot), so it is surveyed here and
not tested now.

---

## Cost ordering (cheapest first)

| # | candidate | test cost |
|---|---|---|
| C2 | denominator artifact | CPU, rows only — minutes |
| C1 | head/slot disagreement | CPU, saved keep-sets — minutes |
| C3 | query-adjacency | CPU, saved keep-sets — minutes |
| C4 | scaffolding retention | CPU, saved keep-sets — minutes |
| C5 | gather-order hygiene | small-n GPU generation (~10 min) — deferred |
| C6 | surface-form integrity | new task + pilot (~1–2 GPU-h) — deferred |
