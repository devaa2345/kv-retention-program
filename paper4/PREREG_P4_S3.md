# PAPER 4 — STAGE 3 PREREGISTRATION (main grid, oracle units)

**Status: frozen. Hash recorded in `PREREG_P4_S3.md.sha256` (sha256 of LF-normalised bytes),
round-trip verified against the committed blob.** No Stage 3 generation row may be produced
before that hash verifies; the runner re-checks it at start-up and refuses to run otherwise
(the procedure Paper 3's `stage4_run.py` uses).

Source of truth for invariants and scope: `PAPER4_CONTEXT_AND_PLAN.md` (v3) PART 4 and PART 5.
Produced on the RTX 5070 (Machine N), single process, pinned WSL toolchain.

---

## 1. Question and scope

Stage 2 established, at one cell (c≈40, C=512): `U-X` beats `X` (3 of 4 pairs, CIs excluding
zero), still loses to `floor_pos` (1.8–3.3×), reaches near-floor completion, and converts worse.
**Stage 3 asks whether that result is a point or a curve**: how the fragmentation effect
`A(U-X) − A(X)` and the position gap `A/A_floor` behave across the whole fact-cost × budget
plane that Paper 3 measured.

**Oracle units ONLY.** Structural units (delimiters/newlines/sentence boundaries) and inferred
units (key-space clustering) are separate later stages of the PART 2 ladder and are out of scope
here; no structural-unit number enters any Stage 3 table. The oracle-vs-structural gap (this
paper's analogue of Paper 2's information share `I`) is measured in a later stage, against this
stage's oracle ceiling.

## 2. Grid — matched to Paper 3 so the curves are directly comparable

| axis | values | source |
|---|---|---|
| fact cost `c` | **{1, 8, 19, 40}** | PREREG_P3 §6 Axis 1 (c=1 via MARK-1; 8/19/40 via LEDGER-C) |
| budget `C` | **{32, 64, 128, 256, 512}** | PREREG_P3 §6 Axis 3 (Paper 2 additionally ran C=16; Paper 3 dropped it; Stage 3 follows Paper 3) |
| models | **M2 = Qwen/Qwen2.5-3B-Instruct, M3 = meta-llama/Llama-3.2-3B-Instruct** | Papers 2/3 |
| arms (10) | `floor_pos`, `oracle_causal`, `snapkv`, `adakv_snapkv`, `expected_attn`, `keydiff`, `U-snapkv`, `U-adakv_snapkv`, `U-expected_attn`, `U-keydiff` | Stage 2 + the two U arms not yet run under generation |
| instances | **n = 50**, ids `s4_00000`–`s4_00049`, H = 4 queries each (200 queries per cell) | Stage 2; same seeds as Paper 3 Stage 4 |
| budget identity | `B = C + n_sink + n_window = C + 72`, `n_sink = 8`, `n_window = 64`. **The retained total is B, not C.** | PREREG_P3 §9.1 |

`n_fields` per `c` comes from `out/c_calibration.json` (Paper 3's measured calibration), not
re-fitted. Estimated cost at Paper 3's measured per-row times: **9.6 GPU-hours** (M2 5.1, M3 4.5).

**Deviation from Paper 3's n, stated:** Paper 3 registered n=200 (captures) / 100 (generation);
Stage 3 registers n = 50, as Stage 2 used. This widens every CI by ~sqrt(2) relative to n=100 and
is a precision choice, not a design change. No cell is dropped.

## 3. Load-bearing quantities — one reading each

Throughout, a **slot** is one (layer, KV-head) pair; `K_s` is the keep-set captured at slot `s`
**during the prefill that generated the answer**; `S` is the set of slots; `H = 4`.

### 3.1 Accuracy
`score_one` is the task scorer, unchanged from Paper 3: LEDGER-C — 1.0 iff the gold content-word
sequence appears as a contiguous run in the generation (`p3/tasks/ledger_c.py:score_one`);
MARK-1 — 1.0 iff the answer word is the first word emitted (`p3/tasks/mark1.py:score_one`).

- per instance: `a_i = (1/H) * sum_h score_one(gen_{i,h}, answer_{i,h})`
- cell accuracy: `A = mean_i a_i`. **All paired statistics are computed on `a_i`, per instance.**

### 3.2 `floor_pos` — THE definition, stated once, for all of Paper 4

> **`floor_pos` retains exactly `B = C + 72` tokens: the first `n_sink = 8` tokens, plus the most
> recent `B − n_sink = C + 64` contiguous tokens. Nothing else.**

This is the total-**B** reading. Paper 2's prereg carried both this and a total-`C` reading
("`n_sink=8` + last `C−8`"), 72 tokens apart at C=512; the ambiguity was found twice
independently, once by each blind reimplementation. It is resolved here in this one place and
nowhere else in Paper 4.

**The harness is verified to implement this exact reading, not assumed to:** `p4/floorcheck.py`
expresses it, and `assert_floor_pos` is called on every `floor_pos` row by the Stage 3 runner.
Checked against Stage 2's captured keep-sets before freezing: **M2 3600/3600 and M3 11200/11200
slots equal `first 8 ∪ last 576` at C=512 (B = 584)**. Both code paths agree
(`harness/ladder.py:floor_pos`, `harness/press.py:FloorPosPress`).

### 3.3 Completion — three measures, no default

- `q_slot = (1/|S|) * sum_s (#{queried facts f : f ⊆ K_s}) / H`  (per-slot mean)
- `q_any  = (1/H) * sum_f 1[exists s in S : f ⊆ K_s]`  (union across slots)
- `q_maj  = (1/H) * sum_f 1[ |{s : f ⊆ K_s}| > |S|/2 ]`  (majority of slots)

A fact `f` is its set of token indices in the templated prefill; `f ⊆ K_s` means every one of
those tokens is retained at that slot. **No one of the three is "the" usability measure**
(`LIMITATIONS_P4.md`, PART 5): all three are reported side by side in every table where
completion appears, and any claim resting on completion must state which measure it uses.

Unit accounting, same definitions as Paper 3's captures: `units_complete = (1/|S|) * sum_s
#{units u : u ⊆ K_s}`, `units_touched = (1/|S|) * sum_s #{u : u ∩ K_s ≠ empty}`, over **all**
units in the context (42 for LEDGER-C incl. the 2 worked-example record lines; 43 entry words for
MARK-1), never only the queried ones. `p_g = (1/|S|) * sum_s |gold ∩ K_s| / |gold|`.

### 3.4 Conversion — a triple, reported together

`V_slot = A / q_slot`, `V_any = A / q_any`, `V_maj = A / q_maj`, each at cell level. Undefined
(reported `n/a`, never imputed, never treated as 0) where its denominator is 0. Stage 2 quoted
`V_slot` alone; Stage 3 does not, per PART 5.

### 3.5 Ratios and paired differences

- **`Δ_U(method) = a_i(U-X) − a_i(X)`, paired per instance** — PRIMARY quantity for the
  fragmentation claim. Reported as mean with CI.
- `ρ_floor(arm) = A_arm / A_floor` — the same quantity as Paper 3's method/position ratio, for
  direct comparison with its curve. Ratio of cell means (not mean of per-instance ratios).
- `F = (A_{U-X} − A_X) / (A_floor − A_X)` — fraction of the position gap that removing
  fragmentation recovers. **Defined only where `A_floor − A_X > 0.05`**; otherwise `n/a`.

### 3.6 Uncertainty

95% percentile bootstrap, **10,000 resamples of instances** (paired: an instance resampled
carries all its arms), seeded `CRC32("p4|s3|<quantity>|<model>|<c>|<C>|<arm-pair>")`. A CI
"excludes zero" iff both bounds are strictly on one side of 0.

### 3.7 Degenerate-cell exclusion

**`A_floor < 0.05` → the cell is excluded** from every ratio table, from every kill-criterion
evaluation and from the c=1 control. Rows are still produced and appear in the raw accuracy
table, marked `degenerate`. Mechanical, applied in the analysis script; never a judgement call.

## 4. Seeding, dedup key, and the two PART 4 gaps found in Stage 1–2

**Seeding:** instance seed = `CRC32` over the canonical record key with `seed` at a sentinel and
`arm`, `B`, `C`, `max_new`, `matched_to` nulled, so every arm at every budget sees the same
instance (`p3/keys3.py:instance_seed`). Python `hash()` is never used. Bootstrap RNGs are seeded
by CRC32 of a fixed string, as in §3.6.

**Dedup key = `p3/keys3.FIELDS`, all 21 fields:** `task, instance_id, model, model_revision, arm,
B, C, protocol, device, backend, torch_version, transformers_version, kvpress_version, dtype,
seed, batch_size, n_fields, n_records, layout, matched_to, max_new`. Stage 3 values:
`device="nvidia"`, `backend="cuda-12.8"`, `torch_version="2.11.0+cu128"`,
`transformers_version="5.2.0"`, `kvpress_version="0.5.4"`, `dtype="bfloat16"`, `batch_size=1`,
`protocol="agnostic"`, `layout="p4_stage3_oracle"`, `matched_to=null`.

Two gaps between PART 4 and what Stage 1–2 actually encoded, both found by inspection before
freezing, both fixed here:

1. **`torch_version` was absent from the key dict** in Stage 1–2 (`stage4_run.ENV` omits it), so
   it hashed as `null` — a version field PART 4 requires was not actually recorded. Stage 3 sets
   it explicitly. Consequence: Stage 3 key digests are not comparable to Stage 1–2 digests; this
   is intended, Stage 3 is a fresh grid, and no Stage 1–2 row is reused in any Stage 3 number.
2. **`batch_size = 1` pinned and "single process, never pipe a runner"** were implemented by the
   Stage 1–2 drivers (single-writer `pgrep` guard, plain redirects) but were not written down in
   `STAGE12_RULES.md`. Registered here explicitly: `batch_size` stays 1 for every Stage 3 row,
   batching is not used anywhere, and no runner is piped through another command.

## 5. Harness invariants (PART 4, applied in full, asserted per row)

1. `position_ids` continue from the **uncompressed** context length — asserted per generation.
2. **Every arm generates tokens**; cache state is additionally checked compressed (length `B`
   for scorer presses, mask present for head-wise AdaKV). Prefill-only checks never stand alone.
3. Mandatory **sink(8) + window(64) floors retained by every arm including method arms** —
   asserted from the captured keep-set of every row (`floor ⊆ K_s` for all `s`).
4. **Realised budget parity asserted per instance from captured keep-sets**, never from the
   requested ratio: `|K_s| = B` per slot, and per-layer total `B * |heads|` for AdaKV.
5. Degenerate-cell rule as §3.7.
6. CRC32 seeding as §4; never Python `hash()`.
7. `device` in the dedup key; **every number inside any Stage 3 ratio comes from the RTX 5070
   only.** No cross-device and no cross-machine quantity enters a ratio.
8. **Single process** on the 5070; runners are never piped; a single-writer guard runs first.
9. Hypotheses and decision rules committed **before** the data they judge exists — this file.
10. `batch_size = 1`, batching unverified and unused (§4.2).
11. **Keep-sets are saved with every generation row** (`runs/nvidia/p4_s3_keepsets_*.npz`,
    packed bitmaps), so no later diagnostic needs a re-capture (PART 6).
12. **No session boundary inside a compared cell.** Every row carries `session_id` (one per
    runner process). A cell is scored only if all of its rows share one `session_id`; an
    interrupted cell is re-run in full, and partial rows are excluded from scoring by the
    analysis script. Rationale: Paper 3's power-loss boundary, and Amendment 1.
13. The U-arms' scoring function is untouched: `score()` is called exactly once per layer and its
    tensor is not modified — asserted per row, as in Stage 2.

## 6. The c = 1 control — pass/fail

At `c = 1` (MARK-1) a queried fact is a single token, so **unit-awareness must be inert**.

- **Test:** for every admissible (non-degenerate) c=1 cell, both models, all four methods:
  `Δ_U` with its 95% CI.
- **PASS:** no admissible c=1 pair has a CI excluding zero, in **either** direction.
- **FAIL → STOP AND DIAGNOSE.** If any does, Stage 3 halts: no further analysis is reported and
  no further runs are launched until the cause is found. A U-arm that beats its X-arm where a
  unit is one token is doing something other than what the wrapper claims.
- **Additional hygiene check (prefill only, before the grid):** the Stage 1 singleton-unit
  identity — with every unit forced to one token, `U-X` must reproduce `X`'s keep-sets up to
  exact score ties — re-run once per (model, budget, method). A violation is also a STOP.

## 7. Kill criteria — registered now, before any grid row exists

- **K1 — c=1 inertness.** §6 FAIL. Halts the stage.
- **K2 — the fragmentation account.** Stage 3's central prediction is that the fragmentation
  effect grows with fact cost. **Kill: if `mean Δ_U` at c=40 is not greater than at c=1, pooled
  over admissible budgets, on both models**, then "fragmentation removal explains Paper 3's
  cost trend" fails as stated. It is then reported as failed. The wrapper is not modified, and
  the grid is not re-cut, to rescue it.
- **K3 — sufficiency.** Stage 2 concluded that removing fragmentation is not sufficient to
  reach the positional floor. **Kill: if `A_{U-X} >= A_floor` in a majority of admissible cells
  with c >= 19 on both models**, that conclusion is wrong and must be restated, not softened.
- **K4 — hygiene.** Any row failing an assertion in §5 invalidates its cell until re-run; if
  more than 1% of attempted rows fail, the stage stops and the cause is reported.
- **K5 — ceiling sanity.** `oracle_causal` is the numerator ceiling and should not fall below the
  zero-cost floor once facts cost more than one token. **Kill: if `A_oracle_causal < A_floor` in
  more than 25% of admissible cells with c >= 8**, the ceiling arm is mis-specified for this grid
  and no `ρ_floor` table is reported until that is resolved.

## 8. Predictions — committed before any Stage 3 row is scored

- **P1.** `Δ_U > 0` with CI excluding zero at c=40 for SnapKV and AdaKV on M2 at C in {128, 256,
  512} — at least 4 of those 6 pairs.
- **P2.** `Δ_U` increases with `c`: `Δ_U(c=40) > Δ_U(c=19) > Δ_U(c=8)`, and `Δ_U(c=1) ≈ 0`, on
  both models (direction only, pooled over admissible budgets).
- **P3.** `ρ_floor(U-X) > ρ_floor(X)` in >= 80% of admissible c >= 19 cells, **and**
  `ρ_floor(U-X) < 1` in >= 80% of them — the wrapper narrows the position gap without closing it.
- **P4.** The Stage 2 conversion deficit replicates: `V_slot(U-X) < V_slot(floor_pos)` in a
  majority of admissible c >= 19 cells, on both models. Reported alongside `V_any` and `V_maj`,
  with no claim that any one of them is the usability measure.
- **P5.** At c=1, `ρ_floor(X) ≈ ρ_floor(U-X)`, both in the 2.5–3.5 band Paper 3 and Stage 2
  measured at C=512, with `Δ_U ≈ 0` (this is K1's quantity, stated here as a prediction too).

Scoring order is fixed: the analysis script is committed before it is run on Stage 3 rows, and
predictions are scored exactly as written above — including the ones that fail.

## 9. Out of scope for Stage 3

- Structural and inferred units (later ladder stages).
- Any re-test of **dispersion** or **distractor load** as explanations of the conversion gap:
  both are closed, preregistered nulls in this task family (PART 5 item 5). A fifth hypothesis
  must be genuinely new.
- The head/layer lead in `LIMITATIONS_P4.md` — scoping only, no Stage 3 data.
- Treating the completion≠usability gap as solved, or designating per-slot completion as the
  usability measure anywhere in the analysis.

## 10. Amendments (made BEFORE any Stage 3 row existed)

Both amendments below were made while the grid was still empty: no generation row, no capture
row, no analysis. Neither changes the grid, the quantities, the kill criteria or the
predictions. The pre-amendment text hashed to
`5339d9b623fae92be8774249931cf848024022e2a330768bc224f4b2f1c3ef8a` (commit `3a44853`); the
digest recorded in `PREREG_P4_S3.md.sha256` after this section is the one the runner enforces.

### A1 — `oracle_prescient` is deliberately excluded from Stage 3

`oracle_prescient` (the arm that knows *which* fact is queried) is **not** in the §2 arm list, and
its absence is a decision, not an oversight.

- Paper 2 needed the prescient/causal split to define its information share
  `I = (A_presc − A_causal) / (A_presc − A_floor)` — the value of knowing the query.
- **Paper 4's analogue of `I` is a different quantity: the unit-source gap**, oracle units vs
  structural units at matched budget (PART 2's ladder; PART 5 item 2). It prices knowing where
  the *fact boundaries* are, which is exactly what this paper's method consumes — not knowing
  which fact is asked. Structural units are a later stage, measured against the oracle-unit
  ceiling this stage establishes.
- Paper 4 has never run `oracle_prescient`: not in the Stage 2 pilot, not in the verify package,
  not in the dispersion probe. Introducing it here would add an arm that no Paper 4 quantity is
  defined in terms of, at 10% of the grid's cost.
- The ceiling arm Stage 3 does carry is `oracle_causal`, which is what `ρ_floor` is read against
  and what K5 sanity-checks.

Paper 3's `oracle_prescient` numbers remain available for context; no Stage 3 quantity is defined
in terms of them, and none is compared across that session boundary.

### A2 — the instance seed excludes `torch_version`; the dedup key records it

§4 adds `torch_version` to the dedup key (PART 4 requires it; Stage 1–2 had it absent, hashing as
null). Taken literally through `keys3.instance_seed`, that would also change the CRC32 **instance
seed**, and therefore the generated instances themselves — which would silently break §2's
"instances `s4_00000`–`s4_00049`, same seeds as Paper 3 Stage 4" and make Stage 3's task content
depend on a toolchain version.

Registered reading, fixing that conflict in one direction only:

> **`torch_version` is nulled in the seed probe**, exactly as `arm`, `B`, `C`, `max_new` and
> `matched_to` already are, so the instance seed is stable across toolchain upgrades and Stage 3
> sees byte-identical instances to Stage 2 and Paper 3 Stage 4. **The row's dedup key still
> records `torch_version`**, so a toolchain change still invalidates a *record* without silently
> changing the *task*.

Verified mechanically before the grid runs (`stage3_preflight.py`): for every (task, n_fields)
in the grid, the rebuilt instance seeds must equal the seeds stored in Paper 3's Stage 4 rows and
Paper 4's Stage 2 rows for the same instance ids. A mismatch is a STOP.

### A3 — the inertness test is re-specified; MARK-1 c=1 is demoted to a diagnostic; the central claim is scope-restricted

Made after the c=1 sweep (5,000 rows, 0 failures, 20 complete single-session cells) and **before
any c>1 row exists**. The pre-A3 text hashed to
`131278082f3ee35593929ff35be3a7a85b32e1f0ca5a8075eec840e70c85838c` (commit `e9a85fb`), which this
amendment supersedes. Evidence and full diagnosis: `DIAGNOSIS_K1.md` (commit `e660cf3`); the c=1
control table as collected under the superseded rule is `out/stage3_control.txt`. The wrapper was
not modified at any point.

**A3.1 — the registered inertness test.** The registered test of unit-awareness inertness is now
the **singleton-unit identity**: with every unit forced to a single token, `U-X` must reproduce
`X`'s keep-set up to exact score ties. It was verified on real scores, **40/40** (2 models × 5
budgets × 4 methods), in this session's preflight (`out/stage3_preflight_identity.json`).

The prior premise — *"at c=1 a queried fact is a single token, so unit-awareness must be inert"*
(§6 as originally frozen) — **is retracted as stated.** It held for the queried fact only. MARK-1's
36 distractor entries are multi-token surnames, so units exist at c=1 and unit-aware allocation
still differs from top-k. `STAGE12_RULES.md` §2 stated this correctly during Stage 2 ("c=1 is a
control rather than an identity"); §6 over-claimed it into inertness, and that over-claim, not the
wrapper, is what K1 detected.

**A3.2 — MARK-1 c=1 becomes a reported diagnostic, not a pass/fail gate.** §6's PASS/FAIL and §7's
K1 no longer gate the stage. The existing c=1 rows **stand as collected and are not regenerated**;
they are now scored diagnostically. Both measured channels are reported per method:

- **Δ p_g — score-variance-driven gold-retention shift.** A multi-token unit competes on its mean
  score under `U-X` but on its best tokens under `X`, so budget freed by a skipped unit can land on
  the gold token. Material for **KeyDiff and ExpectedAttention (+0.008 to +0.021)**; ≈ 0 for
  **SnapKV and AdaKV (±0.001)**.
- **Δ units_complete — context-structure reallocation.** `U-X` completes more whole distractor
  entries at equal budget (up to **+5.0** at M2 C=512, where Δ p_g ≈ 0 — that cell's effect is not
  gold retention at all).

**A3.3 — scope restriction on the central claim, to be reported front-and-centre.**

> **"`U-X` vs `X` isolates fragmentation" holds cleanly for SnapKV and AdaKV. For KeyDiff and
> ExpectedAttention it does not: unit-aware allocation also shifts which individual tokens are
> retained, through score-variance effects at the unit level. For those two methods the wrapper's
> effect is not fragmentation-removal alone.**

This limitation is reported in the writeup at the same prominence as Paper 3's F5 kill-criterion
disclosure — in the main text, with the result it qualifies, **not** in an appendix or a footnote.
Stage 3's fragmentation claim rests on SnapKV and AdaKV; KeyDiff and ExpectedAttention are reported
with the confound named wherever they appear.

**A3.4 — what did not change.** The grid, `n`, the arm list, every quantity in §3, the seeding and
dedup key in §4, the invariants in §5, kill criteria K2–K5 and predictions P1–P5 are unchanged.
K1's identifier is retained and now refers to A3.1's singleton-unit identity, whose violation
remains a STOP. **Watch condition, registered now:** if at c ≥ 8 a KeyDiff or ExpectedAttention
pair shows a CI excluding zero in the positive direction, that is flagged immediately rather than
at the next scheduled report — it would indicate the confound is not confined to c=1, which A3.3
assumes it is.

### A4 — the wrapper gathers in score order; the bf16 accumulation-order confound and the re-run scope

Made after the c=1/c=8 cells and the tier-1 probes, **before any claim-bearing cell is re-run**.
The pre-A4 text hashed to `34a20a7ef35c1510067359a76d05de10f32d912b9058a1c94ac9187e9d8e7a3f`
(commit `e9a85fb`), which this amendment supersedes. Cause and evidence: `DIAGNOSIS_A6.md`.

**The defect.** `kvpress`'s `ScorerPress.compress` gathers the retained keys/values in **top-k
score order**; `p4/unitwrap.py` gathered them in **ascending position order**. The retained SET was
identical -- measured 20/20 instances on both models -- but the cache write order differed. With
RoPE applied before compression, attention is permutation-invariant in exact arithmetic, so this
cannot change attended content; **bf16 accumulation over the key axis in the sdpa kernel is not
order-invariant**, and the perturbation flips greedy decodes wherever the top-2 logits are nearly
tied. Measured with identical keep-sets: generations byte-identical on only 10/20 (M2) and 11/20
(M3) at n=20, and 3/10 (M2) and 5/10 (M3) in a dedicated re-run. Generation itself is exactly
reproducible (repeat runs 10/10 identical on every arm, both models), so this is **a deterministic
order effect, not nondeterminism**. It is the same class as Paper 2's batch-order disagreement
(83/100) and Paper 3's session-boundary tie flips.

**Why it matters:** every `X` vs `U-X` contrast in Stages 2-3 therefore differed in cache order as
well as in allocation. The order component alone measures **+0.025 [+0.000, +0.062] on M2** and
**+0.000 [+0.000, +0.000] on M3** at c=8, C=512 -- on M2 the same order of magnitude as Stage 2's
headline `U-X - X` = +0.045.

**The change.** `p4/unitwrap.py` (and the probe variants in `p4/probes.py`) now gather in
**descending score order**, matching `ScorerPress`. The keep-set, budget, floors and the score
function are untouched; only the write order changes, so `X` vs `U-X` now differs in allocation
alone. AdaKV arms are unaffected (they mask rather than gather). Rows produced after this
amendment carry `gather: "score"`; rows without that field were produced under ascending order.

**Re-run scope, fixed here.** Claim-bearing cells are re-run under the fixed wrapper, one at a
time, the Stage 2 pilot cell first and alone because every later decision depends on it:

1. c=40, C=512, both models (the headline `U-X - X` comparison);
2. then, only if the rule in `DECISION_C40.md` says PROCEED: M2 c=8 C=256 and C=512, and the c=1
   diagnostic cells.

**All other completed cells are QUARANTINED, not re-run**: the M2 and M3 c=1 sweeps beyond the
diagnostic use, M2 c=8 C=32/64/128 (degenerate), and M2 c=19 C=32/64 (degenerate). Their rows stay
on disk and stay in the record, marked as produced under ascending gather, and **no `X` vs `U-X`
contrast from them is reported as isolating allocation**. Degenerate cells carry no claim anyway,
which is why re-running them buys nothing.

**Unchanged:** the grid definition, `n`, the arm list, every quantity in section 3, seeding and the
dedup key in section 4, invariants in section 5, K2-K5 and P1-P5. K1 remains the singleton-unit
identity of A3.1.
