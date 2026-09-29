# PAPER 4 — CONTEXT FILE (v3)

**CRITICAL — READ FIRST:** Save this file to `D:\INNOCREW\Blockage\paper4\PAPER4_CONTEXT_AND_PLAN.md`
before starting any session that needs it. Two prior versions of this file were written in
chat and never reached the machine — the session searched the whole drive, couldn't find
either, and had to reconstruct everything from restated invariants. Do not repeat that.
Download this artifact and place it at that exact path before pasting any prompt that
references it.

---

## PART 0 — WHY THIS FILE EXISTS

This is a four-paper research program on KV-cache retention, run across an RTX 5070 (12GB,
primary), an RX 7900 XTX (24GB, secondary/verification), and one A40 rental (14B scale
replication, now terminated). Papers 1, 2 and 3 are functionally complete — experiments
done, drafts revised, venues chosen. Paper 4 is the one still running. This file exists so
a fresh Claude Code session on the 5070 can pick up Paper 4 exactly where it left off
without re-deriving context that already exists.

**The governing discipline across all four papers, inherited by Paper 4 without exception:**
preregister before running, gate before spending GPU, commit predictions before scoring
them, report negative and null results as plainly as positive ones, never let a session
guess at a missing file — stop and ask. Every paper in this program has caught at least one
real defect (mislabelled estimator, floor definition ambiguity found twice independently, a
stale narrative contradicting its own data) specifically because this discipline was
followed. Do not relax it for Paper 4 because it feels like "just a pilot."

---

## PART 1 — WHAT PAPERS 1, 2, 3 ESTABLISHED (context only, not to be re-run)

### Paper 1 — "Retention and Promotion" (target: MLSys 2027)

Credential-retrieval task, 6 credentials + 20 distractors, quantization-tiered KV cache.
**Core finding:** promotion (choosing which evicted tokens to restore to full precision) is
nearly saturated — protection-alone accuracy 0.423 vs retention-oracle 0.943 at retention
0.25, a 0.52 gap. Promotion signals (attention, epiphany, random, rotation, oracle) show no
meaningful separation in the primary condition. Retention is where the headroom lives, not
promotion.

**Recent corrections (all closed):**
- Found via verification: the reported "interaction" throughout the draft was actually the
  simple effect `arm4−arm2`, not the preregistered difference-in-differences
  `(arm4−arm3)−(arm2−arm1)`. Relabelled; the true DiD has wider CIs and two of three 4-bit
  points now include zero. Recorded as a PREREG deviation, not hidden.
- Random-span control (the "single most valuable unrun experiment"): run and positive.
  Structural protection beats random by +0.17 to +0.89 across budgets; random's own small
  gain over nothing is explained by incidental token overlap. Confirms protection finds
  real signal, not reserved capacity.
- Natural-text validation: retention gap replicates almost exactly on a new ~2048-token
  natural-text dataset (+0.52 at matched retention, same as the credential task).
- H-ORTH cross-implementation disagreement (with AMD's independent implementation): a 2×2
  factorial isolated the cause to **seating-rule implementation, not distractor format** —
  one cell confirmed on RTX, reciprocal cell pending on AMD as of last check.
- `ANALYSIS.md` re-hashed after LF-normalisation; new digest `10575a55…`, documented as a
  deliberate one-time exception to the no-renormalise rule (this file is not a frozen prereg
  artifact, unlike the Papers 2/3 preregs).

**Status:** essentially done pending AMD's final H-ORTH cell. Not blocking Paper 4.

### Paper 2 — "Two Ceilings" (target: TMLR)

Six-arm reference ladder (`full_cache`, `null`, `random`, `floor_pos`, `oracle_causal`,
`oracle_prescient`) on LEDGER (one-hop, N=40 records, H=4 queried, one-line records with an
id, surname, department, 6-digit value). Two headline quantities:

```
G_m = (A_m − A_floor) / (A_causal − A_floor)        capture ratio
I   = (A_presc − A_causal) / (A_presc − A_floor)    information share
```

**Findings:** `G_m ≤ 0` in 48 of 50 method-cells (methods at or below a zero-cost positional
floor). `I` = 0.73–0.75 at the tightest binding budget, structurally zero once budget
exceeds the candidate set's total payable cost. `Δ_head` (head-wise allocation relaxation)
negative at every squeezed budget on all three models (M2 2 KV heads, M3 8, 14B 8) —
per-head allocation, in isolation, *hurts*.

**Recent work (all closed):**
- Floor sensitivity (T1): `floor_pos` has two incompatible readings in the frozen prereg
  (total-C vs total-B, differ by 72 tokens at C=512). Sign of `G_m` is reading-invariant at
  C≥128 (1/27 flips, not significant); diverges only at C≤64 where the literal reading
  starves the floor below its own mandatory minimum. `I` survives both readings within 0.03.
- Natural-text external validation (T6): built a new dataset (real Gutenberg prose + inserted
  gold-annotated fact sentences). Gate passed first attempt. Confirms `G_m`/`I` findings
  outside synthetic data. No crossover observable (natural-text min cost c≈17, no c=1
  analogue).
- Blind reimplementation (T2/T3/T4, "B6" on AMD): independent mechanism (explicit attention
  mask, not kvpress hooks). Zero ordering-invariant violations. Independently reconfirmed the
  `floor_pos` double-definition (two blind attempts, same defect). New finding: dropping the
  BOS/sink token collapses every arm — including both oracles — to 0.0000. This is now a
  stated validity invariant.
- Citations verified, title changed from "irreducibly predictive" framing (overclaimed) to
  one scoped to what `I` actually measures.

**Status:** drafted, revised, submission-ready for TMLR.

### Paper 3 — "The Functional Unit of KV Retention" (target: ACL Rolling Review Oct 2026 → NAACL Findings)

**This is the paper Paper 4 is built directly on top of.** Central finding: the method-to-
position accuracy ratio declines monotonically with fact cost `c` (tokens a fact requires),
in 24/24 admissible method-budget sequences. At the largest budget: ~2.7–3.1× above floor at
c=1 (single-token facts), falling to 0.01–0.43× at c≈40 (multi-field records) — **methods
that beat position on cheap facts lose to it on expensive ones.**

**Mechanism — coherent packing helps:** an oracle-informed intervention that packs whole
facts (using annotated candidate spans) instead of letting a method fragment its budget
across many partial facts raises accuracy 2.6× (0.125→0.325 on M2/SnapKV), all four paired
CIs excluding zero. This is the direct ancestor of Paper 4's wrapper.

**The scaling account fails its own test:** a preregistered Gaussian threshold model (F5)
won the completion-prediction comparison (lowest error) and then failed its own kill
criterion — its residual correlates with log-budget at 0.917–0.948, well above the 0.5
threshold fixed in advance. Reported as a genuine self-falsification, not hidden.

**The open question Paper 4 inherits directly:** *completion is not usability.* A fact can
be fully retained (`q=1`, complete keep-set) and still not produce a correct answer. Four
hypotheses tried and failed to explain this:
1. Union completeness overcalls usability (fixed in Paper 3 by moving to per-slot measures)
2. Per-slot completeness itself doesn't fully predict accuracy either
3. Distractor load — tested, not supported
4. **Positional dispersion — tested in Paper 4's own pilot, not supported (see Part 3)**

**Recent revisions (all closed):**
- Stage 2 narrative/output reconciliation: raw rows are authoritative; a stale narrative
  transcription was corrected. Retroactively applied the preregistered admissibility mask
  (F5 error 0.147→0.207, residual 0.948→0.917 — these masked numbers are now primary).
- **ChunkKV tested and excluded (not merely "provisional").** Passed G3 (oracle overlap,
  increasing with C on both models) but **failed G2 (permutation check)** on both models: at
  C=32 the chunk-sized floor consumes the entire budget so permuted and original keep-sets
  are identical — the scoring signal is structurally undetectable at tight budgets, which is
  itself consistent with the paper's central mechanism (fixed-size units interact badly with
  budget constraints). G1 not attempted (its published eval uses 7-8B models, doesn't fit
  either GPU; G2 already excludes it regardless). Framed as informative exclusion, not gap.
- Blind reimplementation (same B6 run as Paper 2, verifies the same ladder Paper 3's curve
  depends on).
- Natural-text extension (shared dataset with Paper 2): confirms the reversal's high-cost
  end only.

**Status:** drafted, revised for ARR Oct 2026 / NAACL Findings framing.

---

## PART 2 — WHAT PAPER 4 IS

**Positioning, decided explicitly:** the wrapper alone is *not* a top-venue contribution —
"aggregate scores over units, allocate whole units" is a two-line idea, and comparators like
ChunkKV already motivate chunking. **The contribution is the curve and the diagnosis**:
method-vs-position advantage collapses with fact size (already Paper 3's finding), the
field's benchmarks implicitly sit at some point on that curve, and the wrapper is the
constructive proof that fragmentation is a real, removable cause — used as an instrument,
not as the product.

**Target: MLSys 2027, deadline 30 Oct 2026.**

**The method:** `U-X`, a wrapper around any existing kvpress press. Aggregates the press's
per-token scores over *units* (in the pilot, oracle units = true fact boundaries from
LEDGER-C), then allocates budget by whole units, all-or-nothing, greedy by score-per-token.
The scoring function is untouched — only allocation changes — so `X` vs `U-X` isolates
fragmentation as the causal mechanism, cleanly.

**Unit sources, planned as a ladder (oracle → structural → inferred), not yet built beyond
oracle:**
- Oracle units (pilot, done) — true fact boundaries, not deployable, establishes ceiling
- Structural units (not built) — delimiters/newlines/sentence boundaries, zero-cost,
  deployable
- Inferred units (not built) — clustering in key space, general fallback

**Benchmark characterisation (not yet run, cheap, high-value):** measure the implicit `c` of
LongBench, RULER, SCBench, Needle-in-a-Haystack by ablating gold tokens and finding the
minimal set that preserves a correct answer. If these cluster below Paper 3's crossover,
this is the single table that turns the finding into a claim about the field, not about
LEDGER.

---

## PART 3 — WHAT HAS ACTUALLY RUN (Stages 0–2, complete)

### Stage 1 — wrapper built, verified

`U-SnapKV`, `U-AdaKV`, `U-ExpectedAttention`, `U-KeyDiff` implemented as wrappers around the
existing kvpress presses. Identity check: with every unit a single token, `U-X` gives
identical keep-sets to `X` up to exact score ties (32/32 real-score checks). Direction
confirmed: at c=40, `U-X` touches far fewer records and completes far more than `X`.

**Known gap in the no-GPU verification plan:** Paper 3's stored captures hold only
per-instance summary statistics (`touched`, `complete`, `q_complete`, `q_any`, `p_g`), not
raw keep-sets. So the originally-planned "verify on Paper 3's existing captures with no GPU"
step had to be split into (a) a CPU-only synthetic-geometry check (passed, matched all 200
stored instances) and (b) a prefill-only real-score check that does need the GPU briefly
(this became part of the Stage 2 verify gate).

**One real defect found and fixed during Stage 1 verification (Amendment 1, committed):**
M3's re-captured keep-sets did not exactly match Paper 3's stored captures (0/400 rows;
M2 matched 400/400). Root-caused, not assumed: a fresh-process re-capture was deterministic
on both models (40/40), so this is Paper 3's own documented pre-power-loss session effect,
not a Paper 4 wrapper bug. Amended and committed before any generation ran.

### Stage 2 — the pilot, gate PASSED

LEDGER-C, c≈40, C=512, n=50, both models. Arms: `floor_pos`, `snapkv`, `adakv_snapkv`,
`U-snapkv`, `U-adakv_snapkv`. Plus the c=1 control.

**Gate result: PASS.** `U-X` beats `X` at c=40 with a CI excluding zero on 3 of 4 pairs
(M2 SnapKV +0.045 [+0.015,+0.075]; M2 AdaKV +0.045 [+0.010,+0.080]; M3 SnapKV +0.000
[−0.025,+0.025], not significant; M3 AdaKV +0.035 [+0.005,+0.070]). The c=1 control shows
**no effect** (all four intervals include zero; M3 SnapKV identical on every instance) —
exactly as required, since a unit is one token there and unit-awareness must be inert.

**The result is more informative than a clean win would have been:**

| | at c=40, C=512 |
|---|---|
| `U-X` vs `X` | +0.045 typical, small (7–9 more correct answers per 200 queries) |
| `U-X` vs `floor_pos` | still loses, 1.8–3.3× depending on model/method (down from 3.3–5.0× for raw `X`) |
| completion (queried facts complete) | rises 0.07 → 0.23–0.28, near the floor's 0.29–0.31 |
| conversion (accuracy ÷ completion) | M2: 0.40 for `U-X` vs 0.60 for floor; M3: 0.12–0.20 vs 0.33 |

**Fragmentation is causally real and `U-X` removes it — but removing it is not sufficient.**
`U-X` assembles whole facts at nearly the floor's rate, and still converts them to correct
answers far less often. This is a sharper, *constructed* version of Paper 3's open question:
not "does completeness predict usability" (observational, four failed hypotheses already),
but "two arms hold comparably complete facts and diverge in conversion — why."

### Stage 2.5 — the dispersion probe (registered before looking, NOT SUPPORTED)

Following the reframe above, a fifth hypothesis was tested with the same discipline as
Paper 3's distractor-load null: **decision rule and files committed before any keep-set was
inspected** (`DISPERSION_RULE.md`, commit `ab0b92f`).

**What was tested:** does positional dispersion of the retained set (contiguous-run
structure, or isolation of the queried fact) predict conversion, within-arm, controlling for
completion?

**Keep-sets were not free** (same summary-only limitation as above) — rebuilt via a
prefill-only re-capture, validated exactly against the pilot's own generated rows (500/500
match) before any dispersion measure was computed.

**Result: NOT SUPPORTED, one measure mildly contradicted.**

| model | measure | n (correct answers) | effect/SD [95% CI] | verdict |
|---|---|---|---|---|
| M2 | contiguous runs (primary) | 54 | −0.099 [−0.253,+0.030] | not supported |
| M2 | isolation of the fact | 54 | +0.052 [+0.016,+0.090] | **contradicted** (more isolated → answered more, opposite of predicted direction) |
| M3 | contiguous runs (primary) | 27 | +0.013 [−0.272,+0.300] | not supported |
| M3 | isolation of the fact | 26 | +0.010 [−0.014,+0.033] | not supported |

Power is low (26–54 correct answers per cell) — this is weak evidence of absence, not a firm
null, and is reported as such.

**A second, sharper correction to the working hypothesis, from descriptive (not
pre-registered) evidence:** the raw `X` arms are the *most* dispersed of all arms (65–78
contiguous runs vs 20–29 for `U-X` and 1 for the floor) — **and convert as well as or better
than the floor** (M2: 0.67–0.68 vs 0.60; M3 SnapKV: 0.45 vs 0.33). So the true statement is
narrower than first framed: **the conversion deficit belongs to `U-X` specifically, not to
"methods" or "dispersion" in general.** Unit-aware allocation raised completion 3.4× and
accuracy only 2×, and this is not explained by dispersion.

**An untested, recorded-but-not-pursued lead**, now sitting in `LIMITATIONS_P4.md`
specifically so Stage 3 doesn't default to per-slot completion as the usability measure:
`X` keeps fragments that, unioned across attention heads, cover a fact in 19–72% of
instances; `U-X` makes the fact whole in ~a quarter of slots but the union across heads
approaches 1.0. **Which layers/heads hold the fact may matter more than how many do.** This
is a live hypothesis, not a result — flag it to whoever picks up Stage 6+, do not assume it.

**Fourth failed hypothesis for the completion≠usability gap, added to the running list:**
dispersion (Paper 4) joins union-completeness, per-slot-completeness, and distractor-load
(all Paper 3) as tested-and-not-supported.

---

## PART 4 — HARNESS INVARIANTS (apply in full, inherited from Papers 1–3)

Every one of these exists because of an actual defect that produced a plausible, wrong
result at some point in this program. Do not treat any as optional.

- `position_ids` continue from the **uncompressed** context length, asserted not assumed
  (bug: renumbering the compressed cache corrupted RoPE geometry and zeroed every compressed
  arm in Paper 2's original ladder)
- Every press check **generates tokens** — never trust prefill logits or cache length alone
  (bug: `get_seq_length()` does not shrink for head-wise presses; prefill logits are
  identical at every compression ratio for some presses)
- Mandatory sink(8) + window(64) floors honoured by **all** arms including method arms
  (bug: methods were evicting the sink/window, competing at a different effective budget
  than the floor — measured sink retention as low as 0.413 against a required 1.000)
- Realised budget parity asserted **per instance from captured keep-sets**, never from the
  requested compression ratio
- Degenerate-cell rule: floor accuracy < 0.05 → cell excluded from ratio tables (mechanical
  rule in the analysis script, not a post-hoc judgement call)
- Seeds by **CRC32 of the record key**, never Python `hash()` (salted per process,
  irreproducible)
- Device recorded in the dedup key; **no number crosses a device boundary into a ratio** —
  RTX 5070, RX 7900 XTX, and the terminated A40 rental are three separate strata
- **Single process** on the primary machine; never pipe a runner through another command
  (a `grep` pipe once masked a fatal assertion failure and silently produced an incomplete
  run that looked finished)
- **Commit hypotheses and decision rules before looking at data** — every dispersion-style
  probe in this program (Paper 3's distractor-load null, Paper 4's dispersion probe) follows
  this exact pattern: write the rule and commit it, *then* run the analysis.
- **Batching untested/unverified must not be assumed safe** — Paper 2 found generated-token
  agreement against batch-1 as low as 83/100 with wholly different values on mismatches;
  `batch_size` stays in the dedup key.

---

## PART 5 — WHAT'S NEXT (Stage 3 onward, none of this has started)

**Immediate open items, in likely priority order:**

1. **Freeze the Stage 3 preregistration** for the main grid — c-sweep and budget-sweep,
   both models, oracle units first (the deployable unit sources come after oracle units are
   fully characterised). This has not been drafted yet as of the last session.
2. **Build structural units** (delimiters/sentence boundaries) — the actually-deployable unit
   source. Oracle units establish a ceiling; structural units are what a real system would
   use. The gap between them is the analogue of Paper 2's information share `I`.
3. **Benchmark characterisation** — measure implicit `c` on LongBench/RULER/SCBench/NIAH.
   This is cheap, no new task construction, and is the single highest-value remaining item
   for making the paper a claim about the field.
4. **The head-allocation lead from the dispersion probe** — untested, flagged, not yet
   pursued. Decide whether it's worth a Stage 6-style probe or stays as a limitations note.
5. **Do not re-attempt dispersion or distractor-load as explanations for the conversion gap**
   — both have now failed, under proper pre-registered discipline, in this exact task
   family. If a fifth hypothesis is proposed, it must be genuinely new, not a variant of
   these two.

**What must NOT happen:** treating the completion≠usability gap as solved, or defaulting to
per-slot completion as "the" usability measure in any Stage 3+ analysis — `LIMITATIONS_P4.md`
exists specifically to prevent this.

---

## PART 6 — FILE / REPO STATE (as of last session)

- Repo root: `D:\INNOCREW\Blockage\paper4\`
- Latest commit at last check: `b30d65d`
- Key files: `STAGE12_RULES.md` (Stage 1/2 procedure + Amendment 1), `DISPERSION_RULE.md`
  (pre-registered dispersion test, committed at `ab0b92f`), `LIMITATIONS_P4.md` (dispersion
  null + the head/layer lead, flagged not pursued), `out/stage2_pilot.txt` (full pilot
  report)
- Keep-sets are now saved with every generation run going forward
  (`runs/nvidia/p4_keepsets_c40_*.npz`) — this was not true for Stage 1/2's first pass and
  cost real time to work around; do not regress on this.
- This file (`PAPER4_CONTEXT_AND_PLAN.md`) has never successfully reached the machine before
  this version. Verify it is actually present at the stated path before starting Stage 3.
