# Preregistration — the spacing hypothesis (Track 2 follow-up)

**Committed before any spacing-constrained row is generated.** Source: `TRACK2_FAILURE_READ.md`
(commit `a500cb9`) and `TRACK2_SELECTION_CHECK.md` (commit `c122dab`). CPU-only work to date;
this document authorizes the first GPU generation for this hypothesis, not before.

## Scope, stated up front

**This test addresses M2's dominant failure mode (contamination, 17% of its floor-correct/U-wrong
population) and M3's minority contamination cases (5% of its population, roughly a third of M2's
rate). It does not address M3's dominant failure mode (refusal, 31%), which is a separate,
unexplained mechanism and is explicitly out of scope here.** A result on this hypothesis, positive
or negative, says nothing about M3's refusal behavior. Both are reported separately; neither is
allowed to stand in for the other in any summary.

## The claim, as one falsifiable sentence

> Accuracy on complete-but-wrong cases improves when complete records adjacent in the kept
> sequence are separated by at least K non-record tokens (or other constraints), relative to
> unconstrained packing, holding total budget and completion rate fixed.

## Operationalization

**K, fixed before any data is generated: K = 24 tokens.** Chosen as roughly half the mean filler
sentence length in this corpus (`_MENTION_TEMPLATES`, ~30-45 tokens rendered), so it is large
enough to be a real structural change and small enough to be payable inside the existing budget
without gutting completion. Not tuned against any result.

**Spacer-constrained allocator (`spaced-U-X`).** Same wrapper (`p4/unitwrap.py`), same score
function, same budget `C`, same all-or-nothing whole-unit greedy — with one added constraint: a
candidate unit is skipped if accepting it would leave fewer than K non-unit (filler or floor)
tokens between it and the nearest already-chosen unit in **final kept-sequence order** (not source
order), unless skipping it would leave the remaining budget unable to reach the matched completion
rate (see below), in which case the constraint relaxes for that one unit and the relaxation is
logged per instance. This is a new allocator variant, added to `p4/probes.py`, not a change to the
`U-X` arms already in the frozen Stage 3 grid.

**"Holding total budget and completion rate fixed."** Same `C` as the comparator cell (no
budget change). Completion rate is matched **after the fact, not before**: `spaced-U-X` is run at
the same `C`, and the analysis conditions on the subset of instances where its per-instance
`units_complete` is within +/-1 unit of unconstrained `U-X`'s, on the same instance. Instances
outside that band are reported as a separate count (how many, which direction), not discarded
silently. If fewer than 20 instances per model fall inside the band, the comparison is NOT SCORED
for lack of a matched sample, and that is reported as a finding about the operationalization, not
worked around.

## Comparator and cells

- **Arms:** `U-snapkv` (unconstrained, existing Stage 3 data) vs `spaced-U-snapkv` (new). SnapKV
  only for this first test — it carries M2's cleanest contamination signal (6/12 sampled cases,
  and 17% at the full-population level) and has no active confound caveat at this cell (Δp_g
  negative, per `PILOT_C40_POSTFIX.md`).
- **Cell:** c=40, C=512, both models, n=50 (matching the Stage 3 pilot cell so the comparator is
  the exact data already collected, not a fresh unconstrained run).
- **Gather order:** score order (post-A4), so this inherits none of the A4 confound.

## The measure that matters: contamination-tagged accuracy, not overall accuracy

Per the standing instruction, overall accuracy is NOT the primary readout — it is diluted by
hallucination, refusal and no-pattern cases that spacing cannot touch by construction. The primary
readout is:

> **Contamination-tagged accuracy**: among instances where unconstrained `U-snapkv` produced a
> contamination-pattern error (per `TRACK2_FAILURE_READ.md`'s rule: wrong answer matches a real
> record within record-distance <=2 at field-overlap >=6, applied to the FULL disagreement
> population per instance, not the 24-case sample), does `spaced-U-snapkv` answer the SAME queries
> correctly?

This is a **paired, per-query** comparison: for every query where unconstrained `U-snapkv` shows
the contamination tag, score `spaced-U-snapkv` on the identical query (same instance, same fact),
correct or not. Overall accuracy is reported alongside, descriptively, never as the decision
quantity.

## Decision rule

- **Precondition (checked first, before scoring):** >= 20 contamination-tagged queries available
  per model at this cell (population size, not sample) after generating `spaced-U-snapkv`. Given
  M2's full population showed 14/82 contamination cases at n=50, and roughly 4 queries per
  instance, this is expected to clear on M2; M3's 3/59 full-population contamination rate is much
  smaller and may not reach 20 -- **if M3 fails this precondition, M3 is reported as NOT SCORED for
  this test, not forced into a comparison with insufficient power**, and the test proceeds on M2
  alone with that limitation stated.
- **REAL** (per model, independently -- this is NOT gated on both models the way DECISION_C40 was,
  because the scope statement above already concedes M3 may be underpowered for its own reason):
  contamination-tagged accuracy under `spaced-U-snapkv` minus under unconstrained `U-snapkv`, paired
  per query, 95% bootstrap CI (10,000 resamples, instances as the resampling unit, CRC32-seeded)
  excludes zero on the positive side.
- **NOT-IT**: CI includes zero, or negative.
- **Degenerate/confound check, mandatory before interpreting REAL:** report Δp_g and Δunits_complete
  for `spaced-U-snapkv` vs `U-snapkv` exactly as done for every prior pair in this stage --if
  `spaced-U-snapkv` retains MORE gold tokens than `U-snapkv` (not just re-arranges them), a positive
  result would be confounded with retention, not spacing, and must be reported as such rather than
  claimed as a clean spacing effect.

## Kill criterion

If the completeness-matching precondition fails on BOTH models (fewer than 20 matched-band
instances each), the test is reported as **inconclusive by construction** -- the operationalization
of "holding completion fixed" does not work at this cell, and K or the matching band would need to
change before re-attempting. This is a different outcome from NOT-IT and must not be conflated
with it in any summary.

## Cost

New arm `spaced-U-snapkv`, n=50, c=40, C=512, both models: same order of GPU time as one Stage 3
cell for one arm, on top of the (already-collected) `U-snapkv` comparator rows.
**Estimated: ~15-20 GPU-minutes total** (2 models x 50 instances x 1 new arm, at the ~13-15 min/50
instance rate this cell has shown for other single-arm additions).

## What this test does NOT do

- Does not explain M3's refusal failures (31% of its population) -- separate, unaddressed.
- Does not claim K=24 is optimal -- it is one value, fixed before data, chosen for plausibility.
- Does not re-run or touch the frozen Stage 3 grid, which stays paused.
- Is not a new wrapper variant proposed as a replacement for `U-X` -- it is a diagnostic probe of
  the spacing mechanism specifically, reported as such regardless of outcome.

## Amendment S1 -- scope widened to pool contamination-tagged queries across arms

Made after SnapKV alone returned <20 contamination-tagged queries on BOTH models (NOT SCORED,
commit `28530e2`). Before any new row is generated for this amendment.

**Widened scope:** generate `spaced-U-expected_attn` (same allocator, same K=24, same relaxation
logic, base scorer swapped to ExpectedAttention -- a plain `ScorerPress`, identical code path to
SnapKV) at the same cell (c=40, C=512, n=50, both models), and pool its contamination-tagged
queries with SnapKV's toward the >=20 floor, per model.

**`spaced-U-adakv_snapkv` is BLOCKED, not run, not silently dropped.** `U-adakv_snapkv` in the
frozen Stage 3 wrapper does not use the per-head-independent allocation `allocate_spaced` is built
on -- it uses `unit_keep_adakv`, a cross-head joint allocation (per-head safeguard + a global
cross-head top-k across the whole layer). Building a spaced variant on the per-head allocator
would silently change AdaKV's cross-head budget-sharing mechanism as well as adding spacing,
conflating the two. This needs its own allocator (spacing threaded through `unit_keep_adakv`'s
cross-head logic), which is new scope, not a same-allocator swap. Not built here. AdaKV's
contamination-tagged queries (from the existing unconstrained rows) are excluded from the pool.

**Confound check ordering, per instruction: PER ARM, BEFORE pooling, not after.** Delta p_g and
Delta units_complete are computed independently for `spaced-U-snapkv` vs `U-snapkv` and for
`spaced-U-expected_attn` vs `U-expected_attn`, each vs its own unconstrained comparator, at THIS
cell (c=40, C=512) -- not inherited from the c=8 finding in `DIAGNOSIS_EA.md`. An arm found
confounded here is still pooled (the pooling question is about statistical power, not cleanliness)
but its contribution is reported with the confound caveat attached, and the pooled verdict states
which arms carried a caveat.

**Relaxation logs reported per arm, not only pooled** -- the two base scorers are expected to
relax at different rates (ExpectedAttention's score is unsmoothed and heavy-tailed vs SnapKV's
pooled/smoothed attention, per `DIAGNOSIS_A6.md`'s and `DIAGNOSIS_EA.md`'s characterisations).

Nothing else changes: K=24, the completion-matching band, the >=20-query floor (now applied to
the POOLED count), and the REAL/NOT-IT rule on the pooled contamination-tagged accuracy.

## Amendment S2 -- cross-head-aware AdaKV allocator, added as new scope

Made after the n-raise scale-up left both models short (M2 16/20 pooled, M3 3/20, commit
`1da3fe4`), and after checking the arithmetic on AdaKV's own contamination rate from the
original 141-case Track 2 dataset before writing any code: 21.7% (5/23), the HIGHEST of the
three method arms measured (SnapKV 14.8%, ExpectedAttn 15.6%, KeyDiff 0.0% -- KeyDiff excluded,
its own confound runs the opposite direction per `DIAGNOSIS_EA.md`'s analogue and it never
scored a single contamination case). Projected pooled count at M2 n=120 with AdaKV added,
using the same n-ratio scaling already validated against the observed SnapKV/ExpectedAttn
scale-up (naive projection landed within normal small-count sampling variance of what was
measured): actual 8+8 (SnapKV+ExpectedAttn at n=120) plus AdaKV's naive projection
5*(120/50)=12, discounted by the same ~0.75x undershoot the other two arms showed, ~9 --
projected total ~25, clearing the >=20 floor with the margin this amendment's decision rule
required (>=24) before authorizing the build below.

**The allocator is new scope, not a swap onto `allocate_spaced`.** `U-adakv_snapkv` in the frozen
Stage 3 wrapper does not allocate per head independently -- it runs `unit_keep_adakv`'s two-phase
mechanism: (1) a per-head safeguard budget (`alpha * n_kept` per head, via the SAME per-head
greedy as SnapKV/ExpectedAttn), then (2) a single GLOBAL cross-head top-k over the remaining
budget, pooling (head, unit) and (head, singleton) candidates across the WHOLE layer and
spending the shared `h*C` total wherever the score is highest. `p4/probes.py:allocate_spaced_adakv`
replicates both phases exactly, adding the K=24 spacing check to each phase's unit-acceptance
step (checked against that HEAD's own already-accepted unit spans, from either phase), and a
relaxation pass that pools skips from both phases, re-admits the least-violating first, and
evicts the lowest-scoring non-floor/non-unit token in that SAME head when the shared h*C budget
is already spent -- extending `spaced_keep_with_relaxation`'s per-head eviction discipline to
AdaKV's cross-head shared budget.

**Unit-tested the same way `allocate_spaced` was** (`out/_test_spacing_adakv.py`, 4 tests, all
pass): (1) strict spacing rejects at least one too-close pair in a small scenario; (2) the
unconstrained-equivalent call reproduces `unit_keep_adakv`'s own layer total exactly (sanity);
(3) a densely-packed 10-unit/3-head scenario forces the relaxation pass to actually engage
(15 relaxations, not zero) and lands the layer back in the matched band -- the same discipline
`allocate_spaced`'s own test suite required, since a scenario passing the band trivially without
relaxation firing would not validate the relaxation logic at all; (4) without a target, strict
spacing alone measurably reduces completion in the same scenario (units_complete 10.0 -> 4.0).

**Cell, arm, K, target-n, pooling and confound-check ordering: all unchanged from S1.**
`spaced-U-adakv_snapkv` is generated at c=40, C=512, n=120 (M2) / n=150 (M3), and its
contamination-tagged queries are pooled with SnapKV's and ExpectedAttention's toward the same
>=20 floor -- three arms now, not two. Its Δp_g/Δunits_complete confound check is computed
independently, before pooling, exactly as for the other two arms; it is NOT assumed clean.

**Stopping rule, stated before any AdaKV row is generated:** this is the fourth attempt at this
cell (SnapKV alone -> pooled two-arm -> n-raised two-arm -> this). If the three-arm pool still
falls short on M2, the spacing hypothesis is reported as untestable at the n achievable in
reasonable GPU time with the arms available at this cell, and closed -- not grounds for a fifth
attempt. M3 is not re-scored here regardless of what AdaKV's inclusion does to its pooled count,
unless that addition alone clears 20 on M3, in which case it is reported without having been
chased.
