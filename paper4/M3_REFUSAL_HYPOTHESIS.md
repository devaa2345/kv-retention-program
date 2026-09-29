# M3 refusal mechanism — hypothesis and threshold, committed before any further test

Grounded in two already-collected facts, no new data: (1) `TRACK2_SELECTION_CHECK.md` (commit
`c122dab`) established M3's disagreement population is 31% refusal, 5% contamination, the rest
other; (2) the free CPU check run this session (not yet committed) found refusal cases' own q_slot
(mean 0.10-0.23 across arms) sits close to each arm's general population mean, while instances
where M3 got *something* right have much higher q_slot (0.27-0.52) — a signal that most plausibly
just restates "completion predicts success," not something specific to refusal as opposed to
M3's other failure modes (contamination, hallucination) at matched completion.

## The sharper claim

> Among M3's floor-correct/U-wrong disagreement cases, **refusal's share of failures is not
> flat across completion (q_slot) level** — refusal is disproportionately the failure mode at LOW
> completion, and other failure modes (contamination, hallucination) take a larger share as
> completion rises, even though accuracy stays low throughout.

This is falsifiable and distinct from the already-established "completion predicts accuracy":
it says something about the *composition* of failure at a given completion level, comparing
failure modes to each other, not failures to successes.

## Test (CPU only, no GPU; data already on disk)

All M3 disagreement cases (floor-correct, U-wrong, fact complete in ≥1 slot; same population as
`TRACK2_SELECTION_CHECK.md`), each already taggable as refusal / contamination / other by the
existing rules (`REFUSAL_RE`, the record-distance/overlap contamination rule) with no new
extraction needed beyond re-running the existing tagging across all cases with their q_slot
attached. Split into two completion bins by the **median q_slot of the full M3 disagreement
population** (the split point is the data's own median, fixed by the population itself before
any tag is looked at — not chosen post hoc to produce a result).

**REAL**: refusal's share of disagreements is higher in the low-completion bin than the
high-completion bin, difference ≥ 15 percentage points, checked with a bootstrap CI (10,000
resamples of instances, CRC32-seeded) that excludes zero.

**NOT-IT**: the CI includes zero, or the difference is below 15 points, or the direction is
reversed.

**Precondition**: ≥ 10 cases of each failure-mode tag in each bin; a bin failing this is reported
as underpowered for that bin's rate, not filled in or estimated.

## What this does and does not require

- CPU only for this test. No GPU touches anything until this reports.
- Does not explain WHY low completion produces refusal specifically (that would need a further,
  separate hypothesis — e.g. a confidence/margin check analogous to Task 3's calibration test, run
  on M3 refusal specifically rather than pooled with M2's null result).
- Does not touch the malformed/truncation thread (9/35 M2 cases) — parked, per the instruction to
  pick one thread, with the one free fact already recovered (all 9 stop via `eos`, not `cap`,
  ruling out simple length-cap truncation) recorded for whoever picks that thread up.


## Amendment 1 -- precondition corrected before rescoring, no new GPU work

Made after the test ran and reported NOT SCORED (commit `eb22013`), against data already on disk --
no new tag counts, no new q_slot values, no GPU. Supersedes the precondition in the original
threshold above (commit `adb5c2c`).

**Why this correction is legitimate, not post-hoc rule-bending.** The original precondition
required >=10 cases of EACH of three tags (refusal, contamination, other) in both completion bins.
M3's total contamination population is fixed at 3 cases across the entire 80-case disagreement set
(0 in the low bin, 3 in the high bin) -- a fact already established by
`TRACK2_SELECTION_CHECK.md` (commit `c122dab`), BEFORE this test's threshold was written. A
>=10-per-bin requirement on contamination was therefore structurally unsatisfiable regardless of
the true refusal effect size, for any possible bin split of this population -- this is diagnosable
from data that predates today's result, not a bar being lowered because today's number came in
short.

**Corrected rule.** Drop the contamination-count sub-requirement entirely. Precondition is now
**>=10 REFUSAL cases in each bin only**. The effect-size bar is unchanged: refusal-share difference
(low bin minus high bin) >= +0.15, with a bootstrap CI (10,000 resamples, CRC32-seeded) excluding
zero, for REAL; NOT-IT otherwise.

## Rescore, corrected rule, same data as commit `eb22013`

- Low-completion bin (q_slot <= 0.229): n=41, **25 refusal cases** -- clears >=10.
- High-completion bin (q_slot > 0.229): n=39, **8 refusal cases** -- **does NOT clear >=10.
  One case short of the bar, evaluated exactly, not rounded up.**
- Effect size (already computed, unchanged): +0.405, 95% CI [+0.206, +0.602] -- would clear the
  effect-size bar on its own.

**Verdict: NOT SCORED, under the corrected rule, because the high-completion bin narrowly fails
its own refusal-count precondition (8 < 10).** This is a genuine power shortfall for the specific
comparison intended, not a structural impossibility like the original contamination requirement --
it is reported as failing, not rounded up, and the bar is not adjusted a second time in this
session. The effect-size numbers are recorded above for whoever picks this up next; they are not
being called REAL despite clearing the effect-size bar, because the precondition exists precisely
to stop an underpowered bin from being read as a real result.

**No finding is asserted about M3 refusal being completion-dependent.** That claim is NOT made
here -- the test that would support or refute it is one case short of its own registered power
bar. Any follow-up (e.g. widening the bin definition, or collecting more instances) needs its own
separately preregistered test, as instructed.
