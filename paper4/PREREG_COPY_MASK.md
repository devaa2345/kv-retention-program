# Hard-mask confirmatory prereg — committed before generation

Mechanism: attention-guided copy mask, text-level, substring-of-span, fixed twice on concrete
faults found by unit-testing (record-id inclusion + field-order tracking, then EOS-legality) —
`HARDMASK_CONFIRMATORY.md`. Unit-tested at n=30+30 on floor's own held-but-wrong/held-and-correct
populations: **29/30 (96.7%) converted, 2/30 (6.7%) broken, 0/60 invented**, identical for the
schema-ordered and schema-free variants.

## Design

**Instances**: fresh, seeds beyond every range used in this thread's development (0-119 used
throughout tonight for floor/oracle/full_cache/U-floor). **Instances 120-199 (n=80) for both M2
and M3** — M3 was never touched by any part of this copy-mask thread, so any range is fresh for
it; using the same range as M2 keeps the two runs directly comparable.

**Arms**:
- Primary set: `floor_pos` (unbonused/unmasked baseline), `floor + schema-ordered mask`,
  `floor + schema-free mask`.
- Secondary set (schema-free only, since it passed step 3 and is the general form): `U-floor`,
  `oracle_causal`, `full_cache`, each with and without the mask.

**Primary**: floor + mask vs `floor_pos`. **Headline**: U-floor + mask vs `floor_pos` (the
paper's actual baseline, not U-floor unbonused — same convention as the soft-bonus prereg).
**Generality check**: schema-free vs schema-ordered, on the primary arm, at full n (the unit
test already found them identical at n=30+30; this confirms it isn't a small-sample coincidence).

## Ceilings (per-query arithmetic `1/(instances*4)`, using the unit-test rates: conversion
29/30=0.967, breakage 2/30=0.067 — the SAME rate applied to all four arms, an explicitly stated
assumption for U-floor/oracle/full_cache since only floor's own population was unit-tested
directly)

| arm | accuracy | ceiling | gain |
|---|---|---|---|
| floor_pos | 0.175 | 0.284 | **+0.109** |
| U-floor | 0.178 | 0.325 | **+0.148** |
| oracle_causal | 0.695 | 0.943 | +0.248 |
| full_cache | 0.895 | 0.937 | +0.042 |

**Caveat flagged, not resolved**: full_cache's own wrong-case attention-hit-rate was measured
earlier at 80% (lower than floor's ~98%), so borrowing floor's 96.7% conversion rate for
full_cache's ceiling is likely optimistic; its real ceiling may be smaller. Not corrected here —
stated so the eventual result is read against the right expectation.

## Decision rule

**REAL**: paired gain over the unmasked same arm, 95% bootstrap CI excluding zero, AND net
positive after subtracting correct answers the mask broke — both reported in the SAME table, not
separately. **Per-model verdicts, independently** — M3 is a replication of the mechanism with NO
changes, not a second design.

**Interpretation, fixed in advance**: if schema-free accuracy ≈ schema-ordered accuracy (within
noise) on the primary arm at full n, the claim is general (no schema/field-order assumption
needed). If only schema-ordered works at scale, the claim is limited to structured records with
known field boundaries — the unit test's n=30+30 finding of exact equality would need to be
read as a small-sample coincidence in that case, not confirmed.

## Sizing

n=80/model/arm-pair, matching the session's established floor for confirmatory tests. The
ceiling gains here (+0.109 to +0.148 on the primary/headline arms) are the largest of any
mechanism tested tonight — n=80 gives ample margin over what the raw power formula would require
for an effect this size, consistent with not trusting a formula-derived smaller n
(`PAPER4_PHASE1_RECORD.md`'s carried-forward lesson).

## Confound check — first, before any scoring

Decode-only intervention: `p_g`, `units_complete` MUST be identical between each masked arm and
its unmasked counterpart. **If they differ anywhere, stop** — something other than decoding
changed and no accuracy number from that run can be trusted.

## Scope

**Extractive tasks only.** Every ceiling and result here depends on the correct answer being a
literal, contiguous span of context text. Public-benchmark or abstractive-task transfer is
untested and not assumed.
