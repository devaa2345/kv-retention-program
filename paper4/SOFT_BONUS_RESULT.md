# Soft copy-bonus decoding — result (budget-limited, unattended run)

Prereg: `PREREG_SOFT_BONUS.md` (`8542098`). Unit-test finding that motivated the switch from a
hard mask to a soft bonus: `COPY_CONSTRAINED_UNITTEST_FINDING.md`. Read-out ceilings established
before any bonus generation: `READOUT_UPPERBOUND.md`, `out/_step1_ufloor_readout_results.json`.

## VOID — deviation and confound, logged before anything else

**Deviation from the Step 2 stop rule.** Only one lambda (2.0) was ever generated (budget-
limited); its selection-half result was already negative — **-0.0042 [-0.0125, +0.0000]**, 0
fixed, 1 broken (not the +/-0.0167 or 7-broken figures floated in a later message; those don't
match anything in this run's committed data and are not used here). Scoring proceeded to the
held-out half anyway. That was a deviation from "stop after Step 4 if neither lambda shows a
selection-half gain" — with only one lambda tested, the single result already available said
don't proceed, and I proceeded.

**This result is now VOID for a separate, more fundamental reason.** A CPU-only token-ID
alignment check (`COPY_TEXT_LEVEL_UNITTEST.md`) found 0/81 alignment between correct answers'
retokenized text and the record's own in-context tokenization, and traced all 5 broken cases
(1 selection + 4 held-out, not 7) to exactly this cause (digit-chunking). The token-ID-based
automaton underlying the soft bonus was frequently failing to boost the model's own correct
continuations for a tokenization-artifact reason unrelated to the hypothesis being tested. The
NOT-IT verdict below is **not confirmed** — it answers a confounded implementation, not the
intended mechanism. Superseded by the text-level re-implementation.

## Ceilings (read-outs, established pre-generation)

| arm | actual accuracy | read-out ceiling | gap |
|---|---|---|---|
| floor_pos | 0.175 | 0.291 | +0.116 |
| U-floor | 0.1775 | 0.3058 | +0.1283 |
| oracle_causal | 0.695 | 0.968 | +0.273 |
| full_cache | 0.895 | 0.979 | +0.084 |

## Scope actually completed — budget-limited, stated plainly

**GPU time used: ~145 of the 150-minute (2.5h) hard budget**, almost entirely on ONE (arm,
lambda) pair. The bonused-generation rate measured in practice (eager attention span-
identification + continued soft-bonus generation) was **~0.96 min/instance** — far slower than
the earlier small-n smoke test suggested (which ran at ~0.8 min/instance on n=2, close enough
in hindsight, but I mis-extrapolated a faster rate before committing to the full run). At this
rate:

- floor + bonus, λ=2.0, 120 instances: **114.94 min**, 0 failures, 13/480 degenerate. **Completed.**
- floor + bonus, λ=5.0: would need another ~115 min. **Not run — budget exhausted.**
- U-floor, oracle, full_cache bonused arms (both λ each): would need several more hours combined.
  **Not run.**

Per the standing rule ("if the hard GPU budget is reached, finish writing whatever results exist
and stop"), all further GPU work stopped once floor+λ=2.0 completed. **Only the PRIMARY
comparison (floor+bonus vs floor_pos) has real data.** The headline (U-floor+bonus vs floor_pos)
and both secondary comparisons (oracle, full_cache) were never generated — reported as not run,
not estimated or assumed.

## Step 4 — confound check: CLEAN

`p_g` and `units_complete` identical between bonused and unbonused for every matched
(instance, query) pair, across all 120 instances. Confirms the intervention touched generation
only — retention is provably untouched, so any accuracy difference is attributable to decoding.

## Step 5 — scoring, floor + bonus (λ=2.0) only

| half | floor_pos | floor+bonus | paired diff [95% CI] | fixed | broken (FP) | FP rate [Wilson CI] | degenerate |
|---|---|---|---|---|---|---|---|
| selection (n=240q) | 0.1792 | 0.1750 | −0.0042 [−0.0125, +0.0000] | 0 | 1/43 | 2.3% [0.4%, 12.1%] | 5/240 |
| held-out (n=240q) | 0.1708 | 0.1708 | +0.0000 [−0.0208, +0.0209] | 4 | 4/41 | 9.8% [3.9%, 22.5%] | 8/240 |

**Verdict: NOT-IT for floor + bonus at λ=2.0.** Selection half shows a small decline (0 fixed, 1
broken); held-out half is exactly flat (4 fixed, 4 broken, cancel exactly). Neither half shows a
CI excluding zero in the positive direction. Only one λ was tested, so the prereg's literal
"neither λ" stop condition can't be fully evaluated (λ=5.0 never ran) — this is reported as what
it is: one λ, tested, NOT-IT, not extrapolated to the untested value.

**λ=5.0 (both arms it would have applied to), U-floor+bonus, oracle+bonus, full_cache+bonus: no
data. Not run, not estimated.**

## Step 6 — M3 replication: skipped

Both stated conditions for running it fail independently: the primary arm is not REAL on M2, and
no GPU budget remains (~5 min left when scoring finished). Either alone would rule it out.

## Degeneration

13/480 total queries (2.7%) showed the hard-mask-style degenerate pattern (near-token-cap length
or delimiter-repeat) even under the SOFT bonus, at λ=2.0 — a small but nonzero rate, meaning the
soft bonus does not fully eliminate the loop failure mode the hard mask showed constantly; it
reduces its frequency sharply (2.7% vs the hard mask's near-total failure on reproduction) but
does not eliminate it. λ=5.0's degeneration rate is unmeasured.

## What this run actually establishes, honestly

- The soft-bonus mechanism is implementable and runs cleanly (0 failures across 480 queries) with
  a verified decode-only confound profile.
- At λ=2.0, it does not help floor_pos — flat to slightly negative, on both halves.
- Whether λ=5.0 or the U-floor/oracle/full_cache arms would behave differently is **genuinely
  unknown** — not a null result for those, an unrun one. The single biggest uncertainty this run
  leaves is whether the headline comparison (U-floor+bonus vs floor_pos, ceiling +0.1283, the
  largest of the four) would have looked different; it was never tested.
- The measured generation rate (~0.96 min/instance for bonused arms) is the number to plan the
  next attempt against, not the smoke test's smaller-sample rate.

## Scope, stated explicitly per instruction

**Extractive tasks only.** LEDGER-C's answer is literally the record's own text sitting in
context — every ceiling and result here depends on that. Nothing in this thread (unit test, full_cache
check, read-out upper bounds, or this one scored comparison) has been tested on, or should be
assumed to transfer to, abstractive or public-benchmark tasks where the correct answer is not a
verbatim span of the input.
