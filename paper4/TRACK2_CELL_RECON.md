# Reconnaissance — does a different cell offer a higher contamination rate? CPU-only, no GPU.

Source: all Stage 3 cells with both rows and saved keep-sets on disk (`p4_s3_*.jsonl` +
`p4_s3_keepsets_*.npz`). Coverage is uneven and stated plainly: M2 has c in {1, 8, 19-partial, 40}
across all 5 budgets except c=19 (only C=32/64 ran before the pause); M3 has **only c=1 (all
budgets) and c=40, C=512** — c=8 and c=19 were never generated on M3 (blocked at the confound halt,
never resumed). Any claim about M3 outside those two cost levels cannot be checked from existing
data.

## Full table

| model | c | C | floor | disagreements | contam. | rate | legal (SnapKV+EA) disagree | legal contam. | legal rate |
|---|---|---|---|---|---|---|---|---|---|
| M2 | 1 | 32 | 0.075 | 9 | 0 | 0.0% | 2 | 0 | 0.0% |
| M2 | 1 | 64 | 0.085 | 12 | 0 | 0.0% | 3 | 0 | 0.0% |
| M2 | 1 | 128 | 0.130 | 19 | 0 | 0.0% | 5 | 0 | 0.0% |
| M2 | 1 | 256 | 0.205 | 12 | 0 | 0.0% | 5 | 0 | 0.0% |
| M2 | 1 | 512 | 0.300 | 17 | 0 | 0.0% | 7 | 0 | 0.0% |
| M2 | 8 | 32 | 0.025* | 3 | 0 | 0.0% | 1 | 0 | 0.0% |
| M2 | 8 | 64 | 0.025* | 4 | 0 | 0.0% | 2 | 0 | 0.0% |
| M2 | 8 | 128 | 0.035* | 11 | 0 | 0.0% | 6 | 0 | 0.0% |
| M2 | 8 | 256 | 0.085 | 39 | 0 | 0.0% | 18 | 0 | 0.0% |
| M2 | 8 | 512 | 0.200 | 71 | 0 | 0.0% | 27 | 0 | 0.0% |
| M2 | 19 | 32 | 0.005* | 3 | 0 | 0.0% | 2 | 0 | 0.0% |
| M2 | 19 | 64 | 0.020* | 11 | 0 | 0.0% | 5 | 0 | 0.0% |
| **M2** | **40** | **512** | **0.170** | **108** | **14** | **13.0%** | **59** | **9** | **15.3%** |
| M3 | 1 | 32 | 0.080 | 12 | 0 | 0.0% | 2 | 0 | 0.0% |
| M3 | 1 | 64 | 0.085 | 15 | 0 | 0.0% | 4 | 0 | 0.0% |
| M3 | 1 | 128 | 0.100 | 17 | 0 | 0.0% | 6 | 0 | 0.0% |
| M3 | 1 | 256 | 0.150 | 17 | 0 | 0.0% | 8 | 0 | 0.0% |
| M3 | 1 | 512 | 0.275 | 26 | 0 | 0.0% | 12 | 0 | 0.0% |
| **M3** | **40** | **512** | **0.105** | **80** | **3** | **3.8%** | **39** | **1** | **2.6%** |

`*` = degenerate cell (floor < 0.05); still tallied for completeness, contributes nothing.

## Answer to the question as posed

**Contamination is not roughly flat across cells — it is essentially confined to c=40.** Every
non-degenerate c=1 and c=8 cell shows **zero** contamination cases, on both models, across every
budget tested. Disagreements exist at those cells (up to 71 at M2 c=8 C=512), but none of them
match a nearby real record closely enough to tag as contamination — they are near-misses,
hallucinations, or other patterns, not this mechanism. **c=40, C=512 is not merely the best cell
available — among the cells actually run, it is the *only* one where this mechanism appears at
all**, on both models.

This is consistent with the mechanism's own logic, not a coincidence: contamination requires two
adjacent, *complete*, *distinct* records to both be candidates for confusion. At c=1 a "fact" is a
single token (MARK-1) — there is no adjacent complete record to confuse it with in the relevant
sense (the task construction differs entirely from LEDGER-C). At c=8, records are short enough
that the model rarely seems to conflate them even when several sit close together in the retained
set. Only at c=40 (long, multi-field records) does misattributing one whole record for its
neighbor produce a plausible-looking wrong answer the tagging rule catches. **So the mechanism's
raw opportunity count scales with fact cost, not with budget alone** -- confirming the question's
hypothesis about "cells with more candidate units in play," but the driving variable appears to be
`c`, not `C`.

## Consequence for the projection

No untested non-degenerate cell offers a higher contamination rate than c=40, C=512, because no
other cell offers a **nonzero** rate at all, on either model, in the data available. **c=19 is the
one real gap**: it is admissible only at C=256/512 on M2 (untested — pause occurred before those
budgets ran) and was never run on M3 at all. c=19 sits between c=8 (zero contamination) and c=40
(13-15%), so its rate is unknown, not necessarily zero — but nothing currently on disk can settle
it, and it would need new generation (both C=256/512 on M2, all five budgets on M3), not a CPU-only
lookup.

**Verdict: option 3 (a better existing cell) is closed for the cells actually on disk.** c=40,
C=512 remains the best -- in fact the only usable -- cell for this test among what has been run.
The choice reduces to what was already on the table: raise `n` at c=40, build the AdaKV allocator
to add a third arm, or generate the untested c=19 budgets first (new scope, not free, and not
guaranteed to help -- c=19's rate could just as easily land near c=8's zero as near c=40's 13%).

No GPU used. No new prereg written. Nothing decided here beyond ruling out "switch cells" among
existing data.
