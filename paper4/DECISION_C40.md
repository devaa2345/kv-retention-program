# Decision rule for the c=40 claim-bearing cell — fixed BEFORE the result is seen

Committed while the c=40 probes were still running and before any c=40 delta had been computed.
Set by the user; recorded verbatim in substance.

## The quantity

**Allocation-only delta** at c=40, C=512: `U-X` minus an `X` arm gathered in the SAME cache order,
so the bf16 accumulation-order confound found in A6 (`DIAGNOSIS_A6.md`) is held fixed and only
allocation differs. Post-fix (wrapper gathering in score order) this is simply `U-X` minus `X`.

## The rule

- **PROCEED** to the remaining claim-bearing cells **only if**: the allocation-only delta's 95% CI
  excludes zero on the positive side on **at least one model** **AND** its point estimate is
  **>= +0.022** (half the original +0.045 headline).
- **STOP** if the CI includes zero, or the point estimate is positive but below +0.022, **on both
  models**. Do not re-run the remaining cells. Report it as a finding that requires rethinking the
  mechanism, not more data collection.
- **If the two models disagree** against this threshold: report both explicitly and **hold**. Do
  not average them and do not select the favourable one.

## Order of operations (also fixed here)

1. Fix the wrapper to gather in score order, matching `X` (amendment A4).
2. Re-run the **Stage 2 pilot cell (c=40, C=512)** first and alone; report its allocation-only
   delta before anything else runs.
3. Only then, and only if the rule says PROCEED: M2 c=8 C=256 and C=512, and the c=1 diagnostic
   cells.
4. All other completed cells are **quarantined with a note**, not re-run.
5. The c=19/c=40 grid stays paused until the pilot cell reports.

## Scope note

The n=20 probes running when this rule was written measure the same quantity **pre-fix**
(`U-snapkv` minus `ascend-snapkv`). They are a **preview** at lower n and at the screen's own cell;
the rule above is applied to the **post-fix pilot-cell re-run**, not to the preview.
