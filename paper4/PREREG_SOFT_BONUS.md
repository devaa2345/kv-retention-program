# Soft copy-bonus decoding — prereg, committed before any bonus generation

## Mechanism

At each generation step, add `+λ` to the raw logits of tokens that continue some occurrence of
the current generated suffix within the top-attended record (max attention mass over retained
records at the last prompt-token forward call — same signal as Test A/E3/Item 4), or that start
a new field immediately after a field delimiter. **No masking** — every token stays legal at
every step; the bonus only shifts the model's own preference toward span-continuing tokens
without ever forbidding a departure from the span. This directly targets the hard-mask thread's
failure mode (`COPY_CONSTRAINED_UNITTEST_FINDING.md`): a hard mask forces a legal-but-degenerate
loop when the model's own preferences point elsewhere; a soft bonus can be outweighed by the
model's own logits when the span-continuation path is bad, instead of forcing it.

## λ grid

**λ ∈ {2.0, 5.0}, added directly to raw logits.** No stated reason to expect these are the wrong
scale: Qwen2.5-3B's own logit range at a confident greedy step is commonly single-to-low-double
digits (typical top-1 vs top-2 gaps observed in this session's margin work,
`out/_task3_margins.py`, were O(1) in the same units), so +2 is a moderate nudge and +5 is a
strong one, without hardcoding to -inf. Two values, not swept further per instruction.

## Ceilings (read-outs already measured)

| arm | actual accuracy | read-out (ceiling) | gap |
|---|---|---|---|
| floor_pos | 0.175 | 0.291 | +0.116 |
| U-floor | 0.1775 | 0.3058 | +0.1283 |
| oracle_causal | 0.695 | 0.968 | +0.273 |
| full_cache | 0.895 | 0.979 | +0.084 |

## Arms (M2)

floor, U-floor, oracle, full_cache — each **unbonused** (baseline, already on disk or generated
fresh where missing) and **at both λ** (bonused), instances 0-119.

## Split

Instances 0-119 split by instance: **selection half = 60, held-out half = 60** (even/odd
instance index, deterministic, matches this session's established alternating-split
convention). **One λ chosen for all arms** — whichever of {2.0, 5.0} gives the highest MEAN net
gain (bonused minus unbonused accuracy) pooled across the four bases on the selection half. No
per-arm tuning.

## Primary, headline, secondary

- **Primary**: floor + bonus vs `floor_pos`, held-out half.
- **Headline**: U-floor + bonus vs `floor_pos` (not vs U-floor unbonused — the comparison that
  decides whether the compound beats the paper's actual baseline), held-out half.
- **Secondary**, reported separately, never merged into the primary/headline verdict: oracle +
  bonus vs oracle unbonused; full_cache + bonus vs full_cache unbonused.

## Decision rule

**REAL**: held-out gain over the unbonused SAME arm has a paired bootstrap 95% CI excluding
zero, AND net accuracy improves after subtracting correct-case answers the bonus broke (false
positives) — not just gross gain on the wrong-case population. **NOT-IT** otherwise.
Per-query arithmetic throughout: one query = `1/(instances*4)`.

**Stop condition**: if NEITHER λ produces a selection-half gain on the PRIMARY arm (floor+bonus
vs floor_pos), stop after Step 4 (confound check) — do not proceed to held-out scoring across
all arms, do not test a third λ, do not change the mechanism. Write the report with whatever
exists at that point.

## Confound check

Decode-only intervention: `p_g`, `units_complete` (and completion generally) MUST be IDENTICAL
between each bonused arm and its unbonused counterpart, since the bonus never touches the
prefill/press/keep-set — only generation-time logits. Verified by recomputing `keep_metrics` on
the same captured keep-set for both bonused and unbonused runs and diffing, not assumed.

## Scope

**Extractive tasks only.** LEDGER-C's answer is literally the record's own text sitting in
context — read-out achieves 0.29-0.98 precisely because the correct answer is always a literal
substring of a retained record. Nothing in this design or its results (whatever they turn out to
be) has been tested for, or should be assumed to transfer to, abstractive or public-benchmark
tasks where the correct answer is not a verbatim span of the input.
