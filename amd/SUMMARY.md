# Executive summary

Blind reimplementation of `SPEC_REIMPL_v1.md`. No file of the original harness was opened,
imported, listed or searched at any point. Every ambiguity resolved is recorded in
`SPEC_QUESTIONS.md`. **7300 runs, 0 failures.**

## What replicated

| quantity | reference | this implementation |
|---|---|---|
| quantizer error, 8-bit keys / values | 1.17% / 0.88% | **1.18% / 0.91%** |
| quantizer error, 4-bit keys / values | 20.06% / 14.99% | **19.98% / 15.20%** |
| quantizer error, 2-bit keys / values | 101.11% / 74.68% | 103.35% / 75.55% |
| context length | ~1029 (1011–1042) | **median 1030.5 (1017–1041)** |
| iso-memory token grant at budget 257 | 411 (1.6×) | **411** |
| `full_cache_ref` budget-invariance | required | **exactly invariant (0.7989 at all 3)** |
| 4-bit accuracy recovery after 5-bit collapse | 0.123 | **0.1167** |
| interaction @ 257 (8-bit) | +0.002 | **-0.0011** |
| interaction @ 514 (8-bit) | −0.002 | **-0.0056** |

## The three findings

**1. The headline disagreement was a missing line in the specification.** Phase 1's `quant_bits`
is never stated; §7 fixes 4-bit for Phase 2 only. Under 4-bit the interaction is −0.046 and
−0.116 where the spec reports ≈0. Re-running the tiered arms at 8-bit lands within 0.005 of the
reference at both budgets. The interaction is therefore **not null — it is conditional on
bit-width**, and both implementations are right about their own condition. Credential-survival
instrumentation shows why: arms 2 and 4 evict *exactly the same 92.1 credential tokens*, so
tiering adds no eviction at all and its entire effect is precision loss on retained tokens.
*(The 8-bit condition was added post-hoc; the mechanism was predicted before the run.)*

**2. Structural protection degenerates into an oracle when distractors stop matching its
pattern.** Rewriting distractor values to a non-credential form lifts the structural arm from
0.150 to **0.907**, above the full-cache ceiling of 0.880 measured under the same condition, while the
full-cache arm gains only +0.137. Pattern-based KV protection's strength is set by the
ratio of target lines to pattern-matching non-target lines — a property of the mechanism,
not of this task.

**3. Accuracy is not monotone in bit-width while reconstruction error is** — §9's caution
reproduced independently. Error rises strictly (1.15 → 39.92%), yet accuracy collapses to
0.0000 at 5-bit and recovers to 0.1167 at 4-bit under twice the error. The quantizer is cleared:
zero level-count violations at every width on tensors intercepted during live generation.

## What did not replicate, and why

**The ceiling: 0.799 here against the reference's 0.947.** Diagnosed rather than assumed. The
normalisations §2 anticipates (case, whitespace) rescue *exactly zero* failures. Two other
conventions close it: the model often emits the bare 14-hex payload without the `sk-` prefix
(58 of 181 failures), and allowing one character of edit distance recovers 82 more. Together:
**0.9544 against 0.947**. The evidence favours a scoring-convention difference over a capability
difference. The scorer was not changed.

## Corrections made during the study

The §7 manipulation check failed on two of five promotion signals at n=100 — P4 ρ = +0.4974, P5 ρ = +0.8489 against a ~0 target.
Both fell back to the eviction score for non-must positions, correlating them with it by
construction. Signals were corrected (P5 membership-only; P4 rotating over a fixed
permutation), giving P4 -0.0218 and P5 -0.0781, and
Phase 2 was re-run in full. The corrected run is primary; the original is retained to
quantify the contamination. P1 is exactly +1.0000 with a zero-width CI, as it must be.

## iso-memory

§6's iso-memory condition (never run in the original brief) reverses the picture: at
budget 257 the interaction moves from −0.0456 (iso-token) to +0.0133
[-0.0111, +0.0378] (iso-memory). The extra tokens funded by
the cold tier's lower byte cost largely compensate for 4-bit damage — which is exactly
what that condition exists to test.

## Reading this repository

`FINDINGS.md` — full results. `AGREEMENT.md` — every reference value compared, disagreements
traced to named gaps. `SPEC_QUESTIONS.md` — 26 ambiguities and the default chosen for each.
`results/csv/` — per-run data. `README.md` — how to reproduce.

Limitations are enumerated in the *Threats to validity* section of `FINDINGS.md`; the load-bearing
ones are the absorbing failure mode compressing k/6, two floored factorial cells at budgets 154
and 257, the post-hoc 8-bit condition, and a single model / task / GPU architecture.

