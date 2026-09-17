## Threats to validity

Ordered by how much each should change a reader's confidence.

**1. The primary metric is compressed by an absorbing failure mode.** All eviction arms collapse
to zero after turn 1, while arms 5 and 6 stay flat across all six turns. Once a credential is
evicted it cannot return, so turns 2–5 carry no information for arms 1–4 and k/6 behaves as an
effective 2-turn metric there, capped near 0.33. Absolute effect sizes are therefore not
comparable to a design where all six turns are live. The turn-index table is reported alongside
k/6 for this reason; k/6 remains primary because §8 fixes it.

**2. Two of four factorial cells are floored at two of three budgets.** Arms 1 and 3 score 0.0000
with zero variance at budgets 154 and 257. A factorial with dead cells is not really a factorial:
the interaction at those budgets is driven almost entirely by the protected arms. Only budget 514
(arm 1 = 0.0967, arm 3 = 0.0400) gives a clean four-cell estimate — and it is also the budget
where the bit-width contrast is largest, so conclusions rest disproportionately on it.

**3. The 8-bit condition is post-hoc.** The mechanism was predicted before the run and confirmed,
but the condition was added after observing the 4-bit disagreement. Exploratory, not confirmatory.

**4. Depth and dormancy are perfectly confounded by construction.** The [SPEC-GAP 1] turn order
queries credentials in reverse depth, so turn index, dormancy gap and context depth are one
variable. No analysis here separates a dormancy effect from a depth effect. Deliberate, to
maximise dormancy pressure; it costs identifiability.

**5. Calibration gates G1–G4 are this implementation's invention.** The spec defines only G5, so
gate-level agreement with the reference is meaningless; only downstream numbers can be compared.

**6. G2 fails at both long-context conditions** (0.6389 at 2048, 0.6528 at 4096, against a 0.70
bar set before any long-context data was seen). Those cells are reported and marked GATE FAILED;
the threshold was not moved. The 2048/4096 factorial is indicative only.

**7. The ceiling divergence is probably a scoring convention, but that is an inference.** Strict
scoring gives 0.799; prefix-tolerant edit-distance-1 scoring gives 0.9544 against the reference's
0.947. Strong circumstantial evidence, unverifiable without reading the reference code. If the
reference genuinely reached 0.947 under strict scoring, this task is materially harder and every
absolute level here is shifted.

**8. The originally-run Phase 2 failed its manipulation check.** P4 (ρ = +0.4974) and P5
(ρ = +0.8489) were substantially correlated with the eviction signal at n=100. Those arms do not
test orthogonal promotion. Phase 2 was re-run with corrected signals and the corrected run is
primary; the original is retained only to quantify the contamination.

**9. Copy-error noise costs power.** Roughly 20% of trials fail on hex copy fidelity in every arm.
Unbiased in a paired design, but it inflates variance, so intervals here are wider than a cleaner
task would yield.

**10. Single model, single task, single architecture.** Qwen2.5-1.5B-Instruct, one retrieval task,
one GPU (gfx1100/ROCm). Nothing establishes that the bit-width conditionality generalises to
larger models, other tasks, or CUDA numerics. The exact-match scoring is likely to make this task
unusually sensitive to quantization; a semantically-scored task would plausibly show a far weaker
bit-width effect.

**11. The blind protocol cannot catch shared errors.** As the spec states, an error present in
both the specification and an implementation of it will not be detected by this method.
Everything here is conditional on the specification being right about the mechanism.

**12. Structural protection's magnitude is task-parameterised.** The distractor ablation shows the
benefit depends on the ratio of target lines to pattern-matching non-target lines (6:26 here).
The direction should transfer; the magnitude should not be quoted out of context.

**13. Infrastructure note.** Running four concurrent ROCm processes on this GPU triggered a
`hipErrorLaunchFailure` that killed all of them mid-run. No data was lost or corrupted —
checkpointing is append-only with per-row fsync, and all 6,808 rows written at that point parsed
cleanly — and the affected jobs resumed from their last committed row. Concurrency was
subsequently capped at two. This affected wall-clock time only, not results.
