# Paper 1 (Paper 6 line) — writing prompt

Received 2026-09-05. **Not yet actioned — the user said "prompt for making paper but later."**
Target venue: MLSys 2027. Style files already uploaded (`mlsys2025.sty`, `mlsys2025.bst`,
`algorithm.sty`, `algorithmic.sty`, `fancyhdr.sty`).

---

## Framing note that came with the prompt

The three results are now uneven in strength. H-SUB is bulletproof: two families, three
configs, two implementations, zero prompts differing in places. Promotion saturation is one
model, one width, chosen because it was the only width with a measurable band on that model.
The bit-width anomaly turned out to be Qwen-specific. A paper presenting them as three
findings of equal standing will get caught on the weakest one. Grade them explicitly.

---

## The prompt

You are writing a systems measurement paper for MLSys from a completed experimental record.
Everything you need is in the repository: `PROJECT_RECORD.md`, `ANALYSIS.md`, `PREREG.md`, the
addenda, the AMD reimplementation folder (`AGREEMENT.md`, `SPEC_QUESTIONS.md`, `FINDINGS.md`,
and `results*/`), and the raw JSONL under each results directory.

**Read all of it before writing a word.** Do not write from a summary. Several numbers in
circulation are superseded — the pre-mean-correction EOS rates, the interim Q4 figure at
n=142, and the original AMD Phase 2 table contaminated by their `make_report.py` dedup bug.
Every figure cited must be traceable to a file, and where two versions exist use the later one
and mark the earlier as superseded.

### What this paper is

The prior paper in this line, *Where the Cost Goes*, is the model for register and standard.
Read it. It succeeded because it made structural claims backed by measurements taken multiple
independent ways, scoped them hard, and reported its own failures in the body rather than in
limitations. Match that. Do not exceed it in confidence.

The contribution is **a stage-level decomposition of recoverable KV caching, with resource
accounting controlled on both axes.** Recoverable eviction is split into a retention decision
and a promotion decision; each stage is measured against its own oracle; the promotion stage —
the part the literature builds machinery for — has essentially no headroom while the retention
stage retains a large unexploited gap.

"Retention beats precision" is a punchline, not a title. Choose a title in the register of the
prior work: name the measurement, not the verdict.

### Grade the findings — the most important instruction

Three explicitly different confidence levels, stated as such in the paper.

**Established.** The Phase 1 null under iso-token accounting. Two model families, three
configurations, two independently written implementations, several cells with zero prompts
differing out of 150. Lead with this. It carries the paper.

**Scoped.** Promotion saturation. One model, one bit-width. Be honest about why that width:
4-bit was the only precision on Qwen2.5-1.5B with a measurable band between all-FULL and
all-QUANT. At int8 the tier is free and the comparison is degenerate; at 6, 5 and 3 bits output
collapses. On Llama-3.2-3B, 4-bit is indistinguishable from 8-bit, so no band exists there
either. The claim is "no tested promotion signal, including a verified oracle, helps at the one
operating point where the mechanism could act on this model." Not "promotion is useless."

**Reported, not explained.** The Qwen bit-width anomaly. Four runs, four different curves: 1.5B
collapses at 6 and 5 in one implementation and at 5 only in the other, 3B collapses at 4 and
partially recovers at 3, Llama-3.2-3B is flat with zero degenerate outputs at every width down
to 3 bits despite 48.7% reconstruction error. Seven candidate explanations tested and
eliminated. The family contrast is the finding; do not present it as a general property of KV
quantization.

### The results, with the framing each needs

**Iso-token versus iso-memory.** The same experiment scored two ways gives opposite answers —
null under matched tokens, large under matched bytes, because the tiered arm is handed roughly
1.6x the tokens. This is the evaluation-convention result and the second-strongest thing in the
paper. Report the `quant_byte_cost` sensitivity as a range (+0.35 to +0.54 across 0.125–0.50)
and disclose that the constant was fixed after the pre-registration hash.

**The 4-bit harm, and the Q4 reversal.** At 4-bit the tier does not merely fail to help, it
harms, and correcting a retention-ranking defect made the harm *larger* (−0.683 → −0.801). The
mechanism is the paper's sharpest: the cold tier sits directly downstream of the selection
stage, so improving retention feeds it more of exactly the tokens retention correctly identified
as valuable. Frame this as the mechanism unifying the 8-bit null and the 4-bit harm into one
story, not as a magnitude correction.

Qualify with the decomposition: conditional accuracy falls 0.973 → 0.384, so roughly 44 points
are retrieval and 24 are generation, and the design cannot separate them further because the
same quantization causes both. State that as a limitation of the experiment, not a caveat on a
number. The dose-response is monotonic in both accuracy and degeneracy, but realized quantized
tokens are ~31% of target because protection seats FULL first — so it establishes the mechanism
and does not license a rate.

**Why the promotion result is a null and not a failure.** The oracle is verified as a genuine
ceiling: never oversubscribed, zero credential tokens ending quantized. The epiphany signal is
verified as informative, beating random by a margin comparable to attention's own. And the
binding constraint is arithmetic — 52.3% of credential tokens are evicted before promotion runs,
so the ceiling on any promotion signal is bounded before the signal is chosen. **Lead the
saturation argument with that number, not with the tied arms.** The tie confirms; the arithmetic
explains.

Scope it precisely: this bounds the promotion decision under attention-ranked retention, which
is what the methods under study use. It says nothing about a system co-designing both stages,
and the factorial cannot reach that. Say so in the body.

### Two things that must not be smoothed over

**The reproducibility audit is a contribution, not an appendix.** Two defects in the primary
implementation — a biased credential alphabet and sum-versus-mean attention accumulation — were
found only by cross-implementation comparison, after five rounds of internal validation missed
both, because each was self-consistent within its own implementation. The AMD side independently
carried a report-generation dedup bug that produced a spurious result in their own table. Each
implementation had a blind spot the other's design happened to cover: their oracle-versus-full-
cache tripwire could not detect the sum defect in the primary harness, because that oracle never
consults attention. That is a stronger argument for independent reimplementation than anything
in the prior paper, and it should be a numbered section.

**Report the gate failure honestly.** Calibration validated five gates at one budget, then a
five-budget sweep ran without re-deriving competitive room per cell. Two budgets were
arithmetically void — at B=51 the recency floor alone exceeds the budget — and one more was
saturated. The exclusion criterion is arithmetic and was computable before launch, which is what
makes it legitimate rather than post-hoc. Say that plainly. Note also that the same failure
recurs in the opposite direction on Qwen 3B, where protection-alone lands below band at two of
three budgets.

### Structure

Follow the prior paper's shape: abstract stating the results as numbers, introduction,
background and setup, a measurement-methodology section carrying the defect ledger, one section
per result, a reconciliation section on the cross-implementation and cross-model comparison, a
section of predictions that failed, limitations restating scope in full, conclusion. Appendices
for the specification, the gap list, and the reproduction protocol.

Every timing or accuracy figure carries its model, precision, budget as a *retention fraction*
rather than a raw token count, and context length. Llama's context median is 916.5 against
Qwen's 1030, so the same token budget is a different fraction — never compare raw counts across
families.

### Predictions that failed — include these

- The shared-bias-cancels prediction (held at 8-bit, broke at 4-bit for a regime-specific reason).
- The threshold hypothesis for the bit-width collapse (predicted a mixed population, found
  near-universal collapse).
- The prediction of high conditional accuracy at 514 (found the compound case instead).
- H-ORTH itself, the project's own hypothesis, falsified against a verified-informative
  orthogonal signal.

### On writing

Write plainly. Short sentences. No em-dash asides, no "not X but Y" constructions, no rhetorical
questions. Let the numbers carry the argument. Where a claim is uncertain, say what would settle
it rather than hedging with adverbs.

Do not manufacture a relationship to future work. State that identifying an optimal retention
signal is an open problem and stop. Note in related work that this problem is actively contested
— LU-KV (arXiv 2602.08585) formulates head-level budget allocation as combinatorial optimization
over long-horizon marginal utility, and ForesightKV (arXiv 2602.03203, ICML 2026) trains a ranker
distilled from a Golden Eviction oracle using future attention scores. **Verify both against
arXiv before citing; do not cite anything not checked.**

### Adaptation

These instructions encode what the evidence supports, not a fixed outline. If reading the record
suggests a better organization, a sharper framing, or a result under- or over-weighted, follow
the evidence and say what changed and why. If a number contradicts something above, the number
wins.

Two things are not adaptable. Do not upgrade a scoped claim to a general one. Do not omit a
defect, a failed prediction, or an excluded cell.

Deliver the paper as a single markdown file. Alongside it, produce a short list of every claim
that could not be fully sourced from the record, so it can be checked before submission.

---

## Blockers to resolve before this can be executed

1. **`PROJECT_RECORD.md` does not exist in the repo.** The prompt names it as a primary source.
   Either it is on another machine, or it is `ANALYSIS.md` under a different name. Confirm which.
2. **The prior paper, *Where the Cost Goes*, is not in this repo.** The prompt says "Read it"
   and sets it as the register and quality bar. It needs to be supplied.
3. **`amd/results*/` raw JSONL is present**, but the prompt references an AMD
   `make_report.py` dedup bug producing a contaminated Phase 2 table. That bug is not documented
   in their `FINDINGS.md` as read on 2026-09-05 — their stated Phase 2 correction was the
   P4/P5 signal contamination (rho +0.4974 / +0.8489). Confirm whether the dedup bug is a
   separate, later-discovered defect and which AMD Phase 2 table is current.
4. **Llama reconstruction error**: the prompt cites 48.7% at 3-bit; `amd/FINDINGS.md` quotes
   24.28%/18.90% at 4-bit and does not give a 3-bit figure in the same table. Trace before use.
