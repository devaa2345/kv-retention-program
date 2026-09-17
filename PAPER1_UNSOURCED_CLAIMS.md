# Claims in PAPER1.md I could not fully source from the record

Check each before submission. Ordered by how much it would matter if wrong.

## 1. Two numbers from the writing prompt that I could not trace, and therefore did not use

**The AMD `make_report.py` dedup bug.** The prompt says the original AMD Phase 2 table is
"contaminated by their `make_report.py` dedup bug" and must be treated as superseded. I could
not find this defect anywhere in `amd/FINDINGS.md`, `amd/AGREEMENT.md` or `amd/SPEC_QUESTIONS.md`.
What those documents *do* record is a different Phase 2 defect: two promotion signals (P4
rotation, P5 oracle) were contaminated by falling back to the eviction score, measured at
ρ = +0.4974 and +0.8489 at n=100, caught by their own manipulation check, corrected, and Phase
2 re-run in full with the corrected run reported as primary.

**I used their corrected Phase 2 table** (§9.3 of the paper) on the assumption that this is
what "superseded" refers to. If the dedup bug is a separate and later discovery, §9.3 may be
citing the wrong table, and the direction of the cross-implementation disagreement on the
promotion question could change. This is the single most important item to check.

**Llama 3-bit reconstruction error of 48.7%.** The prompt cites this figure. `amd/FINDINGS.md`
gives Llama-3.2-3B error at 8-bit (1.44%/1.12%), 4-bit (24.28%/18.90%) and 2-bit
(133.36%/98.58%), with no 3-bit row. I used the 4-bit figure of 24.28%, which is sourced, and
did not use 48.7% anywhere.

## 2. Sources named in the prompt that are not in this repository

**`PROJECT_RECORD.md` does not exist.** The prompt names it as a primary source. The paper is
built from `ANALYSIS.md`, `PREREG.md`, `ADDENDUM_2026-09-03.md`, the `amd/` tree, and the raw
JSONL under every `results/` directory. If `PROJECT_RECORD.md` exists on another machine and
contains anything not in `ANALYSIS.md`, the paper has not seen it.

**The prior paper, *Where the Cost Goes*, is not in this repository.** The prompt instructs
"read it" and sets it as the model for register and standard, and as the confidence ceiling. I
could not read it. I wrote to the register the prompt describes: plain sentences, structural
claims, failures in the body, scope stated hard. Whether it matches the prior paper's actual
voice is unverified. The instruction "do not exceed it in confidence" is also unverified,
though the grading in §4 through §8 is deliberately conservative.

## 3. Numbers taken from the AMD implementation on their authority

These are sourced to `amd/FINDINGS.md` but were produced by their harness and I did not
re-derive them from their raw JSONL.

- Their Phase 1 8-bit interactions (+0.0011, −0.0011, −0.0056) and Qwen2.5-3B interactions
  (−0.0011, +0.0000, +0.0022). Cited in §4.3 and the abstract.
- Their three-model sweep table in §8.3, including all Llama-3.2-3B accuracies.
- The distractor ablation in §9.2 (arm 2 rising 0.150 to 0.907, no-eviction 0.880).
- Their ceiling diagnosis in §9.3 (0.7989 strict, 0.8633 prefix-tolerant, 0.9544 at edit
  distance 1; 58 of 181 failures being prefix omissions).
- Their quantizer error figures and context median.
- Their promotion-signal contamination values (ρ = +0.4974, +0.8489) and corrected values.
- The claim that arms 2 and 4 evict "exactly the same 92.1 credential tokens", which I
  referenced as mechanism support in §6 without independent verification.

Their raw rows are in `amd/results*/`. Re-deriving the three-model sweep and the distractor
ablation would be the highest-value check, since both carry load in the paper.

## 4. Inferences stated as such, but worth flagging

**§9.2, that the distractor value format explains our protection arm being ~2.5× theirs.** I
wrote this as "plausibly explains". It is a mechanism argument supported by their ablation and
by our confirmed difference in distractor format, not a measurement. Nobody has run our
distractor format through their harness or theirs through ours. That experiment would settle
it.

**§9.3, that our failures are single-character miscopies "in 8 of 10 sampled cases".** This is
from a 30-prompt verification run of the no-eviction arm, 180 answers, of which 10 failed. It
is a small sample and I described it as such. It argues against a purely-scoring explanation
for the ceiling gap but does not establish a behavioural one.

**§8.3, the four-different-curves count.** This aggregates our sweep, their 1.5B sweep, their
3B sweep and their Llama sweep. Our sweep and their 1.5B sweep were run under different
retention rankings (we corrected to mean; they used mean throughout) and different distractor
formats, so "four runs, four curves" is a fair description of the evidence but the four are not
four replications of one protocol.

## 5. Things I changed from the prompt, and why

**Title.** The prompt asked for a title naming the measurement rather than the verdict. I used
"Retention and Promotion: A Stage-Level Measurement of Recoverable KV Cache Eviction".

**I promoted the iso-token/iso-memory result to Established rather than leaving it implicit.**
The prompt calls it "the second-strongest thing in the paper" but the three-tier grading lists
only three findings. It is n=150, two accountings, and the sensitivity sweep covers the one
free parameter, so it belongs at the same confidence level as the Phase 1 null. It is §5.

**I led §7 with the 52.3% arithmetic**, as instructed, and put the tied arms second.

**I did not use the phrase "retention beats precision" anywhere**, per the instruction that it
is a punchline.

**I added §9.4 on the mutual blind spot** because it strengthens the reproducibility argument
in §10 and was in the record but not in the prompt's outline.

## 6. Not verified because it needs a run

Nothing in the paper requires a new measurement. Two open questions are named in the body as
open rather than answered: the direction of the cross-implementation disagreement on the
promotion signal (§9.3), and the mechanism of the Qwen bit-width collapse (§8.3).
