# Paper 3 — Limitation: completeness indexes retention, not usability

**Status: named open question, handed to Paper 4. Deliberately not investigated inside Paper 3.**

Paper 3 asked what the functional unit of KV retention is. The answer it supports is co-retention:
a fact is usable only if its tokens survive together, and completeness — whether a fact survived
whole — is the right measure *of retention*. Both boundary conditions hold exactly (`c_eff` = 1.00
for a contiguous policy in every cell; `q_complete = p_g` at `c` = 1), and holding the same gold
tokens coherently at the same budget raises accuracy 2.6×.

What Paper 3 does **not** establish, and the evidence below shows cannot be established from
completeness alone, is when a complete fact is *usable*. The step from completion to accuracy is not
determined by completion, by any completeness measure we recorded, or by the number of other whole
records retained alongside it. That step is the limit of this paper's framework.

All numbers below are Stage 4, final n: generation n = 100 per cell (a stated deviation from the
registered 200), captures n = 200. Cells are admissible under the frozen PREREG_P3 §9.1 rule.

---

## Evidence 1 — no completion-based predictor can fix the crossover misses

Crossover agreement on the 22 admissible cells, for the registered arms (snapkv, adakv_snapkv),
by what supplies the completion estimate:

| completion estimate | agreement |
|---|---|
| F5 prediction (the registered winning form) | 19 / 22 |
| **measured** per-slot completion `q_complete` | 18 / 22 |
| **measured** union completion `q_any` | 13 / 22 |

F5 does as well as the measured per-slot completion itself, so the misses are not a modelling
error: a perfect predictor of completion would not remove them. The union measure is markedly worse,
replicating Stage 1's finding that union availability over-calls usability.

The three misses share one structure — a method whose measured completion is **below** the floor's
nonetheless scores **above** it:

| model | c | C | floor q / acc | snapkv q / acc | adakv_snapkv q / acc |
|---|---|---|---|---|---|
| M2 | 8.2 | 256 | 0.181 / 0.072 | 0.110 / **0.102** | 0.108 / **0.117** |
| M3 | 8.9 | 256 | 0.166 / 0.080 | 0.087 / 0.040 | 0.090 / **0.098** |
| M3 | 8.9 | 512 | 0.305 / 0.247 | 0.167 / 0.117 | 0.189 / **0.375** |

At M3 c = 8.9, C = 512, adakv_snapkv completes 0.189 of queried facts per slot against the floor's
0.305 and answers 0.375 against the floor's 0.247.

## Evidence 2 — the same failure at c = 1, where co-retention cannot be the cause

At `c` = 1 (MARK-1) a fact is one token, so completeness equals the per-token keep rate and there is
nothing to co-retain. In 3 of 40 (model × method × budget) cells a method retains the gold token
**more** often than the floor and still scores **below** it:

| model | arm | C | method q_mean | method q_any | method acc | floor q | floor acc |
|---|---|---|---|---|---|---|---|
| M2 | keydiff | 32 | 0.091 | 0.985 | 0.045 | 0.046 | 0.075 |
| M2 | keydiff | 64 | 0.168 | 1.000 | 0.055 | 0.061 | 0.100 |
| M3 | keydiff | 32 | 0.142 | 1.000 | 0.060 | 0.064 | 0.085 |

Replicated on both models, confined to KeyDiff at tight budget. The gold token is present in at least
one KV slot in 98.5–100% of instances, and per-slot retention is about twice the floor's, yet accuracy
is lower. **Both** completeness measures over-call usability here, and because `c` = 1 the explanation
cannot be a failure to keep a fact's tokens together.

Evidence 1 and Evidence 2 are independent: different task families, different fact costs, different
budgets. Together they say that no completeness measure recorded in this programme indexes usability.

## Evidence 3 — what the failures look like (descriptive, an upper bound)

When the recency floor fails at C ∈ {256, 512}, the generated answer usually contains **no word of
the gold answer**:

| model | c | failures | no gold word, share of failures | partly correct, share of failures |
|---|---|---|---|---|
| M2 | 8.2 | 696 | 1.000 | 0.000 |
| M2 | 18.9 | 719 | 0.812 | 0.188 |
| M2 | 39.5 | 714 | 0.560 | 0.440 |
| M3 | 8.9 | 669 | 0.949 | 0.051 |
| M3 | 18.6 | 617 | 0.877 | 0.123 |
| M3 | 39.3 | 746 | 0.787 | 0.213 |

So **56–100% of the floor's failures contain no gold word.** Inspected generations are dominated by
answers that reproduce a *different* record's fields, e.g. `R033|Dornolhardt|Procurement|285057` for
gold `Ellachardt | Procurement | 169011`.

Three cautions, stated because they bound what this table can claim:

- **"No gold word" is an upper bound on wrong-record copying, not a count of it.** It also includes
  corrupted near-copies of the right record — `Bardenwick` for `Caldenwick` with the correct value was
  observed — and at `c` ≈ 8, where the gold answer is a single surname, any corruption lands in this
  category by construction (partly correct is impossible there, hence the 0.000). A direct count would
  need each generation matched against the instance's other records, which was not done.
- **The partly-correct share grows with `c`** (M2: 0.000 → 0.188 → 0.440), so at long facts a large
  part of the floor's failure is corrupted or truncated copying of the right record, not retrieval of
  the wrong one.
- **Correction to an earlier interim figure.** An interim update reported "68–87% wrong-record". That
  figure used the wrong denominator (all generations, not failures), omitted the M2 c ≈ 40 cell, and
  named the category more strongly than the measurement supports. The table above supersedes it.

## Evidence 4 — completeness at its maximum, usability halves (observation, not a registered test)

On the `k` axis (one fact split into `k` adjacent labelled parts, fact cost matched at ≈19), the
causal oracle keeps **every** queried record whole at C = 512 for both `k` = 1 and `k` = 2 — its
completion is at the ceiling by construction. Its accuracy on M3 nonetheless falls by half:

| M3, oracle_causal | k = 1 | k = 2 |
|---|---|---|
| C = 512 accuracy | 0.730 | 0.370 |
| C = 64 accuracy | 0.318 | 0.125 |
| full_cache anchor (Stage 3 gate, n = 24) | 0.844 | 0.844 |

Uncompressed, splitting the record costs nothing. Compressed, with the split record kept complete,
usability halves.

This sits inside the accuracy-across-`k` contrast, which is confounded with answer length (matching
`c` across `k` shortens the answer; see the Stage 4 k-axis design limit). The confound, however,
predicts the **opposite** sign: the `k` = 2 answer is shorter (M3 13.3 → 10.2 tokens), which should
make it easier, and on M2 — where the answer shortens more, 14.2 → 6.2 — the same oracle's accuracy
**rises**, 0.492 → 0.820. So the M3 drop is not explained by the confound's direction.

Recorded as an observation with its limits: M3 only; the accuracy contrast across `k` remains
confounded and is not adjusted; the full-cache comparison rests on n = 24; nothing here was
registered. What it adds is the cleanest instance yet of the boundary this document states — the arm
with **no retention error at all** loses half its usability when the retained unit is rearranged,
so completeness cannot be what indexes usability.

## Ruled out — interference from other whole records

A natural mechanism is retrieval interference: a policy that keeps many whole records makes the
queried one harder to find. This was tested with its hypothesis and decision rule committed before it
ran on the final data (commit `32be79f`). Within-cell design, so budget, cost and policy are held fixed
while instances vary in how many non-queried records happen to survive whole (D):

| | effect of one extra whole record on accuracy, at fixed completion | 95% CI (bootstrap over cells) |
|---|---|---|
| M2 floor | +0.016 | [−0.013, +0.041] |
| M3 floor | −0.014 | [−0.025, +0.018] |
| M2 methods | +0.050 | [+0.012, +0.081] |
| M3 methods | +0.049 | [+0.005, +0.075] |

**Verdict under the committed rule: not supported.** The floor is the clean test — one global keep-set,
no averaging across KV heads — and shows no effect on either model (few cells, 5 and 7, so the CIs are
wide for their width). For methods the sign is reversed but rests on almost no usable variation: within
a cell D and per-slot completion are collinear at r = −0.78 (M2) and −0.83 (M3), because budget spent
on other whole records is budget not spent on queried ones. The positive method coefficient is
therefore reported as unexplained, not as evidence that extra records help. One proxy explanation was
eliminated: D is negatively correlated with union availability of the queried facts (r = −0.43, −0.34)
while availability predicts accuracy positively, so D standing in for availability would push the
coefficient negative, not positive.

**The cross-cell version of the same question would have looked like support** — partial correlation
of conversion with D given log B and c is −0.51 (M2) and −0.47 (M3). The two designs disagree in sign;
the cross-cell result is confounded by budget and is not the finding.

## Deliberately not pursued

A second candidate — interference from records retained *partly* but not whole, fragments that look
like records without being answerable — is available in the existing captures and was **not run**. It
would be a post-hoc test immediately after a null, on a question outside PREREG_P3's scope, inside a
stage that was otherwise complete. The completion→accuracy step is a different question from the one
Paper 3 set out to answer, and investigating it here would trade a bounded result for an open-ended
one.

## Why this is Paper 4's question

Paper 4 is meant to propose a set-aware retention policy. Paper 3 shows that such a policy must keep
facts whole — necessary — and that keeping them whole is not sufficient: a retained set can be
complete and still unusable, and whether it is usable is not predicted by completeness, by union
availability, or by how many other whole records surround it. A set-aware policy therefore needs a
model of what makes a retained set **usable**, and Paper 3 supplies the three failure cases above as
its starting evidence:

1. crossover misses where lower completion yields higher accuracy (Evidence 1);
2. a single-token over-call where co-retention cannot be the cause (Evidence 2);
3. failures dominated by answers with no gold word, bounding how much of the gap is wrong-record
   copying versus corrupted copying (Evidence 3);
4. an oracle holding every queried fact whole whose usability halves when the fact is split into
   two adjacent parts, on M3, against the sign the answer-length confound predicts (Evidence 4).

Together with Stage 2.3's finding that `full_cache` itself falls from 0.910 to 0.400 when one fact's
two halves are scattered, this bounds Paper 4's claim before it is made: retention policy can make
facts available, but whether the model can use what is available is a separate capability that
retention does not control.
