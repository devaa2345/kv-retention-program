# Retention and Promotion: A Stage-Level Measurement of Recoverable KV Cache Eviction

## Abstract

Recoverable KV cache eviction adds a quantized middle tier between full precision and
deletion. Tokens demoted to that tier can be promoted back when they appear useful again.
The mechanism has two stages. A retention decision chooses what survives at all. A promotion
decision chooses what among the survivors is held at full precision. The literature builds
machinery for the second stage. We measure both against their own oracles.

We report three results at three explicitly different confidence levels.

**Established.** Under matched-token accounting, adding a recoverable tier to structural
protection produces no measurable effect. On Qwen2.5-1.5B at 8 bits and retention fractions
0.15, 0.25 and 0.50 the recoverability effect given protection (arm 4 minus arm 2) is +0.001,
+0.002 and −0.002, and the pre-registered interaction (difference in differences) is +0.000,
+0.003 and −0.002. All 95% intervals are inside ±0.05, and the arms differ on only 3 to 9 of
150 prompts. An independently written implementation on different hardware reports an
interaction of +0.001, −0.001 and −0.006 at the same points, and −0.001, +0.000 and +0.002 on
Qwen2.5-3B. Two implementations, two model scales, three budgets. Retention 0.15 is a
qualified budget (§3.2), so the clean evidence is retention 0.25 and 0.50.

**Scoped.** At one operating point where the tier is neither free nor destructive, no
promotion signal helps. On Qwen2.5-1.5B at 4-bit and retention 0.25, five signals spanning a
perfectly circular one (Spearman ρ = +1.000 against the eviction score) to a verified oracle
land between 0.154 and 0.216 fraction retrieved. The oracle is 0.188 and does not beat
circular attention at 0.203. The explanation is arithmetic rather than behavioural. 52.3% of
credential tokens are evicted before promotion runs, so the ceiling on any promotion signal
is fixed before the signal is chosen.

**Reported, not explained.** Task accuracy is not monotone in quantization bit-width while
reconstruction error is. This is a property of one model family. Qwen2.5-1.5B collapses to
0.000 at 5 bits and recovers to 0.247 at 4 bits under twice the reconstruction error.
Qwen2.5-3B collapses at 4 bits and partially recovers at 3. Llama-3.2-3B shows no collapse at
any width down to 3 bits, with zero fully degenerate prompts, while carrying higher
reconstruction error than either Qwen at every width. Seven candidate explanations were
tested and eliminated.

A fourth result concerns evaluation rather than mechanism. The same experiment scored under
matched bytes rather than matched tokens reverses sign, because the byte-cost accounting
hands the tiered arm roughly 1.6× the tokens. The recoverability effect moves from +0.002 to
+0.532 at retention 0.25 (interaction +0.003 to +0.533). The effect is robust across the byte-cost constant (+0.35 to +0.54 over
0.125 to 0.50), which was fixed after the pre-registration hash.

Two defects in the primary implementation were found only by comparison against an
independent reimplementation, after five rounds of internal validation missed both.

---

## 1. Introduction

A transformer decoder caches keys and values for every token it has processed. The cache
grows linearly in context length and is often the binding memory constraint in deployment.
Eviction policies discard entries to bound it. Eviction is irreversible: a discarded token
cannot contribute again, even if it becomes relevant later.

Recoverable eviction addresses this. Instead of two states, entries occupy three. High
confidence entries stay at full precision. Intermediate entries are quantized and retained.
Only the lowest tier is discarded. A quantized entry can be promoted back to full precision
when its measured importance rises. The promotion rule is where the design effort goes.

This paper separates the mechanism into two decisions and measures each one.

The **retention decision** determines which positions remain in the cache in any form. The
**promotion decision** determines which of the retained positions are held at full precision
rather than quantized. These are different questions and they can be measured against
different oracles. An oracle for retention knows which positions will be needed. An oracle
for promotion knows which of the retained positions deserve fidelity. The second oracle is
bounded by the first.

Our measurements say the promotion decision is saturated at the operating point we could
test, while the retention decision has substantial unexploited headroom at the same budget.
On Qwen2.5-1.5B at retention 0.25, structural protection with attention-ranked retention
scores 0.423 fraction retrieved. A retention oracle at the same budget scores 0.943. The gap
is 0.520. Within that budget, a verified promotion oracle buys 0.000 over a circular signal.

The result is not that promotion is useless in general. It is that at the one precision on
this model where the promotion decision has a measurable effect on the outcome, no signal we
tested moves it, and the reason is a constraint imposed upstream.

We also report a measurement problem that is independent of the mechanism. Whether a
recoverable tier appears beneficial depends entirely on whether the comparison holds tokens
constant or bytes constant. Under matched tokens the effect is null. Under matched bytes it
is large and positive. Both accountings are defensible. The literature does not consistently
say which it uses.

### 1.1 Contributions

1. A stage-level decomposition of recoverable KV eviction with resource accounting
   controlled on both axes, measured on two model families and reproduced by an independent
   implementation.
2. A quantification of promotion-stage headroom, bounded arithmetically rather than
   inferred from tied arms.
3. Evidence that a recoverable tier at aggressive precision actively harms, with the harm
   growing as the retention policy improves, because the cold tier sits directly downstream
   of the selection stage.
4. A reproducibility result. Two defects in our implementation were invisible to five rounds
   of internal validation and were found only by an independently written implementation of
   the same specification.

---

## 2. Background and setup

### 2.1 The mechanisms under study

Attention-score eviction ranks cached positions by accumulated attention received and
discards the lowest ranked. Structural protection reserves cache for positions matched by a
content-agnostic pattern, independent of their attention score. Recoverable tiering adds a
quantized tier between retention and deletion, with a promotion rule that can return a
quantized position to full precision.

We treat protection and tiering as two factors in a 2×2 design, plus two references. Arm 1
is attention-ranked permanent eviction. Arm 2 adds structural protection. Arm 3 adds
recoverable tiering to arm 1. Arm 4 has both. Arm 5 is a retention oracle that knows which
positions matter and fixes its keep-set once. Arm 6 never evicts.

### 2.2 Task

The task is multi-credential retrieval with induced dormancy. Each prompt contains a
configuration dump with six credential lines of the form `CRED_i_KEY: sk-<14 hex>` and
twenty distractor lines with different labels, separated by filler paragraphs. Six
conversational turns each query one credential in shuffled order. A credential counts as
retrieved if its full 17-character value appears verbatim in the answer to the turn that
asked for it. The prompt score is the fraction retrieved, k/6.

Query order is the dormancy manipulation. A credential asked last has had five intervening
turns during which its cache entry could be demoted and must be recovered.

Context length is 1029 tokens median under the Qwen2.5 tokenizer (range 1011 to 1042,
n = 10). Under the Llama-3.2 tokenizer the same task text is 916 tokens median. Budgets are
therefore reported as retention fractions throughout, since the same token count is a
different fraction across tokenizers.

### 2.3 Configuration

Qwen2.5-1.5B-Instruct, bfloat16, eager attention so attention weights are readable, greedy
decoding. A single NVIDIA RTX 5070, 12 GB. Six credentials of 14 hex characters, twenty
distractors, 50-word filler paragraphs. Sink position 0 and the 64 most recent positions are
retained unconditionally in every arm. Selection runs every decode step. One cache spans all
six turns, so eviction persists and generated tokens are themselves evictable.

The quantized tier is simulated by an affine min-max quantize-dequantize round trip applied
per position over the `(kv_heads, head_dim)` slice, arithmetic in the engine's bfloat16.
Byte cost is `n_full + quant_byte_cost × n_quant` with `quant_byte_cost = 0.25`.

Model, precision, retention fraction and context length are carried on every figure below.

### 2.4 Pre-registration

The analysis was fixed before Phase 1 ran and hashed (`PREREG.md`, sha256 `5d3c7e3b…`).
Primary metric fraction retrieved. The primary estimand is the interaction, defined as the
difference in differences (arm 4 − arm 3) − (arm 2 − arm 1). The estimator is a bootstrapped
median of paired per-prompt differences with 10,000 resamples clustered on prompt, with
Benjamini-Hochberg correction within each accounting condition, minimum effect of interest
0.05, and no functional form assumed across budgets. Deviations are recorded in §3.3 and §3.4
rather than absorbed.

Two quantities are reported throughout and must not be confused. The **interaction** is the
pre-registered difference in differences. The **recoverability effect given protection** is
arm 4 minus arm 2, the effect of adding a recoverable tier when protection is already on. The
draft as first written reported the second under the name of the first (D-09, §3.4). Both are
now stated.

---

## 3. Measurement methodology

### 3.1 Calibration gates

Five gates had to pass before any inferential run. G1 requires the retention oracle to reach
0.95, otherwise the instrument is measuring something other than retention. G2 requires the
unprotected floor at or below 0.10. G3 requires protection alone to land between 0.45 and
0.75, so the factor under test has room to move in either direction. G4 requires the
no-eviction reference at or above 0.95. G5 requires credentials to actually go dormant.

At six credentials and budget 280 tokens (retention 0.27), all five pass: 0.983, 0.033,
0.567, 0.983, 1.000.

G3 exists because of a specific failure. An earlier pilot had structural protection pinning
every credential unconditionally, so the recoverable tier had nothing to recover and all four
cells of the 2×2 were inert. The factor under test could not move the outcome. A gate that
checks for dynamic range catches this; a gate that checks only floor and ceiling does not.

### 3.2 The gate was validated at one budget and applied at none

Calibration cleared all five gates at a single budget. Phase 1 then swept five budgets
derived as fractions of mean context length without re-deriving each cell's competitive room
against those same gates.

Competitive room is `B − recency_window − sink`, which is `B − 65`. At B = 51 that is −14.
The 65-token retention floor exceeds the entire budget, so every arm retains 65 tokens
regardless of policy and all five collapse to an identical 0.013. At B = 103 competitive room
is 38, the budget is honoured, but the retention oracle reaches only 0.271 and G1 fails badly.

Two of five budgets are therefore excluded from the primary analysis. The criterion is
arithmetic and was computable before launch. That is what separates it from a post-hoc
exclusion. The rows remain in the results files.

| retention fraction | budget | competitive room | budget honoured | oracle | status |
|---|---|---|---|---|---|
| 0.05 | 51 | −14 | no, retains 65 | 0.013 | void |
| 0.10 | 103 | 38 | yes | 0.271 | void |
| 0.15 | 154 | 89 | yes | 0.648 | qualified |
| 0.25 | 257 | 192 | yes | 0.943 | valid |
| 0.50 | 514 | 449 | yes | 0.958 | valid |

Only retention 0.25 and 0.50 clear every gate. Retention 0.15 has real dynamic range but its
own oracle sits at 65% of achievable, so it is reported separately and never pooled with the
other two. Any statement about how an effect moves with budget rests on two clean points plus
one qualified one, not a five-point gradient.

The same failure recurs in the opposite direction on Qwen2.5-3B, where the independent
implementation found protection alone below band at two of three budgets and could not bring
the model into the competence band at all across eight configurations.

### 3.3 The estimator

The pre-registration fixed a bootstrapped median of paired per-prompt differences. On this
data that estimator is degenerate. The differences are overwhelmingly exactly zero, so the
median is zero in nearly every resample and the interval collapses to zero width. Reported
medians were `[0.000, 0.000]`, `[0.167, 0.167]` and `[0.500, 0.500]`. The non-zero values are
exactly 1/6 and 3/6, the granularity of a six-credential score. A zero-width interval is an
artifact of summarising a mostly-constant discrete distribution with a median.

We report the mean paired difference with its bootstrap interval, alongside the count of
prompts on which the arms differ at all. The count does not depend on any estimator choice
and is the stronger statement. This is a deviation from the pre-registration and is recorded
as one.

A second deviation concerns the estimand. Every figure first labelled "interaction" in this
work was arm 4 minus arm 2, not the pre-registered difference in differences. The two are
close where arms 1 and 3 sit at the floor and the tier is free, and they separate where the
tier is damaging. At 4 bits the difference in differences is −0.0067 [−0.0178, +0.0044] at
retention 0.15 and −0.0122 [−0.0267, +0.0022] at 0.25, both intervals including zero, and
−0.6767 [−0.7122, −0.6389] at 0.50. The arm 4 minus arm 2 values that excluded zero at all
three budgets (§6) are a different quantity. Both are reported in every table below. Bootstrap
intervals in this paper are recomputed from the raw rows with a fixed seed
(`verification/rederive_interaction.py`) and can differ from earlier interim figures in the
fourth decimal.

### 3.4 Defect ledger

Nine defects changed a reported number or invalidated a comparison. Seven were found
internally. Two were found only by an independent implementation.

| ID | Defect | Found by |
|---|---|---|
| D-01 | Qwen2.5 emits repeated punctuation in fp16 on this stack, including under an unmodified `generate()` call | Reference run with no custom code |
| D-02 | Structural protection covered a fixed 12 characters after each label, shorter than the 27-character credential it was protecting | Inspecting generated output |
| D-03 | No recency floor. Attention ranking evicted the tokens immediately preceding the answer, breaking every arm including the oracle | Oracle below its own ceiling |
| D-04 | The retention oracle sliced an unordered Python `set` when trimming to budget, dropping an arbitrary non-reproducible subset of credentials. It also lacked credential labels, leaving surviving values unattributable | Ceiling shortfall |
| D-05 | Interim analysis pooled matched-token and matched-byte rows for the tiered arms under one nominal-budget label | A query that would not reproduce |
| D-06 | The factor under test was inert, twice. Protection pinned credentials so tiering had nothing to recover; later, the 8-bit tier carried no quality signal | G3 gate, then a smoke test |
| **D-07** | Credential values were not uniform hex. `string.hexdigits.lower()` is a 22-slot string in which a–f appear twice, so a–f were drawn at 9.09% each against 4.55% for digits | **Independent reimplementation** |
| **D-08** | Retention ranking accumulated attention summed over queries, carrying a −0.717 correlation with position | **Independent reimplementation** |
| D-09 | The quantity reported as the interaction was arm 4 minus arm 2, not the pre-registered difference in differences | Pre-submission re-derivation from raw rows |

The transferable practice is to assert an invariant on live data rather than on a
reimplementation of it. Our strongest single check asserted that the number of distinct values
in a quantized tensor is at most 2^bits, on 10,640 quantization calls intercepted during real
generation. That form of check would also have caught D-04 and D-05. Separately, three of the
nine defects surfaced because a reference arm behaved impossibly: an oracle below its own
ceiling, a policy beating a no-eviction baseline, five arms returning byte-identical results.
Reference arms earn their compute as tripwires, not only as denominators.

D-07 and D-08 are treated in full in §8, because how they were found is a result.

---

## 4. Result 1, established: no interaction under matched tokens

Structural protection and recoverable tiering address the same failure, which is information
becoming unreachable. If that is right, once you have either, the other adds nothing. We test
this in a 2×2 factorial at matched retained-token count, using the pre-registered interaction
(difference in differences) as the primary estimand and the recoverability effect given
protection (arm 4 minus arm 2) as a companion.

### 4.1 Arm levels

Qwen2.5-1.5B, 8-bit tier, matched tokens, context 1029, n = 150.

| retention | no protection | protection | tiering only | protection + tiering | retention oracle | no eviction |
|---|---|---|---|---|---|---|
| 0.15 | 0.011 | 0.196 | 0.012 | 0.197 | 0.648 | 0.947 |
| 0.25 | 0.010 | 0.423 | 0.009 | 0.426 | 0.943 | 0.947 |
| 0.50 | 0.010 | 0.964 | 0.010 | 0.962 | 0.958 | 0.947 |

### 4.2 The interaction and the recoverability effect

| retention | interaction (DiD) | 95% CI | prompts differing | recoverability effect given protection (arm 4 − arm 2) | 95% CI | prompts differing |
|---|---|---|---|---|---|---|
| 0.15 | +0.0000 | [−0.0044, +0.0044] | 4 / 150 | +0.0011 | [−0.0022, +0.0056] | 3 / 150 |
| 0.25 | +0.0033 | [−0.0022, +0.0100] | 6 / 150 | +0.0022 | [−0.0033, +0.0089] | 5 / 150 |
| 0.50 | −0.0022 | [−0.0100, +0.0056] | 9 / 150 | −0.0022 | [−0.0100, +0.0056] | 9 / 150 |

Every interval is inside ±0.05, so equivalence is declared at all three budgets under the
pre-registered criterion, for both quantities. Arms 1 and 3 sit at the floor (0.009 to 0.012)
in every one of these cells, which is why the two estimands nearly coincide here.

The equivalence is stronger than the intervals convey. The arms are not merely close on
average. They are frequently identical. At retention 0.25, protection and protection plus
tiering return the same score on 145 of 150 prompts. Of the five that differ, two favour
protection alone and three favour the combination. Retention 0.15 is a qualified budget
(§3.2), so the clean evidence is the last two rows.

Protection itself is large and unambiguous over the same range: +0.169, +0.457 and +0.924
under corrected retention ranking, with intervals far from zero. The null is specific to
adding recoverability on top of protection. It is not a null on the protection factor.

### 4.3 Independent replication

An independently written implementation of the same specification, on an RX 7900 XTX under
ROCm, reports +0.0011, −0.0011 and −0.0056 at the same three retention fractions, with all
three intervals containing zero. On Qwen2.5-3B it reports −0.0011, +0.0000 and +0.0022. Their
estimand is the difference in differences, so the like-for-like comparison is with our
interaction column above. These are their reported values and were not re-derived here.

Two implementations, two model scales, three budgets, and in several cells zero or near-zero
prompts differing out of 150. This is the most robust finding in the work and it carries the
paper.

---

## 5. Result 2, established: the same experiment reverses under byte accounting

A recoverable tier holds evicted content in quantized form. At a budget defined as a count of
retained tokens, it therefore consumes more memory than permanent eviction at the same
budget. Both accountings are defensible and we ran both.

Under matched tokens, arms hold the same number of retained positions. Under matched bytes,
arms hold the same accounted bytes, which at `quant_byte_cost = 0.25` and a half-and-half
split grants the tiered arm `T = B / 0.625 = 1.6 B` tokens.

Recoverability effect given protection (arm 4 minus arm 2):

| retention | matched tokens | 95% CI | matched bytes | 95% CI |
|---|---|---|---|---|
| 0.15 | +0.001 | [−0.002, +0.006] | **+0.199** | [+0.179, +0.219] |
| 0.25 | +0.002 | [−0.003, +0.009] | **+0.532** | [+0.504, +0.559] |
| 0.50 | −0.002 | [−0.010, +0.006] | −0.051 | [−0.068, −0.034] |

Interaction (difference in differences), same rows:

| retention | matched tokens | 95% CI | matched bytes | 95% CI |
|---|---|---|---|---|
| 0.15 | +0.000 | [−0.004, +0.004] | **+0.201** | [+0.181, +0.222] |
| 0.25 | +0.003 | [−0.002, +0.010] | **+0.533** | [+0.507, +0.560] |
| 0.50 | −0.002 | [−0.010, +0.006] | −0.064 | [−0.083, −0.047] |

The same experiment, scored two ways, gives opposite answers at two of three budgets. The
matched-byte gains track having roughly 60% more retained tokens. They are not evidence about
the reversibility mechanism.

`quant_byte_cost` was fixed after the pre-registration was hashed, so it was swept rather
than asserted. At retention 0.25, n = 150:

| quant_byte_cost | multiplier | tokens granted | recoverability effect (arm 4 − arm 2) | 95% CI |
|---|---|---|---|---|
| 0.125 | 1.78× | 457 | +0.543 | [+0.518, +0.569] |
| 0.25 | 1.60× | 411 | +0.532 | [+0.506, +0.559] |
| 0.30 | 1.48× | 395 | +0.536 | [+0.510, +0.561] |
| 0.50 | 1.33× | 343 | +0.354 | [+0.330, +0.379] |

This sweep ran arm 4 only, so it reports the arm 4 minus arm 2 quantity and not the
difference in differences. The qualitative claim survives the whole plausible range, including a deliberately punitive
0.50 that implies only 2× compression where int8 against bfloat16 is genuinely 4×. The
magnitude moves with the constant and must be reported as a range, +0.35 to +0.54, with the
constant stated. It must not be quoted as a point estimate.

This is a claim about how the literature evaluates recoverable eviction, not about any one
system.

---

## 6. Result 3, established at retention 0.50: at aggressive precision the tier harms, and better retention makes it worse

Phase 1 ran at 8 bits, which the specification never stated. At 8 bits the tier is free. §7
shows that holding retention fixed and varying only precision changes nothing at that width.
The 8-bit comparison therefore measured a regime in which recoverability could not act in
either direction.

Re-run at 4 bits, matched tokens, n = 150:

| retention | interaction (DiD) at 4-bit | 95% CI | prompts differing | recoverability effect given protection (arm 4 − arm 2) | 95% CI | prompts differing | arm 4 − arm 2 at 8-bit |
|---|---|---|---|---|---|---|---|
| 0.15 | −0.0067 | [−0.0178, +0.0044] | 23 / 150 | −0.0122 | [−0.0222, −0.0033] | 19 / 150 | +0.0011 |
| 0.25 | −0.0122 | [−0.0267, +0.0022] | 35 / 150 | −0.0189 | [−0.0322, −0.0056] | 30 / 150 | +0.0022 |
| 0.50 | **−0.6767** | [−0.7122, −0.6389] | 149 / 150 | **−0.6833** | [−0.7189, −0.6456] | 149 / 150 | −0.0022 |

The two estimands differ by at most 0.007 in every row, but they lead to different statements
at the smaller budgets. The pre-registered interaction has an interval that includes zero at
retention 0.15 and 0.25, so it is distinguishable from zero at retention 0.50 alone. The arm 4 minus arm 2 effect excludes zero at all three budgets, but it
is small at 0.15 and 0.25 (about one to two points), and its intervals at those two budgets
overlap, so the data do not show growth between them. The one clean contrast is between 0.25
and 0.50, a jump from about −0.02 to about −0.68. The claim is that the tier harms at
retention 0.50, and that the harm at lower retention is at most small.

The independent implementation reports a negative interaction at retention 0.50, at −0.116,
which agrees in direction with ours and is smaller in magnitude. At retention 0.25 its own
documents disagree: one table gives −0.046 and another gives +0.013 with an interval
including zero for the same cell. We therefore make no cross-implementation claim at 0.25,
where our own interaction interval also includes zero. These are their reported values and
were not re-derived here. §10 addresses why the magnitude at retention 0.50 is smaller.

### 6.1 The mechanism, and a reversal that sharpens it

`quant_slots` scales with the budget. A larger budget sends more of the context to the cold
tier. At retention 0.50, roughly 225 positions are quantized to 4 bits. Since 4-bit
all-quantized collapses outright, corrupting that much context destroys generation.
Protection alone at 0.964 falls to 0.281 once the tier is added. At the smaller budgets far
fewer positions are retained, so far fewer are corrupted, which fits the small effects at 0.15
and 0.25. The scaling is an account of three points, only one of which shows a large effect.

Correcting the retention-ranking defect D-08 made the harm larger, not smaller. Under
corrected ranking at retention 0.50, protection alone is unchanged at ceiling (0.963) while
protection plus tiering falls to 0.162. The recoverability effect given protection moves
from −0.6833 to **−0.8011** [−0.8300, −0.7722], with 150 of 150 prompts differing and
non-overlapping intervals. The mean-ranking rerun covers arms 2 and 4 only, so the
difference in differences is not available for this comparison. The DiD at 4 bits under
summed ranking (−0.6767) sits within 0.007 of the arm 4 minus arm 2 value (−0.6833) because
arms 1 and 3 are at the floor, and we expect the same under mean ranking, but that is an
expectation and not a measurement.

This is the sharpest mechanism in the paper. The cold tier sits directly downstream of the
selection stage. Improving retention feeds the tier more of exactly the tokens retention
correctly identified as valuable, where 4-bit quantization destroys them. The better the
retention policy, the more a lossy recoverable tier costs.

This unifies the 8-bit null and the 4-bit harm into one account. At 8 bits the tier is free,
so nothing happens in either direction. At 4 bits the tier is a sink positioned immediately
after the selection stage. The two are one mechanism observed at two precisions.

### 6.2 What the experiment cannot separate

The harm is compound. Holding retention fixed at retention 0.50 and 4 bits, n = 50:

| arm | accuracy | empty | first-token EOS | malformed | well-formed but wrong | correct | conditional accuracy |
|---|---|---|---|---|---|---|---|
| protection alone | 0.967 | 0.000 | 0.000 | 0.013 | 0.027 | 0.960 | 0.973 |
| protection + tiering | 0.287 | 0.010 | 0.007 | 0.243 | 0.460 | 0.287 | 0.384 |

The baseline is clean, with 1.3% malformed output and no empty answers, so the contrast is
not confounded by a degraded reference. Conditional accuracy, restricted to well-formed
non-repetitive output, falls from 0.973 to 0.384. Of the roughly 68-point deficit, about 24
points are outputs that stopped being well formed and about 44 points are well-formed answers
that are simply wrong.

The tier degrades retrieval and generation together. **The design cannot separate them
further, because the same quantization causes both.** This is a limitation of the experiment,
not a caveat attached to a number. A claim of "68 points of retrieval quality" would overstate
it.

### 6.3 Dose-response

Holding budget at retention 0.50 and 4 bits, varying only the number of positions placed in
the cold tier, n = 50:

| target quantized | realized | accuracy | degeneracy |
|---|---|---|---|
| 0 | 0.0 | 0.967 | 0.013 |
| 50 | 15.5 | 0.673 | 0.053 |
| 100 | 31.1 | 0.473 | 0.160 |
| 150 | 46.5 | 0.397 | 0.163 |
| 225 | 70.0 | 0.253 | 0.277 |

Strictly monotone in both columns. That establishes the mechanism.

It does not license a rate. Realized quantized positions are about 31% of target, because
structural protection seats its matched groups at full precision first and the cold tier
receives only what remains. Target and realized are coupled through protection, so the slope
is not a clean dose coefficient.

---

## 7. Result 4, scoped: no promotion signal helps at the one testable operating point

### 7.1 Why one bit-width

The promotion decision can only matter if the FULL and QUANT tiers differ in outcome. We
measured that band directly, holding retention identical and varying only precision.

At 8 bits on Qwen2.5-1.5B: all-full 0.450, half-and-half 0.467, all-quantized 0.467.
Quantization is genuinely applied, at about 1.2% relative error on real cached tensors with
values provably differing. It is simply too mild to matter. At 8 bits the promotion decision
has no effect on the outcome, so P1 ≈ P2 ≈ P3 ≈ P4 ≈ P5 is true by construction rather than
by evidence.

At 6, 5 and 3 bits, output collapses (§8.1) and no promotion policy can help either.

At 4 bits the band is real and measured, not inferred. At retention 0.25 it runs from 0.133
all-quantized to 0.411 all-full, a band of 0.278. At retention 0.15 the band is 0.111, which
is marginal and reported separately.

On Llama-3.2-3B, 4-bit accuracy is indistinguishable from 8-bit (0.320 against 0.310), so no
band exists there either.

**4-bit on Qwen2.5-1.5B is the only operating point we found where the promotion decision can
move the outcome on this model.** That is the scope of this result and it is narrow.

### 7.2 The binding constraint is arithmetic

Lead with this rather than with the tied arms.

At retention 0.25 and 4 bits, **52.3% of credential tokens are evicted at the retention stage
before promotion runs**. The promotion oracle wants 111 tokens at full precision and has 96
slots, but only about 53 of those tokens survive retention. It was never oversubscribed on any
selection call, and zero credential tokens ended quantized under it.

The ceiling on any promotion signal is therefore fixed before the signal is chosen. Promotion
can only reallocate fidelity among survivors. It cannot recover what retention discarded.

### 7.3 The arms confirm it

Qwen2.5-1.5B, 4-bit, retention 0.25, context 1029, n = 150.

| signal | ρ(eviction, promotion) | accuracy | vs P1 | 95% CI | vs random | 95% CI |
|---|---|---|---|---|---|---|
| P1 attention, circular | **+1.000** | 0.203 | — | — | +0.049 | [+0.021, +0.077] |
| P2 epiphany, orthogonal | −0.077 | 0.216 | +0.012 | [−0.014, +0.040] | +0.061 | [+0.034, +0.088] |
| P3 random | −0.050 | 0.154 | −0.049 | [−0.078, −0.021] | — | — |
| P4 rotation | +0.002 | 0.184 | −0.019 | [−0.048, +0.009] | +0.030 | [+0.004, +0.057] |
| P5 oracle | −0.068 | 0.188 | −0.016 | [−0.046, +0.014] | +0.033 | [+0.008, +0.059] |

At retention 0.15 the same ordering holds with everything compressed and nothing clearing
zero.

The manipulation is exact. ρ is +1.000 for the circular arm and near zero for the rest. The
orthogonal candidate does not beat the circular one. Our own hypothesis, that recoverability
has value if and only if the promotion signal is independent of the eviction signal, is
falsified against a signal verified to be informative: epiphany beats random by +0.061, a
margin comparable to attention's own +0.049.

Only a deliberately uninformative ordering separates, and it separates downward. The
promotion signal is not ignored. It is saturated. Any non-degenerate ordering performs about
equally and only actively bad ordering costs anything. Rotation ties attention, so the
blind-rotation-beats-measured-ranking result from adjacent work does not reproduce here.

Unlike §4, the arms do disagree prompt by prompt, on 85 to 96 of 150 prompts. This is
equivalence in expectation with different behaviour, not identical behaviour. The two nulls
are different in kind and should not be described in the same words.

### 7.4 Scope

The promotion oracle bounds the promotion decision only. It cannot rescue the 52.3% of
credential tokens already evicted, because retention is held attention-ranked in every arm by
design. That is what isolates promotion as the sole varied factor.

This result bounds promotion **under attention-ranked retention**, which is what the methods
under study use. It says nothing about a system that co-designs both stages. The factorial
cannot reach that question. A design that allocated budget jointly, rather than filtering
through retention and then reallocating fidelity, is outside what we measured.

For contrast, a retention oracle at the same budget scores 0.943 against protection's 0.423.
The promotion decision is saturated. The retention decision has 0.520 of unexploited headroom
at the same budget.

---

## 8. Result 5, reported but not explained: a model-family bit-width anomaly

### 8.1 The observation

Reconstruction error is monotone in bit-width. Task accuracy is not.

Qwen2.5-1.5B, retention 0.25, n = 50 prompts × 6 turns per width, corrected retention ranking:

| bits | relative error | accuracy | degeneracy | empty | prompts with any degeneracy |
|---|---|---|---|---|---|
| 8 | 1.45% | 0.473 | 0.073 | 0.000 | 0.260 |
| 7 | 2.67% | 0.303 | 0.150 | 0.017 | 0.500 |
| 6 | 4.50% | **0.020** | 0.747 | 0.027 | 0.980 |
| 5 | 8.62% | **0.000** | 0.920 | 0.170 | 1.000 |
| 4 | 17.62% | **0.247** | 0.277 | 0.000 | 0.640 |
| 3 | 37.31% | **0.000** | 0.967 | 0.370 | 1.000 |

4-bit carries roughly four times the reconstruction error of 6-bit and produces roughly four
times fewer degenerate outputs.

### 8.2 Seven explanations tested and eliminated

1. **Level counts wrong.** Asserted at most 2^bits on 10,640 quantization calls intercepted
   during real generation. Correct at every width.
2. **Correct levels, wrong scale.** Relative error measured on those same live calls through
   the production code path, not a reimplementation, is monotone from 1.45% to 37.31%, with
   zero subnormal scales and zero non-finite values.
3. **Policy behaves differently per width.** Same positions quantized, tier composition
   identical to three decimals, call counts within 0.4%.
4. **A per-prompt threshold effect.** Falsified. At 6 bits, 35 of 50 prompts are fully
   degenerate and zero are clean. This is near-universal collapse, not a boundary crossed on
   a subset.
5. **Level-occupancy concentration.** Effective level count, computed as 2^H of the occupancy
   distribution, halves exactly in step with nominal. The occupancy ratio is constant to
   three decimals across every width.
6. **The retention-ranking defect (D-08).** The anomaly survives correction. Under corrected
   ranking the shape is unchanged and the 4-bit recovery is stronger, 0.247 against 0.123.
7. **Early termination.** Empty output is immediate end-of-sequence by construction, verified
   with zero violations across 600 answers. But the 6-bit empty rate under corrected ranking
   is 0.027, close to the independent implementation's 0.007. The 6-bit failure is
   repetition and truncation of the copy, not termination.

### 8.3 The family contrast is the finding

The independent implementation ran the same sweep on three models.

| model | 8 | 7 | 6 | 5 | 4 | 3 | KV group | 4-bit key error |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 0.143 | 0.160 | 0.130 | **0.000** | 0.117 | **0.000** | 256 | 19.98% |
| Qwen2.5-3B | 0.250 | 0.250 | 0.250 | 0.247 | **0.037** | 0.097 | 256 | 19.55% |
| Llama-3.2-3B | 0.310 | 0.307 | 0.313 | 0.307 | 0.320 | 0.260 | **1024** | **24.28%** |

Llama-3.2-3B shows no collapse at any width, with zero fully degenerate prompts, while
carrying higher reconstruction error than either Qwen at every width. Its accuracy is flat
from 8 bits through 4 bits and declines gracefully at 3.

Four runs give four different curves. Our Qwen2.5-1.5B collapses at 6 and 5; the independent
implementation's Qwen2.5-1.5B collapses at 5 only, with 6-bit merely degraded; Qwen2.5-3B
collapses at 4 and partially recovers at 3; Llama-3.2-3B does not collapse at all.

**Catastrophic non-monotone collapse is a model-family property, not a property of KV cache
quantization.** Reconstruction error does not predict accuracy collapse. A bit-width severity
ordering derived from one model family does not transfer to another, and the two Qwen models
do not even agree with each other on which widths collapse.

We do not have a mechanism for the collapse and we do not offer one. The quantizer is cleared
on all three models independently. What remains is how those models' downstream computation
responds to quantization noise. We report the family contrast as the finding and state the
rest as open.

### 8.4 Consequence for §7

Phase 2 ran at 4 bits, which is the width where our own curve is anomalous and where the two
Qwen implementations disagree. The internal validity of §7 does not depend on the anomaly,
because it is a within-width comparison at a width whose band was measured empirically rather
than inferred from a position on this curve. But §7 has no independent corroboration at the
width it was measured, and the choice of that width was licensed by a curve whose central
feature does not replicate. That is stated rather than resolved.

---

## 9. Reconciliation: cross-implementation and cross-model

An independent team implemented the same specification without reading our harness. No file
of ours was opened, imported, listed or searched. They ran 7,300 cells with zero failures and
documented 26 specification ambiguities with the default chosen for each.

### 9.1 What replicated

Quantizer reconstruction error agrees to within 0.03 percentage points at 8 bits (1.18%/0.91%
against our 1.17%/0.88%) and 0.21 at 4 bits (19.98%/15.20% against 20.06%/14.99%), and is
essentially model-independent. Context length agrees at median 1030.5 against 1029. The
matched-byte token grant is identical at 411 for retention 0.25. The 4-bit recovery in the
bit-width sweep reproduces at 0.117 against our 0.123.

The 8-bit null reproduces at every budget and on a second model scale.

Two independent inventions converged. Both implementations arrived at the same
representational-change formula for the orthogonal promotion signal, `mean_l ‖h_l − h_{l−1}‖ /
(‖h_{l−1}‖ + ε)` computed at prefill, and both flagged it as an invention rather than a
specified quantity. Both independently identified the sum-versus-mean attention accumulation
question, and both resolved it by the same diagnostic: the retention oracle retains every
credential span by construction, so it should equal the no-eviction reference. Under mean it
does. Under sum it does not.

### 9.2 What differs, and why

Four implementation differences account for the divergences.

**Distractor value format.** Their distractor lines share the credentials' full value format,
so all 26 lines match `LABEL: sk-<14 hex>` and differ only in the label. Ours use different
value formats and shorter values. Both implementations match 26 lines on the label pattern,
but our protected content is cheaper to seat, so more credential lines fit within the same
budget. This plausibly explains why our protection arm is roughly 2.5× theirs at retention
0.25 (0.423 against 0.166), which in turn drives the difference in 4-bit effect magnitude
at retention 0.50.

Their ablation makes the point sharply. Rewriting distractor values to a non-credential form,
so only 6 lines match the protection pattern instead of 26, lifts structural protection from
0.150 to 0.907, above the no-eviction ceiling of 0.880 measured under the same condition.
With few enough competitors, pattern-based protection stops competing and simply retains every
credential. It becomes a retention oracle wearing a pattern matcher's clothes.

**The strength of pattern-based KV protection is set by the ratio of target lines to
pattern-matching non-target lines.** That is a property of the mechanism, not of this task.
The direction of the protection effect should transfer. The magnitude should not be quoted
outside the 6:26 ratio it was measured at.

**Tier seating.** They assign sink and recency positions to full precision by priority and
split the remainder by attention rank. We seat structurally matched groups at full precision
first and split what remains. This is why our realized quantized positions are about 31% of
target. Cross-implementation 4-bit numbers are therefore not directly comparable, and we do
not compare them as though they were.

**Turn order.** Theirs is deterministic reverse depth, so turn index, dormancy gap and context
depth are one variable. This produces an absorbing failure mode in which all eviction arms
reach zero after turn 1 and the metric behaves as an effective two-turn measure. Ours shuffles
per prompt.

**Value alphabet.** Theirs draws from uniform hex. Ours drew from a 22-slot string in which
a–f appear twice (D-07).

### 9.3 Where we still disagree

On the promotion question the two implementations disagree in direction. Under their corrected
signals, epiphany and the promotion oracle beat random while circular attention does not.
Under ours, attention beats random and epiphany is level with it. Both find the arms clustered
and both find random worst. They read the ordering within the cluster differently.

We do not resolve this. Candidate causes are the different tier-seating rules, the different
distractor competition ratio, and the ceiling difference below. What would settle it is a
single implementation running both seating rules on both distractor formats.

Their no-eviction ceiling is 0.799 against our 0.947, exactly invariant across budgets in both
cases. They diagnosed the gap rather than assuming it. Case folding and whitespace stripping
rescue exactly zero failures, which eliminates the axis the specification anticipated. Their
model frequently answers with the bare 14-hex payload and omits the `sk-` prefix, accounting
for 58 of 181 failures, and allowing one character of edit distance recovers 82 more. Together
those conventions give 0.9544 against our 0.947.

Their conclusion is that the two implementations are probably not scoring the same thing. We
note one point against a pure scoring explanation. Our failures are single-character miscopies
rather than prefix omissions, in 8 of 10 sampled cases. If their model omits the prefix and
ours does not, the difference is at least partly behavioural, arising from prompt phrasing,
rather than purely a matter of the scorer. We verified our own scorer by differential testing:
a from-scratch strict matcher and a case-and-whitespace-normalising matcher both return 0.9444
on 30 fresh prompts, agreeing with the harness on all 180 answers.

### 9.4 Their defects, and the mutual blind spot

Their manipulation check caught two contaminated promotion signals in their own
implementation. Rotation measured ρ = +0.497 and their promotion oracle ρ = +0.849 against a
target near zero, because both fell back to the eviction score for positions outside the
must-promote set. They diagnosed, corrected, and re-ran Phase 2 in full, retaining the original
to quantify the contamination. Our signals were clean from the start on the same check.

The reverse also holds. Their tripwire for the retention-ranking defect was the retention
oracle failing to match the no-eviction reference. That test is structurally blind to the same
defect in our harness, because our retention oracle never consults attention. Its keep-set is
ground truth plus earliest-position filler. We passed their tripwire while carrying the defect
it was designed to catch.

Each implementation had a blind spot the other's design happened to cover.

---

## 10. The reproducibility result

Two defects in our implementation were found only by comparison against an independent
implementation, after five rounds of internal instrument validation missed both. Neither would
have been caught by more of the same, because each was self-consistent within its own
implementation.

**D-07, the biased alphabet.** `string.hexdigits.lower()` produces
`'0123456789abcdefabcdef'`, a 22-character string, because `ABCDEF` folds onto `abcdef`. Values
drawn uniformly from it give a–f a 9.09% chance each against 4.55% for digits, and 3.914 bits
per character against a true-hex 4.000. Every internal check passed, because the values were
consistently generated and consistently scored. Holding seeds and all other parameters fixed
and swapping only the alphabet: as-shipped 0.9444, uniform hex 0.9111, digits only 0.8833. The
bias is worth about 0.033. Because all arms in every phase faced the identical alphabet,
between-arm comparisons are unaffected. Absolute accuracy levels in this paper are inflated by
roughly 0.03 relative to a true-hex task.

**D-08, sum versus mean attention accumulation.** Our retention ranking accumulated attention
mass summed over queries. In a 1029-token prefill, position 0 is attended by about 1029
queries and position 1000 by about 29, so early positions outrank later ones for having been
present longer. Measured Spearman correlation between score and position is **−0.717** under
sum and **+0.587** under mean normalisation. The two readings are near-uncorrelated with each
other at ρ = **+0.014** and share only **13.5%** of the retained set. This is not a refinement.
It selects an almost entirely different cache.

We measured what the correction changes. At 8 bits the null is unchanged. The recoverability
effect given protection is +0.000, −0.001 and +0.001 against the shipped +0.001, +0.002 and
−0.002, and the interaction is +0.002, −0.001 and −0.002 against the shipped +0.000, +0.003
and −0.002, all intervals inside ±0.05.
The protection effect survives at +0.169, +0.457 and +0.924. Arm levels rise where there is
headroom, with protection moving 0.423 to 0.481 at retention 0.25. At 4 bits and retention
0.50 the correction made the harm larger, from −0.683 to −0.801 (§6.1).

**The general point.** Five rounds of internal validation cleared the instrument in five
independent ways: level counts on live intercepted calls, monotone reconstruction error through
the production path, identical policy behaviour across widths, near-universal rather than
per-prompt collapse, and constant level occupancy. All five were correct. None could detect
either defect, because a self-consistent choice does not violate any invariant that the
implementation making the choice would think to assert.

Independent reimplementation of a written specification found both, and simultaneously exposed
26 ambiguities in the specification itself, several of which are load-bearing. The distractor
value format (§9.2) changes what mechanism arm 2 tests. The Phase 1 bit-width, never stated,
accounts for the largest single numerical disagreement between the implementations.

This is a stronger argument for independent reimplementation than internal rigour, and we
report it as a result rather than as an appendix.

---

## 11. Predictions that failed

**Shared bias cancels in a difference.** We predicted that because both arms of the interaction
use the identical retention ranking, correcting D-08 would shift levels without moving the
interaction. This held at 8 bits, where either estimand moved by at most 0.005. It broke at 4
bits, where the recoverability effect given protection moved from −0.683 to −0.801 with
non-overlapping intervals at retention 0.50 (the difference in differences is not available
under mean ranking). The
reason is regime-specific and is the mechanism in §6.1: better ranking feeds more credential
tokens into a destructive cold tier. A bias shared by both arms does not cancel when one arm's
response to it is mediated by a stage the other arm lacks.

**The bit-width collapse is a per-prompt threshold effect.** We predicted that quantization
noise pushes a threshold function across a boundary on some inputs, which implies a mixed
population at a given width. Measured at n = 50, the collapse is near-universal: at 6 bits, 35
of 50 prompts are fully degenerate and zero are clean. The prediction was wrong about the
population structure and the hypothesis was abandoned.

**High conditional accuracy at retention 0.50.** We predicted that the 4-bit harm would prove
to be generation collapse rather than information loss, which implies conditional accuracy
staying high. Measured, conditional accuracy fell from 0.973 to 0.384. The effect is compound
and the design cannot decompose it further (§6.2). The prediction rested on reading the
collapse rows of the bit-width sweep; the mixed rows predicted the outcome better.

**H-ORTH.** Our own hypothesis was that recoverability has value if and only if the promotion
signal is statistically independent of the eviction signal. It is falsified. The orthogonal
signal does not beat the circular one, and the orthogonal signal is verified informative
(§7.3). This is a cleaner falsification than most, because the failure has a mechanism
attached: promotion is bounded upstream, so signal quality cannot be the binding constraint.

---

## 12. Limitations

**Scope of the promotion result.** One model, one bit-width, one task. The bit-width was chosen
because it is the only precision on Qwen2.5-1.5B with a measurable band between all-full and
all-quantized. At 8 bits the tier is free. At 6, 5 and 3 bits output collapses. On
Llama-3.2-3B no band exists at any tested width. The claim is that no tested promotion signal,
including a verified oracle, helps at the one operating point where the mechanism could act on
this model. It is not a claim that promotion is useless.

**The promotion result is conditional on attention-ranked retention.** It bounds the promotion
decision given the retention rule the methods under study use. A system co-designing both
stages is outside what the factorial can reach.

**The compound effect is not decomposable.** §6.2 splits roughly 44 points of retrieval loss
from 24 points of generation loss, but the same quantization causes both and the design cannot
isolate them.

**The dose-response does not license a rate.** Realized quantized positions are about 31% of
target because protection seats full-precision groups first.

**The estimand was mislabelled in the first draft.** The quantity called the interaction was
arm 4 minus arm 2 (D-09, §3.3). Both estimands are now reported. At 4 bits the
pre-registered interaction is distinguishable from zero only at retention 0.50. Under mean
ranking the difference in differences is not available at all, because that rerun omitted
arms 1 and 3. The intervals in this paper were recomputed from raw rows with a fixed seed and
can differ from earlier interim figures in the fourth decimal.

**Budget coverage.** Two of five budgets are void by arithmetic and one is qualified. Any
statement about the budget gradient rests on two clean points plus one qualified one.

**The bit-width anomaly is unexplained and family-specific.** Seven candidates eliminated, no
mechanism offered. It should not be read as a general property of KV cache quantization.

**Quantizer scope.** All results use one quantizer, per-position affine min-max over the
`(kv_heads, head_dim)` slice. A scheme with different group sizes or per-channel scales will
move the free-versus-damaging boundary. The independent implementation's Llama results show
group size matters: 1024-element groups give higher error at every width than 256-element
groups.

**Absolute levels are inflated by about 0.03** by D-07. Between-arm comparisons are unaffected.

**Cross-implementation comparability.** Tier seating and distractor value format both differ
between implementations. Cross-implementation 4-bit numbers are not directly comparable and are
not treated as such.

**The blind protocol cannot catch shared errors.** An error present in both the specification
and every implementation of it is invisible to this method.

**Single task and single scoring convention.** Exact-match retrieval of random hex strings is
unusually sensitive to quantization. A semantically scored task would plausibly show a weaker
bit-width effect. The two implementations may not be scoring the same thing (§9.3).

---

## 13. Related work

Recoverable eviction with a quantized middle tier is the mechanism under test. Structural and
pattern-based protection of dormant or boundary tokens is the alternative under test. Signals
read from hidden-state change rather than the attention matrix motivate our orthogonal
promotion arm.

The retention question this paper leaves open is actively contested. LU-KV (Tang et al.,
arXiv:2602.08585) formulates head-level budget allocation as a global combinatorial
optimization problem maximising the long-horizon marginal contribution of reserved tokens,
solved by convex-hull relaxation and greedy search with offline profiling. ForesightKV (Dong
et al., arXiv:2602.03203, ICML 2026) designs a Golden Eviction algorithm that identifies
optimal evictions using future attention scores, then distils that oracle into a ranker with a
pairwise ranking loss refined by reinforcement learning.

Both target the stage our measurements identify as holding the headroom. Neither is compared
against here.

---

## 14. Conclusion

Recoverable KV cache eviction has two stages. We measured them separately against their own
oracles.

Under matched-token accounting, adding a recoverable tier to structural protection produces no
measurable interaction. This holds at three budgets, on two model scales, in two independently
written implementations, with several cells showing zero or near-zero prompts differing out of
150.

At the one precision on Qwen2.5-1.5B where the promotion decision can move the outcome, no
tested promotion signal helps, including a verified oracle and a verified-informative
orthogonal signal. The constraint is arithmetic. Slightly more than half of the target tokens
are gone before promotion runs.

At aggressive precision the tier does not merely fail to help. It harms, and correcting a
defect in the retention ranking made the harm larger, because the cold tier sits directly
downstream of selection and receives exactly the tokens retention correctly identified as
valuable.

Whether any of this appears as a benefit depends on an accounting choice the literature does
not consistently state. The same experiment is null under matched tokens and strongly positive
under matched bytes.

Identifying an optimal retention signal is an open problem.

---

## Appendix A: reproduction protocol

Repository layout, exact commands, and the append-only checkpoint schema are in `README` and
`PREREG.md`. All raw rows are retained, including for excluded cells.

Primary artifacts: `PREREG.md` (sha256 `5d3c7e3b…`), `ADDENDUM_2026-09-03.md` (sha256
`0c599372…`, sections 9 through 13 carry post-hash corrections), `ANALYSIS.md`. Results
directories: `phase1/`, `phase1_4bit/`, `phase1_meanscore/`, `phase2_4bit/`,
`phase2_8bit_control/`, `q1_decompose_514/`, `q2_dose_response/`, `q3_bitsweep_mean/`,
`q4_phase1_4bit_mean/`, `quant_sensitivity/`, `bitwidth_degeneracy/`, `phase0r_calibration/`.

The independent implementation is in `amd/`, with its own `README`, `AGREEMENT.md`,
`SPEC_QUESTIONS.md`, `FINDINGS.md` and results tree.

`engine.SCORE_MODE` selects sum or mean attention accumulation. `engine.QUANT_BITS` selects
tier bit-width. Both are module-level so a sweep can vary them, and both are recorded per row.

## Appendix B: specification ambiguities

The independent implementation documented 26 ambiguities in the written specification with the
default chosen for each. The load-bearing ones are: Phase 1 bit-width, never stated, which
accounts for the largest numerical disagreement; distractor value format, which determines
whether structural protection competes against 26 lines or 6 and therefore what mechanism it
tests; attention-score aggregation, which the specification leaves open between sum and mean;
the full-precision assignment rule in Phase 1; and the definition of the representational-change
promotion signal, which both implementations had to invent.

## Appendix C: calibration gates

G1 retention oracle ≥ 0.95. G2 unprotected floor ≤ 0.10. G3 protection alone in [0.45, 0.75].
G4 no-eviction reference ≥ 0.95. G5 dormancy in ≥ 0.80 of prompts. Measured at six credentials
and retention 0.27: 0.983, 0.033, 0.567, 0.983, 1.000.

G3 is the gate that catches an inert factor and is the one the sweep failed to re-derive per
cell.
