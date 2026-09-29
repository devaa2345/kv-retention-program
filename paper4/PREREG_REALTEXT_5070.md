# Real-text extractive copy mask on RTX 5070 — preregistration

Status: frozen before **any answer-token generation** for this test. GPU work before this
commit is limited to prefill and last-prompt-token attention measurement, with no decoding.
The first generation gate is `full_cache` on both models and requires explicit user confirmation.

## Data and split

Models: M2 `Qwen/Qwen2.5-3B-Instruct` and M3 `meta-llama/Llama-3.2-3B-Instruct`, eager
attention, bfloat16, RTX 5070. The deterministic constructor
`out/_realtext_5070_data.py` uses CRC32 seed `realtext-5070-v1|index`. Indices 0–23 are the
24-instance pilot, and indices 24 onward are held out. Each instance has one context and four
queries: two independently selected SQuAD validation questions with unique context answers and
two distinct eight-digit access-code needles in natural WikiText filler. Gold answers occur
verbatim **exactly once** in the context, in four distinct sentences. Questions containing the
gold answer are excluded. The pilot data file is `out/realtext_5070_pilot.jsonl`, SHA-256
`84c79f5c9ccd9b15949bbd9da41a99fa24a2eef178a470e758ef4f3d537a58f4`.
The prompt asks for only the exact words from the context; no paraphrase or other wording.
The same instance text, questions, seeds, and split are used for both models and every arm.

Pilot answer lengths: 4–54 characters, median 8, 90th percentile 20; 1–10 whitespace-delimited
words, median 1, 90th percentile 3. Each model gets 96 pilot queries. One query changes a pilot
rate by exactly `1/(24×4) = 1/96 = 0.0104167`; on `N` held-out instances one query changes it
by `1/(4N)`.

Budget: `B = round(0.2802 × n_ctx)`, `C = B − 8 − 64`; every compressed arm retains the first
eight sink positions and last 64 context positions. M2 pilot `n_ctx` is 1538–1989 tokens
(median 1782); M3 is 1485–1947 (median 1714). The baseline and each masked variant must have
identical captured keep-sets for the same instance/query. Captured realised budget and mandatory
floors are asserted per layer/head. Position IDs for question and answer continuation begin after
the **uncompressed** context length.

## Arms, mask, and scoring

Arms are `full_cache` (unmasked anchor), `floor_pos`, `oracle_sentence` (gold sentence retained
within the same budget), and floor-constrained `SnapKV`; the latter three each have an unmasked
baseline plus two literal-mask variants: (1) attention-chosen span, and (2) oracle gold span.
`SnapKV` was selected before inspecting the prefill audit. All masked arms use
`LiteralSpanMask` in `out/_realtext_5070_mask.py`, **not** the whitespace-normalized LEDGER-C
`SchemaFreeSpanAutomaton`. All unmasked arms have no mask. Historical LEDGER-C CI comparisons
retain their original ordered/schema-free mask versions; this study does not relabel them.

The selected span is one sentence. The attention selector considers sentences whose complete
token span appears in the **union** of retained per-head keep-sets, then chooses the one with
maximum mean last-prompt-token attention mass across layers and heads. Captured top-k gather
order maps compressed KV positions back to original positions. A span is counted retained only
if all its token positions are in that union. This is a permissive retention definition for
head-wise methods, so its ceiling is an upper bound, not a guarantee of model access.

The literal mask decodes the entire emitted token sequence after each candidate next token.
The next token is admitted only if the resulting decoded string extends the previous decoded
string and is an **exact, case- and whitespace-sensitive contiguous substring** of the selected
sentence. It permits no free formatting token and no restart delimiter for this single-answer
task. EOS is legal only after at least one character has been copied; EOS is not part of the
answer. If the right passage is attended but a different sentence is selected, or the right
sentence is selected but a wrong substring is copied, the output is scored wrong. A wrong-span
output cannot match a unique gold answer unless that gold occurs there, which the data check
excludes. Empty output and cap termination are recorded as failures. No output is silently
whitespace-normalized for mask-fidelity checks.

Before this commit, CPU unit tests on all 96 pilot gold spans per model reproduced 96/96 exact
gold answers, rejected 96/96 absent invented strings, and kept 96/96 sampled legal walks inside
the selected span, for both M2 and M3. The transition itself asserts that every emitted decoded
prefix is in the selected sentence.

Primary answer accuracy is exact gold equality after trimming only surrounding whitespace from
the final answer; raw output is preserved. Completion means nonempty answer text, with EOS/cap
recorded. Conversion is unmasked wrong → masked right; broken-correct is unmasked right → masked
wrong. The corrected **full-context** invented-value check marks a nonempty raw output invented
when it is not a literal substring of the entire context. Both masked and unmasked arms on both
models use this version; the earlier gold-only check is not used. The per-arm output reports
completion, conversion, accuracy, own-baseline delta, broken-correct, and invented-value count.

## Prefill ceilings and headroom

All counts below are on the 24 pilot instances (96 queries per model) at the frozen budget.
No generated accuracy is available yet. `held` is complete gold-sentence retention. `hit` means
the max-attention eligible sentence is the gold sentence. The maximum attainable absolute
accuracy under a literal attention-chosen mask is `hit/96`; under an oracle-span mask it is
`held/96`. These are optimistic upper bounds, before any failures to copy the correct substring
and before broken correct answers. Full cache has the trivial ceiling 1 and is an anchor, not a
masked comparison. The maximal own-baseline improvement is at most ceiling minus that arm's
unmasked accuracy once generation supplies the baseline.

| model | retention arm | gold held /96 | attention hit /96 | attention-mask ceiling | oracle-span ceiling |
|---|---|---:|---:|---:|---:|
| M2 | floor_pos | 23 | 21 | 0.2188 | 0.2396 |
| M2 | SnapKV | 62 | 37 | 0.3854 | 0.6458 |
| M2 | oracle_sentence | 96 | 89 | 0.9271 | 1.0000 |
| M3 | floor_pos | 24 | 22 | 0.2292 | 0.2500 |
| M3 | SnapKV | 92 | 48 | 0.5000 | 0.9583 |
| M3 | oracle_sentence | 96 | 93 | 0.9688 | 1.0000 |

Attention-hit uncertainty is estimated by resampling **instances** and retaining each set of
four queries together (10,000 seeded replicates). This supersedes a query-independent Wilson
interval because the four questions share a context and cache. The prefill audit is descriptive;
all confirmatory delta CIs likewise use instance-level paired bootstrap.

| model | arm | attention hits / held | hit rate over all queries, 95% instance CI |
|---|---|---:|---:|
| M2 | floor_pos | 21/23 | 0.2188 [0.1354, 0.3125] |
| M2 | SnapKV | 37/62 | 0.3854 [0.2812, 0.5000] |
| M2 | oracle_sentence | 89/96 | 0.9271 [0.8646, 0.9792] |
| M3 | floor_pos | 22/24 | 0.2292 [0.1458, 0.3229] |
| M3 | SnapKV | 48/92 | 0.5000 [0.3854, 0.6042] |
| M3 | oracle_sentence | 93/96 | 0.9688 [0.9375, 1.0000] |

CPU keep-set feasibility at ratio 0.2802: floor retained 23/96 M2 and 24/96 M3; the oracle
retained 96/96 for both. As a predeclared alternative-budget diagnostic, ratio 0.20 yielded
floor 19/96 on each model, while ratio 0.40 yielded 36/96. These are retention counts only;
neither budget is a replacement evaluation cell. `answered correctly when held` and therefore
`held-but-wrong` require generated answers and are deferred by user instruction. Once pilot
generation is authorized, calculate `held ∧ unmasked wrong` for every arm. If floor headroom is
under 10% of queries, stop before held-out evaluation, report the result and propose a newly
preregistered budget; do not switch budgets to rescue this preregistered test.

| model | arm | held /96 | unmasked correct given held | held but wrong /96 |
|---|---|---:|---:|---:|
| M2 | floor_pos | 23 | pending generation | pending generation |
| M2 | SnapKV | 62 | pending generation | pending generation |
| M2 | oracle_sentence | 96 | pending generation | pending generation |
| M3 | floor_pos | 24 | pending generation | pending generation |
| M3 | SnapKV | 92 | pending generation | pending generation |
| M3 | oracle_sentence | 96 | pending generation | pending generation |

## Gates, sample size, and decision rule

After user confirmation, run `full_cache` on the 24 pilot instances for both models. Each
anchor point estimate must be within `[0.55, 0.97]`; if either fails, stop and report with no
within-study redesign. Also stop before scoring if masked and unmasked captured retention
metrics or keep-set hashes differ. No confound mismatch is waived. Only after both anchors
pass may other pilot arms generate answer tokens. Every arm check includes actual token
generation; prefill logits alone cannot count as an arm result. No Stage 3 scaling is included.

The pilot is for variance and feasibility, never for confirmatory claims. For each model/arm,
form paired per-instance differences as the mean of its four masked-minus-own-unmasked query
scores. With pilot variance `s²`, calculate required held-out instances for a 10-percentage-point
effect at two-sided 5% and 80% power as `ceil((1.96+0.84)² × s² / 0.10²)`. Set `N` to the
maximum over the primary floor and oracle attention-mask contrasts and at least 200. Increase
`N` further if needed so the pilot-rate projection provides at least 100 baseline-correct
queries per arm: `ceil(100/(4p_correct))`. If a baseline has zero pilot correct queries, or
required `N` exceeds 500, report infeasibility and do not begin the held-out run without a
newly approved scope. Once calculated, freeze `N` before generating any held-out answer.

For every reported held-out paired delta, draw 10,000 bootstrap samples of **whole instances**
using a CRC32-seeded RNG, keeping the four query outcomes and both arms paired. Report the
2.5th and 97.5th percentiles. `REAL` requires an own-baseline accuracy gain with its 95% CI
excluding zero on the positive side, positive net conversions after broken correct cases, and
at least 100 baseline-correct held-out queries so the false-positive interval is informative.
If these conditions fail, report the scoped negative without tuning on held-out data.

The headline comparison is `floor_pos + literal mask (attention span)` against `floor_pos`.
Separately, `oracle_sentence + literal mask (attention span)` against the same oracle retention
without a mask tests whether the mask lifts that retention ceiling. Oracle-span variants bound
span-localization loss. If only oracle-span masks work, sentence localization is the research
problem. If both attention- and oracle-span masks work, a plug-in method merits a separately
approved scale-up. If neither works, report a scoped negative for these extractive tasks and
do not tune. Whole-system contrasts such as masked oracle vs `floor_pos` are secondary and
reported in a separately labelled column, never as an own-baseline gain.

All output rows are labelled `RTX 5070`; deduplication keys include model, device, instance,
query, and arm. Runs use a single non-piped process and resumable per-query output discipline.

## GPU cost and hold point

The 24-instance prefill/attention audit took 1.3 minutes per model after model load (2.6
minutes total measured GPU processing). Historical 5070 LEDGER-C timing implies roughly
5–15 minutes for the two-model unmasked full-cache anchor and 3–8 GPU hours for all 24 pilot
instances with the compressed arms and two masked selectors. Exact-substring checking may
make the upper end more likely. Held-out time cannot be estimated responsibly until pilot
variance sets `N`; it scales approximately linearly with `N/24`. **No anchor or answer-token
generation begins until the user confirms.**

## Amendment 1 — answer format after failed M2 anchor

Committed before the single permitted M2 anchor rerun and before **any compressed-arm answer
generation**. Compressed-arm prefill/attention diagnostics had run under the original prompt;
no compressed or masked arm has generated answer tokens. The original preregistration and
failed anchor remain visible above and in `out/realtext_5070_anchor_M2.jsonl`.

The original M2 full-cache anchor scored **9/96 = 0.09375**, outside the frozen `[0.55, 0.97]`
gate. We read all 96 outputs before changing code. Of 87 misses, 77 contained the complete
gold answer but had extra text or a case difference; 10 omitted the complete gold answer
(wrong or incomplete answers); none was a refusal or otherwise unclassifiable. Two of the 77
were case-only differences. The dominant observed failure was long sentence-style responses
where the model had located the answer. This diagnosis uses only the anchor outputs.

One format correction is now frozen for **every arm, both models, both span selectors**. The
shared `format_query()` function provides two unrelated examples of short verbatim answers,
then the real question. Its instruction begins: "Answer with only the exact words from the
passage above that answer the question. Give only the short answer, not a sentence, label,
quotation marks, or explanation." The two examples are an observatory opening year and a
vault code; each answer is only the year or code. The question follows the examples and ends
with `Answer:`. The context, four questions, gold answers, split, seeds, budget, masks, and
retention methods do not change. Because `templated_parts()` places the question after the
prefill context, `n_ctx` and retained budgets are unchanged.

The primary scoring rule remains **strict exact match**, never substring-contains. Amendment 1
normalizes only case (Unicode casefold) and whitespace (split/join of whitespace runs) on the
full output and full gold answer, then requires equality. Punctuation and extra words remain
errors. The full-context invented-value check continues to inspect the raw output, unchanged.
Under this normalization alone, the original outputs would score 11/96; the prompt rerun is
needed to test whether the 77 format misses become short exact answers.

The changed query can alter last-prompt-token attention. Consequently, the prefill hit rates
and mask ceilings above are labelled **original-prompt diagnostics**, not accepted as A1
ceilings. If the amended M2 full-cache anchor passes, refresh those diagnostics under the
A1 prompt and freeze A1 ceilings before any compressed-arm answer generation. If the anchor
fails, stop; do not iterate again without user direction. The `[0.55, 0.97]` anchor gate and
all confirmatory decision rules stay fixed. The user narrowed this next run to **M2 only**.

## Amendment 2 — focused, lower-cost M2 comparison

Committed before any compressed-arm **answer generation**, after user requested a quick 5070
answer on whether a scorer beats floor and how much the literal mask improves that scorer.
The amended-prompt M2 full-cache anchor passed at **72/96 = 0.750**. The original two-model,
three-retention-arm generation plan and its 200–500 held-out-instance rule are superseded for
this focused study. No M3 answer generation or RTX Pro 5000 work is included.

The generation arms are now only `floor_pos` and floor-constrained `SnapKV`, both on M2. For
each, compare unmasked, attention-chosen literal mask, and oracle-span literal mask when the
pilot feasibility gates below permit. Oracle **retention** is omitted to save GPU time; the
oracle **span selector** remains an upper-bound control. The primary comparisons are:

1. `SnapKV` unmasked minus `floor_pos` unmasked: whether the method beats floor on real text.
2. `SnapKV + literal mask (attention)` minus unmasked `SnapKV`: the **own-baseline** improvement
   attributable to the proposed plug-in mask.
3. `SnapKV + literal mask (attention)` minus unmasked `floor_pos`: the separately labelled
   whole-system improvement.

`floor_pos + literal mask (attention)` minus unmasked `floor_pos` is a secondary own-baseline
check. Oracle-span variants measure how much accuracy is lost in sentence localization.
All variants share the A1 prompt, same 0.2802 budget, same instances and seeds, same strict
case/whitespace-normalized whole-answer equality, and the full-context invented-value check.
The first 24 instances remain pilot only; indices 24 onward remain untouched held-out data.

The A1-prompt prefill audit (`out/realtext_5070_prefill_M2_A1.jsonl`) measured these ceilings
before compressed answer generation:

| M2 arm | gold sentence held /96 | attention-selected gold /96 | attention-mask absolute ceiling | oracle-span absolute ceiling |
|---|---:|---:|---:|---:|
| floor_pos | 23 | 23 | 0.2396 | 0.2396 |
| SnapKV | 62 | 39 | 0.4062 | 0.6458 |

The attention-hit rate's 95% instance-bootstrap interval is [0.1562, 0.3333] for floor and
[0.3021, 0.5208] for SnapKV. These are optimistic *absolute accuracy* ceilings, not predicted
gains. The net gain ceiling for an attention mask is at most its listed ceiling minus the
corresponding observed unmasked accuracy. One query is 1/96 of this pilot and 1/(4N) of an
`N`-instance held-out set.

To save GPU time, first generate only the two **unmasked** pilot arms and report their accuracy,
the method-minus-floor difference, and each arm's held-but-wrong share. For each arm, let
`H = count(gold sentence held AND unmasked answer wrong)/96` and
`U = attention-hit/96 − unmasked accuracy`. If `H < 0.10` or `U <= 0`, do not generate that arm's
attention-mask variant: it lacks 10-point conversion headroom or cannot beat its baseline even
at its measured localization ceiling. Report the bound without changing the budget. An
oracle-span masked pilot may still run where its own `held/96 − unmasked accuracy > 0`, solely
to diagnose localization; it does not license a deployable-mask claim. For arms passing both
gates, generate both span-selector variants on the **same 24 pilot instances**. This gate uses
pilot data only, before the held-out set is touched.

After the pilot, set one held-out `N` for all available primary contrasts, using the largest
per-instance paired pilot variance `s²` of the method-vs-floor and eligible mask-vs-own-baseline
contrasts. The planning target is a 10-percentage-point effect at two-sided 5% and 80% power:
`N_power = ceil((1.96+0.84)² s² / 0.10²)`. Also require a pilot-rate projection of at least
50 baseline-correct held-out queries for each eligible mask contrast:
`N_FP = ceil(50/(4 × min relevant pilot baseline accuracy))`. Round the maximum of 64,
`N_power`, and `N_FP` up to a multiple of 16. **Cap N at 128 instances.** If a relevant pilot
baseline accuracy is zero or the formula requires more than 128, stop before held-out
generation and report that this low-cost study cannot make the intended claim; do not call a
non-significant small run a negative result or expand N silently.

The held-out comparisons use 10,000 instance-level paired bootstrap resamples with each
instance's four query outcomes kept together. A positive method-vs-floor claim requires a
positive difference and 95% CI excluding zero. A positive mask-uplift claim separately
requires positive own-baseline gain with 95% CI excluding zero, more conversions than broken
correct answers, and at least 50 baseline-correct held-out queries for its false-positive
rate. Report all other results as estimates with CIs and their scope. No evaluation-set tuning,
budget change, or additional scorer search is allowed. Historical LEDGER-C claims remain
separate and are not combined with real-text intervals.

This narrows the expense to 24 M2 pilot instances plus at most 128 held-out instances for two
retention arms. If the attention-ceiling gate fails, the shorter pilot and a numerical upper
bound can answer the mask-uplift question without spending on a masked held-out run. A positive
5070 result would motivate a **separately preregistered** RTX Pro 5000 replication, not start
one automatically.

### Amendment 2 sample-size lock — before held-out generation

The 24-instance M2 pilot is now complete, and **no held-out answer has been generated**. The
unmasked floor scored 22/96 and unmasked SnapKV 15/96. The floor's held-but-wrong count was
3/96, below the predeclared 10% gate, so its mask was not generated. SnapKV's held-but-wrong
count was 48/96, and its measured attention-mask gain ceiling was +24/96, so both SnapKV mask
selectors ran on pilot only. The attention mask scored 22/96 (+7/96 against its own unmasked
SnapKV baseline); the oracle-span mask scored 27/96 (+12/96). The pilot confound check was
clean, and both literal-mask variants produced zero invented values. These are **pilot
diagnostics**, not confirmatory claims.

The preregistered size formula gives `N_power=41` from maximum primary per-instance variance
0.0515172, and `N_FP=80` from SnapKV's pilot accuracy 15/96. Rounding to a multiple of 16
locks the held-out size at **N=80 instances = 320 queries**, indices 24–103. A single query
changes a held-out rate by `1/320 = 0.003125`. The held-out data file
`out/realtext_5070_heldout_M2.jsonl` is frozen at SHA-256
`05a517b13e267b76ae0a1c40daac7e6c063b980195b52ce7c9d55a8d82433323`.
Construction code was extended to retry an index when a candidate gold string also appeared
elsewhere in its context (encountered while building index 96); this enforces the existing
unique-gold criterion and does not select on model outcomes. The first 24 pilot instances and
their answers are unchanged. No budget, prompt, score rule, arm, or statistical threshold has
been changed after observing pilot outputs.

The held-out run will generate only `floor_pos` unmasked; `SnapKV` unmasked; and `SnapKV`
with attention-chosen and oracle-span literal masks, all on M2. It will not generate a floor
mask, oracle-retention arm, M3, or RTX Pro 5000 result. The predeclared comparisons and
instance-bootstrap decision rules above apply without further adjustment.
