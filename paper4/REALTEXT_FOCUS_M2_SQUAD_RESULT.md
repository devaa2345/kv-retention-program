# M2 SQuAD-only focused-evidence win over paired floor controls

The preregistered M2 focused-evidence pipeline **beat both paired floor
controls on SQuAD**. On 76 fresh instances with 152 distinct SQuAD questions,
SnapKV with its frozen lexical sentence pointer, an answer-time copy of the
selected sentence in the prompt, and literal_v1 decoding scored **53/152**.
Equally focused floor_pos scored **24/152**, a +0.1908 difference with 95%
paired instance-bootstrap CI **[+0.1250, +0.2632]**. Unmodified floor_pos
scored **22/152**, a +0.2039 difference with CI **[+0.1184, +0.2895]**.
Both lower bounds exceed zero, satisfying the two prespecified win criteria.

This is a **new system-level method**, not a pure mask or equal-total-KV
result. The focused prompt explicitly reinserts one selected original source
sentence after compression. Both focused arms receive exactly the same
format and selection procedure appropriate to their retained candidate set;
the inserted sentence and resulting extra KV tokens can differ. The
unmodified floor contrast is reported separately. To claim a compression
method win at exactly matched *total* working KV, an additional budget-matched
test would be required. The result does establish that this same-model,
single-sentence focus system beats a floor system given the identical
focus operation on this M2 SQuAD task.

The development gate, sample-size formula, prompt, pointers and dual
decision rule were committed in `PREREG_REALTEXT_FOCUS_M2_DEV.md` before
answer generation. The held-out 76-instance manifest SHA-256 is
`1d25bf5b273ff4cbf6d636e60c92f82799389adf617ad7626fcf0d610bb44641`.
Its 152 SQuAD question strings were excluded from earlier real-text datasets
and are unique within this test. Synthetic needles remain in the context
construction but were not generated or scored. Model M2 was
Qwen2.5-3B-Instruct on RTX 5070. Prefill retained-context ratio was 0.2802
for both floor_pos and SnapKV. Whole-answer exact match normalizes only
casefold and whitespace runs; it never uses substring-contains scoring.

| Arm | Correct /152 | Accuracy | Completed | Full-context invented | Cap stops |
|---|---:|---:|---:|---:|---:|
| floor_pos unmasked, A1 | 22 | 0.1447 | 152 | 110 | 2 |
| SnapKV unmasked, A1 | 16 | 0.1053 | 152 | 84 | 2 |
| floor_pos + selected-sentence focus + literal_v1 | 24 | 0.1579 | 152 | 0 | 10 |
| **SnapKV + selected-sentence focus + literal_v1** | **53** | **0.3487** | **152** | **0** | **7** |

All intervals use 10,000 paired bootstrap resamples of complete two-query
instances, seed 5070. One query changes accuracy by 1/152 = 0.6579 points.

| Comparison | Difference | 95% CI | Converted / broken |
|---|---:|---:|---:|
| **SnapKV+focus − floor_pos+focus (primary)** | **+0.1908** | **[+0.1250, +0.2632]** | **32 / 3** |
| **SnapKV+focus − unmodified floor_pos (second criterion)** | **+0.2039** | **[+0.1184, +0.2895]** | **38 / 7** |
| SnapKV+focus − own unmasked SnapKV | +0.2434 | [+0.1579, +0.3355] | 43 / 6 |
| floor_pos+focus − own unmasked floor_pos | +0.0132 | [−0.0329, +0.0592] | 7 / 5 |

The pre-generation ceiling audit found floor holding and attention-selecting
the gold sentence in **37/152** queries. Focused floor answered 24/37 of
those exactly. SnapKV held it in **98/152**, and its frozen lexical pointer
selected it in **79/152**; focused SnapKV answered 53/79 selected-gold cases
exactly. Thus the focused prompt improved extraction when the sentence was
right, while SnapKV supplied the right selected sentence more often.
These are mechanism diagnostics, not evidence that the unfocused literal
mask alone beats floor on SQuAD: the prior independent unfocused test scored
46/260 versus floor 45/260 with CI crossing zero.

Every focused arm matched its own unmasked retention arm on prefill keep-set
hash, context length, prefill budget and held status for all 152 queries.
Selected sentences matched the frozen prefill-only predictions, and every
focused raw answer was an exact substring of its selected sentence. The
corrected full-context invented count was zero for both focused arms.
These checks passed before interpreting the CIs.

Focus inserted a mean **49.27** post-prefill tokens for floor (maximum 110)
and **43.86** for SnapKV (maximum 105). These tokens were *not* deducted
from the 0.2802 prefill retention budget; they consume additional working
KV and compute. The two prefill-only ceiling captures used 1.23 and 1.20
GPU processing minutes after model load. Held-out answer generation used
**2.51 GPU processing minutes after load**. No M3, RTX Pro 5000, or
post-result prompt iteration ran. Raw rows are in
`out/realtext_focus_M2_confirm_results.jsonl`.
