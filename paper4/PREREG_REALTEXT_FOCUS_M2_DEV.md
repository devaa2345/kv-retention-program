# M2 focused-evidence SQuAD development pilot

This is a new **same-M2** pipeline hypothesis after the negative SQuAD-only
confirmation and bounded semantic-pointer screen. It is not a result on those
spent held-out sets. The original A1 prompt, strict whole-answer exact
scoring (casefold and whitespace-run normalization only), 0.2802 prefill
retention ratio, and literal_v1 substring mask remain fixed where used.

The previous independent confirmation retained the SQuAD gold sentence in
172/260 SnapKV queries, selected it in 142/260, and answered exactly in only
46/142 correct selections. Sentence embeddings and retrieval rerankers did
not materially improve localization on 176 older development queries
(`REALTEXT_SQUAD_POINTER_SCREEN.md`). This pilot tests whether **showing the
selected sentence to M2 again at answer time** improves copying from a
correct selection. It adds prompt tokens and is therefore a new system, not
a pure decoding-mask effect. Both retention arms receive the same focused
prompt rule so the primary comparison has a matched control.

## Data and ceilings before any pilot answer generation

Fresh candidate indices 456–503 were built by the unchanged data constructor.
The CPU filter in `out/_realtext_focus_filter.py` selected the first 32
instances whose two SQuAD questions were not used in prior real-text data
or among earlier selected instances. Seven candidate indices were skipped
for question duplication. The 32-instance, 64-SQuAD-query development
manifest has SHA-256
`6f4242cfa905f52138199f9c346d6163bf48089695bab198ab481a9f9a16ade4`.
The context still contains two synthetic code needles but only SQuAD answers
will be generated or scored.

Before generation, prefill-only capture for both retention arms cost 0.46
(floor_pos) and 0.50 (SnapKV) measured GPU processing minutes after model
load. At the same 0.2802 prefill budget, floor_pos held the gold sentence in
22/64 and its last-prompt-token attention pointer selected it in **21/64**.
SnapKV held the gold sentence in 42/64 and the frozen BM25 lexical pointer
selected it in **34/64**. The absolute exact-answer ceiling under one
selected-sentence literal mask is thus 21/64 for floor+focus and 34/64 for
SnapKV+focus. These are ceilings, not expected accuracies. The choice of
floor's attention pointer and SnapKV's lexical pointer is fixed here before
answer generation; each is the previously studied rule that localized best
for that arm on this prefill-only development set.

## Exact focus rule and paired arms

For each SQuAD query, run the original A1 post-prefill query once to choose
the pointer and obtain the unmasked baseline answer. Then rerun the query
against an identical clone of that arm's compressed prefill cache, inserting
the selected eligible sentence before A1's final instruction as:
`Evidence sentence from the passage:\n{sentence}\nEnd evidence sentence.`
The context prefill, keep-set and budget are unchanged. Decode the focused
answer using literal_v1 restricted to that same selected sentence. Neither
pointer nor focus text sees the gold answer. The four paired arms are
floor_pos unmasked, SnapKV unmasked, floor_pos+focus+literal mask, and
SnapKV+focus+literal mask. The two focused arms receive the **same focus
format** and their arm-specific selected sentence. Log extra post-prefill
tokens for each. Because these tokens add working KV and GPU cost, do not
describe the focus result as an equal-total-KV version of the old baseline.

Pilot checks: all four arms finish all 64 queries; each focused arm matches
its own unmasked arm on prefill keep-set hash, n_ctx, retention budget, and
held status; every focused raw answer is an exact substring of the selected
sentence; every selected sentence matches the corresponding frozen prefill
prediction. No oracle sentence may be supplied.

Proceed to an independent confirmation only if SnapKV+focus exceeds
floor_pos+focus by at least **4/64 queries** in this development pilot and
both focus arms' checks pass. Otherwise stop this method and report the
negative. If passed, compute the paired per-instance variance of the
SnapKV+focus minus floor_pos+focus mean over two SQuAD queries. For a
10-percentage-point target set
`N_power=ceil((1.96+0.84)^2*s^2/0.10^2)` and
`N=min(160,max(64,N_power))`; if N_power>160 report the power limit. Build
fresh distinct-question instances starting at index 504, lock the data hash,
N and unchanged code before any confirmatory generation.

A confirmation would require **both** 95% paired instance-bootstrap CIs
(10,000 resamples, seed 5070) for SnapKV+focus minus floor_pos+focus and
SnapKV+focus minus unmodified floor_pos to have lower bounds >0 before
claiming a SQuAD floor win. Report SnapKV+focus minus own unmasked SnapKV
separately. Report answer counts, conversions/broken cases, gold retention,
pointer hit, focus extra tokens, corrected full-context invented values,
cap stops, and measured GPU processing time. This is one M2 model; no M3
or RTX Pro 5000 generation. If the focus method is promising, its extra
prompt tokens and source-text side channel must be included in any later
deployment comparison.

## Development gate and independent confirmation lock

All four development arms completed 64/64 queries and passed keep-set,
budget, selected-sentence, and literal-output checks. Correct counts were
floor_pos 16/64, SnapKV 15/64, floor_pos+focus 16/64, and SnapKV+focus
24/64. Thus the fixed pilot gate passed by **8/64** queries (24−16 >=4).
The descriptive paired instance CI for SnapKV+focus minus floor_pos+focus
was [+0.015625, +0.234375]; the point estimate was +0.125. Focus added a
mean 53.98 post-prefill tokens to floor and 44.13 to SnapKV, maxima 108
and 113. Development answer generation used 0.98 measured GPU processing
minutes after model load. The observed per-instance variance of the paired
two-SQuAD-query focus difference was 0.09677419. The frozen formula gives
`N_power=76`, so confirmation uses **76 instances / 152 SQuAD queries**.

Fresh candidate indices 504–603 were generated without model outcomes.
The CPU first-distinct-question filter in
`out/_realtext_focus_confirm_filter.py` selected 76 instances (indices
504–596, with 17 skips for repeated questions against prior datasets or
already selected cases). The final manifest SHA-256 is
`1d25bf5b273ff4cbf6d636e60c92f82799389adf617ad7626fcf0d610bb44641`.
No answer generation occurred before the following ceiling audit. The
prefill-only captures used 1.23 (floor_pos) and 1.20 (SnapKV) measured GPU
processing minutes after their model loads. Floor held and attention-selected
the gold sentence in **37/152**; SnapKV held it in **98/152**, and the frozen
lexical pointer selected it in **79/152**. Thus the absolute one-sentence
literal-mask ceilings for the focused arms are 37/152 and 79/152. The
captures' hashes are `839e8d3dbd55ad892334a5fcea7ce09c1d03e44f0c6b741d8e3c3a95e178caf2`
(floor) and `92832b3948642a98b05ddfb66cc0d507c61c9e0816f0c0837cfe04c3690a19f1`
(SnapKV). No pointer, prompt, mask, budget, scoring, or decision-rule change
is allowed in this confirmation. Estimated generation cost is about 2–3
GPU processing minutes after model load, based on development throughput.
