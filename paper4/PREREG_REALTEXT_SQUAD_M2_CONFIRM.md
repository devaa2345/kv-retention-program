# M2 SQuAD-only confirmation of the unchanged lexical literal mask

This is an independent **SQuAD prose-QA** confirmation, not a new method
search. The model is M2 Qwen2.5-3B-Instruct on RTX 5070. The tested method is
exactly the prior SnapKV retention with the frozen BM25 lexical sentence
pointer (`out/_realtext_pointer_screen.py`) and literal_v1 substring mask
(`out/_realtext_5070_mask.py`). It does not use the code-run rule, which never
activated on SQuAD. All arms use the same A1 prompt, 0.2802 retained-context
ratio, and strict whole-answer exact match after only casefold and
whitespace-run normalization. No substring-contains scoring.

## Audit and sample size fixed before new answer generation

In the previous 64-instance mixed test, SQuAD accuracy was 27/128 for the
lexical literal mask and 19/128 for floor, a +0.0625 paired lead with 95%
instance-bootstrap CI [0, +0.125]. The 101 masked misses were 67 pointer
misses, 19 disjoint wrong spans in a correct sentence, and 15 boundary or
overlap errors in a correct sentence. The mask broke six correct unmasked
answers, all with a wrong selected sentence, and converted seven wrong ones.
The audit is in `REALTEXT_SQUAD_AUDIT.md`. These outcomes size this test but
do not change its pointer, mask, prompt, budget, or scoring.

The per-instance paired difference averaged over the two SQuAD queries had
sample variance 0.05952381. For a 6-percentage-point target, two-sided
alpha .05 and nominal 80% power, the normal approximation gives
`ceil((1.96+0.84)^2 * 0.05952381 / 0.06^2) = 130` complete instances,
or **260 SQuAD queries**. One query is 1/260 = 0.3846 percentage points.

The fresh data are constructed deterministically from candidate indices
296–455. The CPU filter in `out/_realtext_squad_filter.py` takes the first
130 instances whose two casefolded SQuAD question strings are both absent
from every earlier real-text dataset and from already selected new instances.
It skips 24 candidates for that data-validity reason; selected indices run
from 296 to 449. Selection uses no model outcomes. The final manifest
`out/realtext_squad_confirm_130.jsonl` has SHA-256
`97595a5337fdb7e309add2fb93533aa9720b5e3437a9b701fbfe5198e020e9fa`.
The generic builder labels contexts as heldout and includes two code needles,
but this test **generates and scores only the two SQuAD queries per instance**.
The needles merely remain as part of the same real-text context construction.

Before answer generation, M2 prefill-only capture on these 130 instances
took 3.24 GPU processing minutes after load. SnapKV held the SQuAD gold
sentence in 172/260 queries; the frozen lexical pointer selected it in
142/260. Therefore the literal single-sentence mask has an **absolute
accuracy ceiling of 142/260 = 0.5462** on this set. The maximum possible
absolute accuracy gain over floor can only be calculated once the paired
floor score is known. This ceiling is not a forecast. Prefill rows have
SHA-256 `04f08b723f9acc985389379fc8305203c35d96924b9a092f40b38ef5922c3c49`.
No answers were decoded during this ceiling check.

## Frozen confirmation

Generate three paired arms on all 260 SQuAD queries: floor_pos unmasked,
SnapKV unmasked, and SnapKV + frozen lexical pointer + literal_v1 mask.
Primary contrast is masked SnapKV minus floor_pos. Claim SQuAD superiority
only if a 95% paired bootstrap interval from 10,000 resamples of whole
two-query instances (seed 5070) has **lower bound >0**. Report the point
estimate and CI even on failure. Secondary own-baseline contrast is masked
SnapKV minus unmasked SnapKV; report conversions and broken correct answers
for both contrasts. Do not tune on the new result or run a second confirmation
after seeing it.

Verify per-query SnapKV baseline and masked arm share the exact keep-set
hash, budget, context length, and held status. Every masked raw answer must
be a literal substring of its selected sentence; compare on-the-fly lexical
selections with the frozen prefill-only predictions. Report completed queries,
cap stops, corrected full-context invented counts, pointer hit, held-but-wrong,
and measured GPU processing time. No M3 or RTX Pro 5000 generation. At the
prior M2 throughput, answer generation is estimated at roughly 3–5 GPU
processing minutes after model load.
