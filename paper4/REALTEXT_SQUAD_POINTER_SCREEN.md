# CPU SQuAD pointer screen after failed SQuAD confirmation

The independent M2 SQuAD-only confirmation in
`REALTEXT_SQUAD_M2_CONFIRM_RESULT.md` scored 46/260 for lexical-mask SnapKV
versus floor_pos 45/260 (paired CI includes zero). This report screens
answer-blind semantic sentence selectors on **earlier development captures
only**; no answer generation or new held-out scoring occurred here.

The development rows are the old 24-instance pilot (0–23), lexical-pointer
development (104–135), and boundary-rule development (200–231): 88 complete
instances, 176 SQuAD queries. Candidate sentences are only those completely
eligible under the original union-of-per-head SnapKV keep definition.
The gold sentence is eligible in 112/176. The frozen lexical BM25 pointer
selects it in 92/176.

| Pointer on development SQuAD | Gold sentence selected /176 | Difference from lexical | Notes |
|---|---:|---:|---|
| Frozen lexical BM25 | 92 | reference | no new model |
| all-MiniLM-L6-v2 sentence cosine | 90 | −2 | CPU sentence embedder |
| Equal-rank lexical + embedding blend | 94 | +2 | paired instance CI [−0.0170, +0.0398] |
| ms-marco-MiniLM-L6 cross-encoder, all candidates | **98** | **+6** | CI [0.0000, +0.0682] |
| ms-marco cross-encoder, lexical top 10 | 97 | +5 | CI [0.0000, +0.0625] |
| BGE reranker base, all candidates | 92 | 0 | extra model cost without gain |
| BGE reranker base, lexical top 10 | 91 | −1 | |
| SQuAD-trained DistilBERT QA confidence, all candidates | 32 | −60 | confidence poorly comparable across sentence inputs |
| SQuAD-trained DistilBERT QA raw start/end score, all candidates | 89 | −3 | still below lexical |

The cross-encoder's six extra hits were distributed as 0/48 on the old
pilot, +3/64 on lexical development, and +3/64 on boundary development.
Its paired interval touches zero, and even this modest gain requires a
second model to score every eligible sentence. No selector passed a
meaningful pointer-only promotion gate; no M2 decoding was run with one.
The auxiliary model revisions are pinned in the CPU scripts. This screen
compares localization only, not final answer quality; raw predictions are
in the matching `out/realtext_*_pointer_dev_predictions.jsonl` files.

More importantly, the latest independent confirmation had 172/260 gold
sentences retained, 142/260 selected by lexical, and only 46/142 exact answers
when selected. Of the 118 pointer misses, **88 had no retained gold sentence**
and cannot be repaired by changing the selector under the current retention.
A perfect selector over the current eligible set can address at most the other
30 pointer misses. Even among already correct selections, 96/142 answers
were wrong, so within-sentence extraction is a substantial second bottleneck.
The earlier statement that "the pointer is the only lever big enough" is
not supported by this decomposition. Continuing to shop semantic pointers
on these development rows would risk overfitting; the next method needs a
fresh development set and must account for both retained evidence and
answer-span selection.

Script-reported CPU processing times (including model initialization) were
about 0.29 minutes for the sentence embedder, 0.42 minutes for the MS MARCO
cross-encoder, 2.51 minutes for BGE, and at least 1.13 minutes for
DistilBERT QA. Actual
end-to-end latency, model-loading memory, and any production cost would
have to be included in a deployable comparison. No RTX Pro 5000 run occurred.
