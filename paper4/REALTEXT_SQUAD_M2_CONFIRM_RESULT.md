# M2 SQuAD-only confirmation: no superiority over floor

The unchanged SnapKV + lexical sentence pointer + literal_v1 mask did **not**
beat floor on a new 130-instance, 260-SQuAD-query confirmation. It scored
46/260 against paired floor_pos 45/260: **+0.00385** (+0.38 percentage
points), 95% paired instance-bootstrap CI **[−0.04231, +0.05000]**. The
preregistered lower-bound->0 decision rule failed. It also did not establish
an own-baseline gain over unmasked SnapKV. This is the honest SQuAD result;
the prior +8/128 lead was not confirmed on an independent sample.

The method, size and sample were frozen in `PREREG_REALTEXT_SQUAD_M2_CONFIRM.md`
before answer generation. The final 130-instance manifest has SHA-256
`97595a5337fdb7e309add2fb93533aa9720b5e3437a9b701fbfe5198e020e9fa`.
It contains 260 SQuAD questions not repeated from any earlier real-text
dataset or within this confirmation. The constructor's two synthetic needles
remain in each context, but only SQuAD questions were decoded or scored.
Model M2 was Qwen2.5-3B-Instruct on RTX 5070. All arms used the same A1
prompt and 0.2802 retained-context ratio. Accuracy is strict whole-answer
exact match after only casefold and whitespace-run normalization; no
substring-contains scoring.

| Arm | Correct /260 | Accuracy | Completed | Invented full-context values | Cap stops |
|---|---:|---:|---:|---:|---:|
| floor_pos unmasked | 45 | 0.1731 | 260 | 160 | 2 |
| SnapKV unmasked | 45 | 0.1731 | 260 | 132 | 2 |
| SnapKV + lexical pointer + literal_v1 | 46 | 0.1769 | 260 | 0 | 2 |

All intervals below resample complete two-query instances 10,000 times with
seed 5070. One query changes accuracy by 1/260 = 0.3846 percentage points.

| Comparison | Difference | 95% CI | Converted / broken |
|---|---:|---:|---:|
| **Masked SnapKV − floor_pos (primary)** | **+0.00385** | **[−0.04231, +0.05000]** | **20 / 19** |
| Masked SnapKV − own unmasked SnapKV | +0.00385 | [−0.03462, +0.04231] | 13 / 12 |

The pre-generation ceiling check found the gold sentence held by SnapKV in
172/260 and selected by the frozen lexical pointer in 142/260; the maximum
absolute literal-mask accuracy was therefore 142/260. The actual mask
answered only 46/142 selected-gold cases exactly. This is a within-sentence
extraction limit in addition to 118 pointer misses, 88 of which were not
retained at all. A better pointer alone cannot solve unretained cases.

The SnapKV baseline and masked arm matched keep-set hash, budget, context
length and held status on all 260 queries. Every on-the-fly lexical choice
matched the frozen prefill-only prediction, every masked raw answer was a
literal substring of its selected sentence, and the corrected full-context
invented count was zero. All checks passed before interpreting the CI.

Prefill-only ceiling measurement used 3.24 GPU processing minutes after
model load. The confirmatory generation used **3.26 GPU processing minutes
after load**. No second confirmation, prompt iteration, M3, or RTX Pro 5000
generation was run after seeing the result. Raw rows are
`out/realtext_squad_M2_confirm_results.jsonl`.

The separately preregistered +22.3-point mixed-set gain in
`REALTEXT_BOUNDARY_M2_RESULT.md` remains valid as an **exact-string lookup**
result driven by synthetic access-code needles. It should not be presented
as a general SQuAD or prose-QA gain.
