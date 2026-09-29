# M2 code-boundary diagnostic: mixed win, narrow scope

The preregistered **mixed-set** contrast passed on 64 new held-out instances
(256 queries): SnapKV with the lexical sentence pointer and code-run boundary
rule scored 112/256 versus floor_pos 55/256, a **+0.2227** difference with
95% paired instance-bootstrap CI **[+0.1719, +0.2773]**. This establishes a
win for the present mixture of synthetic access-code needles in natural filler
and SQuAD prose questions on M2. It does **not** establish broad real-text
question-answering superiority: the improvement was concentrated in the
synthetic code-lookup half, and the SQuAD interval does not exclude zero.

The experiment and decision rule were frozen in
`PREREG_REALTEXT_BOUNDARY_M2.md` before answer generation. New held-out
indices 232–295 have JSONL SHA-256
`439c9958d2011251abaac5755f9926a0e7eb4e0021d1565b463c8cd0d397e202`.
All arms used the same Qwen2.5-3B-Instruct M2 model, A1 prompt, 0.2802
retained-context ratio, and strict whole-answer exact scoring after only
casefold and whitespace-run normalization. The new answer-blind rule triggers
for an `access code` question when the lexically selected sentence contains
exactly one isolated 4–12 digit run. Its mask permits only prefixes of that
whole run and EOS only after completion. Otherwise it falls back to the
unchanged literal_v1 sentence-substring mask. This uses source sentence text
already supplied to the literal mask; it does not receive the gold answer.

All 256 queries completed for all five arms. The SnapKV baseline and all
masked SnapKV arms had identical keep-set hashes, budgets, context lengths
and gold-retention status on each query. Every masked raw answer was a literal
substring of its selected sentence, and the corrected full-context invented
check found zero invented masked outputs.

| Arm | Correct /256 | Accuracy | Full-context invented | Cap stops |
|---|---:|---:|---:|---:|
| floor_pos | 55 | 0.2148 | 157 | 1 |
| SnapKV unmasked | 49 | 0.1914 | 165 | 0 |
| SnapKV + attention pointer + literal_v1 | 63 | 0.2461 | 0 | 0 |
| SnapKV + lexical pointer + literal_v1 | 72 | 0.2812 | 0 | 1 |
| **SnapKV + lexical pointer + code-run rule** | **112** | **0.4375** | **0** | **1** |

All intervals use 10,000 paired bootstrap resamples of complete four-query
instances (seed 5070). One query is 1/256 = 0.3906 percentage points.

| Contrast | Difference | 95% CI | Converted / broken |
|---|---:|---:|---:|
| **Boundary rule − floor_pos (primary)** | **+0.2227** | **[+0.1719, +0.2773]** | **66 / 9** |
| Boundary rule − lexical literal mask | +0.1562 | [+0.1094, +0.2031] | 40 / 0 |
| Boundary rule − own unmasked SnapKV | +0.2461 | [+0.1914, +0.3008] | 69 / 6 |
| Lexical literal mask − floor_pos | +0.0664 | [+0.0156, +0.1172] | 34 / 17 |

| Question type (128 each) | floor_pos | SnapKV | attention literal | lexical literal | boundary rule | Boundary − floor, 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| SQuAD | 19 | 26 | 26 | 27 | 27 | +0.0625 [0.0000, +0.1250] |
| Natural-filler access-code needles | 36 | 23 | 37 | 45 | **85** | **+0.3828 [+0.2891, +0.4766]** |

SnapKV held the gold sentence in 161/256 queries and was wrong in 114 of
those. The old attention pointer hit 103/256; the frozen lexical pointer hit
146/256. On needles specifically, the lexical pointer hit 85/128 and the
code-run rule answered **all 85** of those exactly. The rule activated on
113/128 needle queries; 15 used literal_v1 fallback. It never activated on
SQuAD. All 128 SQuAD boundary-arm raw outputs equalled those of the lexical
literal arm. The SQuAD result is consequently evidence about the unchanged
pointer/mask, not about the new boundary rule.

The parser's strong needle result is expected for a sentence containing one
code-shaped answer. The dataset deliberately uses two synthetic eight-digit
access-code queries per instance; the rule exploits this answer type. This
narrows the claim to exact-string lookup in this benchmark mixture, and any
claim of prose-QA improvement needs a separate SQuAD-focused test with a
within-sentence span selector. It would be misleading to present the +22.3
points as a general real-text QA gain.

Prefill-only development capture used 0.74 GPU processing minutes after model
load; development generation used 1.95 minutes; the independent held-out run
used **3.80 GPU processing minutes after load**. No further generation was
performed after seeing this result. No M3 or RTX Pro 5000 answer generation
occurred. Raw held-out rows are in
`out/realtext_boundary_M2_heldout_results.jsonl`.
