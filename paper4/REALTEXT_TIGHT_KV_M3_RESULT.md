# Fresh M3 SQuAD confirmation with exact per-query pre-answer KV equality

This is the preregistered M3 branch of `PREREG_REALTEXT_TIGHT_KV_M2_M3.md`, committed before M2 development generation in `3c9d15e` and locked for M3 held-out generation in `d8ebfa9`. The unchanged method, scorer and 160-instance / 320-SQuAD-query manifest were used on M3 (`meta-llama/Llama-3.2-3B-Instruct`) and RTX 5070. Manifest SHA-256: `c1b440a6cf10b3070f61e1515239d9cf257b58fc8a12f36f72818fdf1a7fbd46`. The earlier uncommitted M3 exploratory outputs and the earlier clean **extra-KV** M3 test were excluded from this sample.

| Arm | Strict exact answers | Scout selected gold | Final gold held | Mean focus tokens | Cap stops |
|---|---:|---:|---:|---:|---:|
| Plain full-budget floor | 52/320 | 84/320 diagnostic | — | 0 | 25 |
| Plain full-budget SnapKV | 38/320 | 241/320 diagnostic | — | 0 | 29 |
| Two-pass focused floor + literal_v1 | 52/320 | 84/320 | 83/320 | 52.66 | 10 |
| **Two-pass focused SnapKV + literal_v1** | **140/320** | **241/320** | **287/320** | **47.12** | 16 |

All four arms had exactly **604.94 mean pre-answer KV tokens** and the same pre-answer KV count on **each paired query**. The final prefill budget was reduced by that query's measured focus-token cost. No allocation fallback fired. The two compressed caches did not coexist; a full-budget scout pass selected the sentence, followed by a separate lower-budget prefill for answer generation. The final cache still retained the scout-selected sentence in 286/320 focused-floor and 303/320 focused-SnapKV queries; all cases remained in the score.

The 95% CIs use 10,000 paired **instance-level** bootstrap resamples, seed 5070, averaging each instance's two query differences before resampling.

| Comparison | Difference | 95% CI | Conversions / broken |
|---|---:|---:|---:|
| **Focused SnapKV − plain full-budget floor (headline)** | **+27.50 points** | **[+21.88, +33.13]** | 98 / 10 |
| **Focused SnapKV − equally focused floor** | **+27.50 points** | **[+21.56, +33.13]** | 99 / 11 |
| Focused SnapKV − plain SnapKV | +31.88 points | [+25.94, +37.81] | 110 / 8 |
| Focused floor − plain floor | 0 points | [−2.50, +2.50] | 10 / 10 |

**Both preregistered M3 lower-bound criteria are positive.** Complete paired records, saved scout-pointer predictions, strict whole-answer exact scores, literal-substring masked outputs and exact per-query pre-answer KV equality all passed independent scoring checks. Both focused arms had zero corrected full-context invented values. The method, prompt, pointer, mask, budget accounting, scorer and sample did not change after the M2 result.

Together with the separately preregistered [M2 result](REALTEXT_TIGHT_KV_M2_RESULT.md), this establishes the focused SnapKV system's SQuAD advantage over both floor controls at **equal pre-answer KV tokens within each of two 3B models**. This does not establish equal compute, latency, generation-stage KV use, or equal absolute KV bytes *across* models. The two-pass method performs a second context prefill and must be presented with that cost.

M3 prefill-only confirmation capture used 2.67 floor and 2.70 SnapKV measured GPU processing minutes after model load. Held-out answer generation used **10.45 minutes**, with peak allocated GPU memory 7303.0 MiB, totaling **15.82 GPU processing minutes** for M3 confirmation excluding model loads, CPU preparation and scoring. Result artifact SHA-256: `e5ef4d217e6788f38e854bfff5f381fe86eaf785cef1f95a781f8cdcac11d231`.
