# Fresh M2 SQuAD confirmation with exact per-query pre-answer KV equality

The run followed `PREREG_REALTEXT_TIGHT_KV_M2_M3.md`, committed in `3c9d15e` and locked for confirmation in `fa17599` before any held-out answers. It used 160 fresh instances / 320 SQuAD queries, manifest SHA-256 `c1b440a6cf10b3070f61e1515239d9cf257b58fc8a12f36f72818fdf1a7fbd46`, on M2 and RTX 5070. The scout pointer was selected at the original full retention budget; a second prefill reduced retained tokens by the exact token cost of the selected sentence. The scout cache was released before the final cache. All arms had mean **635.67 pre-answer KV tokens**, with per-query equality asserted and checked.

| Arm | Strict exact answers | Scout selected gold | Final gold held | Mean focus tokens | Cap stops |
|---|---:|---:|---:|---:|---:|
| Plain full-budget floor | 63/320 | 84/320 diagnostic | — | 0 | 5 |
| Plain full-budget SnapKV | 58/320 | 181/320 diagnostic | — | 0 | 2 |
| Two-pass focused floor + literal_v1 | 53/320 | 84/320 | 84/320 | 51.67 | 17 |
| **Two-pass focused SnapKV + literal_v1** | **107/320** | **181/320** | **188/320** | **43.79** | 11 |

All four arms completed all 320 queries. There were no allocation fallbacks. The final cache still held the scout-selected sentence in 302/320 focused-floor and 265/320 focused-SnapKV queries; queries where it did not were retained in the score. Both focused arms had zero corrected full-context invented values, and every focused raw answer was an exact substring of its selected source sentence. The saved prefill pointer predictions matched the generated selections, strict whole-answer exact scores recomputed from raw text, and both keep-hash and per-query KV checks passed.

The 95% CIs use 10,000 paired **instance-level** bootstrap resamples, seed 5070, averaging each instance's two query differences before resampling.

| Comparison | Difference | 95% CI | Conversions / broken |
|---|---:|---:|---:|
| **Focused SnapKV − plain full-budget floor (headline)** | **+13.75 points** | **[+8.44, +19.06]** | 63 / 19 |
| **Focused SnapKV − equally focused floor** | **+16.88 points** | **[+11.88, +21.88]** | 64 / 10 |
| Focused SnapKV − plain SnapKV | +15.31 points | [+10.31, +20.31] | 68 / 19 |
| Focused floor − plain floor | −3.13 points | [−5.94, −0.31] | 7 / 17 |

**Both preregistered M2 lower-bound criteria are positive.** Thus this test establishes a matched-*pre-answer-KV-token* SQuAD win over both plain full-budget and equally focused floor **on M2**. The prior fixed-128-reserve test's inconclusive plain-floor interval remains a separate result on different cases. This implementation pays for focus with a second context prefill, so equal KV tokens do not imply equal computation or latency. Generated answer lengths also differ; memory equality is defined at the first answer token, as preregistered. Cross-model replication on M3 is still required for a two-model matched-memory claim.

Measured GPU processing after model load was 2.34 minutes floor prefill audit, 2.38 minutes SnapKV prefill audit and **8.84 minutes** answer generation, or **13.56 minutes** total for M2 confirmation excluding model loads, CPU preparation and scoring. Peak allocated GPU memory after model load during generation was 6888.8 MiB. Result artifact SHA-256: `402c12ccc4d9493e0ade68db838c8848ba9dccd21922a8bd09701d4117e21621`.
