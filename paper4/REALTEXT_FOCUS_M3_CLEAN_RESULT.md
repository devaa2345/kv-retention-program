# Fresh, preregistered M3 focused-evidence SQuAD replication

The M3 run followed `PREREG_REALTEXT_FOCUS_M3_CLEAN.md`, committed as `19a9cd1` before any answer generation. It used 76 fresh, globally distinct-question instances / 152 SQuAD queries, manifest SHA-256 `b63d3e3fab356fae074732f84da500e3aeace25b9e476fba7502e4d7742da095`. The earlier uncommitted M3 outputs on exposed M2 cases were excluded from the design and are not counted here.

| Arm | Strict exact answers | Gold sentence held | Pointer selected gold | Focus extra tokens, mean / max | Cap stops |
|---|---:|---:|---:|---:|---:|
| Plain floor | 20/152 | 36/152 | 33/152 diagnostic | 0 | 14 |
| Plain SnapKV | 14/152 | 142/152 | 118/152 diagnostic | 0 | 15 |
| Focused floor + literal_v1 | 20/152 | 36/152 | 33/152 | 50.53 / 100 | 5 |
| **Focused SnapKV + literal_v1** | **72/152** | **142/152** | **118/152** | **46.43 / 152** | 3 |

The two focused arms used the same focus operation and arm-specific selected source sentence. Focused SnapKV answered 72/118 queries exactly when its pointer selected the gold sentence; focused floor answered 20/33 such queries. This supports the pointer-coverage mechanism, while the absolute success rates and KV sizes need not transfer across models.

The 95% CIs below use 10,000 paired **instance-level** bootstrap resamples, seed 5070; each resampled unit is the mean difference over an instance's two SQuAD queries.

| Comparison | Difference | 95% CI | Conversions / broken |
|---|---:|---:|---:|
| Focused SnapKV − focused floor | **+34.21 points** | **[+25.00, +43.42]** | 55 / 3 |
| Focused SnapKV − plain floor | **+34.21 points** | **[+25.66, +43.42]** | 56 / 4 |
| Focused SnapKV − plain SnapKV | +38.16 points | [+29.59, +47.37] | 61 / 3 |
| Focused floor − plain floor | 0 points | [−3.29, +3.29] | 3 / 3 |

**Both preregistered lower-bound criteria are positive**, so the M3 focused-system SQuAD replication passes. All four arms completed all 152 queries. Within each retention arm, focused and unmasked rows matched on keep-set hash, prefill budget and held status. Every focused selected sentence matched its saved prefill-only prediction, every focused raw answer was an exact substring of that source sentence, and both focused arms had zero corrected full-context invented values. Strict whole-answer exact scoring used only casefold and whitespace-run normalization.

This is a **system-level result with extra KV from sentence reinsertion**, not a matched-total-KV win. Focus added a mean 46.43 M3 tokens to SnapKV and 50.53 to floor outside the `round(0.2802*n_ctx)` per-head prefill budget. M3 has 8 KV heads versus M2's 2, so equal retention ratios do not imply equal absolute KV bytes across models. Prefill-only capture took 1.26 minutes per arm and answer generation took 2.98 minutes of measured RTX 5070 GPU processing after model load, totaling **5.50 GPU processing minutes** excluding load and CPU preparation/scoring. Result artifact SHA-256: `8d9a6be132e8807be8c26f3725366ed7bcf4b09c13cd1b114b147cec8dbe33af`.
