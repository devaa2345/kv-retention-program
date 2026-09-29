# M2 SQuAD matched-preanswer-KV confirmation

The experiment followed `PREREG_REALTEXT_MATCHED_M2.md`, committed in `4d30597` before development generation and locked for confirmation in `3e67627` before confirmation generation. The independent heldout comprised 64 instances / 128 SQuAD questions (manifest SHA-256 `fba0407d6c2097f6fcf29364a773f9763a40418896d42eb5e8aa26e0e3c39043`). M2 ran on RTX 5070. Plain floor used `B0=round(0.2802*n_ctx)`; the focused arms used `B0-128` prefill retention and inserted at most 128 extra post-prefill tokens. Per-query assertions confirmed each focused arm's KV before its first answer token was no greater than plain floor's.

| Arm | Exact correct | Mean pre-answer KV | Gold sentence held | Pointer selected gold | Mean extra focus tokens | Fallbacks |
|---|---:|---:|---:|---:|---:|---:|
| Plain floor, full budget | 21/128 | 644.08 | 35/128 | 33/128 diagnostic only | 0 | 0 |
| Unmasked SnapKV, reduced budget | 13/128 | 516.08 | 59/128 | 51/128 diagnostic only | 0 | 0 |
| Focused floor, reduced budget | 16/128 | 562.25 | 25/128 | 25/128 | 46.17 | 2 |
| Focused SnapKV, reduced budget | **30/128** | **557.05** | **59/128** | **51/128** | 40.98 | 0 |

All counts are from the same heldout set. Plain floor's held and pointer columns are diagnostics computed at its own larger budget; its answer is unmasked. The two focused rows use `literal_v1`, not a changed scoring rule. All 128 queries completed in every arm. Every focused raw answer was an exact substring of its selected source sentence and both focused arms had zero corrected full-context invented values. Selected sentences matched the frozen prefill-only predictions; SnapKV's unmasked and focused rows matched on keep-set hash, budget and held status. The per-query pre-answer KV assertion passed. Cap stops were 0 plain floor, 0 unmasked SnapKV, 8 focused floor and 3 focused SnapKV.

The following 95% CIs use 10,000 **instance-level paired-bootstrap** resamples, seed 5070, averaging each instance's two SQuAD questions before resampling:

| Comparison | Difference | 95% CI | Conversions / broken |
|---|---:|---:|---:|
| Focused SnapKV − focused floor | **+10.94 points** | **[+3.91, +17.97]** | 19 / 5 |
| Focused SnapKV − plain full-budget floor | +7.03 points | **[0, +14.06]** | 17 / 8 |
| Focused SnapKV − unmasked reduced-budget SnapKV | +13.28 points | [+7.81, +19.53] | 19 / 2 |
| Focused floor − plain full-budget floor | −3.91 points | [−9.38, +1.56] | 3 / 8 |

The preregistered dual-CI rule required both focused SnapKV comparisons against focused floor and plain full-budget floor to have lower bounds >0. **It did not pass**, because the latter lower bound was exactly zero. The supported finding is an equal-cap focused-system advantage over the equally focused floor control, plus an inconclusive positive point estimate versus plain full-budget floor. The earlier extra-KV focused-system win remains separate. No matched-memory win over plain floor is claimed.

The 128-token reserve was conservative relative to observed mean focus additions, leaving focused SnapKV with about 87 fewer pre-answer KV tokens than plain floor on average. A future fresh-data test could allocate the reserve per query to match actual focus length more closely; this heldout set must not be reused for that design choice or confirmation. This is a hypothesis from the memory-accounting numbers, not a demonstrated fix.

Prefill-only confirmation audit cost 1.01 floor and 1.02 SnapKV measured GPU processing minutes after model load. Four-arm answer generation cost 2.69 measured GPU processing minutes after load. Together with the development prefill audits (0.50+0.49) and generation (1.35), measured GPU processing for this matched-control experiment was **7.06 minutes**, excluding model-loading time and CPU-only data preparation and scoring.
