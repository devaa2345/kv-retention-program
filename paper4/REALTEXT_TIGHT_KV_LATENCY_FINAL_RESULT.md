# Final RTX 5070 latency cost of tight-KV SQuAD focus

The corrected timing pilot passed the prospectively stated sanity gate, and the frozen 48-instance final manifest and timing code were committed in `eed9871` before either final run. The manifest contained one separate warmup and **48 measured instances / 96 paired questions**. M2 and M3 each ran all three arms once on the same questions. All 288 model-arm-query records per model were present and unique, with nonnegative phase clocks, no focused-budget fallback, zero pointer and second-pass time for plain floor, and valid shared-first-pass attribution. The frozen accuracy method, accuracy scores, prompts, pointers, masks, and retention logic were not edited or rescored.

## Warm per-query wall / post-load processing

Seconds/query below are **mean ± query SD**; first prefill charges half the shared context prefill to each of an instance's two questions. All phases use synchronized wall clocks after model load. The all-night term *GPU processing after load* refers to this synchronized elapsed time, **not pure GPU kernel time**. Therefore the phase-sum per-query post-load processing numbers are numerically the same as warm per-query wall numbers. Plain floor receives one pass, no pointer, and no second pass.

| Model | Arm | First prefill | Pointer | Second prefill | Generation | Total |
|---|---|---:|---:|---:|---:|---:|
| M2 | Plain floor | .207 ± .016 | 0 | 0 | .089 ± .098 | .296 ± .100 |
| M2 | Equally focused floor | .206 ± .016 | .049 ± .005 | .374 ± .030 | .144 ± .144 | .773 ± .156 |
| M2 | Focused SnapKV | .209 ± .015 | .002 ± <.001 | .379 ± .030 | .130 ± .116 | .719 ± .125 |
| M3 | Plain floor | .217 ± .016 | 0 | 0 | .193 ± .123 | .410 ± .125 |
| M3 | Equally focused floor | .218 ± .015 | .060 ± .005 | .386 ± .032 | .109 ± .089 | .774 ± .099 |
| M3 | Focused SnapKV | .219 ± .015 | .005 ± <.001 | .390 ± .033 | .091 ± .084 | .704 ± .096 |

Use **10,000 paired instance-bootstrap resamples**, seed 5070, averaging each instance's two query times before resampling. CIs are 95% percentile intervals for the focused-minus-plain difference and the corresponding resampled mean ratio; questions were never resampled independently.

| Model | Focused arm vs plain floor | Added warm/post-load s/query (95% CI) | Added percent of floor (95% CI) | Wall ratio |
|---|---|---:|---:|---:|
| M2 | Focused SnapKV | **+.423 [.396, .451]** | **+142.9% [127.8%, 158.3%]** | **2.43×** |
| M2 | Equally focused floor | +.477 [.447, .507] | +161.1% [144.8%, 177.2%] | 2.61× |
| M3 | Focused SnapKV | **+.294 [.266, .324]** | **+71.7% [61.6%, 83.5%]** | **1.72×** |
| M3 | Equally focused floor | +.363 [.334, .393] | +88.6% [77.5%, 101.0%] | 1.89× |

The added synchronized post-load processing seconds/query and percentages are the same .423/+142.9% M2 and .294/+71.7% M3, because they are measured by the same post-load phase clocks. The complete **three-arm** post-load processing totals, including timing-loop overhead but excluding model load and warmup, were **173.97 s M2** and **183.93 s M3**. Different answer lengths matter: mean emitted-token counts were M2 3.75 plain floor versus 4.73 focused SnapKV, and M3 9.97 plain floor versus 3.95 focused SnapKV. M3's shorter focused generation offsets part of its second-pass cost; the result is an observed workload latency ratio, not an isolated prefill speed ratio.

## CUDA-event components and outer wall

CUDA-event seconds/query below capture the model-forward intervals. The **raw shared context-prefix event** is reported before dividing it between two questions; the A1 query event is separately measured. For per-query accounting, half the raw prefix event plus the A1 event produces the attributed first-pass event. The pointer is CPU work and has no model-forward CUDA event.

| Model | Arm | Raw shared-prefix CUDA | A1-query CUDA | Second-pass CUDA | Generation CUDA |
|---|---|---:|---:|---:|---:|
| M2 | Plain floor | .342 | .034 | 0 | .087 |
| M2 | Focused floor | .342 | .034 | .374 | .128 |
| M2 | Focused SnapKV | .347 | .034 | .379 | .114 |
| M3 | Plain floor | .364 | .035 | 0 | .193 |
| M3 | Focused floor | .365 | .035 | .392 | .095 |
| M3 | Focused SnapKV | .368 | .035 | .395 | .077 |

The final focused-SnapKV second/raw-prefix CUDA ratios were **1.09 M2** and **1.07 M3**, consistent with the corrected pilot and inside the preregistered 0.5–2 sanity interval. The second prefill is the dominant positive overhead: +.379 s of M2's +.423 s net difference, and +.390 s of M3's +.294 s net difference. M3's generation is .102 s faster than plain floor on this workload; this offset is not a reduction in second-pass compute. Pointer selection adds only .002 s M2 and .005 s M3 for focused SnapKV.

Outer process wall **including WSL launch and model load** was **204.87 s M2** and **213.27 s M3** for each complete three-arm job, with one shared model load per job. The directly logged WSL-to-Python-process-start intervals were 1.52 s and .24 s. Inside Python, model load took 18.48 s for each model; separate warmups took 4.09 and 4.11 s. Python job wall was 196.53 and 206.52 s; outer minus Python-job wall was 8.34 and 6.75 s, comprising the WSL interval plus Python imports/setup before the in-Python timer and process teardown. These one-time costs are **not assigned to an arm or hidden in the per-query multiplier**.

The already committed M2 matched-KV accuracy gain of **+13.75 points** over plain floor comes at **2.43× warm wall time per query** on this 5070 timing set. The already committed M3 gain of **+27.50 points** comes at **1.72× warm wall time per query**. These pair accuracy results with measured latency; no accuracy arm was rerun for this cost test.

This test makes **no comparison** against a hypothetical single-prefill method that might achieve the same accuracy. It also does **not amortize** the first pass across a multi-turn agent session where a cache might be reused. Both are open questions for later work, not mitigating factors in these cost figures.

The final manifest SHA-256 was `899c26faac0c21407ca24a84d6f8ef908256e7fd82a211ddc7e7f9934c9fd085`; timing-harness SHA-256 was `b71b7aedf7b8db18e0e867053538e7848b5672a03294a4d0c84f6e308b5211d7`. Raw result SHA-256: M2 `a995e7d564499648c7f19ad2f0efea68fea729e095f8f452142186300967587c`, M3 `3b840f9e4d9f5f1b85a7b0032c299291885517ce7063272b72e4e43166fb9f7c`. The machine-readable analysis is `out/realtext_tight_latency_final_summary.json`.
