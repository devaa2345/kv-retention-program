## Third model — Llama-3.2-3B-Instruct (cross-architecture)

Obtained after the Meta licence cleared. This is the model the second-model brief originally
asked for, and it changes the conclusion.

**Why it is the decisive test.** The two Qwen models share a tokenizer, an architecture family,
and an identical KV geometry (2 KV heads × 128 = **256-element quantizer groups**).
Llama-3.2-3B differs on all three: 28 layers, 24 attention / **8 KV heads** × 128 =
**1024-element groups**, a different tokenizer (the same task text tokenises to 916 rather than
1030 tokens), and a different architecture.

A larger quantizer group spans a wider min–max range, so the affine step is coarser and the
reconstruction error is **higher at every width**:

| bits | Llama-3.2-3B (1024-elt groups) | Qwen2.5-3B (256) | Qwen2.5-1.5B (256) |
|---|---|---|---|
| 8 | 1.44% / 1.12% | 1.16% / 0.89% | 1.18% / 0.91% |
| 4 | **24.28% / 18.90%** | 19.55% / 14.87% | 19.98% / 15.20% |
| 2 | 133.36% / 98.58% | 96.68% / 74.31% | 103.35% / 75.55% |

### Llama sweep (n=50 per width, budget 257)

| bits | n | mean acc | empty/EOS | repetitive | degenerate | clean | mixed |
|---|---|---|---|---|---|---|---|
| 8 | 50 | 0.3100 | 0.000 | 0.000 | **0** | 0 | 50 |
| 7 | 50 | 0.3067 | 0.000 | 0.000 | **0** | 0 | 50 |
| 6 | 50 | 0.3133 | 0.000 | 0.000 | **0** | 0 | 50 |
| 5 | 50 | 0.3067 | 0.000 | 0.000 | **0** | 0 | 50 |
| 4 | 50 | 0.3200 | 0.000 | 0.000 | **0** | 1 | 49 |
| 3 | 50 | 0.2600 | 0.000 | 0.010 | **0** | 2 | 48 |

**No collapse at any width.** Accuracy is flat from 8-bit through 4-bit (0.307–0.320) and
declines gracefully to 0.2600 at 3-bit. **Zero fully-degenerate prompts at every width**, zero
empty/EOS outputs, essentially no repetition. At 4-bit — where Qwen-3B produced 19 of 50 fully
degenerate prompts — Llama's outputs are clean and well-formed.

### The three-model comparison — and the corrected conclusion

| model | 8 | 7 | 6 | 5 | 4 | 3 | KV group | 4-bit err |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 0.1433 | 0.1600 | 0.1300 | **0.0000** | 0.1167 | **0.0000** | 256 | 19.98% |
| Qwen2.5-3B | 0.2500 | 0.2500 | 0.2500 | 0.2467 | **0.0367** | 0.0967 | 256 | 19.55% |
| **Llama-3.2-3B** | 0.3100 | 0.3067 | 0.3133 | 0.3067 | 0.3200 | 0.2600 | **1024** | **24.28%** |

**This refutes the two-model conclusion.** On the Qwen evidence alone it looked as though
non-monotone collapse was a property of the *phenomenon* — both Qwen models collapsed
catastrophically and then recovered under more error, differing only in which widths. Llama shows
**no collapse at all**, and does so while carrying *higher* reconstruction error than either Qwen
at every width.

The correct conclusion is therefore:

1. **Catastrophic non-monotone collapse is a model-family property, not a property of KV cache
   quantization.** Two models sharing an architecture family both exhibit it; a third with
   comparable parameter count and higher quantization error does not.
2. **Reconstruction error does not predict accuracy collapse.** Llama at 24.28% 4-bit key error
   is unharmed; Qwen2.5-3B at 19.55% loses 6.7× of its accuracy. Error is monotone in bit-width
   on all three models; accuracy is not, on two of three.
3. **A bit-width severity ordering derived from one model family does not transfer.** §9 cautions
   against reading a sweep as a severity ordering; this is direct evidence for that caution and
   extends it — the ordering is not merely non-monotone within a model, it is not stable across
   model families.

The quantizer is cleared as the cause on all three models independently: level counts never
exceed 2^bits, error is strictly monotone in bit-width, and the grouping is the documented
whole-slice scheme in every case. The mechanism producing the Qwen collapse lies in how those
models' downstream computation responds to quantization noise, not in the quantizer.

### Caveats

- Llama's `full_cache_ref` is **1.0000** — its ceiling is saturated, so its Phase 1 arms are
  compressed against a ceiling that leaves less room for protection effects than the 1.5B had.
- Its tokenizer yields **916 tokens** for the identical task text versus Qwen's 1030, so budgets
  154/257/514 represent slightly different retention ratios across models. The sweep holds budget
  257 fixed in *tokens*, not in *fraction of context*.
- Sweep reconstruction errors quoted here are prefill-measured. The intercepted-generation
  measurement was deferred because it could not be given a GPU slot without risking a third
  concurrent job; the two agree to within ~5% on both Qwen models.

