## Second model — Qwen2.5-3B-Instruct

`meta-llama/Llama-3.2-3B-Instruct` was the first choice but is gated (HTTP 401,
`GatedRepoError`, no token configured), so the second model is Qwen2.5-3B-Instruct. Same family
as the 1.5B, which makes the comparison isolate **scale** rather than confounding it with
architecture and tokenizer — at the cost of not testing across architectures. 36 layers,
16 attention / 2 KV heads, head_dim 128, hidden 2048, 3.09B params, bfloat16.

**The KV geometry is identical to the 1.5B** (2 KV heads × 128 = 256-element quantizer groups),
so the quantizer is directly comparable and cannot itself explain any difference.

### Calibration: the competence gate cannot be matched

The brief asked that N and budget be tuned to a comparable competence gate. **They cannot be.**
Across 8 configurations the 3B never fell below ~0.94:

| lever | configurations tried | `full_cache_ref` |
|---|---|---|
| N (credentials) | 6 / 9 / 12 / 16 at ctx ~1030–1250 | 0.9861 / 1.0000 / 1.0000 / 0.9896 |
| context + distractors | ctx 2635 / 5086 / 8061 / 8155 | 0.9861 / 0.8194 / 1.0000 / 0.9583 |
| re-test of the one promising point | ctx 5086, 30 fresh seeds | **0.9833** (the 0.8194 was a 12-seed fluke; pooled n=42 → 0.9365) |

The 1.5B's bottleneck was hex **copy fidelity** — 57–64% of its failures were near-misses. The
3B has largely solved that, so adding credentials only adds more of a task it can already do,
and lengthening context does not degrade it monotonically. **The ceiling difference is therefore
the scale effect itself, not a nuisance to be tuned away**, and it is reported as a measured
covariate. The matched-task configuration (identical to the 1.5B: N=6, 20 distractors, filler 7,
pad 26, budgets 154/257/514) was used so that scale is the only manipulated variable.

### Gates, all re-run on the 3B

| gate | 3B | 1.5B |
|---|---|---|
| G1 context length | median 1030.5, range [1017, 1041] | median 1030.5, range [1017, 1041] |
| G2 competence | **0.9833** | 0.7167 |
| G3 budget arithmetic | pass; budget 51 refused; `full_cache_ref` invariant at 0.9792 | pass; invariant at 0.8125 |
| G4 quantizer integrity | pass | pass |
| G5 dormancy | 100% weak, 0% strict | 100% / 0% |

3B quantizer error on real cached K/V: 8-bit 1.16%/0.89%, 4-bit 19.55%/14.87%,
2-bit 96.68%/74.31% — within ~2% of the 1.5B and of the spec reference at every width.
**Reconstruction error is essentially model-independent.**

### Phase 1 at 8-bit, iso-token

| budget | arm1 | arm2 | arm3 | arm4 | arm5 | arm6 | n |
|---|---|---|---|---|---|---|---|
| 154 | 0.0000 | 0.0811 | 0.0000 | 0.0800 | 0.2767 | 0.9989 | 150 |
| 257 | 0.0000 | 0.2478 | 0.0000 | 0.2478 | 0.9922 | 0.9989 | 150 |
| 514 | 0.1611 | 0.5622 | 0.1611 | 0.5644 | 0.9978 | 0.9989 | 150 |

| budget | protection | tiering | interaction | 95% CI | n differ |
|---|---|---|---|---|---|
| 154 | +0.0806 | -0.0006 | **-0.0011** | [-0.0033, +0.0000] | 1 |
| 257 | +0.2478 | +0.0000 | **+0.0000** | [+0.0000, +0.0000] | 0 |
| 514 | +0.4022 | +0.0011 | **+0.0022** | [-0.0033, +0.0078] | 6 |

**The 8-bit interaction is ≈0 on the second model too**, reproducing both the 1.5B's 8-bit
result (−0.0011 at 257, −0.0056 at 514) and the spec's reference (+0.002, −0.002). The
[GAP-W] finding — that the apparent interaction disagreement was an unstated bit-width, not a
mechanism difference — therefore replicates across scale.

### Bit-width sweep — the headline result

| bits | n | mean acc | empty/EOS | repetitive | degenerate | clean | mixed | recon err | levels/limit | viol |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 50 | 0.2500 | 0.000 | 0.000 | 0 | 5 | 45 | 1.07% | 161/256 | 0 |
| 7 | 50 | 0.2500 | 0.000 | 0.000 | 0 | 4 | 46 | 2.14% | 108/128 | 0 |
| 6 | 50 | 0.2500 | 0.000 | 0.020 | 0 | 5 | 45 | 4.31% | 61/64 | 0 |
| 5 | 50 | 0.2467 | 0.000 | 0.010 | 0 | 5 | 45 | 8.65% | 32/32 | 0 |
| **4** | 50 | **0.0367** | 0.020 | 0.053 | 19 | 1 | 30 | 18.10% | 16/16 | 0 |
| **3** | 50 | **0.0967** | 0.003 | 0.007 | 1 | 3 | 46 | 37.50% | 8/8 | 0 |

### The phenomenon reproduces; the widths do not

| model | 8 | 7 | 6 | 5 | 4 | 3 |
|---|---|---|---|---|---|---|
| **1.5B** | 0.1433 | 0.1600 | 0.1300 | **0.0000** ↓ | **0.1167** ↑ | **0.0000** ↓ |
| **3B** | 0.2500 | 0.2500 | 0.2500 | 0.2467 | **0.0367** ↓ | **0.0967** ↑ |

**Non-monotone accuracy under strictly monotone error reproduces on an independent
model.** The 3B is flat from 8 through 5 bits, drops 6.7× at 4-bit, then *recovers* at
3-bit under twice the reconstruction error — the same qualitative signature as the 1.5B.

**But the widths do not transfer at all.** The 1.5B collapses at 5-bit, where the 3B is
unaffected (0.2467); the 1.5B *recovers* at 4-bit, which is precisely where the 3B fails.
So non-monotonicity is a property of the phenomenon, while *which* widths collapse is
model-specific. This strengthens §9's caution: the ordering is not merely non-monotone, it
is not even stable across two models of the same family sharing a tokenizer and KV
geometry.

The quantizer is cleared independently on both models: identical group structure, error
tables agreeing within ~2%, zero level violations at every width, and strictly monotone
error on tensors intercepted during live generation.

### Fresh-seed replication of the collapse widths

| bits | seeds 0–49 | seeds 50–99 (unseen) | difference | n |
|---|---|---|---|---|
| 5 | 0.2467 | 0.2300 | -0.0167 | 50 |
| 4 | 0.0367 | 0.0333 | -0.0033 | 50 |
| 3 | 0.0967 | 0.0733 | -0.0233 | 50 |

The 4-bit collapse and the 3-bit recovery both hold on 50 unseen prompts, so the
headline is not a seed artefact.

### The flat 8/7/6-bit region is not a stuck value

Accuracy is 0.2500 at three consecutive widths, which warranted a check. Two independent
verifications show it is a genuine aggregate:

- **Per-seed distributions are non-degenerate and identical**: 30 seeds at 0.167, 15 at
  0.333, 5 at 0.500 → mean exactly 0.2500 at 8, 7 and 6 bits (5-bit: 31/14/5 → 0.2467).
- **The quantizer is demonstrably active**: generated *text* differs from the 8-bit run in
  28/50 prompts at 7-bit, 30/50 at 6-bit and 43/50 at 5-bit, and retained-token counts differ
  in 11/50.

So quantization is perturbing the model throughout the flat region; the perturbations simply
do not cross the exact-match scoring threshold until 4-bit. The plateau is *effect below
threshold*, not *absence of effect*.

