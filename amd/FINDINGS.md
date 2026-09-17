# FINDINGS

Blind reimplementation of `SPEC_REIMPL_v1.md`. No file of the original harness was opened,
imported, listed, or searched. Gap resolutions are in `SPEC_QUESTIONS.md`; every one
referenced below is documented there.

## Methods

**Model and decoding.** `Qwen/Qwen2.5-1.5B-Instruct`, bfloat16, HuggingFace `transformers`
with `attn_implementation="eager"` so attention weights are readable, greedy decoding, no
sampling. ROCm on an RX 7900 XTX (gfx1100). `PYTHONHASHSEED=0`.

**Task.** Multi-credential retrieval with induced dormancy. Per seed: 6 credentials
`CRED_<i>_KEY: sk-<14 hex>` (17-char values), 20 distractor lines of identical surface
shape with non-credential labels, and coherent 50-word filler paragraphs interleaved so
credential lines are separated and placed at varied, non-clustered depths. Context lands at
median 1029 tokens (range 1023-1034 at n=10), matching the spec's stated target. Six turns,
one credential each, `max_answer_tokens=22`. A credential counts as retrieved iff its full
17-character value appears verbatim in the generated text of the turn that asked for it.
Primary metric is fraction retrieved k/N per prompt, N=6.

**Cache model.** Every position is FULL (bf16), QUANT (quantized), or EVICT (discarded).
The quantizer is affine min/max, one group per (layer, position) spanning all
`kv_heads x head_dim` = 256 elements, arithmetic in fp32 with storage at the engine's bf16
([GAP-Q], [GAP-G]). Byte cost is `n_full + quant_byte_cost * n_quant`, default 0.25.

**Retention.** Identical in every arm: sink (position 0) and the 64 most-recent positions
are floors applied first; remaining budget is filled by attention-score rank; selection runs
every decode step (`rebalance_every=1`). The attention score is the head-mean, layer-sum,
accumulated over all steps including prefill, and divided by the number of queries that
could causally attend to each position ([SPEC-GAP 4], resolved empirically). One global
retained set is shared by all 28 layers ([GAP-H]). One cache spans all six turns, so
eviction persists across turns ([GAP-E]) and generated tokens are themselves evictable
([GAP-F]).

**Protection.** Structural protection is content-agnostic: it seats whole matched
`LABEL: sk-...` lines atomically, ranked by best member score, and therefore competes for
budget across all 26 matched lines, credential and distractor alike. `oracle_static`
instead knows which 6 lines are credentials and retains their label+value spans by
construction, filling surplus budget by attention rank ([GAP-K]).

**Determinism.** No selection over an unordered Python `set` reaches any output; every
ranking sorts explicitly with an ascending-index tie-break. Two processes with
`PYTHONHASHSEED=0` produce identical retained-token index sets. Independently, 127 cells
that were legitimately computed twice by different stages agreed exactly on accuracy,
effective_tokens and n_quant.

**Checkpointing.** Append-only JSONL, flushed and fsynced per row, resumed on the
`(nominal_budget, iso_condition, arm, seed, quant_bits, promotion, context_target)` key.
Failed runs are written to a separate failures file and never silently dropped, so the
paired seed sets stay identical across arms.

**Analysis.** Bootstrapped median of paired per-prompt differences, 10,000 resamples,
resampling unit = prompt; mean-difference CI and the count of prompts that differ at all
reported alongside, per the spec's own warning that the median CI collapses to zero width
on data discrete at 1/6. Paired sign-flip permutation test; Benjamini-Hochberg within each
iso-condition; equivalence declared when the 95% CI is fully inside +/-0.05.

## Phase 1 — iso-token, arms 1-6

Primary metric: fraction retrieved k/N per prompt, N=6. Paired design: every arm sees the identical seed set, so per-prompt differences are well defined.

### Descriptives

| budget | arm | protection / eviction | n | mean | SD | SE | frac=0 | frac=1 | eff. tokens (med) | comp. room |
|---|---|---|---|---|---|---|---|---|---|---|
| 154 | 1 | none/permanent | 150 | 0.0000 | 0.0000 | 0.0000 | 1.000 | 0.000 | 154 | 89 |
| 154 | 2 | structural/permanent | 150 | 0.0267 | 0.0611 | 0.0050 | 0.840 | 0.000 | 154 | 89 |
| 154 | 3 | none/tiered | 300 | 0.0000 | 0.0000 | 0.0000 | 1.000 | 0.000 | 200 | 89 |
| 154 | 4 | structural/tiered | 300 | 0.0600 | 0.0877 | 0.0051 | 0.663 | 0.000 | 200 | 89 |
| 154 | 5 | oracle_static | 150 | 0.1978 | 0.1145 | 0.0093 | 0.160 | 0.000 | 154 | 89 |
| 154 | 6 | full_cache_ref | 150 | 0.7989 | 0.2696 | 0.0220 | 0.053 | 0.427 | 1251 | n/a |
| 257 | 1 | none/permanent | 150 | 0.0011 | 0.0136 | 0.0011 | 0.993 | 0.000 | 257 | 192 |
| 257 | 2 | structural/permanent | 150 | 0.1656 | 0.0991 | 0.0081 | 0.180 | 0.000 | 257 | 192 |
| 257 | 3 | none/tiered | 600 | 0.0028 | 0.0213 | 0.0009 | 0.983 | 0.000 | 398 | 192 |
| 257 | 4 | structural/tiered | 600 | 0.1686 | 0.1454 | 0.0059 | 0.318 | 0.000 | 398 | 192 |
| 257 | 5 | oracle_static | 150 | 0.8089 | 0.2627 | 0.0215 | 0.040 | 0.447 | 257 | 192 |
| 257 | 6 | full_cache_ref | 150 | 0.7989 | 0.2696 | 0.0220 | 0.053 | 0.427 | 1251 | n/a |
| 514 | 1 | none/permanent | 150 | 0.0967 | 0.0823 | 0.0067 | 0.420 | 0.000 | 514 | 449 |
| 514 | 2 | structural/permanent | 150 | 0.4122 | 0.1729 | 0.0141 | 0.033 | 0.000 | 514 | 449 |
| 514 | 3 | none/tiered | 300 | 0.0383 | 0.0727 | 0.0042 | 0.777 | 0.000 | 668 | 449 |
| 514 | 4 | structural/tiered | 300 | 0.2694 | 0.2335 | 0.0135 | 0.283 | 0.007 | 668 | 449 |
| 514 | 5 | oracle_static | 150 | 0.8056 | 0.2704 | 0.0221 | 0.047 | 0.433 | 514 | 449 |
| 514 | 6 | full_cache_ref | 150 | 0.7989 | 0.2696 | 0.0220 | 0.053 | 0.427 | 1251 | n/a |

`comp. room` = total_budget − recency_window(64) − sink(1); spec §4 requires > 0 and the harness refuses any cell where it is not.

### Dormancy (gate G5) by budget, measured in arm 1

| budget | prompts with >=1 dormant window | mean dormant steps | max run |
|---|---|---|---|
| 154 | 1.000 | 2.09 | 2 |
| 257 | 1.000 | 2.27 | 2 |
| 514 | 0.680 | 0.93 | 1 |

### Paired contrasts (BH-corrected within iso-condition)

Estimator per spec §8: bootstrapped median of paired per-prompt differences, 10,000 resamples, resampling unit = prompt. §8 warns the median CI collapses to zero width because differences are discrete at 1/6 and mostly exactly zero, so the mean CI and the raw count of differing prompts are reported alongside — §8 calls that count the most informative of the three. p is a paired sign-flip permutation test (10,000); p_BH is Benjamini-Hochberg within this iso-condition. `equiv` = 95% mean CI fully inside +/-0.05 (minimum effect of interest).

| budget | contrast | mean diff | 95% CI (mean) | median diff | 95% CI (median) | n differ | a>b | b>a | p | p_BH | equiv | med CI 0-width |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 154 | protection / permanent | +0.0267 | [+0.0178, +0.0367] | +0.0000 | [+0.0000, +0.0000] | 24 | 0.0001 | 24 | 0 | 0.0001 | yes | YES |
| 154 | protection / tiered | +0.0200 | [+0.0111, +0.0289] | +0.0000 | [+0.0000, +0.0000] | 18 | 0.0001 | 18 | 0 | 0.0001 | yes | YES |
| 154 | tiering / unprotected | +0.0000 | [+0.0000, +0.0000] | +0.0000 | [+0.0000, +0.0000] | 0 | 1.0000 | 0 | 0 | 1.0000 | yes | YES |
| 154 | tiering / protected | -0.0067 | [-0.0122, -0.0022] | +0.0000 | [+0.0000, +0.0000] | 6 | 0.0323 | 0 | 6 | 0.0431 | yes | YES |
| 257 | protection / permanent | +0.1644 | [+0.1489, +0.1800] | +0.1667 | [+0.1667, +0.1667] | 122 | 0.0001 | 122 | 0 | 0.0001 | no | YES |
| 257 | protection / tiered | +0.1778 | [+0.1533, +0.2022] | +0.1667 | [+0.1667, +0.1667] | 106 | 0.0001 | 104 | 2 | 0.0001 | no | YES |
| 257 | tiering / unprotected | +0.0022 | [+0.0000, +0.0056] | +0.0000 | [+0.0000, +0.0000] | 2 | 0.5054 | 2 | 0 | 0.5514 | yes | YES |
| 257 | tiering / protected | +0.0156 | [-0.0067, +0.0367] | +0.0000 | [+0.0000, +0.0000] | 69 | 0.1848 | 39 | 30 | 0.2217 | yes | YES |
| 514 | protection / permanent | +0.3156 | [+0.2900, +0.3411] | +0.3333 | [+0.3333, +0.3333] | 141 | 0.0001 | 141 | 0 | 0.0001 | no | no |
| 514 | protection / tiered | +0.2000 | [+0.1711, +0.2300] | +0.1667 | [+0.1667, +0.1667] | 114 | 0.0001 | 107 | 7 | 0.0001 | no | YES |
| 514 | tiering / unprotected | -0.0567 | [-0.0689, -0.0444] | +0.0000 | [+0.0000, +0.0000] | 51 | 0.0001 | 0 | 51 | 0.0001 | no | YES |
| 514 | tiering / protected | -0.1722 | [-0.1989, -0.1456] | -0.1667 | [-0.1667, -0.1667] | 110 | 0.0001 | 6 | 104 | 0.0001 | no | no |

### Interaction: (arm4-arm3) - (arm2-arm1)

| budget | interaction | 95% CI | n differ | reference |
|---|---|---|---|---|
| 154 | -0.0067 | [-0.0122, -0.0022] | 6 | - |
| 257 | +0.0133 | [-0.0089, +0.0356] | 69 | +0.002 |
| 514 | -0.1156 | [-0.1444, -0.0856] | 112 | -0.002 |

### Accuracy by turn index

Under the [SPEC-GAP 1] turn order (reverse depth), turn index and context depth are perfectly anti-correlated by construction: turn *t* queries the credential at depth rank 5−*t*. These are therefore **one analysis, not two** — a turn effect and a depth effect are not separable in this design. Reported as a single table with that confound stated rather than as two tables implying independence. Turn index also equals the dormancy gap, since each credential is queried once.

| budget | arm | turn0 (deepest) | turn1 | turn2 | turn3 | turn4 | turn5 (earliest) |
|---|---|---|---|---|---|---|---|
| 154 | 1 none/permanent | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 154 | 2 structural/permanent | 0.160 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 154 | 3 none/tiered | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 154 | 4 structural/tiered | 0.327 | 0.033 | 0.000 | 0.000 | 0.000 | 0.000 |
| 154 | 5 oracle_static | 0.740 | 0.433 | 0.013 | 0.000 | 0.000 | 0.000 |
| 154 | 6 full_cache_ref | 0.753 | 0.787 | 0.773 | 0.847 | 0.840 | 0.793 |
| 257 | 1 none/permanent | 0.007 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 257 | 2 structural/permanent | 0.753 | 0.240 | 0.000 | 0.000 | 0.000 | 0.000 |
| 257 | 3 none/tiered | 0.017 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 257 | 4 structural/tiered | 0.568 | 0.327 | 0.085 | 0.000 | 0.003 | 0.028 |
| 257 | 5 oracle_static | 0.740 | 0.840 | 0.827 | 0.800 | 0.807 | 0.840 |
| 257 | 6 full_cache_ref | 0.753 | 0.787 | 0.773 | 0.847 | 0.840 | 0.793 |
| 514 | 1 none/permanent | 0.580 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 514 | 2 structural/permanent | 0.773 | 0.733 | 0.360 | 0.047 | 0.073 | 0.487 |
| 514 | 3 none/tiered | 0.213 | 0.017 | 0.000 | 0.000 | 0.000 | 0.000 |
| 514 | 4 structural/tiered | 0.507 | 0.413 | 0.193 | 0.183 | 0.130 | 0.190 |
| 514 | 5 oracle_static | 0.733 | 0.807 | 0.787 | 0.827 | 0.853 | 0.827 |
| 514 | 6 full_cache_ref | 0.753 | 0.787 | 0.773 | 0.847 | 0.840 | 0.793 |

**full_cache_ref by budget (spec §5 requires invariance): {154: 0.7988888888888889, 257: 0.7988888888888889, 514: 0.7988888888888889} — INVARIANT**

## Phase 2 — promotion signals (4-bit, budget 257)

Ceiling-normalised columns divide each arm by its own implementation's `full_cache_ref` (mine 0.7989, reference 0.947). This is a **post-hoc comparability aid, not the pre-registered metric** — the spec fixes raw k/N. It exists because [GAP-U] leaves my ceiling ~0.15 below the reference's, which shifts every level without necessarily changing the mechanism ordering.

| id | signal | mean acc | n | reference | mine / ceiling | ref / ceiling |
|---|---|---|---|---|---|---|
| P1 | attention | 0.1686 | 600 | 0.203 | 0.2111 | 0.2144 |
| P2 | epiphany | 0.1411 | 150 | 0.216 | 0.1766 | 0.2281 |
| P3 | random | 0.1133 | 150 | 0.154 | 0.1419 | 0.1626 |
| P4 | roundrobin | 0.1167 | 150 | 0.184 | 0.1460 | 0.1943 |
| P5 | oracle | 0.1378 | 150 | 0.188 | 0.1725 | 0.1985 |

Retention is held fixed and attention-ranked in every Phase 2 arm; protection is ON and tiered eviction is ON. Only the FULL/QUANT promotion decision varies, so these contrasts isolate promotion from retention.

### Phase 2 paired contrasts vs P3 (random control)

| contrast | mean diff | 95% CI (mean) | median diff | 95% CI (median) | n differ | p | p_BH | equiv |
|---|---|---|---|---|---|---|---|---|
| P1 (attention) vs P3 (random) | +0.0678 | [+0.0478, +0.0889] | +0.0000 | [+0.0000, +0.0000] | 63 | 0.0001 | 0.0002 | no |
| P2 (epiphany) vs P3 (random) | +0.0278 | [+0.0144, +0.0411] | +0.0000 | [+0.0000, +0.0000] | 36 | 0.0001 | 0.0002 | yes |
| P4 (roundrobin) vs P3 (random) | +0.0033 | [-0.0067, +0.0133] | +0.0000 | [+0.0000, +0.0000] | 21 | 0.6708 | 0.6708 | yes |
| P5 (oracle) vs P3 (random) | +0.0244 | [+0.0122, +0.0378] | +0.0000 | [+0.0000, +0.0000] | 35 | 0.0009 | 0.0012 | yes |

### Level-occupancy of quantized tensors

| bits | limit 2^bits | max levels seen | mean levels used | occupancy | violations |
|---|---|---|---|---|---|
| 8 | 256 | 152 | 113.05 | 0.442 | 0 |
| 7 | 128 | 102 | 73.98 | 0.578 | 0 |
| 6 | 64 | 61 | 45.44 | 0.710 | 0 |
| 5 | 32 | 32 | 26.76 | 0.836 | 0 |
| 4 | 16 | 16 | 14.87 | 0.929 | 0 |
| 3 | 8 | 8 | 7.85 | 0.981 | 0 |

### Phase 2 manipulation check — Spearman rho(eviction, promotion)

| id | signal | measured rho | target |
|---|---|---|---|
| P1 | attention | +1.0000 | +1.000 |
| P2 | epiphany | +0.0880 | ~0 |
| P3 | random | +0.0905 | ~0 |
| P4 | roundrobin | +0.4000 | ~0 |
| P5 | oracle | +0.8496 | ~0 |

**P5 ceiling verification:** oversubscribed on 0/12 prompts — PASS - never oversubscribed. P5 bounds the promotion decision only; it cannot rescue credentials already evicted at the retention stage.

## Context length at fixed retention ratio (~13%)

[GAP-J] **recency_window held FIXED at 64 at every context length.** Rationale and the cost of that choice are in SPEC_QUESTIONS.md; the alternative (scaling the window with context) is defensible and would make the competitively-selected fraction constant, partly cancelling the manipulation.

| target | realised ctx | full-cache len | budget | retention | comp. room | gates |
|---|---|---|---|---|---|---|
| 2048 | 2044 | 2251 | 293 | 0.130 | 228 | **GATE FAILED - cell not trusted** |
| 4096 | 4084 | 4287 | 557 | 0.130 | 492 | **GATE FAILED - cell not trusted** |

| target | arm1 | arm2 | arm3 | arm4 | interaction | 95% CI | n differ |
|---|---|---|---|---|---|---|---|
| 2048 | 0.000 | 0.132 | 0.000 | 0.090 | -0.0417 | [-0.0567, -0.0283] | 25 |
| 4096 | 0.000 | 0.333 | 0.000 | 0.175 | -0.1583 | [-0.2017, -0.1150] | 64 |

## Agreement report — this implementation vs the spec's references

| quantity | reference | this implementation | delta | traced to |
|---|---|---|---|---|
| quantizer err 8-bit keys (%) | 1.17 | 1.1800 | +0.0100 | [GAP-Q] resolved |
| quantizer err 8-bit values (%) | 0.88 | 0.9100 | +0.0300 | [GAP-Q] resolved |
| quantizer err 4-bit keys (%) | 20.06 | 19.9800 | -0.0800 | [GAP-Q] resolved |
| quantizer err 4-bit values (%) | 14.99 | 15.2000 | +0.2100 | [GAP-Q] resolved |
| quantizer err 2-bit keys (%) | 101.11 | 103.3500 | +2.2400 | [GAP-Q] resolved |
| quantizer err 2-bit values (%) | 74.68 | 75.5500 | +0.8700 | [GAP-Q] resolved |
| full_cache_ref | 0.947 | 0.7989 | -0.1481 | [GAP-U] copy fidelity, unresolved |
| interaction @ budget 257 | 0.002 | 0.0133 | +0.0113 | [SPEC-GAP 4] / [GAP-U] ceiling |
| interaction @ budget 514 | -0.002 | -0.1156 | -0.1136 | [SPEC-GAP 4] / [GAP-U] ceiling |
| Phase 2 P1 (attention) | 0.203 | 0.1686 | -0.0344 | [SPEC-GAP 7] P2 / [GAP-U] ceiling |
| Phase 2 P2 (epiphany) | 0.216 | 0.1411 | -0.0749 | [SPEC-GAP 7] P2 / [GAP-U] ceiling |
| Phase 2 P3 (random) | 0.154 | 0.1133 | -0.0407 | [SPEC-GAP 7] P2 / [GAP-U] ceiling |
| Phase 2 P4 (roundrobin) | 0.184 | 0.1167 | -0.0673 | [SPEC-GAP 7] P2 / [GAP-U] ceiling |
| Phase 2 P5 (oracle) | 0.188 | 0.1378 | -0.0502 | [SPEC-GAP 7] P2 / [GAP-U] ceiling |

Disagreements >0.05 are traced to a specific gap in the column above. The dominant one is [GAP-U]: the full-cache ceiling is 0.72 here against the spec's 0.947, which shifts every accuracy level downward. Paired *differences* are the estimands and are less affected, but copy-error noise costs statistical power.


---

## Phase 1 as a three-factor design: protection × tiering × bit-width

Arms 1 and 2 are permanent-eviction (`n_quant = 0`) and therefore bit-width invariant;
they serve as the shared reference level rather than being re-run. Complete n=150 data
exists at both widths for the tiered arms.

| budget | bits | protection | tiering | interaction | 95% CI | n differ |
|---|---|---|---|---|---|---|
| 154 | **8** | +0.0272 | +0.0006 | **+0.0011** | [+0.0000, +0.0033] | 1 |
| 154 | 4 | +0.0233 | -0.0033 | -0.0067 | [-0.0122, -0.0022] | 6 |
| 257 | **8** | +0.1639 | -0.0006 | **-0.0011** | [-0.0056, +0.0033] | 5 |
| 257 | 4 | +0.1417 | -0.0239 | -0.0456 | [-0.0611, -0.0311] | 47 |
| 514 | **8** | +0.3128 | -0.0017 | **-0.0056** | [-0.0189, +0.0067] | 32 |
| 514 | 4 | +0.2578 | -0.1144 | -0.1156 | [-0.1444, -0.0856] | 112 |

**Both 8-bit interaction CIs contain zero; both 4-bit CIs exclude it.** The tiering main
effect at 8-bit is indistinguishable from zero at every budget.

### The bit-width factor, tested directly

| budget | tiering @8-bit | tiering @4-bit | difference | 95% CI |
|---|---|---|---|---|
| 154 | +0.0006 | -0.0033 | **-0.0039** | [-0.0067, -0.0017] |
| 257 | -0.0006 | -0.0239 | **-0.0233** | [-0.0311, -0.0161] |
| 514 | -0.0017 | -0.1144 | **-0.1128** | [-0.1278, -0.0983] |

The width effect is significant at every budget and grows with budget: larger budgets
retain more credential tokens and therefore expose more of them to quantization damage.

### Interpretation

The interaction the spec reports as ≈0 is **not a null result; it is a result conditional
on bit-width.** Protection's benefit survives near-lossless (8-bit) tiering essentially
intact and is progressively destroyed by aggressive (4-bit) tiering, in proportion to how
much benefit there was to destroy. The reference's ≈0 and this implementation's −0.116 are
two slices of the same surface; both are correct about their own condition.

The mechanism is established independently by the credential-survival instrumentation:
arms 2 and 4 evict **exactly the same 92.1 credential tokens**, so tiering adds no eviction
at all. Its entire effect is precision loss on tokens that were retained — which is why it
tracks reconstruction error (1.18% at 8-bit vs 19.98% at 4-bit) rather than anything about
the retention policy.

> **This condition was added post-hoc.** The 4-bit Phase 1 ran first, the disagreement with
> the reference was observed, and the 8-bit condition was then specified and run to test a
> stated mechanism. The prediction was made before the run and confirmed, but this is an
> exploratory inference, not a pre-registered confirmatory one.



---

## Bit-width sweep — full table (common seed set, n=50 at every width)

8-bit and 4-bit had n=150 available; they are restricted to seeds 0–49 so all six widths
are measured on identical prompts. Full-n values: 8-bit 0.1644, 4-bit 0.1189.

A turn output is *well-formed* if it contains an `sk-<14 hex>` token. A prompt is *clean*
if all 6 turns are well-formed, *degenerate* if none are. `empty/EOS` is the fraction of
turn outputs empty after stripping. Reconstruction error is measured on tensors
**intercepted during live generation**.

| bits | n | mean acc | empty/EOS | repetitive | degenerate | clean | mixed | recon err | levels/limit | violations |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 50 | 0.1433 | 0.000 | 0.017 | 7 | 28 | 15 | 1.15% | 152/256 | 0 |
| 7 | 50 | 0.1600 | 0.000 | 0.043 | 7 | 23 | 20 | 2.28% | 102/128 | 0 |
| 6 | 50 | 0.1300 | 0.007 | 0.030 | 9 | 13 | 28 | 4.56% | 61/64 | 0 |
| **5** | 50 | **0.0000** | 0.223 | 0.127 | 48 | 0 | 2 | 9.36% | 32/32 | 0 |
| 4 | 50 | 0.1167 | 0.010 | 0.057 | 9 | 9 | 32 | 19.31% | 16/16 | 0 |
| **3** | 50 | **0.0000** | 0.513 | 0.050 | 46 | 0 | 4 | 39.92% | 8/8 | 0 |

### The headline claim reproduces

**Accuracy is not monotone in bit-width; reconstruction error is.** Error rises strictly
across all six widths (1.15 → 2.28 → 4.56 → 9.36 → 19.31 → 39.92 %), yet accuracy
collapses to exactly 0.0000 at 5-bit and **recovers to 0.1167 at 4-bit while carrying more
than twice the reconstruction error**, then collapses again at 3-bit.

The reference reports 4-bit recovery at 0.123; this implementation measures 0.1167 (n=50)
and 0.1189 (n=150). **The curve shape is independently confirmed.** The widths that
collapse differ — the reference reports 6- and 5-bit, here it is 5- and 3-bit with 6-bit
merely degraded — so the *phenomenon* replicates while the *specific widths* do not. That
is itself the substance of §9's caution against reading a sweep as a severity ordering.

**The quantizer is cleared.** Level counts never exceed 2^bits at any width (0 violations,
checked per group on tensors actually written during generation), occupancy rises smoothly
from 0.44 to 0.98 as width falls, and error is strictly monotone on those same intercepted
tensors. The collapse is how the model responds to quantization noise, not a quantizer bug.

The collapse signature is degenerate *output*: at 5-bit, 48 of 50 prompts produce no
well-formed value on any turn and empty/EOS jumps to 0.223; at 3-bit, 46 of 50 and 0.513.
At every working width empty/EOS is ≤0.018.



---

## [GAP-U] revisited: the ceiling gap is a scoring-convention difference

**The scorer was not changed.** This reports exactly what it does and measures how much of
the 0.799-vs-0.947 gap each alternative convention would close.

### What the scorer does

`score_turn` returns `target.value in generated_text` — a plain substring test of the full
17-character value (`sk-` + 14 lowercase hex) against the **raw** decoded text of that turn
(`skip_special_tokens=True`, nothing else). No lower-casing, stripping, whitespace
collapsing, unicode normalisation or tokenisation. Credit only on the turn that asked
([GAP-P]). Prompt score is k/6. This is the literal reading of §2 and of [SPEC-GAP 2]'s
stated default.

### What each convention rescues (arm 6, 900 turn judgements, n=150)

| scoring rule | accuracy | vs 0.947 |
|---|---|---|
| **1. exact 17-char substring — the actual scorer** | **0.7989** | −0.1481 |
| 2. case-insensitive substring | 0.7989 | −0.1481 |
| 3. substring after whitespace stripping | 0.7989 | −0.1481 |
| 4. 14-hex payload, `sk-` prefix optional | 0.8633 | −0.0837 |
| **5. payload within edit distance 1** | **0.9544** | **+0.0074** |
| 6. payload within edit distance 2 | 0.9722 | +0.0252 |

### What this establishes

**The normalisations [SPEC-GAP 2] contemplates rescue nothing.** Case-folding and
whitespace-stripping recover exactly zero failures, so "exact vs normalised match" is not
the axis the gap lies on. That eliminates the ambiguity the spec itself flagged.

Two other conventions do close it. The model frequently answers with the bare 14-hex
payload and omits the `sk-` prefix — e.g. `9d268ef50038b6` for `sk-9d268ef50038b6`. This is
a per-prompt behavioural mode, not scattered noise: 12 of 150 prompts do it on ≥3 of their 6
turns, accounting for 58 of 181 failures. Allowing a single character error in the payload
recovers a further 82. Together they land at **0.9544 against the reference's 0.947 — a
difference of 0.0074**, an order of magnitude below the 0.1481 gap under strict scoring.

### Conclusion, with its limits

The evidence favours a **scoring-convention difference over a capability difference**. A
0.947 ceiling is hard to reach under strict 17-character exact-substring scoring — this
model emits the correct payload far more often than this scorer credits — but sits almost
exactly on a prefix-tolerant, edit-distance-1 convention.

This is evidence, not proof: edit-distance tolerance is an unusual convention, §2's text
does not license it, and the reference's code cannot be inspected under the blind protocol.
What can be said firmly is that **the two implementations are probably not scoring the same
thing, and the axis is prefix-tolerance plus fuzzy matching — not the exact-vs-normalised
axis the spec anticipated.** §2 should state whether the `sk-` prefix is required and
whether any edit tolerance is permitted.

**The scorer remains unchanged; every other number in this document uses rule 1.**



---

## Structural protection degenerates into an oracle when distractors stop matching

Structural protection matches lines by surface pattern. Rewriting distractor values to a
non-credential form (`ref_<hex>` instead of `sk-<hex>`), n=50, budget 257:

| arm | shared shape | distinct shape | diff | 95% CI | p |
|---|---|---|---|---|---|
| arm2 | 0.1500 | 0.9067 | **+0.7567** | [+0.7133, +0.8000] | 0.0001 |
| arm4 | 0.1167 | 0.3400 | **+0.2233** | [+0.1567, +0.2900] | 0.0001 |
| arm6 | 0.7433 | 0.8800 | **+0.1367** | [+0.0533, +0.2300] | 0.0053 |

`arm2` (structural/permanent) jumps from 0.150 to **0.907 — above the full-cache ceiling
of 0.880 measured under the same condition**. With only 6 lines matching its pattern
instead of 26, structural protection stops competing and simply retains every credential:
it *becomes* `oracle_static`. The full-cache arm gains only +0.137, so most of arm 2's
+0.757 is degeneration of the mechanism, not the task becoming easier.

**Implication for pattern-based KV protection generally:** its strength is set by the ratio
of target lines to pattern-matching non-target lines. This is a property of the mechanism,
not a quirk of this task — and it means the spec's phrase "same surface shape" ([GAP-V])
is load-bearing in a way the document never states. The *direction* of the protection
effect should transfer to other designs; its *magnitude* should not be quoted out of the
6:26 ratio it was measured at.



---

## Phase 2 manipulation check (§7) at n=100

§7 requires ρ(eviction score, promotion score) to be +1.000 for P1 and near zero for the rest.
An earlier n=12 estimate was too small to act on; this is n=100 prompts per signal with
bootstrap CIs over prompts. Both the **original** signal definitions (those in force when the
completed Phase 2 data was generated) and the **corrected** ones are reported.

| id | signal | ρ original | 95% CI | ρ corrected | 95% CI | target | original verdict |
|---|---|---|---|---|---|---|---|
| P1 | attention | +1.0000 | [+1.0000, +1.0000] | +1.0000 | [+1.0000, +1.0000] | +1.000 | pass |
| P2 | epiphany | +0.0964 | [+0.0887, +0.1041] | +0.0964 | [+0.0887, +0.1041] | ~0 | pass |
| P3 | random | +0.0958 | [+0.0929, +0.0989] | -0.0560 | [-0.0592, -0.0526] | ~0 | pass |
| P4 | roundrobin | +0.4974 | [+0.4944, +0.5006] | -0.0218 | [-0.0250, -0.0186] | ~0 | **FAIL** |
| P5 | oracle | +0.8489 | [+0.8251, +0.8707] | -0.0781 | [-0.0971, -0.0604] | ~0 | **FAIL** |

**P1 is exactly +1.0000 with a zero-width CI**, as it must be: the P1 promotion score *is* the
eviction score by construction. That the check returns exactly 1.000 rather than approximately
1.000 is itself evidence the probe measures the quantity the policy actually uses.

### Two signals failed, and the check is what caught them

P4 (round-robin) at **+0.4974** and P5 (oracle) at **+0.8489** are not marginal — their CIs
exclude zero by a wide margin at n=100. The causes are specific and were found by inspection
once the check flagged them:

- **P5** assigned `1e9` to credential positions the oracle must promote and then *fell back to
  the eviction score* for every other position. An "oracle" signal that reuses the eviction
  ranking for 95% of positions is correlated with it by construction.
- **P4** rotated over raw position index. Position correlates with accumulated attention, so a
  positional rotation inherits that correlation.

Both are defects in this implementation of the signals, not in the spec. The corrected
definitions give P5 membership-only scores (1.0 / 0.0, no fallback) and rotate P4 over a fixed
seed-stable permutation instead of position. Under those, **every signal passes**: P4 −0.0218
and P5 −0.0781, with P1 still exactly +1.0000.

### Consequence for Phase 2, stated plainly

**The originally-completed Phase 2 arms are not a valid test of orthogonal promotion.** Two of
the five signals were substantially correlated with the eviction signal, which is precisely the
condition §7's manipulation check exists to rule out. Reporting those contrasts as though the
manipulation had held would be wrong.

Phase 2 has therefore been **re-run in full with the corrected signals**, and the corrected run
is reported as primary. The original run is retained and reported alongside, because the
difference between them is the measurable cost of the contamination — and because a
manipulation check that fires, is diagnosed, and is acted on is a stronger result than one that
quietly passes.

> Note on scope: the correction changes only the `random`, `roundrobin` and `oracle` promotion
> branches. The `attention` path — which is what Phase 1's tiered arms use — is untouched, and
> that was verified bitwise on 8 cells spanning all arms and budgets. No Phase 1 result is
> affected.



---

## Threats to validity

Ordered by how much each should change a reader's confidence.

**1. The primary metric is compressed by an absorbing failure mode.** All eviction arms collapse
to zero after turn 1, while arms 5 and 6 stay flat across all six turns. Once a credential is
evicted it cannot return, so turns 2–5 carry no information for arms 1–4 and k/6 behaves as an
effective 2-turn metric there, capped near 0.33. Absolute effect sizes are therefore not
comparable to a design where all six turns are live. The turn-index table is reported alongside
k/6 for this reason; k/6 remains primary because §8 fixes it.

**2. Two of four factorial cells are floored at two of three budgets.** Arms 1 and 3 score 0.0000
with zero variance at budgets 154 and 257. A factorial with dead cells is not really a factorial:
the interaction at those budgets is driven almost entirely by the protected arms. Only budget 514
(arm 1 = 0.0967, arm 3 = 0.0400) gives a clean four-cell estimate — and it is also the budget
where the bit-width contrast is largest, so conclusions rest disproportionately on it.

**3. The 8-bit condition is post-hoc.** The mechanism was predicted before the run and confirmed,
but the condition was added after observing the 4-bit disagreement. Exploratory, not confirmatory.

**4. Depth and dormancy are perfectly confounded by construction.** The [SPEC-GAP 1] turn order
queries credentials in reverse depth, so turn index, dormancy gap and context depth are one
variable. No analysis here separates a dormancy effect from a depth effect. Deliberate, to
maximise dormancy pressure; it costs identifiability.

**5. Calibration gates G1–G4 are this implementation's invention.** The spec defines only G5, so
gate-level agreement with the reference is meaningless; only downstream numbers can be compared.

**6. G2 fails at both long-context conditions** (0.6389 at 2048, 0.6528 at 4096, against a 0.70
bar set before any long-context data was seen). Those cells are reported and marked GATE FAILED;
the threshold was not moved. The 2048/4096 factorial is indicative only.

**7. The ceiling divergence is probably a scoring convention, but that is an inference.** Strict
scoring gives 0.799; prefix-tolerant edit-distance-1 scoring gives 0.9544 against the reference's
0.947. Strong circumstantial evidence, unverifiable without reading the reference code. If the
reference genuinely reached 0.947 under strict scoring, this task is materially harder and every
absolute level here is shifted.

**8. The originally-run Phase 2 failed its manipulation check.** P4 (ρ = +0.4974) and P5
(ρ = +0.8489) were substantially correlated with the eviction signal at n=100. Those arms do not
test orthogonal promotion. Phase 2 was re-run with corrected signals and the corrected run is
primary; the original is retained only to quantify the contamination.

**9. Copy-error noise costs power.** Roughly 20% of trials fail on hex copy fidelity in every arm.
Unbiased in a paired design, but it inflates variance, so intervals here are wider than a cleaner
task would yield.

**10. Single model, single task, single architecture.** Qwen2.5-1.5B-Instruct, one retrieval task,
one GPU (gfx1100/ROCm). Nothing establishes that the bit-width conditionality generalises to
larger models, other tasks, or CUDA numerics. The exact-match scoring is likely to make this task
unusually sensitive to quantization; a semantically-scored task would plausibly show a far weaker
bit-width effect.

**11. The blind protocol cannot catch shared errors.** As the spec states, an error present in
both the specification and an implementation of it will not be detected by this method.
Everything here is conditional on the specification being right about the mechanism.

**12. Structural protection's magnitude is task-parameterised.** The distractor ablation shows the
benefit depends on the ratio of target lines to pattern-matching non-target lines (6:26 here).
The direction should transfer; the magnitude should not be quoted out of context.

**13. Infrastructure note.** Running four concurrent ROCm processes on this GPU triggered a
`hipErrorLaunchFailure` that killed all of them mid-run. No data was lost or corrupted —
checkpointing is append-only with per-row fsync, and all 6,808 rows written at that point parsed
cleanly — and the affected jobs resumed from their last committed row. Concurrency was
subsequently capped at two. This affected wall-clock time only, not results.


---

## iso-memory condition (§6) and the [SPEC-GAP 3] byte-cost sensitivity

Under iso-memory the tiered arms receive more raw positions, funded by the cold tier's
lower byte cost, so total bytes match the permanent arms:
`T = budget / (f + (1-f)·cost)`. With `f = 0.5` and `cost = 0.25` this is `T = 1.6 ×
budget`, reproducing §6's worked example exactly (257 → 411). Permanent arms are unaffected
by the condition, so their iso-token rows are reused rather than recomputed.

| budget | T | arm3 iso-token | arm3 iso-memory | arm4 iso-token | arm4 iso-memory | arm4 diff | 95% CI |
|---|---|---|---|---|---|---|---|
| 154 | 246 | 0.0000 | 0.0000 | 0.0200 | 0.1000 | **+0.0800** | [+0.0633, +0.0967] |
| 257 | 411 | 0.0000 | 0.0044 | 0.1189 | 0.1822 | **+0.0633** | [+0.0411, +0.0867] |
| 514 | 822 | 0.0400 | 0.0367 | 0.2400 | 0.2989 | **+0.0589** | [+0.0233, +0.0944] |

### Interaction under each iso-condition

§8 requires BH correction within each iso-condition separately, which presupposes both
conditions exist. Both are now reported.

| iso-condition | budget | interaction | 95% CI | n differ |
|---|---|---|---|---|
| iso_token | 154 | -0.0067 | [-0.0122, -0.0022] | 6 |
| iso_token | 257 | -0.0456 | [-0.0611, -0.0311] | 47 |
| iso_token | 514 | -0.1156 | [-0.1444, -0.0856] | 112 |
| iso_memory | 154 | +0.0733 | [+0.0556, +0.0900] | 77 |
| iso_memory | 257 | +0.0133 | [-0.0111, +0.0378] | 75 |
| iso_memory | 514 | -0.0533 | [-0.0978, -0.0078] | 122 |

### [SPEC-GAP 3] sensitivity to `quant_byte_cost`

§3 registers 0.25 but notes it ignores per-group scales and zero-points, and that a real
int8 scheme with bf16 scales is nearer 0.28–0.31. Changing the constant changes only the
iso-memory token grant, so this is the condition where it can bite.

| cost | T @ budget 257 | arm3 | arm4 |
|---|---|---|---|
| 0.25 | 411 | 0.0044 | 0.1822 |
| 0.28 | 402 | 0.0033 | 0.1922 |
| 0.31 | 392 | 0.0033 | 0.1811 |

Iso-token results are unaffected by this constant by construction, and Phase 1 and
Phase 2 both run iso-token — so the headline results carry no exposure to it. Every output
row records `quant_byte_cost` and a derived `physical_byte_cost = bits/16` ([GAP-N]) so the
grant is re-derivable.



---

## Phase 2 with corrected promotion signals (primary)

The originally-run Phase 2 failed its §7 manipulation check on two of five signals
(P4 ρ = +0.4974, P5 ρ = +0.8489 at n=100). Those arms do not test orthogonal promotion, so
Phase 2 was re-run in full with corrected signals. Both runs are shown; the corrected run
is primary.

| id | signal | original | corrected | Δ | reference | n |
|---|---|---|---|---|---|---|
| P1 | attention | 0.1189 | **0.1189** | +0.0000 | 0.203 | 150 |
| P2 | epiphany | 0.1411 | **0.1411** | +0.0000 | 0.216 | 150 |
| P3 | random | 0.1133 | **0.1133** | +0.0000 | 0.154 | 150 |
| P4 | roundrobin | 0.1167 | **0.1122** | -0.0044 | 0.184 | 150 |
| P5 | oracle | 0.1378 | **0.1344** | -0.0033 | 0.188 | 150 |

### Corrected-signal contrasts vs P3 (random control), BH-corrected

| contrast | diff | 95% CI | n differ | p_BH |
|---|---|---|---|---|
| P1 vs P3 | +0.0056 | [-0.0067, +0.0178] | 29 | 0.6169 |
| P2 vs P3 | +0.0278 | [+0.0156, +0.0411] | 34 | 0.0008 |
| P4 vs P3 | -0.0011 | [-0.0089, +0.0067] | 10 | 1.0000 |
| P5 vs P3 | +0.0211 | [+0.0089, +0.0333] | 33 | 0.0030 |

Retention is held fixed and attention-ranked in every Phase 2 arm, with protection ON
and tiered eviction ON, so these contrasts isolate the promotion decision from retention.



---

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



---

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


