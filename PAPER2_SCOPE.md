# Paper 2 — Scope, Inherited Context, and Proposed Plan

**Status:** proposal. Nothing in Part III or IV is pre-registered yet; this document exists to be
argued with, then hardened into `PREREG_P2.md`.
**Date:** 2026-09-05
**Predecessor:** Paper 1 (Protection × Recoverability), governed by `PREREG.md`,
`ADDENDUM_2026-09-03.md`, `ANALYSIS.md`, and the independent reimplementation in `amd/`.
Paper 1 is **closed**; no further runs are planned against it.

This document is self-contained on purpose. Part I carries every fact from Paper 1 that Paper 2
depends on, so Paper 2 can be reasoned about without re-reading four other documents.

---
---

# PART I — INHERITED CONTEXT (Paper 1, closed)

## I.1 What Paper 1 was

A mechanism-decomposition study of KV-cache eviction, asking whether a **recoverable quantized
tier** (keeping evicted tokens in a degraded form so they can be promoted back) adds anything
beyond a **good retention decision** (choosing well what to keep in the first place).

Two hypotheses:

- **H-SUB** — recoverability is *subordinate* to retention: the protection × recoverability
  interaction is null.
- **H-ORTH** — the promotion signal must be *orthogonal* to the eviction signal to help
  (i.e. the standard practice of ranking promotion by the same attention score used for eviction
  is circular and self-defeating).

Both were tested. **H-SUB confirmed** at deployable precision under matched-token accounting.
**H-ORTH falsified** — no promotion signal beats any other, including a perfect-knowledge oracle.

## I.2 The instrument

**Task** — `kvcache_harness/tasks/multi_credential.py`. Synthetic multi-credential retrieval with
induced dormancy: a dump of labelled credentials plus distractors and filler, then N sequential
questions, each asking for one credential's value. Dormancy means the answer's tokens have not
been attended to for a long stretch before being asked for, so a retention policy that only tracks
recent attention will have discarded them.

**Calibrated configuration** (all five gates passed at n=10):

| parameter | value |
|---|---|
| model | Qwen/Qwen2.5-1.5B-Instruct, bfloat16, **eager** attention |
| n_credentials (N) | 6 |
| credential value | `sk-` + 14 hex chars (17 chars total) |
| n_distractors | 20 |
| words_per_paragraph (filler) | 50 |
| full_fraction | 0.5 |
| recency_window | 64 |
| keep_sink | true (1 token) |
| max_answer_tokens | 22 |
| protect_after_chars | 2 |
| rebalance_every | 1 |
| decoding | greedy, no sampling |
| mean context (dump + first question) | **1029 tokens** (n=10, range 1011–1042) |

**Calibration gates** (`results/phase0r_calibration/`):

| gate | criterion | value | result |
|---|---|---|---|
| G1 ceiling (`oracle_static`) | ≥0.95 | 0.983 | PASS |
| G2 floor (no protection, permanent) | ≤0.10 | 0.033 | PASS |
| G3 headroom (protection alone) | 0.45–0.75 | 0.567 | PASS |
| G4 no-eviction reference | ≥0.95 | 0.983 | PASS |
| G5 dormancy check | ≥0.80 | 1.000 | PASS |

N was reduced from 8 to 6 during calibration: at N=8, G1/G4 landed at 0.90/0.925 while tracking
each other almost exactly — the signature of compounding per-turn copy error over 8 sequential
turns (a base-model capability ceiling), not a cache artifact.

**Scorer** — strict exact substring on raw generated text, no normalization of any kind:

```python
def score_turn(generated_text: str, credential: Credential) -> bool:
    return credential.value in generated_text
```

One definition, all 11 call sites. Verified by differential testing (harness scorer, a
from-scratch strict matcher, and a lowercase+strip variant agree on 180/180 answers).

**Arms**, six of them:

| # | arm | protection | eviction |
|---|---|---|---|
| 1 | `1_no_protect_permanent` | off | permanent |
| 2 | `2_protect_permanent` | structural | permanent |
| 3 | `3_recoverable_alone` | off | recoverable tiered |
| 4 | `4_recoverable_protected` | structural | recoverable tiered |
| 5 | `oracle_static` | ground-truth keep-set | — (ceiling) |
| 6 | `full_cache_ref` | — | none (no eviction at all) |

**Budgets** as fractions of the 1029-token mean context: 0.05/0.10/0.15/0.25/0.50 →
**51 / 103 / 154 / 257 / 514** tokens.

**Two accountings**, run as separate conditions:
- **iso-token** — same raw retained-token count across arms.
- **iso-memory** — same *bytes*, FULL tokens costing 1× and QUANT tokens `quant_byte_cost` (0.25
  under the 8-bit scheme). Tiered arms therefore receive ~1.6× more tokens at the same nominal
  budget.

## I.3 The budget-validity defect (ADDENDUM §§1–3) — carries directly into Paper 2

The retention floors (sink + recency window) are applied **inside** the budget, not on top of it.
So the budget actually available for *ranked competition* is:

```
competitive_budget = total_budget − recency_window − sink = B − 64 − 1
```

| B | competitive budget | classification |
|---|---|---|
| 51 | **−14** | **VOID** — budget-honoring violation; the policy cannot honor B at all |
| 103 | 38 | **VOID** — degenerate; too little for ranking to express anything |
| 154 | 89 | **PARTIAL** — reported separately, never pooled |
| 257 | 192 | **VALID** |
| 514 | 449 | **VALID** |

**This is design-time computable arithmetic, not a discovered result.** Any budget axis in Paper 2
must be checked against it before a single GPU hour is spent. It is why Paper 1's budget gradient
rests on two clean points plus one qualified one instead of the intended five.

## I.4 Paper 1 results

### I.4.1 Phase 1 — arm means, iso-token, 8-bit, n=150, seeds 3000–3149

| budget | no prot | protection | recoverable alone | prot+recovery | `oracle_static` | `full_cache` |
|---|---|---|---|---|---|---|
| 51 *(void)* | 0.013 | 0.013 | 0.013 | 0.013 | 0.013 | 0.947 |
| 103 *(void)* | 0.012 | 0.102 | 0.012 | 0.101 | 0.271 | 0.947 |
| 154 *(partial)* | 0.011 | 0.196 | 0.012 | 0.197 | 0.648 | 0.947 |
| **257** | 0.010 | 0.423 | 0.009 | 0.426 | **0.943** | 0.947 |
| **514** | 0.010 | **0.964** | 0.010 | 0.962 | 0.958 | 0.947 |

**Read this table carefully — it is the single most important input to Paper 2's design.** Three
properties of it constrain everything downstream:

1. The floor arm is **absorbing**: 0.010–0.013 at *every* budget, including 514. It is not a
   graded baseline; it is a trapdoor.
2. `oracle_static` is **not budget-invariant**: 0.013 → 0.271 → 0.648 → 0.943 → 0.958. The
   "ceiling" moves by 0.945 across the budget axis.
3. At budget 514 the ordering **inverts**: protection alone (0.964) > `oracle_static` (0.958) >
   `full_cache` (0.947). The ceiling is below a method, and below the no-eviction reference.

### I.4.2 The interaction — the headline dissociation

| budget | iso-token | 95% CI | iso-memory | 95% CI |
|---|---|---|---|---|
| 154 | +0.001 | [−0.002, +0.004] | +0.199 | [+0.179, +0.219] |
| 257 | +0.002 | [−0.003, +0.009] | +0.532 | [+0.506, +0.559] |
| 514 | −0.002 | [−0.010, +0.004] | −0.051 | [−0.068, −0.034] |

**H-SUB confirmed under iso-token.** The apparent iso-memory benefit tracks having ~60% more raw
retained tokens, not a reversibility mechanism. The dissociation — null under one accounting,
large under the other — is a claim about how this literature evaluates, not about any one system.

`quant_byte_cost` was fixed after `PREREG.md` was hashed, so it was swept rather than asserted:

| `quant_byte_cost` | multiplier | true tokens | interaction @257 | 95% CI |
|---|---|---|---|---|
| 0.125 | 1.78× | 457 | +0.543 | [+0.518, +0.570] |
| **0.25 (hashed)** | 1.60× | 411 | +0.532 | [+0.506, +0.559] |
| 0.30 | 1.48× | 395 | +0.536 | [+0.510, +0.561] |
| 0.50 | 1.33× | 343 | +0.354 | [+0.331, +0.379] |

The qualitative claim survives the whole plausible range; the magnitude must be reported as a
range (+0.35 to +0.54) with the constant stated, never as a point estimate.

### I.4.3 Phase 1 at 4-bit — the tier becomes a sink

Phase 1 originally ran at 8-bit only; the spec never stated a bit-width (our defect, raised by the
independent reimplementation). At 8-bit the quantized tier is a **no-op**, so recoverability was
being compared in a regime where it could not act in either direction.

Re-run at `quant_bits=4`, iso-token, n=150 (`results/phase1_4bit/`):

| budget | interaction @ 4-bit (sum) | 95% CI | prompts differing | @ 8-bit |
|---|---|---|---|---|
| 154 | **−0.0122** | [−0.0222, −0.0033] | 19 / 150 | +0.0011 |
| 257 | **−0.0189** | [−0.0322, −0.0056] | 30 / 150 | +0.0022 |
| 514 | **−0.6833** | [−0.7200, −0.6456] | 149 / 150 | −0.0022 |

Arm means at 4-bit, budget 514: protection alone **0.964** → protection+tiered **0.281**.

Mechanism: `quant_slots` scales with the budget, so a *larger* budget sends *more* context to the
cold tier. At 514 roughly 225 tokens are quantized to 4 bits, and 4-bit all-QUANT collapses
outright, so corrupting that much context destroys generation.

**Q4 (mean-corrected, n=150 FINAL):** interaction at 514 = **−0.8011 [−0.8289, −0.7722]**, vs
−0.6833 under sum; CIs non-overlapping; 150/150 prompts differ. Protection alone unchanged at
ceiling (0.963); protection+tiered drops 0.281 → 0.162. *(An interim n=142 figure of −0.8016 is
superseded and must not be cited.)*

**The framing that survives from Q4, and it matters for Paper 2:**

> The better your retention policy, the more a lossy recoverable tier costs you — because the tier
> sits directly downstream of your best selection and receives exactly the tokens retention
> correctly identified as valuable.

This is also Paper 1's one **falsified prediction with a mechanism attached**: "a bias shared by
both arms cancels in the difference" held at 8-bit and broke at 4-bit, for a reason specific to
the regime.

**Q1 decomposition of the −0.683** (`results/q1_decompose_514/`): a **compound** effect, not
generation collapse. Conditional accuracy (correct | non-degenerate) falls 0.973 → 0.384 — roughly
44 points retrieval and 24 points generation. Baseline arm 2 is clean (1.3% malformed, no
empties). "68 points of retrieval quality" overstates it.

**Q2 dose-response** (`results/q2_dose_response/`): strictly monotonic — target quantized tokens
0/50/100/150/225 → accuracy 0.967/0.673/0.473/0.397/0.253, degeneracy 0.013 → 0.277. But
**realized quantized tokens are only ~31% of target** (225 → 70), because protection seats its
groups FULL first. Target and realized are coupled *through protection*, so monotonicity carries
the mechanism claim but the slope is **not** a clean dose coefficient.

### I.4.4 Phase 2 — H-ORTH falsified

Run at 4-bit, protection on, tiering on, eviction attention-ranked throughout; the promotion
signal is the sole varied factor. Budget 257, n=150. Usable band at 4-bit: all-QUANT 0.133 →
all-FULL 0.411.

| signal | ρ(eviction, promotion) | mean | vs P1 | 95% CI | prompts differing |
|---|---|---|---|---|---|
| P1 attention *(ref)* | **+1.000** | 0.203 | — | — | — |
| P2 epiphany | −0.077 | 0.216 | +0.012 | [−0.016, +0.039] | 85 / 150 |
| **P3 random** | −0.050 | 0.154 | **−0.049** | **[−0.077, −0.021]** | 91 / 150 |
| P4 rotation | +0.002 | 0.184 | −0.019 | [−0.047, +0.009] | 87 / 150 |
| P5 oracle | −0.068 | 0.188 | −0.016 | [−0.046, +0.013] | 95 / 149 |

The manipulation is exact (ρ = +1.000 circular, ≈0 otherwise) and the orthogonal candidate does
not beat the circular one. Only a deliberately uninformative ordering separates, and it separates
*downward* — so the promotion signal is **saturated**, not ignored. Rotation ties measured
ranking, so the blind-rotation-beats-ranking result from adjacent block-diffusion work does not
reproduce in this architecture.

P5's scope was checked (`debug_p5_oversubscription.py`): the future-need set is 111 tokens against
96 FULL slots, but only ~53 survive retention, and **0.000 of selection calls were
oversubscribed** — every retained credential token is promoted. P5 is a genuine ceiling, and its
landing marginally below P1 is noise.

**But P5 bounds the promotion decision only, not the policy.** It cannot rescue the 52.3% of
credential tokens evicted at the *retention* stage, because retention is held attention-ranked in
every arm by design. The composite claim:

> The promotion decision is information-saturated — a perfect signal adds nothing — while
> retention carries **~0.52 of unexploited headroom** at the same budget.

**That sentence is the seed of Paper 2.**

At int8 the same comparison is untestable rather than false: holding retention fixed and varying
only precision gives all-FULL 0.450, half/half 0.467, all-QUANT 0.467. A quantized credential
reads exactly as well as a full-precision one. Quantization is genuinely applied (~1.2% relative
error, values provably differ) — it is simply too mild to matter. The first Phase 2 run was
stopped at 39 prompts (retained in `results/phase2_8bit_control/` as the no-op control).

### I.4.5 The bit-width anomaly — closed as model-family-specific

Reconstruction error is monotone in bit-width; task accuracy is not.

| bits | rel. error | ours (sum) | ours (mean, Q3) | AMD 1.5B | Qwen2.5-3B | **Llama-3.2-3B** |
|---|---|---|---|---|---|---|
| 8 | 1.45% | 0.410 | 0.473 | 0.1433 | 0.2500 | 0.3100 |
| 7 | 2.67% | 0.293 | 0.303 | 0.1600 | 0.2500 | 0.3067 |
| 6 | 4.50% | **0.000** | 0.020 | 0.1300 | 0.2500 | 0.3133 |
| 5 | 8.62% | **0.000** | 0.000 | 0.0000 | 0.2467 | 0.3067 |
| 4 | 17.62% | 0.123 | 0.247 | 0.1167 | **0.0367** | 0.3200 |
| 3 | 37.31% | **0.000** | 0.000 | 0.0000 | 0.0967 | 0.2600 |

Six hypotheses were tested and ruled out (level counts wrong; correct levels wrong scale; policy
differs per width; per-prompt threshold effect; level-occupancy concentration; the sum-vs-mean
ranking artifact). The quantizer is cleared on all three models: level counts never exceed 2^bits,
error is strictly monotone, grouping is the documented whole-slice scheme.

**The Llama-3.2-3B result settles it.** No collapse at any width, zero fully-degenerate prompts,
zero empty/EOS outputs — *while carrying higher reconstruction error than either Qwen at every
width* (4-bit 24.28%/18.90% vs Qwen's 19.98%/15.20%), and with different KV geometry
(1024-element quantizer groups vs 256). So:

1. **Catastrophic non-monotone collapse is a model-family property, not a property of KV cache
   quantization.**
2. **Reconstruction error does not predict accuracy collapse.**
3. **A bit-width severity ordering derived from one model family does not transfer.**

**Paper 2 does not touch this.** It is Paper 1's closed secondary finding.

## I.5 The independent reimplementation (`amd/`)

A blind reimplementation of `SPEC_REIMPL_v1.md` on an RX 7900 XTX, written without opening,
importing, listing, or searching any file of `kvcache_harness/`. **7300 runs, 0 failures.**
Deliverables: `amd/SUMMARY.md`, `amd/FINDINGS.md`, `amd/AGREEMENT.md` (32 reference values
compared), `amd/SPEC_QUESTIONS.md` (26 ambiguities and the default chosen for each).

**What replicated:** quantizer errors at every width to within 0.03pp; context length (median
1030.5 vs 1029); the iso-memory token grant at 257 (411, exactly); `full_cache_ref`
budget-invariance; the 8-bit interaction at both budgets (−0.0011 vs +0.002; −0.0056 vs −0.002);
the 4-bit direction, significance and budget-growth.

**What did not:** the ceiling — their `full_cache_ref` is 0.7989 against our 0.947. Diagnosed
rather than assumed: case/whitespace normalization rescues *exactly zero* failures, but the model
often emits the bare 14-hex payload without the `sk-` prefix (58 of 181 failures) and one
character of edit distance recovers 82 more — together 0.9544 against our 0.947. **The evidence
favours a scoring-convention difference over a capability difference.** The scorer was not changed
on either side.

Also unreplicated: the magnitude at 514 (ours −0.683/−0.801, theirs −0.116) — order-of-magnitude
agreement at 257, ~6× divergence at 514, unexplained and deliberately not reconciled by argument.

### I.5.1 Two real defects the audit found in our code

1. **Biased value alphabet.** `string.hexdigits.lower()` is `'0123456789abcdefabcdef'` — 22 slots,
   because `ABCDEF` folds onto `abcdef`. Our credential values drew a–f twice as often as digits
   (9.09% vs 4.55%), giving 3.914 bits/char against a true-hex 4.000. Holding seeds fixed and
   swapping only the alphabet: as-shipped 0.9444, uniform hex 0.9111, digits-only 0.8833. Costs
   ~0.033; does **not** explain the ceiling divergence. (Their `HEX` is uniform.)

2. **Sum-vs-mean attention accumulation.** `engine._score_step` takes the mean over layers and
   heads, then **sums over queries** and accumulates across steps. In a ~1025-token prefill,
   position 0 is attended by ~1025 queries and position 1000 by ~25, so early positions outrank
   later ones on longevity alone.

   | quantity | value |
   |---|---|
   | Spearman(score, position) — sum, as shipped | **−0.717** |
   | Spearman(score, position) — mean-normalized | +0.587 |
   | **Spearman(sum, mean)** | **+0.014** |
   | top-192 retained-set overlap between readings | **0.135** |
   | credential percentile rank — sum / mean | 0.693 / 0.654 |

   **The two readings are effectively uncorrelated and share only 13.5% of the retained set.**
   Sum-vs-mean selects an almost entirely different cache — a first-order design choice, not a
   refinement. Measured at n=150, 8-bit: the interaction is **unchanged** (+0.0000 / −0.0011 /
   +0.0011, all CIs inside ±0.05) and the protection effect survives (+0.169 / +0.457 / +0.924),
   but *levels* move where there is headroom (protection 0.423 → 0.481 at 257; floor 0.010 →
   0.024). The defect cost ~6 points at mid budget and changed no 8-bit conclusion.

   **Their implementation sums too** (`amd/kvre/engine.py:126`: `.mean(dim=0).sum(dim=0)`
   accumulated across steps). So this is a shared convention, not a divergence — and any of their
   rate statistics known to be sum-inflated on our side carry the same risk on theirs.

**Neither defect was caught by five rounds of internal instrument validation, and neither would
have been caught by more of the same, because both are choices that are self-consistent within one
implementation.** That is the argument for blind reimplementation over additional internal
checking, and it is the most transferable result in Paper 1.

### I.5.2 The distractor ablation — the most important AMD result for Paper 2

Rewriting distractor values to a **non-credential surface form** lifts the structural-protection
arm from **0.150 to 0.907** — above the full-cache ceiling of 0.880 measured under the same
condition — while the full-cache arm gains only +0.137.

> Pattern-based KV protection's strength is set by the ratio of target lines to pattern-matching
> non-target lines — a property of the mechanism, not of this task.

**Structural protection degenerates into an oracle when distractors stop matching its pattern.**
Paper 2 must treat this as a live threat, not a footnote (§II.3).

### I.5.3 Promotion-signal manipulation checks — the practice to reuse

The §7 manipulation check **failed on two of five signals at n=100**: P4 ρ = +0.4974 and
P5 ρ = +0.8489 against a ~0 target. Both fell back to the eviction score for non-must positions,
correlating them with it *by construction*. Signals were corrected (P5 membership-only, P4
rotating over a fixed permutation), giving P4 −0.0218 and P5 −0.0781, and Phase 2 was re-run in
full. The corrected run is primary; the original is retained to quantify the contamination. P1 is
exactly +1.0000 with a zero-width CI, as it must be.

**This is the single most reusable methodological device in the project** and Paper 2's
credibility depends on an analogue of it (§III.6).

## I.6 Defects that changed a reported number

| ID | Defect | Caught by |
|---|---|---|
| D-01 | fp16 degeneracy — Qwen2.5 emits repeated punctuation in fp16 on this stack, including under a plain `generate()` | Reference run with no custom code |
| D-02 | Structural protection covered a fixed 12 characters after each label, shorter than the 27-char credential | Inspecting generated output |
| D-03 | No recency floor — attention ranking evicted the tokens immediately preceding the answer, breaking every arm including the oracle | Oracle below its own ceiling |
| D-04 | `oracle_static` sliced an unordered `set` when trimming to budget, dropping an arbitrary non-reproducible subset; also lacked credential labels | Ceiling shortfall |
| D-05 | Interim analysis pooled `iso_token` and `iso_memory` rows for tiered arms under one nominal-budget label | A query that would not reproduce |
| D-06 | The factor under test was inert — twice (protection pinning credentials; the 8-bit tier carrying no signal) | G3 gate, then a smoke test |

**Transferable practice:** assert an invariant on live data, not on a reimplementation. The
level-count check on 10,640 intercepted calls is the strongest single piece of instrument
validation in the project, and the same form would have caught D-04 and D-05.

Separately: **three of six defects surfaced from a reference arm behaving impossibly** — an oracle
below its ceiling, a policy beating a no-eviction baseline, five arms returning byte-identical
results — rather than from any test suite. **Reference arms earn their compute as tripwires, not
just as denominators.** Paper 2 is a paper *made of* reference arms, so this is not optional
there; it is the design.

Two ledger notes: Q2's realized-vs-target logging is the first confound in this project caught by
instrumentation designed *before* the run rather than by a reference arm behaving impossibly.
Q4's arm-2-rerun decision (arm 2 is bit-invariant but **not** score-mode invariant) is a near-miss
of the D-05 class, caught in advance.

## I.7 Deviations from `PREREG.md`, all recorded

- **Median → mean estimator.** The pre-registered bootstrapped median of paired per-prompt
  differences is **degenerate on this data**: paired differences are overwhelmingly exactly zero,
  so the median is zero in essentially every resample and the interval collapses to zero width
  (`[0.000, 0.000]`, `[0.167, 0.167]`, `[0.500, 0.500]` — the non-zero ones being exactly 1/6 and
  3/6, the granularity of a six-credential score). The mean paired difference with its bootstrap
  interval is what the data supports, reported alongside **the count of prompts on which the arms
  differ at all**.
- Budgets 51 and 103 excluded by the arithmetic criterion; 154 reported separately, not pooled.
- The budget gradient rests on two clean points plus one qualified one, not the intended five.
- `quant_byte_cost` fixed post-hash, therefore swept.

**Two kinds of null.** Phase 1's null is "the same behaviour" (arms differ on 3–9 of 150 prompts).
Phase 2's null is "different behaviour, same outcome" (arms differ on 55–95 of 150 and still
average to the same place). They should never be described in the same words.

## I.8 Paper 1's scope limits, inherited wholesale unless changed

One quantizer (per-position affine min–max over the `(kv_heads, head_dim)` slice), one primary
model, one synthetic task, one GPU per implementation. The structural claims survive; boundary
locations are implementation-dependent and must be stated as such.

**Explicitly not run, and named in `ANALYSIS.md` §7 as the single most valuable unrun
experiment:**

> **Phase 3's random-span control — is "protection" really just reserved capacity?**

It tests the mechanism that *did* carry the task. **Paper 2 cannot skip it** (§III.5).

---
---

# PART II — CRITIQUE OF THE PROPOSAL AS WRITTEN

The core instinct is right and I would keep it: **a ceiling-measurement paper, not a method
paper.** "How much of the available headroom do existing methods capture?" is a better question
than another leaderboard, it follows directly from Paper 1's composite claim (§I.4.4), and it is
preregisterable in a way that "we built a better evictor" is not. The six prohibitions are also
right and I would keep all of them.

But the proposal has **four defects that would break the paper as specified**, and one large
missed opportunity. In order of severity.

## II.1 CRITICAL — the normalized metric breaks at both ends of Paper 1's own budget axis

The proposed metric is

```
G_m = (A_m − A_random) / (A_oracle − A_random)
```

Substituting Paper 1's actual numbers (§I.4.1) rather than the hypothetical ones:

**At budget 257:** floor 0.010, oracle 0.943, denominator 0.933. So
`G_m ≈ (A_m − 0.010)/0.933 ≈ A_m × 1.07`. **The normalization does essentially nothing.** It
rescales accuracy by a constant and subtracts a rounding error. Every claim of the form "attention
closes 42% of the gap" is, on this data, "attention scored 0.42" with extra steps.

**At budget 514:** floor 0.010, oracle 0.958, protection alone **0.964**. So
`G_protection = 0.954/0.948 = 1.006`. **A method exceeds the ceiling and G_m > 1.** Not by noise —
`full_cache_ref` (no eviction at all) is 0.947, *below both*. The oracle is partly denoising: by
discarding distractors it makes the task easier than having the whole context.

**At budgets 51 and 103:** oracle is 0.013 and 0.271 while floor is ~0.012. At 51 the denominator
is **0.001** and G_m is pure noise amplification; at 103 it is 0.259 and every method is pinned
near zero anyway.

So on the budget axis the proposal actually names — 5/10/15/25% of a 1029-token context =
**51/103/154/257** (§I.3) — **three of the four points are VOID or PARTIAL under Paper 1's own
pre-registered criterion**, and the one valid point is where the metric degenerates into rescaled
accuracy.

**This is not a tuning problem. The metric and the budget axis are both wrong, and both are
detectable with arithmetic before any GPU time.**

**Fixes required:**

- **Do not use random as the denominator floor.** Paper 1's floor arm is *absorbing*, not graded —
  0.010–0.013 at every budget including 514. A trapdoor is not a baseline; a metric anchored to it
  measures nothing. Use a **graded, budget-responsive floor**: position-only retention (keep the
  most recent B tokens, no ranking) is the honest "no importance signal, but not sabotage"
  reference, and unlike random it moves with the budget. Report random separately as a *sanity
  tripwire* (any method below random is broken), not as the anchor.
- **Fix the ceiling's identity.** `oracle_static` is not "the attainable ceiling"; it is *the best
  static keep-set built from ground truth, trimmed to B*. It is non-monotone against
  `full_cache_ref` and can exceed it. Paper 2 needs the bracket in §III.2, not a single number
  called "oracle."
- **Report raw accuracies alongside every normalized number, always**, and pre-register that
  `G_m` is **not reported at all** for any cell whose denominator is below a stated threshold. A
  ratio with a denominator of 0.001 is not a measurement.
- **Re-derive the budget axis from the competitive budget** (§I.3), not from percent-of-context.
  Either lengthen the context (the AMD folder already carries 2048/4096 calibration) so 5–25%
  lands above the floors, or move the fractions up. **Settle this before preregistration.** My
  recommendation is to lengthen context — it also improves external validity, since 1029 tokens is
  short enough that reviewers will ask.

## II.2 CRITICAL — the gap is not all closable, and the headline depends on saying so

This is the missed opportunity, and I think it is the difference between a decent paper and a
genuinely good one.

The proposal's oracle **knows which credential will be asked for in the future.** No deployable
method can know that. So the measured gap decomposes into two parts the proposal currently merges:

```
        A_prescient  ──────────────────────  oracle that knows future queries
            ▲
            │  (2) INFORMATION GAP — closable by NO causal method, ever
            ▼
   A_causal_ceiling  ──────────────────────  best achievable knowing only the past
            ▲
            │  (1) METHOD GAP — the actual research problem
            ▼
              A_m    ──────────────────────  the method under test
            ▲
            │
            ▼
          A_floor    ──────────────────────  graded position-only reference
```

Reporting "method X leaves 40% of the headroom on the table" is **actively misleading** if 35 of
those 40 points are the future-query information gap. It would motivate a Paper 3 aimed at a
target that does not exist.

Paper 1 already contains the tooling. P5 was exactly a prescient signal, and the oversubscription
check (`debug_p5_oversubscription.py`) is exactly the instrument that proves an oracle is a genuine
ceiling rather than "oracle on whatever subset fits."

**So the central metric should not be one ratio. It should be a two-term decomposition:**

```
Headroom        H = A_prescient − A_floor              (total, as the proposal has it)
Method gap        = A_causal_ceiling − A_m             ← the research problem
Information gap   = A_prescient − A_causal_ceiling     ← irreducible for causal methods
```

and the headline number is the method's share of the **causally closable** headroom:

```
              A_m − A_floor
G_m =  ──────────────────────────────
       A_causal_ceiling − A_floor
```

**Constructing `A_causal_ceiling` is the hard part and is the paper's real methodological
contribution.** Concretely: an oracle that may use ground-truth token identity but is denied
knowledge of *which* credential is queried next — it must retain all N credentials' tokens, or
allocate under the query's marginal distribution, rather than the realized query. That is a
strictly weaker oracle, it is implementable in the existing `oracle_static` code path, and the
difference between it and the prescient oracle is **a measurement of how much of KV eviction is
inherently a prediction problem.**

That measurement is more interesting than the leaderboard, and it is what tells you whether Paper 3
should exist and what it should attack.

## II.3 SERIOUS — the task has a structural shortcut, and it rigs the comparison

`multi_credential` was built so that *structural protection* — a pattern match on `sk-` lines —
carries the task. Paper 1's numbers show it working: protection alone spans 0.013 → 0.964 across
the budget axis with the model fixed.

The AMD distractor ablation (§I.5.2) shows how contingent that is: rewriting distractors to a
non-credential surface form lifts the structural arm **0.150 → 0.907**, above the full-cache
ceiling under the same condition. The mechanism's strength is set by the target-to-matching-
distractor ratio.

Now consider what Paper 2 proposes: put ForesightKV, LU-KV and Expected Attention into this
harness and measure how close they get to an oracle. **On this task, "the oracle" is very nearly
"a two-line regex."** A learned long-horizon importance signal is being asked to rediscover a
surface pattern the task's construction made trivially available. Whatever comes out, the honest
description is *"learned importance signals underperform a regex on a task built around a
regex-detectable answer"* — and a reviewer will say so in the first paragraph.

**Fixes required:**

- **Add at least one task with no structural shortcut**, where the answer's tokens are not
  identifiable by surface form. This is the largest new engineering item in Paper 2 and should be
  scoped as such, not as an afterthought. Candidates: multi-hop retrieval over paraphrased facts;
  a needle whose surface form matches the distractors exactly and differs only semantically; an
  aggregation question whose answer is not present verbatim anywhere.
- **Run the distractor-ratio ablation as a registered axis**, not a robustness check. It is the
  knob controlling how much of the task is a pattern-match, and Paper 2's conclusions are a
  function of where that knob sits.
- **Structural protection must be reported as one of the methods under test**, not as harness
  furniture. On this task it is a *very strong method with privileged information about the task's
  surface form* — a fascinating table entry and a dishonest baseline.

## II.4 SERIOUS — four reimplementations means four reimplementations' worth of risk

Paper 1's single most transferable finding is that **blind reimplementation found two real defects
that five rounds of internal validation missed**, because both were self-consistent within one
implementation. Paper 2 proposes to write *four* new implementations of other people's methods,
in-house, and publish a result of the form "these methods leave headroom."

Every negative result will be attacked as "you implemented it wrong," and on Paper 1's own
evidence that attack will sometimes be **correct**. Recall that the promotion-signal manipulation
check failed on 2 of 5 signals (§I.5.3) — and those were signals *we designed ourselves* with a
clearly specified target.

**Fixes required — this is the admission gate, and it should be preregistered as hard as G1–G5:**

- **Every method must pass a reproduction gate before admission.** Each must reproduce a number
  from *its own paper*, on *its own reported setting*, within a stated tolerance. A method that
  cannot is reported as **"not admitted — could not reproduce published behaviour"**, which is
  itself a finding, rather than being quietly included at whatever number it produces.
- **Prefer the authors' released code over reimplementation** wherever it exists — wrapped, not
  rewritten, with the wrapper's fidelity asserted on live data.
- **Every method gets a manipulation check** in the P1–P5 style: an assertion on live tensors that
  the signal does what its name claims (e.g. that a future-utility signal correlates with future
  utility and not with current attention). §I.5.3 is the template.
- **Pre-register the free parameters chosen on each method's behalf, and name them.** Sum-vs-mean
  accumulation (§I.5.1) is canonical: a first-order choice (ρ(sum,mean) = +0.014, 13.5%
  retained-set overlap) that every attention-derived method has. Silently fixing it differently
  per method would contaminate the entire comparison. **Pin one convention across all methods and
  sweep it as a robustness axis on the attention baseline.**
- **Budget accounting must be normalized across methods and asserted, not assumed.** Head-wise
  methods, methods with their own recency floors, and methods that reserve capacity all count
  tokens differently. D-05 (pooling two accountings under one label) is exactly this class of
  defect and the easiest one to repeat here.

## II.5 MODERATE — the model axis is not free, contrary to the proposal

"You already have these from Paper 1" is not true in the way that matters. From §I.4.5 and the AMD
calibration work:

- **Llama-3.2-3B's `full_cache_ref` is 1.0000.** Saturated. There is no headroom to measure.
- **Qwen2.5-3B's is 0.9833**, and across **8 configurations it never fell below ~0.94** — N of
  6/9/12/16 gave 0.9861/1.0000/1.0000/0.9896; contexts of 2635/5086/8061/8155 gave
  0.9861/0.8194/1.0000/0.9583, and the one promising point (0.8194) was a 12-seed fluke that
  re-tested at 0.9833.
- The 1.5B's bottleneck was hex **copy fidelity** (57–64% of failures were near-misses). The 3B
  has largely solved that, **so adding credentials only adds more of a task it can already do**,
  and lengthening context does not degrade it monotonically.

**The competence gate cannot be matched by tuning N or context** — a documented AMD finding, not a
guess. A ceiling paper on a saturated ceiling measures nothing.

Tokenizer differences also bite: Llama-3.2 tokenizes the identical task text to **916 tokens**
against Qwen's 1030, so reusing absolute budgets 154/257/514 silently changes the retention
*fraction*, the protected-token count, and what fraction of a turn the 64-token recency floor
covers.

**Fixes required:**

- Treat the model axis as **work, budgeted up front**: per-model recalibration to a common
  competence gate, budgets re-derived per tokenizer as fractions of *that model's own* context
  length, and the competitive-budget check (§I.3) re-run per model.
- If the gate cannot be matched for a model, **say so and report that model as a covariate rather
  than an axis** — which is what the AMD work already concluded and is defensible.
- **Do not make Paper 2 depend on a larger model.** The proposal already says this; I agree
  emphatically. Appendix it if a GPU slot appears.

## II.6 MINOR — tighten, but do not restructure

- **"Outcome A is bad news" is wrong.** If existing methods nearly reach the *causal* ceiling,
  that is a **strong** result given Paper 1's Phase 2 measured ~0.52 of unexploited retention
  headroom. It would mean the headroom is real but not causally accessible — exactly the §II.2
  decomposition earning its keep. **Pre-register all three outcomes as publishable**, with the
  write-up framing fixed in advance for each. That is what makes this a measurement paper rather
  than a fishing expedition, and it is the discipline `PREREG.md` §4's kill gates imposed on
  Paper 1.
- **The estimator must be fixed in advance and must not be the median** (§I.7 — degenerate on this
  data structure). Use the **mean paired difference with a prompt-clustered bootstrap, reported
  alongside the count of prompts on which arms differ at all** — the latter is estimator-
  independent and was the stronger statement in Paper 1. Carry forward the two-kinds-of-null
  distinction.
- **No quantized tier anywhere in Paper 2.** "Don't study recoverable precision again" is right
  but not operational. Stronger: run **permanent eviction only, no cold tier at all**, so precision
  cannot confound any cell. Paper 1 established the tier is a no-op at 8-bit and a sink at 4-bit;
  either way it only adds variance here.
- **Head allocation deserves promotion from the extension to the main design.** The current engine
  takes the mean over heads *before* ranking (§I.5.1), so **head-wise allocation is entirely absent
  from the apparatus** — every method is forced through a head-agnostic bottleneck, and some
  methods on the list (LU-KV especially) are specifically about allocation. A head-wise oracle
  variant is cheap against the existing `oracle_static` and is my leading prior for where the gap
  lives.
- **`oracle_static` is static** — one keep-set decided once. A per-step re-deciding oracle is a
  strictly stronger ceiling. Measuring the gap against the static one may **understate** it, which
  is the direction that makes G_m > 1 (§II.1) more likely.
- The "task-level oracle, not universal token importance" disclaimer the proposal drafts belongs in
  the **abstract**, not only in limitations.

---
---

# PART III — REVISED SCOPE

## III.1 The question, restated

> **Of the task-level retention headroom that a causal policy could in principle capture, how much
> do current KV-eviction methods actually capture — and where does the rest of it live?**

Two clauses matter. *"Causal"* excludes the prescient component that no deployable method can
reach (§II.2). *"Where does the rest of it live"* is the decomposition, promoted from optional
extension to co-headline, because it is what makes the measurement actionable rather than
merely a number.

## III.2 The reference ladder — five arms, not two

Every cell is bracketed by the same ladder, all at identical retained-token budget:

| # | arm | role | what it knows |
|---|---|---|---|
| R0 | `random_retention` | **tripwire only** | nothing (sanity: any method below this is broken) |
| R1 | `position_only` | **the floor / denominator** | position only — most recent B tokens, no ranking |
| — | *methods under test* | — | whatever the method defines |
| R2 | `oracle_causal` | **the ceiling that matters** | ground-truth token identity, **not** which query comes next |
| R3 | `oracle_prescient` | upper bracket (= Paper 1's `oracle_static`) | ground truth **and** the realized future query |
| R4 | `full_cache_ref` | no-eviction reference | everything; no eviction at all |

R1 replaces random as the denominator because Paper 1's floor arm is absorbing (§II.1). R2 is new
and is the paper's methodological contribution. R3 is Paper 1's existing oracle, retained so the
information gap `R3 − R2` is measurable. R4 is retained as a tripwire — **any arm exceeding R4 is
denoising, not retaining**, and that condition fires at budget 514 in Paper 1's own data.

**Registered invariants** (assert on live data, per §I.6):
- `R0 ≤ R1 ≤ R2 ≤ R3` at every cell. Any violation halts the cell and is reported, not smoothed.
- `R2 ≤ R4` — if the causal oracle beats the full cache, the arm is denoising and the cell is
  disqualified from `G_m`.
- Retained-token counts are equal across arms to the token, asserted per step, not per config.
- No arm is oversubscribed (the `debug_p5_oversubscription.py` check, generalized).

## III.3 The central metric

```
              A_m − A_R1
G_m =  ────────────────────────
              A_R2 − A_R1
```

reported **only** where `A_R2 − A_R1 ≥ 0.15` (threshold pre-registered; below it the ratio is noise
amplification). Raw accuracies are reported for every cell regardless.

Alongside it, always:

| quantity | meaning |
|---|---|
| `A_R3 − A_R2` | **information gap** — the share of headroom no causal method can reach |
| `A_R2 − A_m` | **method gap** — the research problem |
| prompts on which arms differ | estimator-independent; the stronger statement (§I.7) |

## III.4 Axes

**Methods** — four plus the ladder, per the proposal's own "don't make the table big" rule:

| method | why it is in |
|---|---|
| Attention ranking | current standard; also the sum/mean robustness carrier |
| Expected Attention | future-query prediction |
| ForesightKV | learned long-horizon contribution |
| LU-KV | future utility / global (head-wise) allocation |
| **Structural protection** | Paper 1's winner — a method with privileged surface information, reported as such (§II.3) |

LKV is explicitly deferred: admitted only if it clears the reproduction gate cheaply, dropped
without ceremony otherwise.

**Budgets** — re-derived from the competitive budget (§I.3), not from percent-of-context, and
re-derived **per model per tokenizer**. Recommendation: move to a longer context (2048 or 4096,
already calibrated in `amd/`) so that four retention fractions land above the floors with real
competitive room. Provisional, pending that calibration: fractions giving competitive budgets of
roughly 150 / 300 / 600 / 1200 tokens.

**Models** — Qwen2.5-1.5B primary (the only one with a non-saturated ceiling). Qwen2.5-3B and
Llama-3.2-3B admitted **only** if a competence gate can be matched by recalibration; otherwise
reported as covariates with the saturation stated (§II.5).

**Tasks** — `multi_credential` (inherited, calibrated) **plus one shortcut-free task** (§II.3).
The second task is a gate on the paper's main claim, not a robustness appendix.

**Fixed out of scope, asserted in code:** no quantized tier at all; permanent eviction only;
greedy decoding; bf16; eager attention; one accumulation convention pinned across all methods.

## III.5 Prerequisites that must run before the main matrix

Two Phase-0 items, both cheap, both blocking:

1. **The random-span control** (Paper 1's §7 unrun experiment). Is structural protection an
   importance signal, or just reserved capacity? Give a random span of equal token count the same
   protection treatment. If the random span performs comparably, then Paper 1's winning mechanism
   was capacity reservation and **every** `G_m` in Paper 2 is measured against a misdescribed
   reference. This must be answered first.
2. **The distractor-ratio calibration** (§I.5.2, §II.3). Establish where on the 0.150–0.907 curve
   the shipped task sits, and register the ratio as a fixed condition.

## III.6 Method-admission protocol (the gate that makes negatives defensible)

For each method, in order, before any comparison cell is run:

1. **Provenance** — authors' code wrapped in preference to reimplementation; the choice recorded
   per method.
2. **Reproduction gate** — reproduce one number from the method's own paper on the method's own
   reported setting, within a tolerance stated in advance. Fail ⇒ **not admitted**, and reported
   as such.
3. **Manipulation check** — assert on live tensors that the signal measures what its name claims
   and is not collapsing onto the eviction score by construction (§I.5.3 template; ρ against
   attention reported for every method).
4. **Budget-accounting assertion** — retained tokens counted identically, verified per step on
   live data.
5. **Free-parameter declaration** — every choice made on the method's behalf named in the
   preregistration, with the accumulation convention pinned globally.

## III.7 The decomposition (co-headline, conditional on a gap existing)

If `A_R2 − A_m` is materially non-zero for the admitted methods, decompose it by constructing
partial oracles that differ from R2 in exactly one faculty:

| partial oracle | isolates |
|---|---|
| R2 with head-agnostic allocation | **head allocation** (my leading prior — the current engine has no head-wise axis at all, §II.6) |
| R2 with a static keep-set | **temporal re-decision** (static vs per-step) |
| R2 with position-blind ranking | **positional effects** (the sum/mean artifact lives here, §I.5.1) |
| R2 restricted to a method's own objective | **objective misalignment** |

Each partial oracle is a *removal* from R2, so its cost is directly attributable. This is the
bridge into Paper 3 and the reason Paper 2 does not need to propose a signal.

## III.8 Pre-registered outcomes — all three publishable

| outcome | condition | what the paper says |
|---|---|---|
| **A — headroom is mostly not causally accessible** | `A_R3 − A_R2` large, `A_R2 − A_m` small | Paper 1's ~0.52 headroom is real but largely a prediction problem. Retention methods are near their causal ceiling. **Paper 3 should not target retention ranking.** |
| **B — large method gap remains** | `A_R2 − A_m` large across methods | A genuine, causally-closable research problem, with §III.7 saying where it lives. **Paper 3 earns its existence.** |
| **C — regime-dependent** | ordering changes across budget / task / model | A regime map: no universally dominant retention objective; the captured fraction depends on retention pressure and task structure. |

Publication framing for each is fixed **before** the run. A fourth possibility — methods fail the
admission gate — is reported as a reproducibility finding rather than absorbed.

## III.9 Scope paragraph (draft, for the preregistration)

> Paper 2 measures the attainable ceiling of task-level KV-cache retention under fixed token
> budgets. Rather than proposing another eviction heuristic, we place representative existing
> retention methods — attention-based, learned long-horizon, and future-utility — in a common
> evaluation framework with identical prompts, budgets, scoring and execution, and measure how
> much of the available headroom each captures. Critically, we separate that headroom into a
> component reachable by a causal policy and a component that requires knowledge of future
> queries, and report method performance against the causal ceiling rather than a prescient one.
> Every method passes a published-behaviour reproduction gate and a signal-manipulation check
> before admission. We evaluate across retention pressures and task structures, including a
> condition in which the target tokens carry no distinguishing surface form. The goal is to
> determine whether current methods approach the causally attainable retention ceiling or leave
> systematic headroom, and — if headroom remains — which faculty of the oracle accounts for it.

---
---

# PART IV — PROPOSED PLAN

No GPU time is spent before Stage 2 completes.

| stage | work | blocking? | output |
|---|---|---|---|
| **0. Settle** | Resolve the open decisions in §IV.1 with you. Nothing else starts first. | yes | decisions recorded |
| **1. Arithmetic** | Re-derive budgets from competitive budget per model per tokenizer; verify no cell is VOID/PARTIAL; verify the `G_m` denominator threshold is reachable. Pure arithmetic, no GPU. | yes | budget table |
| **2. Preregistration** | `PREREG_P2.md` — ladder, metric, thresholds, axes, admission protocol, all three outcome framings. Hashed before any run, per Paper 1 practice. | yes | hashed doc |
| **3. Phase 0 prerequisites** | Random-span control; distractor-ratio calibration (§III.5). Small n. | yes | go / redesign |
| **4. Reference ladder** | Build and validate R0–R4, especially `oracle_causal` (R2). Assert every §III.2 invariant on live data. | yes | validated ladder |
| **5. Method admission** | §III.6 gate, one method at a time. Methods that fail are recorded, not fixed into passing. | per method | admission table |
| **6. Shortcut-free task** | Build and calibrate the second task to the same five gates. Largest new engineering item. | for the main claim | calibrated task |
| **7. Main matrix** | methods × budgets × tasks × models, n≥150, seeds disjoint from everything in Paper 1. | — | `G_m` table |
| **8. Decomposition** | §III.7 partial oracles — **only if Stage 7 shows a gap.** | conditional | attribution |
| **9. Independent check** | Blind reimplementation of the *ladder*, not the whole harness — R1 and R2 are where a self-consistent error would be invisible and fatal. | strongly recommended | agreement report |

Stage 9 is not optional in my view. Paper 1's clearest lesson is that the two defects that mattered
were invisible from inside one implementation, and Paper 2's entire result is a *ratio of reference
arms*. An error in R1 or R2 does not produce a wrong detail — it produces a wrong headline, in a
paper whose only output is that headline.

## IV.1 Open decisions — needed from you before Stage 1

1. **Context length.** Stay at ~1030 (cheap, calibrated, but forces budgets into the void zone and
   invites a reviewer objection) or move to 2048/4096 (already calibrated in `amd/`, better
   external validity, more GPU)? *My recommendation: move to 2048.*
2. **`oracle_causal` construction.** "Retain all N credentials" (simple, strong, arguably too
   strong) versus "allocate under the query's marginal distribution" (principled, more moving
   parts). *My recommendation: build both, report the pair as a bracket — the same instinct that
   made R2/R3 worth separating.*
3. **Method provenance.** Wrap authors' code where it exists, or reimplement uniformly for
   controllability? *My recommendation: wrap, with the fidelity assertion; uniformity is worth less
   than defensibility here.*
4. **Second task.** Which shortcut-free construction? This is the biggest engineering decision in
   the paper and it should be yours.
5. **Model axis.** Attempt gate-matching recalibration on the 3B models, or accept the 1.5B as the
   only measurement model with the others as covariates? *My recommendation: one recalibration
   attempt each, time-boxed; accept covariate status if it fails, since AMD already showed it
   resists tuning.*
6. **Stage 9.** Is a second implementation of the ladder available (the same collaborator, or a
   fresh one)?

## IV.2 Prohibitions carried from your draft, unchanged

- No new eviction signal. That is Paper 3, and Paper 2's job is to establish whether Paper 3 has a
  target.
- No chasing the Qwen quantization anomaly. Closed in Paper 1 as a model-family property.
- No recoverable precision. Operationalized as: **no cold tier in the code path at all.**
- No giant benchmark.
- No claim of an absolute theoretical optimum. The oracle is task- and setup-specific, and the
  disclaimer goes in the abstract.
- No expansion to 20 models and 50 datasets.
