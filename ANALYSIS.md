# Analysis — Protection × Recoverability in Autoregressive KV Eviction

**Date:** 2026-09-04
**Status:** Phases 0-R, 1, 2 complete. §2 and §6 revised 2026-09-05 following an
independent blind reimplementation's agreement report; open items are recorded in §9.
**Governing documents:** `PREREG.md` (sha256 in `PREREG.sha256`), `ADDENDUM_2026-09-03.md`
(sha256 in `ADDENDUM_2026-09-03.sha256`, §§9–13 carry corrections made after the hash).
**Data:** `results/phase1/`, `results/phase2_4bit/`, `results/phase2_8bit_control/`,
`results/quant_sensitivity/`, `results/bitwidth_degeneracy/`, `results/phase0r_calibration/`.

This document exists because the published report is neither hashed nor version-controlled.
Where the two differ, this one governs.

---

## 1. Headline

Two independent routes converge on one conclusion:

> At deployable precision, a recoverable quantized KV tier's entire value is byte-efficiency
> of retention. The promotion machinery — the part that decides what to bring back — is
> decoration, and only has a job in precision regimes where the tier is itself damaging the
> cache.

Route one (Phase 1): the protection × recoverability interaction is null under matched-token
accounting at every valid budget, and large only under matched-byte accounting, where the
tiered arm is handed ~1.6× the tokens. Route two (Phase 2 and its diagnostics): the
FULL/QUANT distinction carries no quality signal at int8 at all, and at 4-bit — where it
does — no promotion signal beats any other, including one with perfect knowledge of future
need.

**Revised 2026-09-05 (§3.4).** Phase 1 originally ran at 8-bit only, which the spec never
stated — our defect. Re-run at 4-bit, where the cold tier is not free, the interaction is
**negative at every budget with CIs excluding zero, and grows with budget** (−0.012 / −0.019
/ −0.683 at 154 / 257 / 514). So the sharper statement is:

> Recoverable tiering does not merely fail to help. Once the cold tier is lossy enough to
> matter, it actively harms — worst at exactly the budgets where retention is otherwise
> working. The 8-bit null and the 4-bit harm are one mechanism seen at two precisions.

---

## 2. Instrument audits

Prompted by an independent blind reimplementation on an RX 7900 XTX whose agreement report
raised four discrepancies (2026-09-05).

### 2.1 The scorer is strict, and the 0.947 is real — but our values are not uniform hex

The scorer is exact substring on raw generated text, with no normalization of any kind:

```python
def score_turn(generated_text: str, credential: Credential) -> bool:
    return credential.value in generated_text
```

One definition, all 11 call sites. Verified by differential testing rather than inspection:
on 30 fresh `full_cache_ref` prompts the harness scorer, a from-scratch strict matcher, and
a lowercase+strip variant all return **0.9444**, agreeing on 180 / 180 answers. Our failure
*mode* also matches theirs — 8 of 10 failures are single-character hex miscopies (theirs 10
of 11). **No rescoring was required and no headline number moves.**

Their objection ("Qwen2.5-1.5B copying 14 random hex characters cannot yield 0.947") is
sound in principle, and it surfaced a genuine defect — just not in the scorer.
`string.hexdigits.lower()` is `'0123456789abcdefabcdef'`: 22 slots, because `ABCDEF` folds
onto `abcdef`. **Our credential values draw a–f twice as often as digits** (9.09% vs 4.55%),
giving 3.914 bits/char against a true-hex 4.000.

Holding seeds and every other parameter fixed and swapping only the alphabet:

| value alphabet | full_cache accuracy |
|---|---|
| as-shipped (a–f 2×) | 0.9444 |
| uniform hex (16 symbols) | 0.9111 |
| digits only (10 symbols) | 0.8833 |

The bias costs ~0.033. **It does not explain the divergence** — uniform hex still gives
0.911 against their 0.72–0.79. The residual is an unlocalized cross-implementation
difference in task construction. All arms in every phase faced the identical alphabet, so
between-arm comparisons are unaffected; absolute accuracy levels in this document are
inflated by ~0.03 relative to a true-hex task.

### 2.2 Retention ranking accumulates sum, and carries a large positional artifact

`engine._score_step` takes the mean over layers and heads, then **sums over queries** and
accumulates across steps. Every attention-ranked policy consumes that score. In a ~1025-token
prefill, position 0 is attended by ~1025 queries and position 1000 by ~25, so early positions
outrank later ones on longevity alone.

Measured (n = 5 prompts, real prefills):

| quantity | value |
|---|---|
| Spearman(score, position) — sum, as shipped | **−0.717** |
| Spearman(score, position) — mean-normalized | +0.587 |
| **Spearman(sum, mean)** | **+0.014** |
| top-192 retained-set overlap between readings | **0.135** |
| credential percentile rank — sum / mean | 0.693 / 0.654 |

The artifact is present and large. The two readings are effectively uncorrelated and share
only 13.5% of the retained set — sum-vs-mean selects an almost entirely different cache, so
this is a first-order design choice, not a refinement.

**Their diagnostic is silent for us.** They detected this via `oracle_static` ≠
`full_cache_ref` under sum. Ours passes that test (257: 0.943 vs 0.947; 514: 0.958 vs 0.947,
mostly tied, sign-test p = 0.39) because **our oracle never touches attention** — its
keep-set is ground truth plus earliest-position filler, structurally immune to the choice.
The tripwire does not fire here while the defect is present, which is worth reporting back.

**Not yet established:** whether switching to mean moves our headline numbers. Both Phase 1
interaction arms share the identical ranking, so a common bias does not obviously create or
destroy the null, and credentials rank poorly under both readings (0.693 vs 0.654). That is
reasoning, not measurement. **Required before the sum reading can stand: rerun the
attention-ranked arms under mean.**

### 2.3 The estimator problem

**PREREG.md §3 fixed a bootstrapped median of paired per-prompt differences. On this data
that estimator is degenerate and was replaced with the mean. This is a deviation from the
pre-registration and is recorded as one, not folded away.**

The paired differences are overwhelmingly *exactly zero* — the two arms return identical
scores on most prompts — so the median is zero in essentially every resample and the
interval collapses to zero width. Reported medians looked like `[0.000, 0.000]`,
`[0.167, 0.167]`, `[0.500, 0.500]`; the non-zero ones are exactly `1/6` and `3/6`, the
granularity of a six-credential score. A zero-width interval is not a precise result, it is
an artifact of summarising a mostly-constant discrete distribution with a median.

The mean paired difference with its bootstrap interval is what the data supports, and it is
reported alongside **the count of prompts on which the arms differ at all** — which is the
stronger statement and does not depend on any estimator choice.

| Budget (iso-token) | prompts where arms differ | mean diff | 95% CI |
|---|---|---|---|
| 154 | **3 / 150** | +0.0011 | [−0.0022, +0.0044] |
| 257 | **5 / 150** | +0.0022 | [−0.0033, +0.0089] |
| 514 | **9 / 150** | −0.0022 | [−0.0100, +0.0044] |

At budget 257 the arms produce identical scores on 145 of 150 prompts. Of the five that
differ, two favour protection alone and three favour protection-plus-recovery. Equivalence
is declared per PREREG.md §3 (CI within ±0.05) at all three budgets.

Note the contrast with Phase 2, where the arms *do* disagree prompt-by-prompt (55–95 of 150)
and still average to the same place. Phase 1's null is "the same behaviour"; Phase 2's null
is "different behaviour, same outcome". They are different kinds of null and should not be
described in the same words.

---

## 3. Phase 1 — the dissociation

n = 150, seeds 3000–3149. Budgets classified per `ADDENDUM_2026-09-03.md` §3:
{51, 103} VOID, {154} PARTIAL, {257, 514} VALID.

### 3.1 Arm means, iso-token (arms 3/4 filtered by `iso_condition`)

| budget | no protection | protection | recoverable alone | protection+recovery | oracle_static | full_cache |
|---|---|---|---|---|---|---|
| 51 *(void)* | 0.013 | 0.013 | 0.013 | 0.013 | 0.013 | 0.947 |
| 103 *(void)* | 0.012 | 0.102 | 0.012 | 0.101 | 0.271 | 0.947 |
| 154 *(partial)* | 0.011 | 0.196 | 0.012 | 0.197 | 0.648 | 0.947 |
| **257** | 0.010 | 0.423 | 0.009 | 0.426 | 0.943 | 0.947 |
| **514** | 0.010 | 0.964 | 0.010 | 0.962 | 0.958 | 0.947 |

### 3.2 The interaction, both accountings

| budget | iso-token | 95% CI | iso-memory | 95% CI |
|---|---|---|---|---|
| 154 | +0.001 | [−0.002, +0.004] | +0.199 | [+0.179, +0.219] |
| 257 | +0.002 | [−0.003, +0.009] | +0.532 | [+0.506, +0.559] |
| 514 | −0.002 | [−0.010, +0.004] | −0.051 | [−0.068, −0.034] |

**H-SUB confirmed under iso-token.** The apparent iso-memory benefit tracks having ~60% more
raw retained tokens, not a reversibility mechanism. This dissociation — null under one
accounting, large under the other — is the paper-ready result, and it is a claim about how
this literature evaluates rather than about any one system.

### 3.3 `quant_byte_cost` sensitivity

`quant_byte_cost = 0.25` sets the iso-memory token multiplier and was chosen *after*
PREREG.md was hashed, so it was swept rather than asserted. Budget 257, n = 150:

| quant_byte_cost | multiplier | true tokens | interaction | 95% CI |
|---|---|---|---|---|
| 0.125 | 1.78× | 457 | +0.543 | [+0.518, +0.570] |
| **0.25 (hashed)** | 1.60× | 411 | +0.532 | [+0.506, +0.559] |
| 0.30 | 1.48× | 395 | +0.536 | [+0.510, +0.561] |
| 0.50 | 1.33× | 343 | +0.354 | [+0.331, +0.379] |

The qualitative claim survives the whole plausible range, including a deliberately punitive
0.50 (implying only 2× compression where int8-vs-bf16 is genuinely 4×). **The magnitude
moves with the constant and must be reported as a range (+0.35 to +0.54) with the constant
stated — never as a point estimate.**

---

## 4. Phase 2 — H-ORTH falsified

Run at 4-bit (see §5 for why 8-bit could not test it). Protection on, tiering on, eviction
attention-ranked throughout; the promotion signal is the sole varied factor. n = 150.

### 4.1 Budget 257 (usable band at 4-bit: all-QUANT 0.133 → all-FULL 0.411)

| signal | ρ(eviction, promotion) | mean | vs P1 | 95% CI | prompts differing |
|---|---|---|---|---|---|
| P1 attention *(ref)* | **+1.000** | 0.203 | — | — | — |
| P2 epiphany | −0.077 | 0.216 | +0.012 | [−0.016, +0.039] | 85 / 150 |
| **P3 random** | −0.050 | 0.154 | **−0.049** | **[−0.077, −0.021]** | 91 / 150 |
| P4 rotation | +0.002 | 0.184 | −0.019 | [−0.047, +0.009] | 87 / 150 |
| P5 oracle | −0.068 | 0.188 | −0.016 | [−0.046, +0.013] | 95 / 149 |

Budget 154 (qualified, band 0.111): same ordering, all effects compressed, nothing clears
zero including P3 (−0.013, [−0.030, +0.003]).

**Findings.** The manipulation is exact (ρ = +1.000 circular, ≈0 otherwise) and the
orthogonal candidate does not beat the circular one. Circularity was not what made
recoverability inert. Only a deliberately uninformative ordering separates, and it separates
*downward* — so the promotion signal is **saturated**, not ignored: any non-degenerate
ordering performs equally and only actively bad ordering costs anything. Rotation ties
measured ranking, so the blind-rotation-beats-ranking result from the adjacent
block-diffusion work does **not** reproduce in this architecture.

### 4.2 P5 scope — what the oracle does and does not bound

P5 landing marginally below P1 prompted a check that it was not silently "oracle on whatever
subset fits" (`kvcache_harness/debug_p5_oversubscription.py`, budget 257, 4-bit): the
future-need set is 111 tokens against 96 FULL slots, but only ~53 of those tokens survive
retention, and **0.000 of selection calls were oversubscribed** — every retained credential
token is promoted, none end quantized. The inversion is noise; P5 is a genuine ceiling.

**But P5 bounds the promotion decision only, not the policy.** It cannot rescue the 52.3% of
credential tokens evicted at the retention stage, because retention is held attention-ranked
in every arm by design — that is precisely what isolates the promotion signal as the sole
varied factor. The retention ceiling is a different arm entirely: `oracle_static` scores
0.943 at the same budget against protection-alone's 0.423.

This is the composite claim, and the scope line is load-bearing:

> The promotion decision is information-saturated — a perfect signal adds nothing — while
> retention carries ~0.52 of unexploited headroom at the same budget.

---

## 5. Why Phase 2 was run twice

At int8, holding retention fixed and varying only precision: all-FULL 0.450, half/half
0.467, all-QUANT 0.467. A quantized credential reads exactly as well as a full-precision
one. Quantization is genuinely applied (~1.2% relative error on real cached tensors, values
provably differ) — it is simply too mild to matter.

That makes P1 ≈ P2 ≈ P3 ≈ P4 ≈ P5 true *by construction* rather than by evidence, and
**H-ORTH untestable at int8 rather than false**. The first Phase 2 run was stopped at 39
prompts (rows retained in `results/phase2_8bit_control/` as the no-op control) and re-run at
4-bit, where a usable band was measured to exist.

**A framing claim that died and should not be repeated:** "4-bit is the informative regime
because it sits between free and destructive." The precision/accuracy curve is not ordered
(§6), so 4-bit sits between nothing. What actually licensed the run is empirical: at 4-bit
the band is non-degenerate, measured rather than inferred, and 4-bit is the one width with a
genuinely mixed prompt population (34/50 partial).

---

## 6. An unexplained behaviour of *our implementation*, not reproduced independently

**Rescoped 2026-09-05.** This section previously stated a general claim about KV
quantization on exact-match tasks. An independent blind reimplementation on an RX 7900 XTX
does not reproduce the central feature, so the claim cannot be stated generally and has
been narrowed to a property of this codebase.

In our harness, reconstruction error is monotone in bit-width and task accuracy is not:

| bits | rel. error (live path) | accuracy (ours) | degeneracy | prompts fully degenerate | prompts clean |
|---|---|---|---|---|---|
| 8 | 1.45% | 0.410 | 0.097 | 0 / 50 | 34 / 50 |
| 7 | 2.67% | 0.293 | 0.107 | 0 / 50 | 33 / 50 |
| 6 | 4.50% | **0.000** | 0.947 | 35 / 50 | 0 / 50 |
| 5 | 8.62% | **0.000** | 0.987 | 46 / 50 | 0 / 50 |
| **4** | 17.62% | **0.123** | 0.393 | 6 / 50 | 10 / 50 |
| 3 | 37.31% | **0.000** | 0.987 | 46 / 50 | 0 / 50 |

n = 50 prompts × 6 turns per width; reproduced across two independently written scripts
**within this codebase**.

### 6.1 The independent implementation disagrees at 6-bit

| bits | ours | theirs (RX 7900 XTX) | agree? |
|---|---|---|---|
| 8 | 0.410 | 0.143 | functional both |
| 7 | 0.293 | 0.160 | functional both |
| **6** | **0.000** | **0.130** | **NO — ours collapses, theirs does not** |
| 5 | 0.000 | 0.000 | collapse both |
| 4 | 0.123 | *(sweep file empty)* | **untested on their side** |
| 3 | 0.000 | 0.000 | collapse both |

Absolute levels are not comparable — their whole task is harder (their `full_cache_ref` is
0.72–0.79 against our verified 0.947; see §2.1). What is comparable is the *shape*: whether
a width degrades relative to that implementation's own 8-bit baseline. Theirs shows
essentially no 6-bit degradation (0.130 vs 0.143). Ours shows total collapse (0.000 vs
0.410).

**The 6-bit cliff is ours alone.** The two implementations agree that 8 and 7 work and that
5 and 3 collapse; they disagree precisely on the width that made our curve non-monotone in
the first place.

### 6.2 What survives, and what does not

**Does not survive:** the methodological caution. "Retrieval accuracy under KV quantization
is not monotone in bit-width even where reconstruction error is, so bit-width sweeps on
exact-match tasks cannot be read as severity orderings" was stated as a general claim about
evaluation. It rested on the 6-bit collapse, which does not replicate. **It is withdrawn as
a general claim** and retained only as a description of this implementation's behaviour.

**Survives:** the five ruled-out explanations below. They establish that whatever produces
the 6-bit collapse here, it is not in the quantizer, the scale, the policy, the prompt
population, or level occupancy — which narrows where an implementation difference could
live, and is the useful thing to hand back to the other implementation.

**Newly relevant:** the sum-vs-mean attention artifact (§2.2) is a live candidate for the
divergence and was not known when this section was first written. Our retention ranking
carries a −0.717 position correlation; if theirs uses mean, the two implementations retain
almost disjoint caches (ρ(sum, mean) = +0.014, 13.5% overlap), which could plausibly produce
different degradation profiles at the same nominal bit-width. This is a hypothesis, not a
finding — it has not been tested.

**Five hypotheses tested, all ruled out:**

1. **Level counts wrong** — asserted ≤ 2^bits on 10,640 quantization calls intercepted
   during real generation. Correct at every width (4-bit ≤ 16, 5-bit ≤ 32, 6-bit 62/64).
2. **Correct levels, wrong scale** — relative error measured on those same live calls
   through the `QUANT_BITS` global path (not with an explicit `bits=` argument, and not with
   a float32 reimplementation) is perfectly monotone 1.45% → 37.31%, with zero subnormal
   scales and zero non-finite values.
3. **Policy behaves differently per width** — same tokens quantized, tier composition
   identical to three decimals, call counts within 0.4%.
4. **Per-prompt threshold effect** — falsified: at 6-bit, 35/50 prompts are *fully*
   degenerate and **zero** are clean. Near-universal collapse, not a boundary crossed on a
   subset.
5. **Level-occupancy concentration** — effective level count (`2^H` of the occupancy
   distribution) halves exactly in step with nominal; the occupancy ratio is constant to
   three decimals (K ≈ 0.381–0.391) across every width. 5- and 6-bit do not concentrate mass
   relative to 4-bit.

**Established (structurally, score-mode independent):** empty output *is* immediate
end-of-sequence. If the first sampled token is EOS the decode loop breaks at once and
`decode(skip_special_tokens=True)` returns `""`, so `first_tok_eos ⟹ empty` by
construction — verified with **0 violations across 600 answers** (one empty answer was
*not* EOS, so EOS is a strict subset). Empty rate is therefore an upper bound on the EOS
rate under any score mode.

**Superseded rates.** The per-width EOS figures previously recorded here (5-bit 12.5% EOS
vs 16.7% empty; 3-bit 58.3% vs 62.5%) were measured under `SCORE_MODE="sum"` at budget
257, n=4 seeds — **pre-correction, and not to be cited**. The sum accumulation was
manufacturing terminations (§2.2). Recomputed empty rates, n=50 prompts × 6 turns per
width:

| bits | empty — sum (pre-correction) | empty — mean (corrected) | Δ |
|---|---|---|---|
| 8 | 0.000 | 0.000 | +0.000 |
| 7 | 0.003 | 0.017 | +0.013 |
| **6** | **0.167** | **0.027** | **−0.140** |
| 5 | 0.180 | 0.170 | −0.010 |
| 4 | 0.000 | 0.000 | +0.000 |
| 3 | 0.483 | 0.370 | −0.113 |

The artifact inflated terminations **selectively** — heavily at 6-bit and 3-bit, negligibly
at 5-bit. So 5-bit's termination failure is genuine while most of the 6-bit one was an
artifact of the ranking defect. Our corrected 6-bit empty rate (0.027) is close to the
independent implementation's 0.007; the 6-bit divergence between builds is therefore about
**copy fidelity, not early termination** (§6.1).

**Still unmeasured:** the EOS *point estimate* under mean. Only
`results/q1_decompose_514/` carries `first_tok_eos` and it ran under sum; the mean sweep
stored answers but not token IDs. The bound (EOS ≤ empty) holds; a GPU pass is required for
the rate.

**Not established:** why a noisier width produces cleaner output *in this implementation*.
No mechanism is offered, and the behaviour does not replicate elsewhere at 6-bit.

### 6.3 Consequence for Phase 2 — the uncomfortable part

Phase 2's internal validity is intact on its own terms: it is a within-bit-width comparison
at a width verified five ways, against a band measured at that width rather than a position
inferred on this curve. All five arms faced identical conditions.

But **Phase 2 ran at 4-bit, which is the one width where the two implementations have no
comparable data** — their 4-bit sweep file is empty. So Phase 2 sits precisely in the gap
between an agreed-functional region (8, 7) and an agreed-collapsed region (5, 3), at the
width our own curve behaves anomalously and theirs is silent. That does not invalidate the
H-ORTH result, but it means the result has no independent corroboration at the width it was
measured, and the choice of that width was licensed by a curve whose central feature does
not replicate.

The honest statement is: H-ORTH is falsified *in this implementation at 4-bit*, and
confirming it requires either the other implementation's 4-bit sweep or a rerun at a width
both implementations agree is functional-but-lossy — 7-bit is the obvious candidate, since
both show real degradation there (ours 0.293 vs 0.410; theirs 0.160 vs 0.143) without
collapse.

---

## 7. Scope, deviations, and what was not run

**Deviations from PREREG.md**, all recorded rather than absorbed:

- Median → mean estimator (§2.3), because the median is degenerate on this data.
- Budgets 51 and 103 excluded by an arithmetic, design-time-computable criterion
  (`ADDENDUM_2026-09-03.md` §§1–3); 154 reported separately, not pooled.
- The budget gradient rests on two clean points plus one qualified one, not the intended
  five (`ADDENDUM_2026-09-03.md` §5).
- `quant_byte_cost` fixed post-hash, therefore swept (§3.3).

**Scope limits.** One quantizer (per-position affine min–max over the
`(kv_heads, head_dim)` slice), one model (Qwen2.5-1.5B-Instruct, bfloat16, eager attention),
one synthetic task, one GPU. A scheme with different group sizes or per-channel scales — and
QEvict's actual scheme is one — will move the free/damaging boundary. The structural claim
survives that (there exists a precision above which promotion logic is inert, and int8 is
above it here); the boundary's location is implementation-dependent and must be stated as
such.

**Not run.** Per PREREG.md §4's kill gate, this is the "P1 ≈ P2 ≈ P5" branch: no promotion
signal helps, so Phases 3–6 are not worth compute beyond what supports the negative claim.
Phase 3's random-span control (is "protection" really just reserved capacity?) remains the
single most valuable unrun experiment, since it tests the mechanism that *did* carry the
task.

---

## 8. Defects that changed a reported number

| ID | Defect | Caught by |
|---|---|---|
| D-01 | fp16 degeneracy — Qwen2.5 emits repeated punctuation in fp16 on this stack, including under a plain `generate()` | Reference run with no custom code |
| D-02 | Structural protection covered a fixed 12 characters after each label, shorter than the 27-char credential | Inspecting generated output |
| D-03 | No recency floor — attention ranking evicted the tokens immediately preceding the answer, breaking every arm including the oracle | Oracle below its own ceiling |
| D-04 | `oracle_static` sliced an unordered `set` when trimming to budget, dropping an arbitrary non-reproducible subset; also lacked credential labels | Ceiling shortfall |
| D-05 | Interim analysis pooled `iso_token` and `iso_memory` rows for tiered arms under one nominal-budget label | A query that would not reproduce |
| D-06 | The factor under test was inert — twice (protection pinning credentials; the 8-bit tier carrying no signal) | G3 gate, then a smoke test |

**The transferable practice:** assert an invariant on live data, not on a reimplementation.
The level-count check on 10,640 intercepted calls is the strongest single piece of instrument
validation in the project, and the same form would have caught D-04 and D-05. Separately,
three of six defects surfaced from a *reference arm behaving impossibly* — an oracle below
its ceiling, a policy beating a no-eviction baseline, five arms returning byte-identical
results — rather than from any test suite. Reference arms earn their compute as tripwires,
not just as denominators.

---

## 9. Open items from the cross-implementation audit (2026-09-05)

Raised by an independent blind reimplementation (RX 7900 XTX). Two are closed, two are open.

| # | Item | Status |
|---|---|---|
| 1 | Scorer permissiveness | **Closed** — scorer is strict exact substring, verified by differential testing; 0.947 replicates at 0.9444. Surfaced a separate real defect: biased value alphabet (§2.1), worth ~0.033, does not explain the divergence. |
| 2 | Sum vs mean attention accumulation | **In progress** — we accumulate sum; artifact confirmed at ρ(score, position) = −0.717, two readings share 13.5% of the retained set (§2.2). `engine.SCORE_MODE` now switches sum/mean; regression check confirms sum reproduces shipped values (0.013 vs 0.010; 0.413 vs 0.423 at n=25). Mean rerun of arms 1–4, 8-bit, iso-token, n=150 running to `results/phase1_meanscore/`. Arms 5/6 are attention-independent and not re-run. |
| 3 | Phase 1 at 4-bit | **Closed** — see §10. Interaction negative at all three budgets, CIs excluding zero, growing with budget (−0.012 / −0.019 / −0.683). Direction, significance and budget-growth replicate their finding; magnitude at 514 is ~6× theirs and unexplained. Materially revises the headline (§1). |
| 4 | §6 rescoped | **Closed** — narrowed from a general claim about KV quantization to a property of this implementation; the methodological caution is withdrawn as a general claim (§6.2), and Phase 2's position in the untested 4-bit gap is stated (§6.3). |

**The two defects this audit found in our code are both ours and both real**: the value
alphabet (§2.1) and the sum accumulation (§2.2). Neither was caught by five rounds of
internal instrument validation, and neither would have been caught by more of the same,
because both are choices that are self-consistent within one implementation. That is the
argument for blind reimplementation over additional internal checking, and it is the most
transferable result in this document.

---

## 10. Phase 1 at 4-bit (added 2026-09-05)

Phase 1 ran at 8-bit only and the spec never stated a bit-width — our defect, raised by the
independent reimplementation. At 8-bit the quantized tier is a no-op (§5), so the tiered arms
were compared in a regime where recoverability could not act at all, in either direction.

Re-run at `quant_bits=4`, iso-token, n=150, seeds 3000–3149
(`results/phase1_4bit/raw_results.jsonl`). Arms 1 and 2 are permanent eviction with no
quantized tier and are therefore bit-invariant; they are reused from `results/phase1/` rather
than re-run.

### 10.1 Arm means

| budget | no prot | protection | recoverable alone | prot + recovery | oracle | full_cache |
|---|---|---|---|---|---|---|
| 154 | 0.011 | 0.196 | 0.006 | 0.183 | 0.648 | 0.947 |
| 257 | 0.010 | 0.423 | 0.003 | 0.404 | 0.943 | 0.947 |
| 514 | 0.010 | **0.964** | 0.003 | **0.281** | 0.958 | 0.947 |

### 10.2 Interaction, paired, 10k bootstrap

| budget | interaction @ 4-bit | 95% CI | prompts differing | @ 8-bit |
|---|---|---|---|---|
| 154 | **−0.0122** | [−0.0222, −0.0033] | 19 / 150 | +0.0011 |
| 257 | **−0.0189** | [−0.0322, −0.0056] | 30 / 150 | +0.0022 |
| 514 | **−0.6833** | [−0.7200, −0.6456] | 149 / 150 | −0.0022 |

All three negative, all three CIs excluding zero, magnitude growing with budget.

The recoverability main effect with protection off is also negative at every budget
(−0.006 to −0.007, CIs excluding zero), though small in absolute terms because that arm sits
at the floor regardless.

### 10.3 Mechanism, and why the effect grows with budget

`quant_slots` scales with the budget, so a *larger* budget sends *more* of the context to the
cold tier. At 514 roughly 225 tokens are quantized to 4 bits; since 4-bit all-QUANT collapses
outright (§6), corrupting that much context destroys generation — protection alone 0.964
falls to 0.281 once the tier is added. At 154 little is retained at all, so little is
corrupted. This accounts for the budget gradient in both implementations.

### 10.4 Agreement with the independent implementation

| budget | ours | theirs (RX 7900 XTX) |
|---|---|---|
| 257 | −0.019 | −0.046 |
| 514 | −0.683 | −0.116 |

**Direction, significance and budget-growth all replicate.** Magnitude agrees in order at 257
and diverges ~6× at 514. Given the two implementations already differ in task difficulty
(§2.1) and possibly in attention accumulation (§2.2), the qualitative agreement is the
meaningful part; the magnitude gap is not yet explained and should not be reconciled by
argument.
