# Stage 2 — Adversarial probes

Designed to break the theory, not confirm it. Each probe is reported separately against the
breaking condition the plan fixed in advance. **A failure is never pooled against a pass.**

Machine N, single process throughout. ~7 GPU-hours (plan allowed ~4; the overrun is 2.3, which
was run three times and is still unmeasurable — see below). 13,850 generation records,
3,000 capture records, **zero failures**.

---

## Verdicts

| # | Probe | Breaking condition | Verdict |
|---|---|---|---|
| 2.1 | Cost sweep, `c` ∈ {8,19,40} × C ∈ {64,512} | deficit does not grow with `c` | **PASS** |
| 2.2 | Contiguity control | accuracy does not improve | **PASS** |
| 2.3 | Shuffled-fact control | floor's advantage does not collapse | **VACUOUS** — unmeasurable, three layouts tried |
| 2.4 | Single-token facts, `c` = 1 | methods still lose to the floor | **PASS** (required) |

**GATE (2.4 AND ≥ 2 of 2.1–2.3): PASS**, at 2/3.

2.3 is scored VACUOUS rather than FAIL. It never reached a cell that could express its question
in either direction, so it is not evidence for the theory and not evidence against it.

---

## Design change from the plan, and why

The plan specified 2.1 at one budget. Stage 1 found F2's residual correlating **+0.879 with
log C** and flipping sign at the calibration budget, so `c` and `C` were confounded in
everything Paper 2 measured. 2.1 was therefore run at **two** budgets, which is what makes
§2.1b below possible.

---

## Competence anchors — a precondition, not a footnote

Paper 2's own Stage 4 ablation reported PASS at anchor 0.000, where "bindings deleted ≤ chance"
is satisfied by there being nothing to answer. Every Stage 2 cell therefore carries a
`full_cache` anchor, and any cell whose arms are all at the floor of the metric is scored
VACUOUS rather than passed.

| task | M2 | M3 | band [0.55, 0.97] |
|---|---|---|---|
| LEDGER-C `c`=8 | 0.844 | 0.938 | in |
| LEDGER-C `c`=19 | 0.927 | 0.958 | in |
| LEDGER-C `c`=40 | 0.865 | 0.771 | in |
| MARK-1 `c`=1 | 0.792 | 0.917 | in |
| SPLIT adjacent N=20 | 0.906 | 0.938 | in |
| SPLIT scattered N=20 | 0.542 | 0.656 | M2 marginally **out** |
| SPLIT scattered_uniform N=20 | 0.400 | 0.680 | M2 **out** |

`c` calibration solved `n_fields` per model; achieved costs M2 {8.24, 18.92, 39.51} and
M3 {8.93, 18.61, 39.28}, with `L` held at ~2070. M2's mid-point reproduces Paper 2's 18.90
exactly. All three `c` anchors are in band, so **the task family can express the cost axis** —
which is the Stage 3 gate condition, checked early and cheaply.

---

## 2.4 — Single-token facts (`c` = 1) — **PASS**

The sharpest test in the programme. At `c` = 1 there is nothing to co-retain, `p_g^c_eff`
collapses to `p_g`, and a pointwise scorer — strictly better informed than position — should
win. If it still lost, co-retention would not be what limits these methods.

| arm | M2 C=64 | M2 C=512 | M3 C=64 | M3 C=512 |
|---|---|---|---|---|
| full_cache | 0.820 | 0.820 | 0.930 | 0.930 |
| null | 0.020 | 0.015 | 0.000 | 0.000 |
| random | 0.015 | 0.105 | 0.015 | 0.110 |
| **floor_pos** | **0.100** | **0.290** | **0.105** | **0.280** |
| snapkv | 0.315 | 0.810 | 0.440 | 0.875 |
| adakv_snapkv | 0.350 | 0.825 | 0.550 | 0.910 |
| expected_attn | 0.680 | 0.775 | 0.670 | 0.905 |
| keydiff | 0.075 | 0.805 | 0.130 | 0.945 |

At the top budget **4/4 methods beat the floor on both models**, by ~3×, landing at or above the
`full_cache` anchor. At C=64, 3/4 on M2 and 4/4 on M3.

**The mechanism is confirmed, not just the outcome.** From the capture pass, `|q_complete − p_g|`
= **0.0000** in every MARK-1 cell on both models — the collapse to `p_g` is exact to machine
precision, which is what the task was built to produce. And the retention ratio accounts for the
accuracy ratio directly: at C=512 methods hold the single gold token in 75–77 % of slots against
the floor's 27 % (≈ 2.8×), matching the observed 2.8–3.3× accuracy ratio. The win is per-slot
retention of one token, not anything peculiar to the task.

`null` (0.000–0.020) and `random` (0.015–0.110) place the guessing floor empirically rather than
by assertion, so the methods' margin is scorer information and not task triviality.

---

## 2.1 — Cost sweep — **PASS**

### Retention ratio, method ÷ floor_pos, at C=512 — combined with 2.4

| `c` | M2 snapkv | M2 adakv | M3 snapkv | M3 adakv |
|---|---|---|---|---|
| 1 | 2.79× | **2.84×** | 3.13× | **3.25×** |
| 8 | 0.77× | 1.02× | 0.50× | 1.64× |
| 19 | 0.81× | 0.87× | 0.20× | 0.59× |
| 40 | 0.44× | 0.48× | 0.00× | 0.24× |

Monotone decline on both models across four fact costs spanning 1 to 40, crossing 1.0 — the point
where methods stop beating position — between `c` = 1 and `c` = 19. The registered prediction is
that the pointwise deficit grows with `c`; it does, decisively.

**The ratio is the measure, and that choice is stated rather than assumed.** Absolute accuracies
span an order of magnitude across these cells, so a fixed 0.05 gap means something very different
at floor 0.020 than at floor 0.235. The raw differences are reported beside it
(`out/stage2_report.txt`); they also grow with `c`, but they compress the effect.

**C=64 is near-degenerate and is excluded by a named precondition.** The floor scores 0.006–0.041
there at every `c`, so the cell cannot express a deficit in either direction. The verdict rests
on C=512, on 4/4 qualifying cells.

---

## 2.1b — `c_eff` fitted independently at each budget

This is the theoretically important result and it goes **against both registered forms**.

`c_eff* = ln(q_complete) / ln(p_g)`, measured per (model, `c`, C, arm) from the retained sets.
Under F2 and F3, `c_eff` is a function of `c` and `ρ` only: it should rise with `c` and be flat
in C.

| | `c`≈8 | `c`≈19 | `c`≈40 | slope d(c_eff)/d`c` |
|---|---|---|---|---|
| M2 snapkv C=64 | 1.208 | 1.238 | 1.214 | **+0.0000** |
| M2 snapkv C=512 | 1.770 | 1.988 | 1.933 | +0.0042 |
| M3 adakv C=64 | 1.187 | 1.252 | 1.334 | +0.0047 |
| M3 adakv C=512 | 1.648 | 1.831 | 2.080 | +0.0139 |

**Budget dependence at fixed `c`** (`c_eff` at C=512 minus at C=64):

| arm | `c`≈8 | `c`≈19 | `c`≈40 |
|---|---|---|---|
| M2 snapkv | +0.561 | +0.750 | +0.718 |
| M2 adakv | +0.552 | +0.764 | +0.731 |
| M3 snapkv | +0.680 | +0.703 | +0.810 |
| M3 adakv | +0.461 | +0.579 | +0.746 |

**Stage 1's budget dependence was not a `c` dependence in disguise.** It is real, it survives
varying `c`, it is positive at every `c` on both models and both methods, and neither registered
form contains a budget term. Worse for the forms: the measured `c` slope (+0.000 to +0.014) is
*smaller* than F2's own predicted (1−ρ) ≈ 0.03–0.06. `c_eff` is close to independent of `c` and
strongly dependent on budget — the opposite of what F2 and F3 assert.

**The estimator is validated against a known answer.** `floor_pos` reads `c_eff` = 1.00 in every
cell (0.994–1.071, with `q_complete` matching `p_g` to three decimals). That is exactly right for
a contiguous policy — a record is either wholly inside the recency block or not — so the method
is not manufacturing the effect.

**So how does 2.1 still hold?** Through a different channel than the theory names. `p_g` itself
falls with `c` (M2 C=512: 0.389 → 0.348 → 0.257) while the exponent stays flat, so completion
falls (0.188 → 0.123 → 0.073) against a floor whose completion is nearly flat in `c`
(0.350 → 0.325 → 0.280). The co-retention penalty at larger `c` operates through the scorer
keeping a smaller *fraction* of gold tokens, not through a steeper exponent.

**No fifth form is registered here**, per instruction. The mechanism this suggests — a larger
budget spreads the keep-set and lowers the within-fact correlation — is an observation to be
derived from mechanism, not fitted to this residual.

---

## 2.2 — Contiguity control — **PASS**

`contig_matched_X` completes as many queried facts as X's own realised gold-token count allows,
then fills the rest from `floor_pos`'s order. It spends **exactly B = 584**, so unlike Paper 2's
`floor_matched` diagnostic it holds budget parity and stays comparable to every other arm.

| model | arm | accuracy | gold tokens held | whole facts assembled |
|---|---|---|---|---|
| M2 | snapkv | 0.125 [0.075, 0.180] | 26.4 | 0.49 |
| M2 | **contig_matched_snapkv** | **0.325** [0.250, 0.400] | **16.2** | **0.94** |
| M2 | adakv_snapkv | 0.135 [0.085, 0.190] | 26.5 | 0.49 |
| M2 | **contig_matched_adakv** | **0.330** [0.250, 0.410] | **16.2** | **0.94** |
| M3 | snapkv | 0.050 [0.025, 0.080] | 20.0 | 0.32 |
| M3 | **contig_matched_snapkv** | **0.285** [0.225, 0.350] | **9.8** | **0.58** |
| M3 | adakv_snapkv | 0.140 [0.095, 0.190] | 20.5 | 0.38 |
| M3 | **contig_matched_adakv** | **0.305** [0.240, 0.375] | **11.2** | **0.66** |

Paired 95 % CIs on the gain: M2 [+0.140, +0.265] and [+0.135, +0.265]; M3 [+0.175, +0.300] and
[+0.095, +0.235]. All four exclude zero.

**The effect is understated, not flattered.** The matched arm takes only *whole* facts, so it
ends up holding **fewer** gold tokens than its target (16.2 against 26.4 on M2) and still scores
2.6× higher. It wins with less gold, by arranging it coherently. Coherence is the operative
variable.

---

## 2.3 — Shuffled-fact control — **VACUOUS, three layouts, not scored**

The probe never reached a cell that could answer its question. Reported in full because the
failure is mine and is instructive.

### Layout 1 — disjoint halves (defective)

Every `/A` in the first half of the body, every `/B` in the second. Measured: **0 `/A` and 12
`/B` lines in the recency tail, in every instance**. `floor_pos` therefore completes nothing *by
construction*, and every compressed arm scored exactly 0.0000 on both models. A control that
forces its own answer is not a control.

### Layout 2 — `scattered_uniform`, C=512 and C=1024

Both halves drawn from the same uniform grid; separation enforced by pairing slot `j` with slot
`j+N`, orientation randomised per record. This **did** fix the positional privilege — the recency
tail now holds 2–10 `/A` lines instead of 0.

| M2 | adjacent C=512 | scattered C=512 | adjacent C=1024 | scattered C=1024 |
|---|---|---|---|---|
| full_cache | 0.910 | 0.400 | 0.910 | 0.400 |
| floor_pos | 0.115 | **0.000** | 0.205 | **0.000** |
| snapkv | 0.040 | **0.000** | 0.265 | **0.000** |
| adakv_snapkv | 0.075 | **0.000** | 0.340 | **0.000** |
| oracle_causal | 0.410 | 0.135 | 0.430 | 0.080 |

Doubling the budget did not rescue it. **My second structural artifact**: pairing `j` with `j+N`
makes the separation *constant* at exactly half the body — measured 61–63 lines in a 127-line
body — so a recency window shorter than half the context cannot hold both halves at any budget.
The first layout barred the floor by putting `/A` out of reach; this one bars it by putting the
halves permanently too far apart. Same failure, opposite mechanism.

Only the oracles score under scattering (0.080–0.135), because only they can place both halves
deliberately — so the task is answerable under compression, just not by any budget-limited
positional or pointwise policy.

M3 at C=1024 was not run: the M2 result is unambiguous across 50 instances and C=512 already
covers both models. That saved ~30 GPU-minutes on a cell that could not have spoken.

### The tension is inherent to the design as specified

Enforcing a large **minimum** separation is precisely what makes "both halves inside one recency
window" impossible — which removes the floor advantage that the probe exists to watch collapse.
`MIN_SEP_LINES` is itself the wrong constraint. A working control needs the separation **drawn
from a distribution** spanning short to long, so that a window captures both halves sometimes.
That is a task redesign, not another run, and is left to Stage 3.

**2.3 is therefore neither support for nor evidence against the theory.** One negative
observation does survive and is worth carrying: scattering did **not** spare the pointwise
scorers, which the layout-independence intuition would have predicted. Both they and the floor
collapse together. That is a hint, on a vacuous cell, and is recorded as no more than that.

---

## Harness invariants — asserted, not assumed

- `position_ids` continue from the **uncompressed** context length.
- Every arm check **generates tokens**; no verdict rests on prefill logits or cache length.
- Method arms honour the mandatory sink and window floors.
- **Realised budget parity asserted per instance from captured keep-sets** — `B = C+72` on all
  3,000 captured cells, with AdaKV's per-layer-total exception handled separately.
- `max_new_tokens` scales with the answer (up to ~37 tokens at `c`=40) and an EOS-vs-cap flag is
  recorded per generation, so Paper 2's B9 confound cannot recur.
- Machine N single-process throughout.

**Unplanned determinism check.** Two runners briefly wrote the same file concurrently during an
interrupted restart. All 176 overlapping keys agreed **byte-for-byte in score and in
generations**. Deduplicated by key digest; no contamination (every `scattered_uniform` row shows
`sep_lines_min` 61–63, i.e. the fixed layout).

---

## Defects found and fixed, each of which would have corrupted a result

1. **The first scorer read working retrieval as retention failure.** It required the joined field
   string as a substring and scored 0 for `R034 /A | Barenwell | Payments\nR034 /B | 711505 |
   Granthill` — correct content, correct order, line markers kept. Same class as Paper 2's B9,
   where a pinned `max_new` measured instruction-following and was reported as retrieval. Now
   word-level with record ids and part markers dropped, unit-tested against the real generations.
   The adjacent anchor moved **0.219 → 0.729**.
2. **SPLIT-LEDGER at N=40 was failing by aliasing, not retention.** 80 near-identical half-lines,
   and the model was picking the wrong `/A`. At N=20 the scattered anchor moved 0.177 → 0.542
   (M2) and 0.313 → 0.656 (M3).
3. **The scatter layout, twice** — see §2.3.
4. **The driver masked a crash.** It piped the runner through `grep`, so the pipeline exit code
   hid an assertion failure that killed M2's anchor run at instance 9 and reported nothing. The
   driver no longer pipes.

---

## Gate

```
2.1 cost sweep ................... PASS
2.2 contiguity control ........... PASS
2.3 shuffled-fact control ........ VACUOUS  (not scored either way; three layouts)
2.4 single-token facts ........... PASS     (REQUIRED)

GATE (2.4 AND >= 2 of 2.1-2.3): PASS  (2/3)
```

**What this licenses.** Stage 3 — preregister and build — per the plan. Nothing more.

**What it does not license.** The claim that `c_eff ≈ 1 + (c−1)(1−ρ)` is the right form. §2.1b
shows both registered interpolations are wrong in the same direction: `c_eff` barely moves with
`c` and moves strongly with budget. Stage 3's preregistration must confront that, and the fifth
form must be derived from mechanism rather than fitted to this residual.

**Carried into Stage 3 as work items.**

1. Derive a `c_eff` with a budget term, from mechanism. Not registered here.
2. Redesign the scattered control with a *distribution* of separations, and anchor it before
   spending GPU time.
3. M2's SPLIT anchors are out of band (0.542 scattered, 0.400 scattered_uniform). If a scattered
   condition is wanted at all, the task needs to be easier at N=20, or run on M3 only.

Stage 3 has not been started.
