# Stage 4 — The designed experiment

Scored against `PREREG_P3.md`, frozen at
`sha256 c920c404fd43777e522916ef0188362805a958aa7253512ec2be40641a846eee`. `stage4_run.py`
refused to run unless that hash matched, so the grid could not drift from the registered one.

Machine N (RTX 5070), single process. 81,800 records, **zero failures**. Every file verified
complete: no malformed lines, no duplicate key digests, exact per-cell counts.

---

## Verdicts

| | Result |
|---|---|
| Registered comparison of five `c_eff` forms (Mode A, primary) | **F5 wins** — 0.1472 mean \|log10 err\|, 91.4% inside a factor of 2, on 58 non-boundary held-out cells |
| Registered kill criterion (§10): F5 residual trend in budget | **FIRES** — r = +0.9482 with log B, above the 0.5 threshold |
| Cost axis: method advantage over position falls with fact cost | **Holds** in 24 of 24 admissible (arm, budget) sequences, both models |
| Crossover (F5, Mode A) | **19 of 22** admissible cells |
| `k` axis: adjacent parts behave as one unit (completion) | **Holds** — form-free `c_eff` does not rise from k=1 to k=2 in 6 of 8 pairs, and never by more than 0.02 |
| Distractor-load mechanism (exploratory, rule committed first) | **Not supported** |
| Generation determinism | **Deterministic and path-independent within a session, on both models.** The M3 oracle mismatch is a cross-session effect at the power-loss boundary, and no between-arm contrast is affected (§6) |

**The framework's out-of-sample prediction is good and its mechanism is not the operative one.**
F5 predicts per-fact completion within a factor of 2 in 91% of held-out cells, beating the naive
bound and the calibration-free baseline by an order of magnitude. Its residual nonetheless tracks
budget at r = +0.948, as strongly as the residual that refuted F2 at Stage 1. By the prereg's own
rule, that makes §5.3's stated empirical regularity the honest stopping point for the budget term.

---

## Stated deviations from the registered design

1. **Generation at n = 100, not 200.** Captures ran at the full registered n = 200 and carry the
   scoring of the registered completion predictions. Generation was halved because the full grid
   costed at 27–49 GPU-hours against the plan's 15–18. Accuracy CIs are √2 wider; no cell and no
   arm was dropped.
2. **The first-reported primary scoring used the wrong constants, now corrected.**
   `stage4_score.py` re-fitted ρ on Stage 4's own calibration cell. PREREG §7 commits Mode A to the
   constants fitted on Stage 2. `stage4_score_registered.py` uses those and is primary. The verdict
   is unchanged (§1.3). The earlier scorer is kept unchanged so its output stays reproducible.
3. **F4's base cost.** The prereg says "the smallest cost on the plane", but c = 1 is a different
   task family. F4 extrapolates from c ≈ 8 within LEDGER-C and is scored only where defined.
4. **Arms beyond the registered two.** ExpectedAttention and KeyDiff were calibrated on the same
   declared cell by the same rule and are reported separately, never pooled into the primary.
5. **A power loss during plane M3.** Recovered from checkpoint `3a1d0fb` with every flushed row
   intact, resumed at instance 51. §9 describes the recovery tooling.

---

## 1. Registered comparison of the five forms

### 1.1 Mode A (primary): measured `p_g`, registered constants

Constants frozen in PREREG §7, fitted on Stage 2: ρ F2 0.9867 / 0.9874, F3 0.9275 / 0.9310, F5 0.9696
/ 0.9713 for snapkv / adakv_snapkv.

**Excluding the c = 1 cells (58 held out).** At c = 1 every parametric form predicts q = p_g by
construction, so those cells are free marks; F1's entire hit rate comes from them.

| form | mean \|log10 err\| | hit rate | r vs log B | r vs c |
|---|---|---|---|---|
| F1 naive | 1.6800 | 0.0% | −0.845 | +0.427 |
| F2 linear | 0.1975 | 79.3% | +0.763 | −0.564 |
| F3 power | 0.1609 | 86.2% | +0.953 | −0.123 |
| F4 non-parametric | 1.3767 | 0.0% (n=38) | +0.277 | −0.678 |
| **F5 threshold** | **0.1472** | **91.4%** | **+0.948** | −0.068 |

All 78 held-out cells: F5 0.1097 / 93.6%, F3 0.1196 / 89.7%, F2 0.1468 / 84.6%, F1 1.2493 / 25.6%,
F4 1.3767 / 0.0%.

**Winner: F5**, reported as the prereg requires. Including the Stage 4 counterpart of the
calibration cell, which is out of sample under registered constants, gives F5 0.1438.

### 1.2 Mode B: registered interpolated `p_g`

| | F5 MAE / hit | F3 | F2 | F1 |
|---|---|---|---|---|
| 58 held-out cells | **0.1639 / 91.4%** | 0.1906 / 81.0% | 0.2381 / 72.4% | 1.6800 / 0.0% |
| excluding the C = 32 extrapolation (46) | **0.1602 / 93.5%** | 0.1951 / 80.4% | 0.2314 / 73.9% | 1.7303 / 0.0% |

The registered `p_g` interpolation itself is accurate: mean |log10 err| 0.0491.

### 1.3 The correction does not change any verdict

| c > 1 | registered constants | first-reported re-fit |
|---|---|---|
| F5 MAE / hit | 0.1472 / 91.4% | 0.1545 / 89.7% |
| F5 residual r vs log B | +0.9482 | +0.9475 |
| winner | F5 | F5 |

### 1.4 Kill criteria, evaluated as registered (§10)

| criterion | outcome |
|---|---|
| No form beats F4 out of sample → drop the interpolations | **Does not fire.** F5, F3 and F2 each beat F4 by roughly 7–9×. |
| F5 residual \|r\| with log B above 0.5 → the mechanism is not the operative one | **FIRES** at +0.948. |
| Explains everything observed, predicts nothing unobserved → publish the empirics without it | **Does not fire as stated.** F5 made registered out-of-sample predictions that held. But the budget term it was derived to explain is not the one operating. |

**The honest statement.** The co-retention structure predicts completion out of sample, and the
threshold mechanism gets its shape in `c` right. The dependence on budget is a regularity the
framework measures and does not derive, exactly as PREREG §5.3 already registered.

---

## 2. The accuracy plane (c × C), generation n = 100

Cells are admitted by the frozen §9.1 rule, applied mechanically. A floor below 0.05 excludes the
cell (X); all compressed arms at or below 0.02 make it vacuous (V).

### 2.1 Method accuracy ÷ floor accuracy

**M2** — 10 of 20 (c, C) cells admissible; 5 excluded, 5 vacuous.

| arm | C | c=1 | c=8 | c=19 | c=40 |
|---|---|---|---|---|---|
| snapkv | 256 | 3.94 | 1.41 | 0.43 | X |
| snapkv | 512 | 2.87 | 1.07 | 0.77 | 0.29 |
| adakv_snapkv | 256 | 4.24 | 1.62 | 0.61 | X |
| adakv_snapkv | 512 | 2.87 | 1.43 | 0.77 | 0.31 |
| expected_attn | 512 | 2.72 | 0.52 | 0.32 | 0.01 |
| keydiff | 512 | 2.77 | 0.56 | 0.47 | 0.10 |

At C ≤ 128, only c = 1 is admissible on M2 (ratios 0.55–8.20).

**M3** — 12 of 20 cells admissible; 5 excluded, 3 vacuous.

| arm | C | c=1 | c=9 | c=19 | c=39 |
|---|---|---|---|---|---|
| snapkv | 512 | 2.94 | 0.47 | 0.17 | 0.19 |
| adakv_snapkv | 256 | 4.65 | 1.22 | 0.30 | X |
| adakv_snapkv | 512 | 2.97 | 1.52 | 0.47 | 0.43 |
| expected_attn | 512 | 2.92 | 0.47 | 0.24 | 0.24 |
| keydiff | 512 | 3.10 | 0.77 | 0.35 | 0.16 |

### 2.2 Cost axis

Measured from the first to the last admissible `c`, the ratio falls in **24 of 24** sequences (M2
8/8, M3 16/16). The fall is **not always monotone**: M3 snapkv at C = 512 goes 0.17 → 0.19 between
c = 19 and c = 39, and M3 expected_attn is flat at 0.24 there. The first-to-last criterion is the one
the analysis uses, and it is stated so the result is not read as stronger than it is.

At c = 1, SnapKV, AdaKV and ExpectedAttention beat position by 2.7–8.2× at every admissible budget
on both models. KeyDiff is the exception: it **loses** to position at c = 1 on tight budgets (0.60 and
0.55 at C = 32 and 64 on M2, 0.71 at C = 32 on M3). Those are the over-call cells of §8 item 2. By
c ≈ 40, every method loses to position, by 2.3–100×. Restated T2 locates the cause in `p_g`: the
scorer's gold advantage over the floor inverts as facts lengthen.

### 2.3 Coverage sits at the cutoff and is noise-sensitive

Several cells have floor accuracy within about 0.02 of the 0.05 cutoff, where the standard error at
n = 100 is comparable. M3 coverage moved from 13 to 12 cells between n ≈ 62 and n = 100 as one cell
crossed the cutoff. The registered cutoff is kept, and its instability is reported rather than
corrected.

---

## 3. Crossover

**19 of 22 admissible cells** agree under F5 (Mode A) on the registered arms.

Replacing F5's prediction with the **measured** completion does not help, so no model of
completion could fix the misses:

| completion estimate | agreement |
|---|---|
| F5 prediction | 19 / 22 |
| measured per-slot completion | 18 / 22 |
| measured union completion | 13 / 22 |

All three misses share one structure: a method with **lower** measured completion than the floor
scores **higher** accuracy.

| model | c | C | floor q / acc | snapkv q / acc | adakv_snapkv q / acc |
|---|---|---|---|---|---|
| M2 | 8.2 | 256 | 0.181 / 0.072 | 0.110 / 0.102 | 0.108 / 0.117 |
| M3 | 8.9 | 256 | 0.166 / 0.080 | 0.087 / 0.040 | 0.090 / 0.098 |
| M3 | 8.9 | 512 | 0.305 / 0.247 | 0.167 / 0.117 | 0.189 / 0.375 |

That limits the step from completion to accuracy, not `c_eff`. It is taken up in §8 and in
`LIMITATIONS_P3.md`.

---

## 4. The `k` axis

### 4.1 Completion, matched on fact cost: clean

| | form-free `c_eff*` rises from k=1 to k=2 | registered P-k part 1: q non-increasing in k | part 2: inside bounds at k=2 |
|---|---|---|---|
| M2 | 2 of 4 (by +0.02 at most) | 4 of 4 | 2 of 4 |
| M3 | 0 of 4 | 2 of 4 | 2 of 4 |

**The registered claim stands: adjacent parts behave as one unit.** Where raw completion rises at
k = 2 (M3, C = 64), `p_g` rose too (0.0559 → 0.0613). That is the retained fraction moving, not a
cost of splitting.

**Flaw in the registered containment test, stated and not re-decided.** At k = 1 the k-inert and
independent-spans bounds coincide. "Inside" at k = 1 therefore only asks whether F5 is exactly
right, and F5's budget error is already registered. Containment is informative at k = 2 only, and
it inherits F5's error there too.

### 4.2 Accuracy across `k`: confounded by design

Holding fact cost at ≈19 while adding a part forces fewer content fields, so the answer shortens:
M2 goes from 3 fields / 14.2 tokens to 2 / 6.2, and M3 from 4 / 13.3 to 3 / 10.2.

**Why it was unavoidable.** Every part carries a fixed id-and-label overhead: about 6 tokens on M2
and 4 on M3. With that overhead, matching `c` across `k` and matching answer length across `k` are
incompatible, so the axis can hold at most one of them constant. It holds `c`, the axis the theory
is about. The same arithmetic removed k = 4 at calibration, where overhead alone forced
c ≥ 42.5 / 30.5 against a target of 19. Accuracy across `k` is reported with the confound named
and is **not adjusted**; a post-hoc correction on a two-point axis would be worse than the plain
statement.

Only C = 512 is admissible on either model. At C = 64, M2 is vacuous at k = 1 and excluded at k = 2,
and M3 is excluded at both.

| C = 512 | floor | snapkv | adakv | expected_attn | keydiff | oracle_causal |
|---|---|---|---|---|---|---|
| M2 k=1 | 0.060 | 0.048 | 0.055 | 0.030 | 0.070 | 0.492 |
| M2 k=2 | 0.163 | 0.077 | 0.107 | 0.043 | 0.048 | 0.820 |
| M3 k=1 | 0.212 | 0.037 | 0.133 | 0.040 | 0.110 | 0.730 |
| M3 k=2 | 0.142 | 0.018 | 0.107 | 0.013 | 0.065 | 0.370 |

On M2, accuracy rises at k = 2 across the ladder, including the floor (0.060 → 0.163) while its
completion falls (0.298 → 0.255). That is the shorter answer, not an effect of `k`.

**On M3 the sign reverses, and the confound cannot explain it.** The causal oracle keeps every
queried record whole at both `k`, and its accuracy halves: 0.730 → 0.370 at C = 512 and
0.318 → 0.125 at C = 64. Uncompressed, the two are identical (`full_cache` 0.844 at both k, n = 24).
The confound predicts a rise, since M3's k = 2 answer is shorter. This is recorded as an
observation, not a test, and added to `LIMITATIONS_P3.md` as Evidence 4.

---

## 5. Ladder anchors

### 5.1 Competence and tripwires

| full_cache | c=1 | c≈8 | c≈19 | c≈40 |
|---|---|---|---|---|
| M2 | 0.810 | 0.820 | 0.910 | 0.897 |
| M3 | 0.935 | 0.885 | 0.953 | 0.725 |

Every cell is in the band [0.55, 0.97]. `null` scores 0.000–0.028. `random` is at or below 0.018
everywhere except c = 1 at C = 512 (0.092 on M2, 0.077 on M3), which is what chance retention of a
single token gives when about 28% of tokens are kept.

### 5.2 Information share against T1's parameter-free prediction, c ≈ 19

| C | M2 measured | M3 measured | T1 |
|---|---|---|---|
| 32 | 0.701 | 0.744 | 0.780 |
| 64 | 0.214 | 0.250 | 0.264 |
| 128 | 0.000 | 0.000 | 0.000 |
| 256 | 0.000 | 0.008 | 0.000 |
| 512 | 0.000 | −0.029 | 0.000 |

Where the budget no longer binds, T1 is exact. Where it binds, T1 slightly overstates I. T1
assumes the prescient oracle scores 1, but it reaches 0.613 (M2) and 0.807 (M3) at C = 32. The
completion→accuracy gap reduces the ceiling arm too, and less on M3, which is why T1 fits M3 more
closely.

### 5.3 Oracle identity where all candidates fit

At c ≈ 19, C ≥ 128, the prescient and causal oracles must hold the same keep-set.

| | same score | byte-identical generation |
|---|---|---|
| M2, C = 128 / 256 / 512 | 100 / 100 / 100 of 100 | 100 / 100 / 100 of 100 |
| M3, C = 128 / 256 / 512 | 98 / 94 / 94 of 100 | 91 / 86 / 88 of 100 |

For every differing M3 instance, rebuilding both keep-sets on CPU shows them **identical**, so the
ladder is not at fault. Identical kept tokens produce different text: mostly formatting near-ties
(`A|B|C` against `A | B | C`), sometimes a dropped field. §6.2 locates the cause. Every
mismatch is on an instance whose causal row was generated **before** the power loss. Where both
oracles were generated after it, they agree 49 / 49 byte for byte and I(C) = 0.000 exactly at
C ≥ 128.

---

## 6. Generation determinism, and a session effect on M3

### 6.1 Within one session, both models are deterministic and path-independent

`stage4_determinism.py` ran after every Stage 4 package had finished, in a single process. It took
12 c ≈ 19 instances at C = 512 and the causal keep-set, and generated each query four ways: each
path run twice, both paths on the identical keep-set, and two independent prefills compared on
their last-position logits.

| model | causal rerun identical | prescient rerun identical | cross-path identical | max \|Δ logit\|, two prefills |
|---|---|---|---|---|
| M2 | 48 / 48 | 48 / 48 | 48 / 48 | 0 |
| M3 | 48 / 48 | 48 / 48 | 48 / 48 | 0 |

Neither run-to-run nondeterminism nor path dependence exists on either model.

### 6.2 The M3 oracle mismatch is a session effect at the power-loss boundary

The determinism sample was not too small to see the mismatch. It ran at C = 512 only, and at that
budget it **contained** four of the Stage 4 mismatching instances — s4_00001, 00006, 00007 and
00010. Inside one session both paths reproduced all four exactly. Six further mismatching
instances fall in the same index range at C = 256 (s4_00002, 00005, 00007, 00008, 00009, 00010),
but the determinism test did not run that budget, so they are not counted as reproduced here.

Splitting the Stage 4 oracle comparison by the session that wrote the `oracle_causal` row locates
it. The prescient rows were all written after the power loss. The causal rows for instances 0–50
were written before it, and for 51–99 after it:

| C | causal row written | n | prescient | causal | floor | I | same score | same generation |
|---|---|---|---|---|---|---|---|---|
| 128 | before power loss | 51 | 0.779 | 0.779 | 0.098 | 0.000 | 49 / 51 | 42 / 51 |
| 128 | after | 49 | 0.811 | 0.811 | 0.051 | **0.000** | **49 / 49** | **49 / 49** |
| 256 | before power loss | 51 | 0.775 | 0.765 | 0.157 | 0.016 | 45 / 51 | 37 / 51 |
| 256 | after | 49 | 0.837 | 0.837 | 0.158 | **0.000** | **49 / 49** | **49 / 49** |
| 512 | before power loss | 51 | 0.775 | 0.804 | 0.299 | −0.062 | 45 / 51 | 39 / 51 |
| 512 | after | 49 | 0.872 | 0.872 | 0.301 | **0.000** | **49 / 49** | **49 / 49** |

**All 35 mismatching (instance, budget) pairs at C ≥ 128 have causal rows written before the power
loss; none after.** Where both oracles were generated after it, they agree byte for byte and I(C)
is exactly zero. M2 crossed the same boundary — plane rows before it, anchor rows after — and stayed
100 / 100 identical, so the effect is specific to M3.

**A fresh session identifies which run differs.** `stage4_session_check.py` regenerated both
oracles in a new session on 24 mismatching (instance, budget) pairs — 12 at C = 512 and 12 at
C = 256, 96 generations — and compared each fresh generation byte for byte with both stored
versions:

| on generations where the two stored runs disagree (28) | count |
|---|---|
| fresh matches the stored **after-power-loss** version only | **28** |
| fresh matches the stored before-power-loss version only | 0 |
| fresh matches neither | 0 |

| fresh arm reproduces its own stored arm | |
|---|---|
| oracle_prescient (stored after the power loss) | **96 / 96** |
| oracle_causal (stored before the power loss) | 68 / 96 — the 28 misses are exactly the disagreements |

Inside the fresh session the two oracles agree on every instance. So three independent sessions
after the power loss — the anchors package, the determinism test and this check — agree with each
other byte for byte, and the session before it is the only one that differs. That does not make
its output wrong: both are valid greedy decodes. It makes the pre-restart session the one whose
generation-level ties resolved differently.

**Cause: not identified.** The boundary coincides with the power loss and the restart of WSL and
the GPU driver stack. A change in kernel selection or library state across the restart would
produce exactly this pattern: deterministic within a session, different between sessions, and
visible only on generation near-ties. It was not investigated further.

### 6.3 What the session effect touches, and what it does not

- **Every between-arm contrast is session-matched.** The plane, anchors and kaxis packages each
  generate all arms for an instance in one pass. The cost-axis ratios, the crossover, the k-axis
  contrasts and the distractor check therefore compare arms generated in the same session.
- **Completion is unaffected.** Every completion measure comes from prefill keep-sets, with no
  generation.
- **The only cross-session generation comparison in this report is I(C) on M3**, since prescient
  comes from the anchors package and causal and floor from the plane. On the instances where both
  were generated after the power loss, I(C) = 0.000 exactly at C ≥ 128. The earlier apparent
  inversion (−0.052 at n = 58, −0.029 at n = 100) is **fully accounted for** by the session
  boundary. It is not an inversion, and it does not need to be read as noise.
- **Size on the affected instances** (0–50, causal from before the power loss minus prescient from
  after): +0.029 at C = 512, −0.010 at C = 256, 0.000 at C = 128. Small, and of both signs.
- **The distractor check** mixes instances from both sessions within a cell. D comes from
  keep-sets and does not depend on the session, so the session effect adds noise to that test, not
  bias.

### 6.4 Corrections to earlier readings

- Stage 2's determinism evidence (176 byte-identical duplicate keys) came entirely from M2. It now
  also holds for M3 within a session (§6.1).
- An interim reading of this section, generated automatically from the determinism verdict,
  said a 48-generation sample "can plausibly miss" the mismatch. That was wrong: at C = 512 the
  sample contained four mismatching instances and reproduced them exactly. §6.2 replaces it. A
  follow-up draft of §6.2 then overstated the correction, counting ten reproduced pairs where the
  test had covered only the four at C = 512. That is also corrected.
---

## 7. Distractor load: exploratory, rule committed before running

Hypothesis and decision rule committed in `32be79f` before the check ran on final data.

**H-D.** At fixed completion of the queried record, more complete non-queried records (D) means
lower accuracy. The primary test uses within-cell variation only, because D and conversion both rise
with budget.

| within-cell, c > 1 | effect of one extra whole record | 95% CI (bootstrap over cells) |
|---|---|---|
| M2 floor | +0.016 | [−0.013, +0.041] |
| M3 floor | −0.014 | [−0.025, +0.018] |
| M2 methods | +0.050 | [+0.012, +0.081] |
| M3 methods | +0.049 | [+0.005, +0.075] |

**Verdict under the committed rule: not supported.** The floor has one global keep-set, so it is
the clean test, and it shows no effect on either model (on only 5 and 7 cells). The methods' sign is
reversed but rests on almost no usable variation: within a cell, D and per-slot completion correlate
at r = −0.78 / −0.83. The reversal is reported as unexplained. One proxy explanation is eliminated:
D correlates negatively with union availability (r = −0.43 / −0.34), which would push the effect
negative, not positive.

**The cross-cell version would have looked like support**, with partial r −0.51 (M2) and −0.47 (M3)
given log B and c. The two designs disagree in sign. That is why the within-cell design was fixed as
primary before running.

---

## 8. The boundary: completeness indexes retention, not usability

Stated in full in `LIMITATIONS_P3.md` as the named open question for Paper 4, and deliberately not
investigated inside Paper 3. Four independent pieces of evidence:

1. **Crossover misses no completion model can fix:** F5 19/22 against measured completion 18/22.
2. **A single-token over-call:** 3 of 40 cells, all KeyDiff at tight budget, on both models. At
   c = 1, co-retention cannot be the cause.
3. **Floor failures:** 56–100% of them contain no gold word. That is an upper bound on wrong-record
   copying, not a count of it.
4. **An oracle with every fact whole whose usability halves when the fact is split** (M3, §4.2).

Ruled out under a pre-committed rule: interference from other whole records (§7). Not pursued, by
decision: interference from partly retained fragments. It would be a post-hoc test straight after a
null, on a question outside the prereg.

---

## 9. Integrity and process

| package | M2 | M3 | per-cell n |
|---|---|---|---|
| capture | 24,000 | 24,000 | 200 |
| plane | 12,000 | 12,000 | 100 |
| anchors | 2,500 | 2,500 | 100 |
| kaxis | 2,400 | 2,400 | 100 |

- **Power loss mid-run.** Every flushed row was intact, with no partial lines and no duplicates.
  The run resumed at instance 51 of plane M3.
- **`stage4_repair.py`** ran before every package after the interruption. It drops only unparseable
  lines and keeps a backup, and it changed nothing each time.
- **Single-writer guard.** Every package refuses to start if another `stage4_run.py` exists.
- **Invariants carried from Paper 2 and asserted.** `position_ids` continue from the uncompressed
  length. Every arm generates tokens. Method arms honour the mandatory floors. Realised budget
  parity is asserted per instance from captured keep-sets. `max_new_tokens` scales with answer
  length.
- **Instance identity across packages.** Every plane row maps to its capture instance with the same
  seed and context length (all 24,000 plane rows checked, zero mismatches).

---

## 10. What Stage 4 licenses

- **Supported and measured out of sample.** Co-retention is the functional unit of retention.
  Completion follows the retained fraction of a fact's tokens. Methods' advantage over position
  collapses as facts lengthen. Adjacent parts of a fact behave as one unit.
- **Predicted but not derived.** The budget dependence of `c_eff`. F5 gets its sign and shape in
  `c`; the registered kill criterion fires on its magnitude.
- **Not established, and bounded.** When a complete fact is usable. That is the named question
  handed to Paper 4.

Paper 3's experimental programme is complete. Every registered test has been run and scored,
every deviation is stated, and the one unresolved question — when a complete fact is usable — is
named and handed to Paper 4 in `LIMITATIONS_P3.md`.
