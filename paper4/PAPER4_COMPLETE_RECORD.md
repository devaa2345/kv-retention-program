# Paper 4 — Complete Record

Everything done in Paper 4 to date, in order, with every number and commit that was
reported. This is a factual record for re-orientation and for drafting. Interpretation is
kept to what each result directly supports. Commits are cited where they were reported;
where no commit was reported, none is invented.

Hardware: RTX 5070 12 GB (Machine N, primary, single-process, WSL2, torch 2.11.0+cu128,
transformers 5.2.0, kvpress 0.5.4). RX 7900 XTX 24 GB (Machine A, verification only).
Models: M2 = Qwen2.5-3B-Instruct (2 KV heads), M3 = Llama-3.2-3B-Instruct (8 KV heads).
Budget convention throughout: `B = C + 8 (sink) + 64 (window)`.

---

## 0. Where Paper 4 came from

Paper 3 established that method advantage over a zero-cost positional floor (`floor_pos`
= 8 sink tokens + most recent B−8) declines with fact cost `c`: methods beat the floor
~2.8–3.3× at c=1 and lose 3–4× at c=40, falling from first to last cost in 24/24
admissible sequences. Paper 3's coherence intervention (contiguity-matched packing)
raised accuracy with all four paired CIs excluding zero. Paper 3 left one question open:
**completeness indexes retention, not usability** — facts can be retained whole and still
not converted into correct answers.

Paper 4's original aim: build a unit-aware retention method that fixes fragmentation, and
use its failures to probe the completion→accuracy gap.

---

## 1. The wrapper: U-X

**Design.** A wrapper, not a new scorer. Take any press X, aggregate its per-token scores
over units, allocate budget by whole units all-or-nothing, greedy by score-per-token.
Arms: `U-snapkv`, `U-adakv_snapkv`, `U-expected_attn`, `U-keydiff`. Scoring function
unchanged, so X vs U-X isolates allocation.

**Stage 1 verification.** Singleton-unit identity: with every unit a single token, U-X
reproduces X's keep-set up to exact score ties on all real-score checks (later 40/40
including ExpectedAttention at all five budgets on both models). At c=40 all four methods
on both models: units touched falls (SnapKV 32 → 11), units complete rises (3 → 11).

**Amendment 1.** M3's X keep-sets matched stored Paper 3 captures in 0/400 rows (M2
400/400). A fresh-process re-capture was deterministic on both models (40/40). Cause:
M3's stored captures predated Paper 3's power loss — a known session effect, not a
wrapper bug. Amended before any generation.

## 2. Stage 2 pilot (c=40, C=512, n=50)

| | M2 | M3 |
|---|---|---|
| U-snapkv − snapkv | +0.045 [+0.015, +0.075] | +0.000 [−0.025, +0.025] |
| U-adakv − adakv | +0.045 [+0.010, +0.080] | +0.035 [+0.005, +0.070] |

c=1 control: all four U-X − X intervals include zero; queried-fact completion unchanged.
Gate passed. Queried-fact completion rose ~0.07 → 0.23–0.28 (floor 0.29–0.31). Accuracy
0.043 → 0.091 against floor 0.177. **U-X still loses to floor ~2×.**

Conversion (correct answers per completed fact): M2 U-X 0.40 vs floor 0.60; M3 0.12–0.20
vs 0.33.

**Dispersion diagnostic (NOT supported).** Keep-sets rebuilt by prefill-only re-capture
(500/500 rows matched pilot summaries). Within-arm, contiguous-runs measure: M2 −0.099
[−0.253, +0.030], M3 +0.013 [−0.272, +0.300]. Isolation of the fact: M2 **+0.052
[+0.016, +0.090] — contradicted** (more isolated answered more often). Across arms: raw X
is the *most* dispersed (65–78 runs vs U-X 20–29, floor 1) yet converts as well as or
better than floor (M2 0.67–0.68 vs 0.60). So the conversion deficit is **specific to U-X**,
not general to methods. Keep-sets are now saved with every generation run.

## 3. Stage 3 grid and amendments

**Amendment A3** (digest `34a20a7e…`, superseding `131278082f3ee355…838c`, diagnosis
commit `e660cf3`). A3.1: inertness test is now the singleton-unit identity (40/40); the
old premise held for the queried fact only, not MARK-1's multi-token distractors. A3.2:
MARK-1 c=1 is a reported diagnostic with two channels — Δp_g (+0.008 to +0.021 for
KeyDiff/ExpectedAttention, ±0.001 SnapKV/AdaKV) and Δunits_complete (up to +5.0 at M2
C=512). A3.3: scope restriction stated in main text. A3.4: confound watch wired into the
runner (any KeyDiff/ExpectedAttention pair at c≥8 whose CI excludes zero positive).

**c=8 on M2.** C=32, 64, 128 degenerate (floor 0.025, 0.025, 0.035). C=256 admissible.

**The ExpectedAttention confound.** Watch fired at M2 c=8 C=256: U-EA − EA +0.035
[+0.010, +0.065], Δp_g +0.0083, Δunits +6.37. Then at M2 c=8 C=512: **+0.190
[+0.130, +0.250], Δp_g +0.0836, Δunits +15.17.** Same cell: SnapKV +0.035 (n.s.), AdaKV
+0.010, KeyDiff −0.030 (Δp_g −0.0799).

**Diagnosis** (`DIAGNOSIS_EA.md`, `dee8add`). Not a bug: identical `_U` code path;
singleton identity 40/40 for ExpectedAttention. ExpectedAttention's score is
`softmax(·)×||V||`, heavy-tailed and not position-smoothed (SnapKV is avg-pooled).
Keep-sets differ in kind: at c=8 C=512 U-EA moves ~28 tokens/slot from non-record text
into record lines (+2.8 gold tokens), 12% of keep-set replaced; U-SnapKV keeps gold/record
/other counts unchanged while completion doubles. Across 12 admissible cells Δ_U vs Δp_g
r = **+0.944**. Conclusion: for ExpectedAttention the U-gain is mostly extra gold
retention, not fragmentation removal. SnapKV/AdaKV carry the fragmentation claim;
ExpectedAttention/KeyDiff reported confound-tagged with Δp_g and Δunits always shown.

## 4. Mechanism survey C1–C6

Rules committed before testing (`bf05e1b`).

| | Candidate | Result |
|---|---|---|
| C1 | Head/slot disagreement | **AMBIGUOUS.** Regression supports it (M2 +0.054 [+0.042, +0.067]; M3 +0.048 [+0.029, +0.067]) but U-X agreement is *slightly higher* than X (0.481 vs 0.475), so it can't explain X→U-X. Script printed REAL by implementing half the rule; written rule governs. |
| C2 | Denominator artifact | Fired REAL on a mis-specified rule (ratio of means, denominators ~0.04–0.09). Not trusted. |
| C3 | Query adjacency | **NOT IT, contradicted** (M2 +0.032, M3 +0.065, wrong direction) |
| C4 | Scaffolding retention | **NOT IT** |
| C5 | Gather-order hygiene | proposed, became A6 |
| C6 | Surface-form integrity | proposed, not run |

## 5. Candidate screen A1–A9

All rules committed before any test (`CANDIDATES.md`, `3156e6c`).

CPU: **A7** budget-gated policy — unscorable by construction (a gate selecting one of two
arms cannot exceed both). **A8** unit composition — NOT IT (M2 −0.085, wrong sign). **A9**
layer-selective — NOT IT (2.16× on M2, 1.91× on M3 against a >2× bar on both).

GPU, M2, c=8, C=512, n=20 (CI half-width ≈ 0.10, so NOT-IT means "no large effect"):

| | Arm | Result |
|---|---|---|
| A1 | head-consensus | +0.000 [−0.062, +0.062] |
| A2 | early-prior | −0.013 |
| **A3** | **floor-hybrid** | **0.237 vs floor 0.237, +0.000 [−0.100, +0.100]; ≥ U-snapkv** |
| A4 | supra-units | −0.062; sub-units +0.000 |
| A5 | sink32 | −0.037 |
| A6 | gather-order | **REAL — hygiene violation** |

Arm means M2: floor 0.237, hybrid 0.237, ascend-snapkv 0.225, U-snapkv 0.212, snapkv 0.200.

## 6. The gather-order confound (A6) and Amendment A4

Generation deterministic on every arm (10/10 identical reruns, both models) — not noise.
`snapkv` vs `ascend-snapkv`: keep-sets identical 20/20, generations identical only 10/20
(M2), 11/20 (M3). Cause: bf16 accumulation order along the key axis in sdpa flips
near-tied logits; greedy decoding diverges. **X gathered in score order, U-X in ascending
position order** — every X vs U-X contrast also differed in cache order.

At c=8: order-only M2 +0.025 [+0.000, +0.062]; allocation-only (U-snapkv vs
ascend-snapkv) M2 −0.013, M3 +0.013 — collapsed. At c=40 preview: order-only +0.000 on
both models; allocation-only M2 +0.050 [+0.000, +0.113], M3 +0.000.

**Amendment A4** (digest `843c788c…`, superseding `34a20a7e…`): wrapper gathers in
descending score order, matching `ScorerPress`. Non-claim-bearing cells quarantined, not
re-run. Decision rule committed before the result (`DECISION_C40.md`, `6cb4b11`): proceed
only if the allocation-only CI excludes zero positive on at least one model **and** point
≥ +0.022; if models disagree, report both and hold.

## 7. Post-fix pilot (c=40, C=512, n=50, gather=score)

| pair | M2 Δ_U | M3 Δ_U |
|---|---|---|
| snapkv | **+0.035 [+0.010, +0.065]** | +0.000 [−0.020, +0.020] |
| adakv_snapkv | **+0.045 [+0.010, +0.080]** | +0.010 [−0.020, +0.040] |
| expected_attn | **+0.035 [+0.015, +0.060]** | −0.005 |
| keydiff | +0.005 | **+0.025 [+0.005, +0.045]** |

M2 floor 0.170, oracle 0.695; M3 floor 0.105, oracle 0.240. **Δp_g negative on every M2
pair** (−0.012 to −0.055) while accuracy rose — the gain is arrangement, not retention.
Verdict HOLD (models disagree). U-X still loses to floor (M2 0.080–0.090 vs 0.170).

**Power check.** M3 power to detect an M2-sized effect: snapkv 0.93, expected_attn 0.95,
adakv 0.68. Cross-model gap: snapkv +0.035 [+0.000, +0.070], adakv +0.034 [−0.015,
+0.080] — not significant. Data do not separate a shared effect from model-dependence;
~n=100/model would.

Track 1 summary committed (`7108998`).

## 8. Track 2 — reading the failures

141 disagreement cases (floor correct, U-X wrong, queried fact complete in ≥1 slot —
"every slot" was unsatisfiable and corrected, reported).

Initial 24-case sample: M2 contamination 6/12, hallucination 4, misattribution 1; M3
refusal 7/12, misattribution 3.

**Selection check** (`c122dab`): the "confidently wrong" filter hid M3's contamination
(refusals are zero-overlap by construction). Full population: **M2 contamination 17%
(14/82), refusal 0%; M3 refusal 31%, contamination 5% (3/59).**

## 9. The spacing intervention (contamination fix attempt)

Hypothesis from Track 2: adjacent complete records cause the model to answer with the
neighbor's content. Prereg `PREREG_SPACING.md` (`31f318d9`, `9c181c9`): K=24, ±1
completion band, contamination-tagged accuracy only, per-model verdicts, floor of 20
tagged queries. Position-order adjacency (RoPE-relevant), not cache write order.

| Attempt | Scope | Pooled M2 / M3 | Verdict |
|---|---|---|---|
| 1 | SnapKV, n=50 | 4 / 1 | NOT SCORED |
| 2 (S1, `1201f616`) | + ExpectedAttention | 9 / 1 | NOT SCORED |
| recon (`e1f7cac`) | per-cell rates | contamination exists **only at c=40** | — |
| 3 (`1da3fe4`) | n=120 M2 / 150 M3 | 16 / 3 | NOT SCORED |
| 4 (S2, `893ca661`) | + AdaKV (cross-head allocator) | **26** / 9 | **NOT-IT, M2** |

AdaKV contamination rate checked before building: 21.7% (vs SnapKV 14.8%, EA 15.6%,
KeyDiff 0%). Final (`3d06852`): spaced 0.000 vs unconstrained 0.000 on all 26 M2 queries.
All confound checks clean (Δp_g negative on every arm). Relaxation heavy (M2 35–41
per instance; M3 AdaKV 152).

**Read-through of the 26:** 6 still contaminated, 20 degraded into weak/distant matches
(exemplar bleed, records 20+ away). **Gap recovery** (`172db42`): the 6 residual cases had
median achieved gap **2 tokens** (relaxation re-admitted the neighbor); the other 20 had
median **44–98 tokens**. So spacing genuinely held in 20 cases and still recovered nothing;
in 6 it was never applied.

## 10. Hallucination and calibration

**Classification of 35 M2 hallucination cases:** 25 plausible-fabrication (full 8-field
schema, all values invented), 9 malformed (7 truncation, 2 degeneracy), 1 mixed. All 9
malformed and all 33 M3 refusals stop via `eos`, not the length cap.

**Headroom arithmetic (M2, c=40, C=512).** Floor: accuracy 0.170, completion 0.285,
conversion 0.5965. U-X (SnapKV/AdaKV): completion 0.2322, conversion 0.3658.
- Completion alone (conversion fixed ~0.40): need completion 0.425 = **1.83×** current,
  1.49× floor's own completion.
- Joint (conversion rises to 0.60): need 0.283 = **1.22×**.

**Calibration test.** Logits were never stored; re-extracted. **23/25 fabrications diverge
at generated token 0.** Paired top-1 − top-2 margin (fabrication − matched correct), M2
n=24: **+0.339 [−1.536, +2.156] — no calibration signal, wrong direction.** M3 n=6:
−2.094 [−4.208, −0.406], reported as underpowered only.

Consolidation committed (`1380326`).

## 11. M3 refusal is completion-dependent

Rule committed (`adb5c2c`). First run NOT SCORED (contamination sub-requirement
structurally unsatisfiable; refusal 25/8). Correction (`9168cf8`, fix `a116804`) still
NOT SCORED (8 < 10). Additional on-disk data (instances 50–149): 183 cases, bins 69/114,
refusals 36/32. **Refusal share low − high completion: +0.241 [+0.096, +0.384] — REAL.**

## 12. Contamination: the base scorer investigation

20 genuinely-separated contamination cases.

- Field similarity 8/8 in all 20 — corpus constant, uninformative.
- **Base scorer ranks the substituted record above the queried one in 18/20**, before any
  allocation.
- Document order: no direction (12/20 vs 8/20).

**Recency** (`6d65f79`): position–score Spearman ρ = +0.61 across all ~41 records per
instance. Both records in the top half 20/20. **After position detrending, substituted
still wins 18/20.** Recency sets up rivals; something residual picks the winner.

**Five candidate explanations** (`f45f855`): token length null (r=+0.19); numeric
magnitude null (r=+0.01); earlier-query compounding structurally impossible (cache clone
per variant); query–filler lexical overlap null (r=−0.36, wrong direction); record-ID
stickiness r=+0.709, then +0.668 on residuals — but only 9 record IDs, all at indices
27–39.

**Slot-identity probe** (`SLOT_IDENTITY_PROBE.md`, `12997c1`; result `52095cc`): record
slot was a deterministic function of index (`ledger_c.py:170-187`), so the 9 IDs had never
appeared outside the tail. Permuted slots, 20 fresh instances: **r = −0.099 — NOT-IT,
weakly reversed.** Stickiness was a tail-detrending artifact.

Accuracy ceiling (computed before the probe): a 100%-effective contamination fix → snapkv
0.080→~0.213, adakv 0.090→~0.223, EA 0.035→~0.168 vs floor 0.170.

**Contamination status: unsolved; residual cause unidentified; context file committed
(`CONTAMINATION_UNEXPLAINED_CONTEXT.md`, `230039a`).**

## 13. Hallucination: the attention finding

One extraction pass, 19 matched pairs (AdaKV excluded — eager mode unsupported).

| Test | Measure | Result |
|---|---|---|
| **A** | attention mass on queried record's span | **−0.0122 [−0.0170, −0.0074] — REAL** |
| B | attention entropy, full context | +0.0175 [−0.0018, +0.0371] — NOT-IT |
| C | attention on record-ID token | +0.0009 [−0.0002, +0.0019] — NOT-IT |

**Position check:** pooled r = −0.370 (p=0.022), within-pair r = −0.360 (n.s.);
fabrication records cluster at positions 45–54.

**Position-detrended Test A** (n=18): raw −0.01341 [−0.01795, −0.00884]; **residual
−0.01337 [−0.01744, −0.00935]** — essentially unchanged. **Confirmed: fabrication cases
under-attend to the queried record's own content, independent of position.**

## 14. Current scoreboard

| Thread | Status |
|---|---|
| Fragmentation (M2) | **Confirmed**, arrangement-not-retention |
| Gather-order confound | Found, fixed (A4) |
| Contamination | Diagnosed, **unsolved**; spacing rejected; residual cause unknown |
| Hallucination | **Mechanism confirmed** (attention deficit on queried content), no fix yet |
| Calibration | Rejected |
| M3 refusal | **Completion-dependent, confirmed**, mechanism unknown |
| Beat or match floor | **Not achieved.** Only A3 floor-hybrid tied floor (n=20) |

## 15. Planned and never run

From the Paper 4 plan v2, displaced by the mechanism search: **T3** benchmark fact-cost
characterisation (LongBench/RULER/SCBench/NIAH), **T4** crossover localisation
(c ∈ {3, 5, 12}), **T5** length sweep (L ∈ {2048, 4096, 8192}), **T6** deployment-cost
measurement, and the natural-text task. Also paused: the Stage 3 c=19/c=40 grid.
