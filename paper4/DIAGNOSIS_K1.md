# K1 FIRED — the c = 1 control failed. Diagnosis.

**Status: Stage 3 halted after the c=1 sweep, before any c=8/19/40 cell, exactly as
`PREREG_P4_S3.md` §6/§7 K1 requires. The wrapper has NOT been modified. No further run is
launched until this is resolved by a human decision.** Produced on the RTX 5070 (Machine N).

Data: 5,000 rows (2 models × 5 budgets × 10 arms × 50 instances), 0 failures, all cells
complete and single-session.

## 1. What fired

4 of 40 registered comparisons have a 95% CI excluding zero, **all positive (U-X > X), none
negative**:

| model | C | pair | Δ_U [95% CI] | Δ p_g | Δ units_complete |
|---|---|---|---|---|---|
| M2 | 32 | U-KeyDiff − KeyDiff | +0.020 [+0.005, +0.040] | +0.0087 | +0.12 |
| M2 | 512 | U-ExpectedAttn − ExpectedAttn | +0.020 [+0.005, +0.040] | −0.0003 | +5.00 |
| M3 | 64 | U-ExpectedAttn − ExpectedAttn | +0.050 [+0.025, +0.080] | +0.0119 | +0.09 |
| M3 | 128 | U-KeyDiff − KeyDiff | +0.050 [+0.020, +0.085] | +0.0200 | +0.55 |

**Not multiplicity alone.** 4/40 at 95% is above the ~2 expected, every exclusion is positive,
and the pooled per-instance effect over all 20 comparisons is `+0.0083 [+0.0020, +0.0145]` on M3
(M2 `+0.0043 [−0.0012, +0.0095]`, includes zero). Small — ~1.7 extra correct answers per 200
queries on M3 — but real on one model.

## 2. Why: the control's premise was false, and the prereg said so in §6

§6 asserts "at c=1 a queried fact is a single token, so **unit-awareness must be inert**". That
inference does not hold for MARK-1. **Only the queried fact is one token. The 36 distractor
entries are surnames of several tokens each**, so units still exist and unit-aware allocation
still differs from top-k — `STAGE12_RULES.md` §2 stated exactly this ("c=1 is a **control**
rather than an identity"), and the Stage 3 prereg's §6 wording over-claimed it into inertness.

Two measured channels, both pure allocation effects, neither of them "keeping the queried fact
whole":

1. **Gold retention itself changes.** A multi-token surname competes on its **mean** score under
   U-X but on its **best tokens** under X. A surname holding one high-scoring token is taken by
   X and skipped by U-X, freeing budget that falls to other singletons — sometimes the gold
   token. Measured Δ p_g at c=1: ≈ ±0.001 for SnapKV/AdaKV, but **+0.008 to +0.021 for KeyDiff
   and ExpectedAttention** — exactly the two methods that violated. Their scores vary far more
   within a line than attention-based scores do, so mean-vs-max diverges most for them.
2. **Context structure changes.** U-X completes more whole distractor entries at equal budget
   (Δ units_complete up to **+5.0** at C=512). The M2 C=512 ExpectedAttention violation has
   Δ p_g ≈ 0 with Δ units_complete = +5.0, so that cell's gain is not gold retention at all.

**The genuinely inert case passed.** The singleton-unit identity — every unit forced to one
token, where U-X must reproduce X's keep-set — passed **40/40** in this session's preflight
(2 models × 5 budgets × 4 methods), on real scores. That is the structural invariant; MARK-1
c=1 is not it.

## 3. Why Stage 2 did not catch it

Stage 2's c=1 control ran **C=512 only**, and only **SnapKV and AdaKV** — the two methods whose
Δ p_g is ≈ 0. It never generated ExpectedAttention or KeyDiff at c=1, and never at tight budget.
Stage 3's first act was to widen exactly that cell, and it found the hole.

## 4. What this does and does not threaten

- It does **not** invalidate Stage 2's c≈40 result: there the compared arms are SnapKV/AdaKV,
  whose c=1 Δ p_g is ≈ 0, and the effect sizes are 0.045 against ≤0.005 here.
- It **does** mean "U-X vs X isolates fragmentation" is too strong as stated for KeyDiff and
  ExpectedAttention: for those scorers the wrapper also shifts which single tokens survive. Any
  Stage 3 claim from those two arms would have carried this confound unlabelled.
- The correct reading of the c=1 cell is **not** "the wrapper is broken" but "MARK-1 at c=1 is a
  weak control, because the distractors are still multi-token units".

## 5. Options — for the human, not to be chosen unilaterally

1. **Re-specify the control and amend the prereg** (before any c>1 row): make the registered
   inertness test the **singleton-unit identity** (already passing 40/40), and demote MARK-1 c=1
   to a reported diagnostic with its two measured channels. Cost: one amendment, no re-run; the
   c=1 rows already exist and stay.
2. **Keep K1 as written and report Stage 3 as killed at the control.** Defensible and maximally
   conservative, but it stops a 9.6 GPU-hour grid over an effect of ≤0.05 on 4 of 40 cells that
   the diagnosis explains.
3. **Narrow the arm set** to SnapKV/AdaKV (Δ p_g ≈ 0 at c=1) for the fragmentation claim, and
   report ExpectedAttention/KeyDiff separately with the confound named.

**Not an option: changing the wrapper to pass the control.** Nothing in `p4/unitwrap.py` has been
touched, and the failing behaviour is the wrapper doing what it was specified to do.
