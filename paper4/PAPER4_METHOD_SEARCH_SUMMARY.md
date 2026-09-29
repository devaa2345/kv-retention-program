# Paper 4 — Method Search: Consolidated Record

Factual record of the method-search work following the Stage 3 pilot (c=40, C=512, M2/M3). Every
number below traces to a specific committed result; this document synthesizes nothing new and is
the input for the next drafting pass, not a draft itself.

---

## 1. Fragmentation — confirmed

The wrapper mechanism (aggregate a press's per-token scores over units, allocate whole units,
all-or-nothing) works as designed. Post-A4 (gather-order fix; `p4/unitwrap.py`, commit `91fc795`,
re-frozen digest `0eeef01`), reproduced at the post-fix pilot cell (c=40, C=512, n=50, commit
`68dc570`, `PILOT_C40_POSTFIX.md`):

| pair | M2 Δ_U [95% CI] | M3 Δ_U [95% CI] |
|---|---|---|
| snapkv | +0.035 [+0.010, +0.065] | +0.000 [−0.020, +0.020] |
| adakv_snapkv | +0.045 [+0.010, +0.080] | +0.010 [−0.020, +0.040] |
| expected_attn | +0.035 [+0.015, +0.060] | −0.005 [−0.020, +0.010] |
| keydiff | +0.005 [−0.020, +0.030] | +0.025 [+0.005, +0.045] |

M2 clears three of four pairs; M3 clears only keydiff. Applying `DECISION_C40.md`'s pre-registered
rule (proceed only if ≥1 model clears with point estimate ≥ +0.022) to the cross-model contrast
directly: the M2−M3 gap itself is not significant (snapkv +0.035 [+0.000, +0.070]; adakv_snapkv
+0.034 [−0.015, +0.080]; commit `68dc570`, power analysis reported inline, not separately
committed as a file). **It remains statistically unresolved whether the model disagreement is real
or a power artifact** — M3's SnapKV/ExpectedAttn nulls were adequately powered (0.93, 0.95 to
detect an M2-sized effect) but AdaKV's was not (0.68).

On M2, the mechanism is arrangement, not retention: Δp_g is negative on **every** pair at this cell
(snapkv −0.0220, adakv_snapkv −0.0120, expected_attn −0.0204, keydiff −0.0548; commit `68dc570`) —
accuracy rises while gold-token retention falls, ruling out the confound that the wrapper wins by
quietly retaining more gold.

---

## 2. Contamination / spacing — rejected, well-powered, with a surviving mechanistic reading

**Finding the failure mode.** Track 2's read-through of the floor-correct/U-wrong/complete-in-≥1-
slot population (`TRACK2_FAILURE_READ.md`, commit `a500cb9`) found M2's dominant failure mode is
**contamination** (wrong answer matches a real, nearby, complete record — 6/12 sampled cases) and
M3's is **refusal** (7/12). The full-population check (`TRACK2_SELECTION_CHECK.md`, commit
`c122dab`) confirmed this at scale and corrected the initial sampled read: M2 contamination 17%
(14/82), M3 contamination 5% (3/59, not zero as the 12-case sample implied), M3 refusal 31% (18/59
in the sample check; the later full extraction under Task 1's scope found 33/... — see §3).

**The test.** `PREREG_SPACING.md` (commit `9c181c9`, K=24 fixed before data, hash
`31f318d9…`), amended twice before any AdaKV row: Amendment S1 (`814464e`, pools SnapKV +
ExpectedAttn contamination-tagged queries) and Amendment S2 (`2434918`, a cross-head-aware
allocator for AdaKV replicating `unit_keep_adakv`'s two-phase safeguard-then-global-top-k
mechanism, unit-tested including a forced-relaxation scenario before any GPU row). Scored at
n=120 (M2) / n=150 (M3), three arms pooled (commit `3d06852`): **M2 pooled contamination-tagged
accuracy 0.000, NOT-IT, CI [0.000, 0.000], 0/26 pooled queries correct.** M3 never cleared the
20-query floor (9/20 at n=150, commit `1da3fe4`), not scored, not chased further per instruction.

**The mechanistic follow-up.** A prefill-only GPU re-run recovered per-slot keep-sets for the 26
still-wrong M2 cases (`Track3`, commit `172db42`), tagging each by whether its wrong answer still
met the original near+confident contamination threshold (record-distance ≤2, word-overlap ≥6)
under spacing:

| category | n | median gap at complete-slot | median gap, all slots w/ match present |
|---|---|---|---|
| contamination-despite-spacing | 6 | 2.0 tokens | 4.1 tokens |
| other (weak/distant match, hallucination-like) | 20 | 44.0 tokens | 98.5 tokens |

The 6 residual cases are precisely the ones where the completion-matching band's relaxation clause
re-admitted a close neighbor the strict K=24 constraint had correctly excluded — separation was not
actually achieved there. **The 20 other cases are the genuine test of the hypothesis: real
separation (median 44–98 tokens) was achieved, and the model still did not answer correctly.**

**Surviving mechanistic reading:** the model shows no distance-sensitive record disambiguation.
Denied its nearest wrong answer by genuine separation, it does not recover the correct one; it
produces a more diffuse wrong answer instead — 20/26 cases moved from a confident near-neighbor
match to a weak/distant match or hallucination-like output (`TRACK2_CELL_RECON.md` context;
per-case table in the Track 3 conversation record, commit `172db42`). Spacing is rejected as a
sufficient mechanism, with the failure mode it targeted characterized rather than merely negated.

---

## 3. Calibration — rejected

**Classification.** M2's full hallucination-tagged population (35 cases, all four arms; commit
`fe5fbf7`), classified by objective schema rules (8-field structure, 6-digit value format, no
repeated blocks): **25/35 plausible-fabrication** (schema-conforming, fully invented values),
**9/35 malformed** (7 truncated, 1 decoding-loop repeat, 1 wrong-digit-count value field),
**1/35 mixed** (schema intact, one field populated with filler-sentence prose rather than an
invented value). Plausible-fabrication dominates >2:1 over malformed.

**The test.** Top-1/top-2 logit margin (raw logit gap, equivalent to the log-softmax margin) at
the first token of divergence from gold (23/25 cases diverge at token index 0; commit `7f169e2`),
paired per case against a completion-matched (±0.05 q_slot, same arm, nearest single match) correct
case from the same arm: **n=24 matched pairs on M2 (1 keydiff case had no match, excluded)**.
Result: **paired difference (fabrication − matched-correct) = +0.339, 95% CI [−1.536, +2.156]** —
CI includes zero, point estimate in the wrong direction (higher margin on fabrication cases, not
lower). Case margin was lower than its match in only 10/24 pairs — not even a majority.

**No calibration signal at this token position distinguishes fabrication from correct generation
on M2.** M3's parallel refusal comparison (n=6 matched pairs of 33 refusal cases; only 6 had any
completion-matched correct case within the tight band) gave paired difference −2.094, 95% CI
[−4.208, −0.406] — reported per the pre-registered ceiling on interpretation as "not obviously
inconsistent with the M2 pattern's direction," explicitly not a standalone claim about M3's
refusal mechanism (commit `7f169e2`).

---

## 4. What remains open — limitations, not failures

- **M3's refusal mode (31% of its disagreement population, 33 cases at full count) is untested as
  a mechanism.** It may not be a retention problem at all — it more closely resembles Paper 2's B9
  (`max_new_tokens` as a model-dependent instruction-following/format confound) than anything
  addressed in this search. No test in §§1–3 speaks to it beyond the underpowered n=6 margin check.
- **The 9/35 malformed hallucination cases on M2 are unexplained.** Truncation (7 cases) and
  decoding-loop repetition (1 case) point toward generation mechanics, not retention or confidence,
  but no mechanism was tested for them.
- **Plausible-fabrication's cause remains unidentified.** The search ruled out logit-margin
  confidence at the divergence token as the explanation — it did not identify what does explain it.
- The A6/A4 gather-order confound (`0df289f`, `91fc795`) was found and fixed as part of this
  search's own machinery, not as a method-search finding per se, but it is the reason the c=40
  reproduction in §1 is trustworthy rather than an artifact.

---

## 5. Net effect on the U-X vs floor_pos gap

**Zero of the tested mechanisms recovered any accuracy toward closing the gap.** Spacing: 0/26
pooled correct (§2). Calibration: no signal to build an intervention on (§3).

The headroom arithmetic (Task 2, this conversation) frames why conversion-side mechanisms were the
right place to search, even though both tried failed: at M2 c=40, C=512, closing the gap via
completion alone requires U-X's completion to reach 0.425 — **1.83× its current 0.2322, and 1.49×
higher than floor_pos's own completion rate (0.285)**. The joint scenario (conversion also rising
to floor_pos's 0.60) requires completion of only 0.283 — **1.22× current**. The arithmetic does not
by itself prove conversion must improve, but it shows the completion-alone path demands U-X
out-complete the very floor it is chasing, while the joint path needs a substantially smaller
completion gain. Spacing and calibration were both conversion-side candidates consistent with that
framing; both are now closed, negative results, reported as such.
