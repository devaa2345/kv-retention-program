# H-ORTH factorial, our half (Paper 1, Option B item 3)

**What was run.** Our tier-seating rule (`cache/protected_split_tier.py`: protection decides retention, the
promotion signal decides FULL vs QUANT among retained) against two distractor formats.

- **Cell A, our format** (existing, `results/phase2_4bit/`): distractors are short values of mixed type
  (`cache_ttl: 37.18`, `log_level: ugiwtb`).
- **Cell B, AMD format** (new, this run): all 26 lines share the credential shape
  `LABEL_i_ID: sk-<14 hex>`, with AMD's label set (`amd/kvre/task.py`). The hex alphabet is our own
  (`string.hexdigits.lower()`) so that only the distractor format differs. Implemented as
  `make_multi_credential_prompt(distractor_format="amd")`; the default path is unchanged and was checked to give
  byte-identical dump text.

Both cells: Qwen2.5-1.5B-Instruct bf16, 4-bit tier, budget 257, seeds 3000 to 3149, n = 150, `SCORE_MODE="sum"`,
promotion signals P1 attention, P2 epiphany, P3 random, P4 rotation, P5 oracle. AMD runs the reverse pairing.

**Context changes with the format.** Cell B contexts average 1,243.7 tokens against about 1,029 in cell A, so
budget 257 is 0.207 retention in B and 0.25 in A. The cells differ in retention fraction as well as in format.

Reference arms were also run in cell B, at the same budget and seeds, so P1 to P5 can be read against something:
R1 no protection, R2 structural protection, R5 retention oracle, R6 full cache, and BAND_full / BAND_quant (same
protected retention, all retained FULL or all retained 4-bit QUANT).

## Results

| | Cell A (our format) | Cell B (AMD format) |
|---|---|---|
| P1 attention | 0.203 | 0.102 |
| P2 epiphany | 0.216 | 0.083 |
| P3 random | 0.154 | 0.078 |
| P4 rotation | 0.184 | 0.084 |
| P5 oracle | 0.188 | 0.064 |
| rho(P1) / P2 / P3 / P4 / P5 | +1.000 / −0.077 / −0.050 / +0.002 / −0.068 | +1.000 / −0.188 / −0.029 / +0.038 / +0.090 |
| structural protection (arm 2) | 0.423 | 0.161 |
| no protection (arm 1) | 0.010 | 0.010 |
| retention oracle (arm 5) | 0.943 | 0.763 |
| full cache (arm 6) | 0.947 | 0.824 |
| band: all-FULL minus all-QUANT | 0.278 (earlier measurement, not re-derived here) | **+0.111 [+0.092, +0.129]** (all-FULL 0.161, all-QUANT 0.050) |

Paired contrasts, mean difference [95% CI], prompts differing (cell B, n = 150):

| signal | vs P1 attention | vs P3 random |
|---|---|---|
| P2 epiphany | −0.019 [−0.036, −0.002], 54 | +0.006 [−0.012, +0.023] |
| P3 random | −0.024 [−0.041, −0.008], 49 | n/a |
| P4 rotation | −0.018 [−0.033, −0.001], 50 | +0.007 [−0.008, +0.021] |
| P5 oracle | −0.038 [−0.054, −0.021], 58 | −0.013 [−0.030, +0.003] |
| P1 attention | n/a | **+0.024 [+0.008, +0.041]**, 49 |

Cell A for comparison (existing): P2 vs P1 +0.012 [−0.014, +0.039]; P3 vs P1 −0.049 [−0.078, −0.022]; P4 vs P1
−0.019 [−0.048, +0.009]; P5 vs P1 −0.016 [−0.046, +0.013].

## Findings

1. **Neither cell supports H-ORTH.** The orthogonal signal (P2) does not beat the circular one (P1) in either cell.
   In A it is level (+0.012, CI spans zero). In B it is slightly below (−0.019, CI excludes zero, but the effect is
   under two points).
2. **Moving our seating rule onto AMD's distractor format does not reproduce AMD's ordering.** AMD's corrected
   signals have P2 and P5 beating random and P1 not. In cell B, P1 is the one that beats random (+0.024), and P2
   (+0.006) and P5 (−0.013) do not. So the format difference alone does not account for the disagreement on the
   promotion question. This says nothing yet about the seating-rule axis, which is AMD's half.
3. **Promotion has less room under the AMD format, and everything is compressed.** The FULL-vs-QUANT band is 0.111
   against 0.278, and the five signals span 0.064 to 0.102 against 0.154 to 0.216. Structural protection collapses
   from 0.423 to 0.161 because protection now competes against 26 credential-shaped lines. The retention oracle
   still scores 0.763, so the gap between protection and the oracle is +0.602 here (0.520 in A). The main
   qualitative finding of Paper 1, that retention has far more headroom than promotion, holds and gets stronger.
4. **Two AMD-reported figures line up with our cell B numbers**: their structural-protection arm at 257 (0.166 in
   their notes, not re-derived) against our 0.161, and their full-cache ceiling (0.799) against our 0.824 with
   credential-shaped distractors (0.947 in our own format). Wrong-line retrieval among credential-shaped
   distractors is a plausible reason for a lower ceiling in their build, but this run does not isolate it.

## Caveats

- **P5 was not re-verified under the AMD format.** In cell A a check showed the oracle was never oversubscribed and
  put every retained credential token at full precision (`debug_p5_oversubscription.py`). Here P5 scores below P1
  (−0.038, CI excludes zero). Retention is different (fewer credential tokens survive, and 26 credential-shaped
  lines compete for protection), so the same check must be repeated before P5 is called a ceiling in cell B.
- Budget 257 corresponds to 0.207 retention in B, so the format and the retention fraction move together in this
  comparison. A retention-matched cell would need budget about 310 in B.
- Only budget 257 and only the 4-bit tier were run. Cell B's BAND arms use attention-ranked promotion, not the
  earlier standalone band script, so the band values are not from the same code path in the two cells.
- One model, one task. The ρ values for P2 and P5 differ from cell A (−0.188 vs −0.077, +0.090 vs −0.068); the
  manipulation is still near zero for the non-circular signals, but P5's sign and size changed and should be
  read with caveat 1.
- Two of AMD's figures above are AMD-sourced and were not re-derived.

Files: `raw_results.jsonl` (1,650 rows), `analysis.json`; code `kvcache_harness/run_phase2_amdfmt.py`,
`analyze_phase2_amdfmt.py`, `tasks/multi_credential.py` (new `distractor_format` option).
