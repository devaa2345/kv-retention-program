# M2 real-text 5070 result — focused floor versus SnapKV test

This is the preregistered Amendment 2 study in `PREREG_REALTEXT_5070.md`. Model: M2
Qwen2.5-3B-Instruct on NVIDIA GeForce RTX 5070. The amended-prompt full-cache pilot anchor
passed at 72/96 = 0.750. The 24-instance pilot set fixed the held-out size at 80 instances,
indices 24–103, 320 queries. The held-out data SHA-256 is
`05a517b13e267b76ae0a1c40daac7e6c063b980195b52ce7c9d55a8d82433323`.

Accuracy is strict whole-answer exact match after **only** casefolding and whitespace-run
normalization. Masked variants use the exact decoded-substring continuation mask. All variants
use the same A1 few-shot prompt and a 0.2802 retained-context ratio. All 320 queries completed
in each arm. Confound check was clean: SnapKV baseline and both masked variants had identical
captured keep-set hashes, budgets, context lengths, and gold-sentence retention on every query.
Both masked variants' raw answers were literal substrings of their selected sentences. The
corrected full-context invented-value check was used for every arm.

| Arm | Correct /320 | Accuracy | Full-context invented | Cap stops |
|---|---:|---:|---:|---:|
| floor_pos | 68 | 0.2125 | 199 | 3 |
| SnapKV | 50 | 0.1563 | 204 | 1 |
| SnapKV + literal mask, attention span | 71 | 0.2219 | 0 | 3 |
| SnapKV + literal mask, oracle span | 104 | 0.3250 | 0 | 0 |

All intervals below are 95% **paired instance-bootstrap** intervals (10,000 resamples of
80 complete four-query instances). Pilot results were excluded.

| Comparison | Difference | 95% CI | Conversion / broken |
|---|---:|---:|---:|
| SnapKV − floor_pos (unmasked method vs floor) | −0.0563 | [−0.1156, +0.0031] | 23 / 41 |
| SnapKV+attention mask − SnapKV (own gain) | **+0.0656** | **[+0.0250, +0.1094]** | **37 / 16** |
| SnapKV+attention mask − floor_pos (whole system) | +0.0094 | [−0.0438, +0.0594] | 31 / 28 |
| SnapKV+oracle span mask − SnapKV | +0.1688 | [+0.1250, +0.2188] | 55 / 1 |
| Oracle-span mask − attention-span mask | +0.1031 | [+0.0719, +0.1375] | diagnostic |
| SnapKV+oracle span mask − floor_pos | +0.1125 | [+0.0531, +0.1719] | nondeployable diagnostic |

The attention-selected mask passes the preregistered **own-baseline** rule: its positive CI
excludes zero, it has 37 conversions against 16 broken correct cases, and its unmasked
baseline has exactly 50 correct cases. Broken-correct rate is 16/50 = 0.320, with an
instance-bootstrap interval [0.208, 0.440]. Conversion rate among baseline-wrong cases is
37/270 = 0.137 [0.094, 0.183]. Thus the gain is real on the frozen **mixed extractive
set**, despite substantial breakage. The unmasked scorer did **not** beat floor, and the
deployable masked scorer did **not** establish superiority over floor.

The aggregate gain comes from the natural-filler needle half, not ordinary SQuAD questions:

| Kind, 160 queries each | floor_pos | SnapKV | SnapKV+attention | SnapKV+oracle span |
|---|---:|---:|---:|---:|
| SQuAD extractive | 25 | 23 | 17 | 37 |
| Needle in natural WikiText filler | 43 | 27 | 54 | 67 |

SnapKV's attention-mask own gain is −0.0375 on SQuAD [−0.0813, +0.0063] and +0.1688 on
needles [+0.1000, +0.2438]. The large oracle-minus-attention gap shows sentence localization
is one bottleneck; the negative SQuAD point estimate means the current deployable mask does
not establish transfer to ordinary extractive QA. These subgroup intervals are descriptive;
the preregistered claim is for the combined mix.

Floor held the gold sentence in 90/320 queries, with only 27/320 held-but-wrong, consistent
with the pilot headroom gate that omitted its mask. SnapKV held it in 216/320; the attention
selector hit it in 128/320, and 169/320 were held-but-wrong. The oracle-span variant is an
upper-bound diagnostic, not a deployable result. There was no M3 or RTX Pro 5000 run, no
budget tuning, and no change to `CONFIRMATORY_RESULTS.md`.

Measured held-out GPU processing time after model load was about 1.4 minutes for floor and
2.8 minutes for the three SnapKV variants, or about 4.2 minutes total. Raw 5070-labelled
results are in `out/realtext_5070_M2_heldout_results.jsonl`.
