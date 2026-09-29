# CPU audit of M2 SQuAD failures after code-boundary test

This is a post-hoc diagnostic of the **128 SQuAD queries** in
`REALTEXT_BOUNDARY_M2_RESULT.md`, not a new confirmatory claim. The boundary
rule never activated on SQuAD, so its 27/128 SQuAD answers are identical to
the lexical-pointer literal-mask arm. Paired floor scored 19/128 and
unmasked SnapKV 26/128. Raw failures and selected sentence text are in
`out/realtext_boundary_M2_squad_failure_audit.jsonl`; the classification
script is `out/_realtext_squad_audit.py`. No GPU was used for this audit.

| Masked SQuAD error source | Count /101 wrong |
|---|---:|
| Lexical pointer selected a different sentence | **67** |
| Gold sentence selected; a disjoint wrong span copied | **19** |
| Gold sentence selected; answer copied with wrong boundaries | **15** |

SnapKV's gold sentence was held in 75/128 queries, but the lexical pointer
hit it in only 61/128. Of the 67 pointer misses among wrong masked answers,
14 had the gold sentence held and 53 did not. A perfect within-sentence
extractor cannot fix these 67 under the current single-sentence mask.

The 15 boundary errors are eight copies starting inside the gold span, six
starting correctly but ending early, and one partially overlapping it. None
were empty. A manual, qualitative subdivision of the 19 **disjoint** copies
(one label per case) found: six clearly wrong answer types (e.g. year for
"how many" or organization for "who"), four question-word echoes, four
wrong semantic roles within the right sentence, three short fragments, and
two different entities of the same broad type. These interpretive tags are
not a validated classifier. The disjoint errors do not have one dominant
simple pattern that justifies a rule tuned on this held-out set.

The mask **broke six correct unmasked SnapKV answers**, and all six had a
wrong sentence selected. It converted seven unmasked-wrong SQuAD answers to
correct, leaving a net +1/128 against its own SnapKV baseline. When the
pointer selected the correct sentence, the mask never broke an unmasked
correct answer in this sample. Trigger-gating could at most rescue the six
observed broken answers if it were perfect, while potentially losing some
of the seven conversions. The bigger losses are 67 pointer misses and 34
within-sentence extraction errors. This audit therefore does not justify a
trigger as the first SQuAD-specific modification.

The unchanged lexical-mask SQuAD arm led floor by +8/128 = +0.0625, with
paired instance-bootstrap CI [0.0000, +0.1250]; this is inconclusive.
Across the 64 complete instances, the paired **two-SQuAD-query** difference
had sample variance 0.05952381. A two-sided normal-approximation calculation
for 80% power at a six-percentage-point target gives
`ceil((1.96+0.84)^2 * 0.05952381 / 0.06^2) = 130` instances, or 260 SQuAD
queries. Because the next test would use the **unchanged** pointer and mask,
it can be run as an independent confirmation without developing a new
extractor on these spent outcomes. A new extractor, if needed, requires a
separate fresh development and test split.
