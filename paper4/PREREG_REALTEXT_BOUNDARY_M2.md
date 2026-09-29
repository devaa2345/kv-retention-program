# M2 code-boundary diagnostic on real-text mix

This is a new, **narrow exact-string lookup** experiment following
`REALTEXT_POINTER_M2_RESULT.md`. It does not make a general real-text QA claim.
On the previous 64-instance test, the lexical pointer chose the gold sentence
in 162/256 queries but the literal mask was right in only 71 of those. A
post-hoc character-interval audit found 60/91 wrong-boundary or overlapping
copies (38/41 needle failures) and 31/91 disjoint wrong spans (28/50 SQuAD
failures). Those spent test rows motivate this hypothesis but are **not** used
to choose thresholds or score the new rule.

## Frozen rule and ceiling before answer generation

The pointer remains the exact lexical BM25 rule in
`out/_realtext_pointer_screen.py`. For a question containing the whole phrase
`access code` (case-insensitive), inspect its selected eligible sentence. If
it has exactly one isolated digit run of length **4–12**, the new mask allows
only prefixes of that entire run and allows EOS only after all its digits.
It preserves exact case/whitespace and emits a literal substring of the
selected sentence. The fixed 4–12 range is a generic code-length guard, not
the gold answer length; the rule receives no gold, sentence label, or future
answer tokens. If the trigger or uniqueness test fails, use the unchanged
literal_v1 sentence-substring mask. SQuAD is left unchanged by design.
The code in `out/_realtext_boundary_mask.py` is frozen before generation.

A fresh 32-instance development set (indices 200–231, 128 queries) was built
before any answer generation; SHA-256
`c01d59cf1feb9df78eeb159cc42f16078f916deac1ce626b5d899478c00864e3`.
The generic builder labels all indices >=24 as `heldout`, but this range is
designated development here. Prefill-only M2 SnapKV capture took 0.74 GPU
processing minutes after load. SnapKV held the gold sentence in 89/128;
the frozen lexical pointer hit 79/128 (31/64 SQuAD, 48/64 needles). Among
the 64 needle queries, 57 selected sentences have a unique 4–12 digit run;
all **48/64** correct-pointer needles have exactly the gold code as that unique
run. Therefore the rule's development-set **needle absolute ceiling is
48/64 = 0.75** (wrong selected sentence cannot contain the unique gold code).
The mixed-set ceiling is at most 79/128 = 0.6172 if SQuAD extraction were
also perfect; this is not a prediction. CPU unit tests on all 64 real needle
gold sentences reproduce the gold, prohibit EOS on incomplete codes, and
permit EOS after the full code. No answer generation has run at this point.

## Development generation and new confirmation

On development indices 200–231, run M2 RTX 5070 with the same A1 prompt,
0.2802 budget and strict whole-answer exact scoring (casefold and whitespace
normalization only). Paired arms: floor_pos unmasked; SnapKV unmasked;
SnapKV + attention-pointer literal_v1; SnapKV + lexical-pointer literal_v1;
SnapKV + lexical-pointer code-boundary rule with literal_v1 fallback. All
SnapKV arms must have identical keep-set hashes, budgets, context lengths,
and held status. Every masked output must be a literal substring of its
selected sentence; log code-rule activation and fallback counts. This is a
development pilot only.

Proceed to a new independent confirmatory set only if the boundary-rule arm
beats unmasked SnapKV on development and the rule activates on at least
32/64 needle queries. Compute per-instance variance of boundary-rule minus
floor accuracy on development, with four query outcomes averaged per instance.
For a 10-point target, set
`N_power=ceil((1.96+0.84)^2*s^2/0.10^2)` and
`N=min(128,max(64,N_power))`; disclose if N_power>128. Build fresh indices
starting at 232 and lock their hash, N, and rule before any confirmatory
generation. Never tune using prior indices 24–103 or 136–199.

Primary new contrast: boundary-rule SnapKV minus floor_pos on the **mixed**
set, using a 95% paired bootstrap of complete four-query instances with
10,000 resamples. A mixed exact-string lookup win requires its lower CI bound
>0. Prespecified secondary contrasts: boundary-rule minus lexical literal
mask (isolating the code rule), boundary-rule minus own unmasked SnapKV,
and lexical literal mask minus floor. Report SQuAD and needle results and
their paired intervals separately even if the mixed primary wins. Do not
claim broad prose-QA improvement unless SQuAD versus floor separately has
lower CI bound >0. Report conversions/broken cases, code-rule activation,
retention/pointer hit/held-but-wrong, cap stops, invented full-context
values, and GPU processing time. No M3 or RTX Pro 5000 generation.

## Development gate and independent held-out lock

Development indices 200–231 completed on all five arms. The confound and
literal-output checks passed. Correct counts were floor_pos 32/128, SnapKV
28/128, attention-mask SnapKV 35/128, lexical-mask SnapKV 42/128, and
boundary-rule SnapKV 58/128. The code rule activated on 57/64 needle queries
(fallback on 7), and boundary-rule accuracy exceeded own SnapKV accuracy
(58 > 28). Both preregistered gates therefore passed. The development
boundary-rule minus floor difference was +26/128 = +0.203125; its descriptive
paired instance CI was [+0.125, +0.28125]. The code-rule gain over the
lexical literal mask was +16/128. Development SQuAD scored 10/64 for boundary
versus 9/64 for floor; needles scored 48/64 versus 23/64. Thus any projected
mixed benefit is driven by the synthetic code-lookup half.

The per-instance variance for boundary-rule minus floor was 0.05821573.
The locked formula gives `N_power=46`, so the minimum sets **N=64 instances**
(256 queries). A new independent held-out set uses indices 232–295. It was
built without outcome inspection and has JSONL SHA-256
`439c9958d2011251abaac5755f9926a0e7eb4e0021d1565b463c8cd0d397e202`.
The prompt, pointer, trigger, digit-run range, mask, budget, scoring, and
claim scope are unchanged for this run. Development generation consumed
1.95 measured GPU processing minutes after model load; the estimated
held-out processing cost is about 3.9 minutes at the same per-instance rate.
