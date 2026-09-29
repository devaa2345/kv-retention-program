# M2 real-text question-aware pointer test

This is a new experiment following `REALTEXT_5070_M2_RESULT.md`. It does not
reinterpret the prior held-out test. Model M2, Qwen2.5-3B-Instruct, runs on the
RTX 5070 only. The prior A1 prompt, strict whole-answer exact match after only
casefold and whitespace-run normalization, 0.2802 retained-context ratio,
SnapKV and floor_pos retention, and literal_v1 substring mask remain fixed.

## Development completed before answer generation

The old 24-instance pilot (indices 0–23) was used to screen seven deterministic,
answer-blind sentence pointers. A separately built 32-instance development set
(indices 104–135; SHA-256
`3861e69c217a52a6edae2487580fb7c5a5100f002bc8f9b89ad04a6441ee638c`)
was then used for prefill-only validation. The generic data builder writes
`split=heldout` for all indices >=24, but these 32 are designated **development**
here, before any answer generation; the previous 80-instance test indices 24–103
are not used to choose the pointer. Captured candidate data hashes are
`59e1965de9316e16cec90634d69fa3e7f066f9159052a37f96845dc0e2e06de7`
(old pilot) and `ce6d1d89a8819415b9599cf86ffc96a34fd277fea4df8d3af6ecc2b028637cfd`
(new development). No model answer was decoded during this screen.

| Pointer | old pilot hit /96 | new development hit /128 |
|---|---:|---:|
| last prompt token attention | 39 | 49 |
| last 8 prompt tokens | 25 | 27 |
| question-token attention | 30 | 45 |
| lexical BM25 | **58** | **79** |
| last attention + lexical rank | 45 | 62 |
| last-8 attention + lexical rank | 42 | 57 |
| question attention + lexical rank | 46 | 64 |

The lexical rule is frozen in `out/_realtext_pointer_screen.py`: tokenize
lowercased alphanumeric words, remove its fixed stopword set and one-character
terms, score each completely retained eligible sentence by BM25 using the
current question and the eligible sentences' document frequencies, then take
the highest score (lowest original sentence index on ties). It receives no
gold answer, gold sentence, or future answer tokens. Eligibility is exactly the
original union-of-per-head-keeps definition. The selected sentence may still
be fragmented across heads; this limitation is unchanged from the prior test.

On new development, SnapKV held the gold sentence in 86/128 queries. Lexical
selection hit 79/128 (37/64 SQuAD, 42/64 needles); the previous attention
selector hit 49/128 (19/64 SQuAD, 30/64 needles). Therefore the measured
single-sentence literal-mask absolute accuracy ceiling is **79/128 = 0.6172**
for lexical selection, versus 49/128 = 0.3828 for attention selection. These
are ceilings, not predicted accuracies; the mask can select the wrong substring
even within the right sentence. The oracle-span arm from the prior test was
nondeployable and included 12 correct answers on unretained gold sentences;
its 104/320 score is not used as a recovery ceiling here.

## Pilot and locked follow-up rule

Run answer generation on the 32 new development instances only after this
preregistration is committed. Arms on every query: floor_pos unmasked,
SnapKV unmasked, SnapKV + old last-token-attention literal mask, and SnapKV +
frozen lexical-pointer literal mask. All share the A1 prompt and identical
retention budget. This development generation is for feasibility and variance,
not a confirmatory claim. Before scoring, verify the masked SnapKV variants
have identical keep-set hashes, budget, context length and gold retention to
the SnapKV baseline; every masked raw output must be a literal substring of
its selected sentence.

Proceed to a new independent held-out set only if the lexical mask beats its
own unmasked SnapKV baseline in the 32-instance pilot and at least 20/128 of
the pilot queries are held-but-wrong under SnapKV. Compute the per-instance
variance of lexical-mask minus floor accuracy, using each instance's four
queries. Size the new held-out test for a **10 percentage-point** target:
`N_power = ceil((1.96+0.84)^2 * s^2 / 0.10^2)`, then
`N = min(128, max(64, N_power))`. If N_power > 128, report the power limit
explicitly. The fresh held-out indices begin at 136 and may only be built
after this pilot gate; no prior result rows from indices 24–103 enter selection
or confirmatory scoring.

In the new held-out test, primary contrast is lexical-mask SnapKV minus
unmasked floor_pos. Claim a real-text win only if its 95% paired
instance-bootstrap interval (10,000 resamples of complete four-query
instances) has lower bound >0, and report its conversions and broken correct
cases. Secondary contrasts: lexical-mask minus own SnapKV baseline, old
attention-mask minus own SnapKV baseline, and lexical-mask minus old
attention-mask. Report SQuAD and needle strata separately; do not claim a
SQuAD gain unless its own interval excludes zero. Preserve per-query arithmetic,
gold retention, pointer hits, held-but-wrong share, invented-value check,
completion/cap stops, and measured GPU processing time. No M3 or RTX Pro 5000
answer generation is authorized by this preregistration.

## Development-generation gate and held-out lock

The frozen runner produced all 128 queries in all four arms on development
indices 104–135. Confound checks passed, both masked arms emitted only literal
substrings of selected sentences, and all 128 on-the-fly lexical and attention
selections matched the prefill-only predictions. Correct counts were floor_pos
21, SnapKV 24, attention-mask SnapKV 28, lexical-mask SnapKV 31. SnapKV held
86/128 gold sentences, with 64/128 held-but-wrong. Thus the two pilot gates
passed: 31 > 24 and 64 >= 20. Lexical-minus-floor was +10/128 = +0.078125
on this development set, with descriptive paired instance CI
[+0.0234375, +0.140625]. The planned floor contrast's measured per-instance
variance was 0.02998992. The preregistered calculation gives `N_power=24`,
and the minimum sets **N=64 instances (256 queries)**.

The independent confirmatory set is indices 136–199, generated without any
model outcome inspection; its JSONL SHA-256 is
`2f9c4b1aad0dd6d1a29ae9941a204bbe12a6a3b791c2e3cb27afc32b93bd588a`.
No pointer, prompt, mask, budget, scoring or decision-rule changes are allowed
after this lock. The development run used 1.66 minutes of measured GPU
processing after model load. At the same per-instance cost, the 64-instance
held-out run is estimated at about **3.3 GPU processing minutes** after load.
