# Exact-recall pilot — short prereg (diagnostic only, committed before generation)

**Purpose**: the SQuAD-style pilot found a boundary condition — the copy-mask mechanism adds
breakage without gains when the baseline model already answers retained prose correctly via
paraphrase. This pilot tests whether the mechanism helps when placed back in LEDGER-C's own
regime (a single implanted value the model cannot paraphrase around) but with **real prose as
the haystack** instead of synthetic key-value records — isolating whether LEDGER-C's result was
about "synthetic records" specifically, or about "an unparaphrasable exact value," independent of
whether the surrounding text is synthetic or real.

## Construction

- Haystack: real, long public-domain text (Project Gutenberg or similar; concatenated paragraphs
  to reach target length — same source-material spirit as the LEDGER filler text being replaced
  by something a model has plausibly seen in training, unlike SQuAD's already-QA-oriented
  passages).
- Needle: a randomly generated 6–8 digit passkey (or short alphanumeric code), inserted as a
  single sentence ("The access code is 483920.") at a **random position** in the haystack.
- Question: asks for the code directly ("What is the access code mentioned in the text?").
- Contexts: 1500–2200 tokens (matching the SQuAD pilot's own validated working range, avoiding
  its earlier OOM at 2–4k tokens).
- Model: M2 only, diagnostic.
- n = 20–30 cases, seeded (CRC32 of case index) for reproducibility.

## Arms

- `floor_pos` (unmasked): same floor mechanism as LEDGER-C and the SQuAD pilot.
- `floor_pos + schema-free mask` (escape-hatch FIXED, per the SQuAD round-2 fix — empty
  always-legal set, EOS only when the automaton allows it): constrained to the top-attended
  sentence's literal text.

Single span width (top-1 sentence) only — the SQuAD pilot's top-3 variant did not help and added
complexity; not repeated here.

## Budget

Same "LEDGER ratio" convention as the SQuAD pilot: retained fraction ≈28.02% (LEDGER-C's own
`B/n_ctx` at c=40, C=512), scaled to each instance's actual context length.

## Prompt

Extractive instruction included by default this time (validated as non-collapsing in the SQuAD
round-2 test): "Answer using ONLY the exact words copied verbatim from the text above."

## Metrics reported

- **Needle-sentence retention rate**: how often the sentence containing the code survives
  compression at all (unconditional — this is itself informative, since a 6-8 digit code sentence
  is short and could easily NOT be prioritized by floor_pos's own attention-based scoring).
- **Span-hit rate**, conditional on the needle sentence being retained (top-attended sentence ==
  needle sentence).
- **Conversion** (floor_pos wrong → masked right), **breakage** (floor_pos right → masked wrong).
- **Invented**: masked output not a literal substring of the top-attended sentence (expected 0,
  per the escape-hatch fix); unmasked output containing a digit string not present anywhere in
  the haystack (a real fabrication check, since a code is a distinctive, checkable string unlike
  SQuAD's free-form answers).

## Scope

Diagnostic only, n=20-30. A positive result (unlike the null SQuAD pilot) would suggest the
mechanism's value is specifically about protecting exact, unparaphrasable values regardless of
whether the surrounding text is synthetic or real — motivating a larger, separately-prereg'd
follow-up. A null result here would suggest LEDGER-C's result is more specific to its synthetic,
schema'd record structure than to "exact value forcing" alone.
