# Real-text pilot — short prereg (diagnostic only, committed before generation)

**Purpose**: check whether the copy-mask mechanism (attention-guided, text-level, literal-span
copy constraint) transfers from LEDGER-C's synthetic key-value records to real prose, where the
"span" is a sentence rather than a schema'd record. **Diagnostic only** — not a confirmatory test,
no REAL/NOT-IT decision rule, no bootstrap CI required. n=15-20 reflects that.

## Construction

- Source: SQuAD v1.1 (`rajpurkar/squad`, validation split).
- Each case: one target SQuAD (context, question, answer) triple, embedded at a **random
  position** among several other, unrelated SQuAD passages used as filler, concatenated to a
  target length of ~2–4k tokens (Qwen2.5-3B tokenizer).
- Model: M2 (Qwen2.5-3B-Instruct) only — diagnostic, not run on both models.
- Seeds: CRC32 of the case index, for reproducible filler selection and insertion position.

## Arms

- `floor_pos` (unmasked): press retains sink(8) + a middle budget + recency window(64), same
  floor mechanism as LEDGER-C.
- `floor_pos + schema-free mask`: same press, decoding constrained to a literal substring of the
  **top-attended sentence** (by the same attention-mass reconstruction used throughout tonight —
  `mass_by_rid_from_attn`, applied to sentence spans instead of record spans).

No schema-ordered arm (prose has no field structure to order).

## Budget — "at the LEDGER ratio"

LEDGER-C's own retained fraction: `B = C + n_sink + n_window = 512 + 8 + 64 = 584` out of
`n_ctx ≈ 2084` → retains ~28.02% of context. For the pilot, this **fraction** is held fixed
(not the absolute token count `B`, since pilot contexts vary 2–4k tokens): for each instance,
`B_pilot = round(0.2802 * n_ctx_pilot)`, `C_pilot = B_pilot - n_sink - n_window`, same
`n_sink=8, n_window=64` floors as LEDGER-C.

## Metrics reported

- **Span-hit rate**: top-attended sentence == the sentence containing the gold answer,
  **conditional on that sentence being held whole** in the floor's keep-set (same conditioning
  used for every LEDGER match-rate figure tonight — an unconditional rate is a denominator
  mismatch when the answer sentence can be evicted).
- **Conversion**: floor_pos wrong -> masked right.
- **Breakage**: floor_pos right -> masked wrong.
- **Invented**: masked output not a literal substring of the top-attended sentence (should be
  ~0 by construction of the mask); unmasked floor_pos output not a literal substring of ANY
  sentence in the context (a real "unconstrained hallucination" check, unlike the masked arm).
- Scoring: SQuAD-style — generated text counted correct if it contains the gold answer string
  (case/whitespace-normalized substring), consistent with LEDGER-C's own substring-based scoring
  convention.

## Scope

Diagnostic only. A positive result here suggests the mechanism is not LEDGER-specific; a null or
negative result does not retract the LEDGER-C confirmatory finding, which stands on its own
prereg and decision rule. No claim beyond "does this transfer at a glance" is made from n=15-20.
