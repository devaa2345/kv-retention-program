# Copy-constrained decoding — unit test finding, mechanism thread stopped

**Precision (part b): 15/15, clean, across all three implementation variants.** The substring
mask never permits a fully-emitted invented 6-digit number. This part of the mechanism works
exactly as designed.

**Recall (part a): 4/10 -> 2/10 -> 0/10, getting WORSE across three iterations, not better.**

1. Span includes the record-id prefix: 4/10. Failure: the model loops re-emitting the record-id
   token itself (`R030|R030|R030|...`) — a legal restart position after every delimiter.
2. Record-id excluded from the span (matches `ledger_c`'s own grading, which drops it): **2/10,
   worse.** New failure: looping on an already-copied field (`994313|994313|94|...`) — restart
   anywhere let the model jump backward.
3. Monotonic restart (copy left-to-right only): **0/10, worse still.** With backward jumps
   closed, the model mostly emits delimiter/whitespace tokens back-to-back until the token cap —
   the only tokens that stayed reliably "safe" and locally preferred once the search space
   narrowed.

**Stopped patching per instruction.** This is recorded as a finding, not a bug still being
chased: a hard per-token substring mask under plain greedy decoding has excellent precision
(never fabricates) but the model does not reliably navigate the legal path toward the *correct*
field sequence — it degenerates into legal loops instead. Whether a softer intervention (logit
bonus rather than a hard mask) avoids this is the open question the next steps are scoped to
answer, not assumed either way.

## Correction — this finding is CONFOUNDED, not confirmed

A later CPU-only check (token-ID alignment, no GPU) found that the constraint automaton was
built on TOKEN-ID identity between the record's own in-context tokenization and the
independently-retokenized generated text — and that alignment is **0/81 (0.0%)** on correct,
unbonused answers, almost entirely from digit-chunking (BPE re-tokenizes a short standalone
string with different boundaries than the same text got mid-document). This means the mask (and,
separately, the soft bonus below) was very often forbidding or failing to boost the model's own
CORRECT continuation tokens at digit/field boundaries, for a reason that has nothing to do with
whether attention found the right record. **The "greedy degenerates into legal loops" reading
above is confounded by this token-ID mismatch, not a clean finding about greedy search under a
sound constraint.** Superseded by a text-level re-implementation; see
`COPY_TEXT_LEVEL_UNITTEST.md`.
