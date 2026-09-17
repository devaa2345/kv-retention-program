## [GAP-U] revisited: the ceiling gap is a scoring-convention difference

**The scorer was not changed.** This reports exactly what it does and measures how much of
the 0.799-vs-0.947 gap each alternative convention would close.

### What the scorer does

`score_turn` returns `target.value in generated_text` — a plain substring test of the full
17-character value (`sk-` + 14 lowercase hex) against the **raw** decoded text of that turn
(`skip_special_tokens=True`, nothing else). No lower-casing, stripping, whitespace
collapsing, unicode normalisation or tokenisation. Credit only on the turn that asked
([GAP-P]). Prompt score is k/6. This is the literal reading of §2 and of [SPEC-GAP 2]'s
stated default.

### What each convention rescues (arm 6, 900 turn judgements, n=150)

| scoring rule | accuracy | vs 0.947 |
|---|---|---|
| **1. exact 17-char substring — the actual scorer** | **0.7989** | −0.1481 |
| 2. case-insensitive substring | 0.7989 | −0.1481 |
| 3. substring after whitespace stripping | 0.7989 | −0.1481 |
| 4. 14-hex payload, `sk-` prefix optional | 0.8633 | −0.0837 |
| **5. payload within edit distance 1** | **0.9544** | **+0.0074** |
| 6. payload within edit distance 2 | 0.9722 | +0.0252 |

### What this establishes

**The normalisations [SPEC-GAP 2] contemplates rescue nothing.** Case-folding and
whitespace-stripping recover exactly zero failures, so "exact vs normalised match" is not
the axis the gap lies on. That eliminates the ambiguity the spec itself flagged.

Two other conventions do close it. The model frequently answers with the bare 14-hex
payload and omits the `sk-` prefix — e.g. `9d268ef50038b6` for `sk-9d268ef50038b6`. This is
a per-prompt behavioural mode, not scattered noise: 12 of 150 prompts do it on ≥3 of their 6
turns, accounting for 58 of 181 failures. Allowing a single character error in the payload
recovers a further 82. Together they land at **0.9544 against the reference's 0.947 — a
difference of 0.0074**, an order of magnitude below the 0.1481 gap under strict scoring.

### Conclusion, with its limits

The evidence favours a **scoring-convention difference over a capability difference**. A
0.947 ceiling is hard to reach under strict 17-character exact-substring scoring — this
model emits the correct payload far more often than this scorer credits — but sits almost
exactly on a prefix-tolerant, edit-distance-1 convention.

This is evidence, not proof: edit-distance tolerance is an unusual convention, §2's text
does not license it, and the reference's code cannot be inspected under the blind protocol.
What can be said firmly is that **the two implementations are probably not scoring the same
thing, and the axis is prefix-tolerance plus fuzzy matching — not the exact-vs-normalised
axis the spec anticipated.** §2 should state whether the `sk-` prefix is required and
whether any edit tolerance is permitted.

**The scorer remains unchanged; every other number in this document uses rule 1.**

