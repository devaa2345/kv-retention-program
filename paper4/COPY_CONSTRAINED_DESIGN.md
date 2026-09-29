# Attention-guided copy-constrained decoding — design, ceiling, prereg (nothing run yet)

## Item 2 — Design, written before any code

**Reframed by Item 1's finding**: max-attention-on-queried-record is ~98% in BOTH wrong (49/50)
and correct (79/81) cases — statistically indistinguishable. The intervention therefore cannot be
"point attention at the right record" (already true almost always). It must instead **force the
generation to literally copy from that already-correctly-identified span**, addressing a
copy/transcription failure downstream of correct targeting.

**Span selection**: exactly as measured — max attention mass over all retained records, at the
last prompt-token forward call (same computation as Test A / E3 / Item 4). **One span** (the
single top-attended record) — no multi-span extension attempted first.

**Allowed tokens during generation**: at every generation step, mask the logits to only:
1. Tokens whose surface text appears within the top-attended record's own retained token span
   (its record-id, both surnames, both site/dept/status words, both 6-digit numbers — literally
   copy-constrained to that line's own content).
2. A minimal fixed formatting set: the `|` separator token(s) and whitespace token(s) as they
   actually tokenize in this vocabulary (verified empirically before running, not assumed —
   BPE tokenizers often encode a leading-space variant differently from a mid-word variant, so
   both surface forms occurring in the record's own line are already covered by (1); the fixed
   set only needs to cover any tokenization variant NOT already present in the span itself).
3. The EOS token (required for `CM.generate_checked`'s existing stop condition to fire at all).

**Trigger**: **fires always**, not gated on the pre-generation signal. Item 1 is the reason —
the signal is ~98% present regardless of outcome, so gating on it would filter out almost
nothing and add a threshold with no real selectivity to tune (mirrors E3's own "nothing to
tune, ceiling already known" situation).

**What this does NOT fix, stated explicitly**: if the top-attended record IS the queried one but
the model copies the WRONG FIELD from within it (e.g. the second 6-digit number where the first
was asked for, or a site name where a status was expected), copy-constrained decoding does
**nothing** — every token in that wrong field is still a legal token under the mask, since the
constraint operates at the vocabulary level (which record), not the field-sequence level (which
position within the record). This is the residual failure mode any read of the result must check
for before concluding the mechanism, not just the record, is fixed.

## Item 3 — Ceilings, prereg

**Ceilings, per-query arithmetic (`1/(instances*4)`), discounted by the measured hit rate**:

| arm | own accuracy | wrong-but-held-whole (n, hit rate) | ceiling | gain over own baseline |
|---|---|---|---|---|
| floor_pos (120-instance pop.) | 0.1750 | 60, 98.0% (Item 1, n=50 measured) | **0.2975** | **+0.1225** |
| oracle_causal (n=50 pop.) | 0.6950 | 61, 89.5% (Item 4, **n=57 only, NOT re-expanded — same caution as any n≈18-20 rate applies, not re-verified at scale like floor's was**) | 0.9679 | +0.2729 |
| U-floor (n=100 pop.) | 0.1780 | ~66 est., **hit rate ASSUMED 98% (floor's rate, NOT independently measured for U-floor)** | 0.3397 | +0.1617 |

**floor's ceiling (+0.1225) is by far the largest of any intervention tested tonight** (E2/E3/
Lever2 all topped out at ≤0.11), because Item 4's hit rate (94-98%) is far higher than E3's
expected_attn rate (53.7%) that grounded every earlier grounding-lever ceiling. Oracle's and
U-floor's ceilings carry explicit caveats (unverified-at-scale hit rate; assumed, not measured,
hit rate respectively) and should be read as more speculative than floor's.

**Decision rule**:
- **REAL**: gain over the unconstrained SAME arm (floor+copy vs `floor_pos`, on held-out data)
  has a 95% CI excluding zero, AND net accuracy improves after accounting for false positives on
  the correct-case set (not just gross gain on the wrong-case set).
- **NOT-IT**: either condition fails.
- **Headline / secondary question**: does floor+copy exceed `floor_pos`? For the floor arm this
  IS the same comparison as "gain over the unconstrained same arm" — they coincide by
  construction, since floor+copy's baseline is floor_pos itself.

**Held-out split**: the 60 wrong cases split 30/30 (alternating index, same convention as E3),
even though the design has no threshold to actually fit on the selection half (fires
unconditionally) — kept as a discipline check, not because a parameter is being tuned.
**False-positive sample: all 81 correct cases already on hand from Item 1** (not 2/10 — this was
exactly the concern raised, and it's already resolved by Item 1's expansion, no new GPU needed to
build this set).

**Sizing**: the ceiling's own implied effect (delta=0.1225) would formally justify n as small as
~6 per the standing power formula — **explicitly rejected** as a sizing basis. This program's own
carried-forward lesson (splits at n≈18-20 moved 30+ points, `PAPER4_PHASE1_RECORD.md`) argues
against trusting a formula-derived small n regardless of how large the point-estimate ceiling
looks. **Using n=80 as the floor**, matching every other confirmatory GPU test this session
(E2/E3/Lever2) — this session's own 60 wrong + 81 correct cases already exceed that on both
sides, so no additional generation is needed beyond what Item 1 already produced for the
population; the NEW work is only the constrained-decoding generation pass itself on these
existing 141 cases.

**Confound check, decode-only**: since this is a decode-time constraint, `p_g` and
`units_complete` (retention metrics) MUST be identical to unconstrained `floor_pos` — same
prefill, same press, same keep-set, only the generation step differs. This will be VERIFIED by
recomputing `keep_metrics` on the SAME captured keep-set used for the constrained run and
diffing against the already-recorded `floor_pos` values for these instances, not assumed from
the design alone.

## Item 4 — Related work to check, not cited from memory

Areas that need verification against the actual papers before anything goes in a related-work
section (none of this is being cited as fact here, only listed as a checklist):

- **Copy mechanisms in generation**: Pointer Networks (Vinyals et al.); CopyNet /
  copy-mechanism summarization (Gu et al.; See et al., "Get To The Point" — pointer-generator
  networks that mix a copy distribution over the source with the model's own vocabulary
  distribution — closely related in spirit to constraining generation to a source span).
- **Lexically/vocabulary-constrained decoding**: grid beam search and other lexically-constrained
  decoding work (Hokamp & Liu; Post & Vilar's dynamic beam allocation) — constraining allowed
  output tokens is an established decoding-constraint family, though usually for enforcing
  REQUIRED words rather than restricting to a SOURCE span.
- **Extractive / span-copy QA**: reading-comprehension models that literally predict a start/end
  span in the source (BiDAF, and pointer-based extractive QA generally) — structurally the
  closest prior art to "restrict the answer to a source span," though not applied as a
  decoding-time constraint over a generative LM.
- **Faithfulness/hallucination-reduction decoding**: context-aware decoding (contrasting
  with/without-context output distributions); PASTA (post-hoc attention steering toward
  specified spans, already flagged in `PAPER4_PHASE2_PLAN.md` as needing a novelty check);
  attention-based hallucination detection such as "Lookback Lens."
- **DoLa-style contrastive decoding** (layer-contrast to suppress hallucination) — a different
  mechanism (logit contrast across layers, not a source-span vocabulary mask) but adjacent enough
  in aim that it needs distinguishing explicitly in any related-work framing.

Every item above needs to be checked against the actual paper (title, mechanism, what it
constrains and how) before any novelty claim is made — this list is a starting point for that
check, not a substitute for it.

## GPU cost (projected, not run)

The wrong/correct populations (60 + 81 = 141 cases) already exist from Item 1 — no new GPU work
needed to build them. New work: one constrained-generation pass per case (single query, not a
full 4-query instance), reusing the already-identified top-attended span (no new eager-attention
extraction needed for THIS step — span identity is already on file from Item 1; only the
constrained decode itself needs to run, under standard sdpa). Estimated at roughly a quarter of a
full-instance's generation cost per case (one query vs four): **~141 single-query decodes,
projected ~10-15 minutes** including model load, using the measured ~23.4-rows/min rate as a
rough per-unit basis. Not run. Waiting for confirmation.
