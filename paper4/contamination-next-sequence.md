# Cross-lane sequencing — contamination and hallucination

Shared record for both investigation lanes, so neither re-derives what the other already found.

## Overlap check: fabrication vs. contamination (resolved, CPU only, no GPU)

**Question**: do the 25 M2 plausible-fabrication cases actually show contamination's structure
(invented value traceable to a specific real record elsewhere in context), or are they genuinely
unsourced? Using contamination's own already-established classification rule (record-distance
<=2, word-overlap >=6 — `CONTAMINATION_UNEXPLAINED_CONTEXT.md` §1, `a500cb9`/`c122dab`), applied
to the `match_rec`/`match_dist`/`match_score` fields already captured for every fabrication case
in `out/_task1_cases.json` (Task 1, commit `fe5fbf7`) — no new extraction needed, this data was
already on disk.

**Result: 0/25 (0.0%) of fabrication cases meet the contamination-shaped threshold.** Every case's
nearest real-record match falls short on at least one criterion; the two closest cases
(snapkv inst37: dist=1, score=5; expected_attn inst23: dist=1, score=5) still miss the score>=6
bar. The invented values in every fabrication case are genuinely unsourced — not traceable to a
specific real record anywhere in the document at the strictness contamination itself uses.

**Reverse direction**: all 20 of contamination's residual cases trivially clear the same threshold
(dist 1-2, score 6-7 in every case, `out/_step2_cases.json`) — expected, since that threshold is
how the contamination population was defined in the first place. No contamination case looks like
an "unsourced" fabrication either.

**Verdict: DISTINCT.** Zero overlap in either direction, out of 45 total hallucination+contamination
cases examined (25 fabrication + 20 contamination). Fabrication and contamination are not the same
failure mode wearing different labels — this was fully resolvable from data already on disk, no
new GPU spend needed to settle it.

## Test A's result, recorded for both lanes

From the hallucination attention-test thread (`HALLUCINATION_ATTENTION_TESTS.md`, commits
`b65888f`, `2f871f6`, `bef6bff`, `da5965b`): attention mass on the queried record's own content, at
the divergence token, is measurably LOWER in fabrication cases than in completion-matched correct
cases. Raw paired difference -0.01341 [-0.01795, -0.00884]; **position-detrended residual
difference -0.01337 [-0.01744, -0.00935]** — essentially unchanged by detrending, confirmed as a
genuine, largely position-independent effect (n=18-19, one pair excluded for a floor-region edge
case). Entropy (Test B) and record-ID-token attention (Test C) both came back NOT-IT — the deficit
is specific to the record's own field content, not generic diffuseness or record-boundary tracking.

**Given the DISTINCT verdict above, this result stays scoped to the hallucination lane.** Test A
does not, by itself, imply anything about contamination's mechanism — the overlap check found no
shared cases, so there is no basis yet to treat attention-mass-on-queried-record as the same
underlying cause behind both failure modes. It remains the leading, confirmed mechanism for
fabrication specifically.

## Sequencing decision, per the overlap result

Overlap resolved as **DISTINCT**, so per the pre-agreed branching: proceed with the contamination
lane's own Step 1 (harness capability check — does the extraction infrastructure expose attention
weights) rather than redirecting all GPU budget into re-running Test A's methodology on
contamination's 20 residual cases. That redirection was conditioned on a MERGED finding, which
did not happen.

**Step 1 can be marked done, not re-verified from scratch.** Test A already demonstrated tonight,
on this exact model/cell/wrapper stack, that attention weights ARE extractable (`output_attentions
=True` under eager attention, GQA head-mapping solved, per-head retained-set inversion for
non-headwise presses solved, floor-region edge case identified and handled) — the same
infrastructure (`out/_task_hallu_extract.py`, `out/_task_hallu_extract_v2.py`) applies directly to
contamination's cell (same model, same c=40/C=512, same arms except one exclusion). The one
carried-over caveat: AdaKV pairs still cannot be extracted this way (eager-mode assertion in
`p4/unitwrap.py:222`); any contamination arm breakdown that includes adakv_snapkv needs the same
disclosed exclusion Test A used.
