# Real-text tight-KV reinterpretation

New file under `out/_retpolicy_dev/`, per the standing convention for this exploratory thread.
No frozen `PREREG`/`RESULT`/manifest file is edited, thawed, or reinterpreted in place. The
frozen tight-KV accuracy and latency numbers stand exactly as measured and reported in
`REALTEXT_TIGHT_KV_M2_RESULT.md`, `REALTEXT_TIGHT_KV_M3_RESULT.md`, and
`REALTEXT_TIGHT_KV_LATENCY_FINAL_RESULT.md`. What follows is a reading of what those numbers, plus
the exploratory work in this directory, together imply -- not a correction of them.

## The core finding

The frozen tight-KV system's headline win (focused SnapKV **107/320** vs plain floor **63/320**,
+13.75 points, both preregistered dual-CI gates passed) is real and stands. The component
ablation run in this directory shows what is actually producing that gain:

| arm | accuracy (M2, n=320 SQuAD queries) |
|---|---|
| plain floor (frozen) | 63/320 |
| **(a) question + retrieved sentence, NO context cache at all** | **151/320** |
| (b) floor@Bfinal + retrieved sentence, unmasked | 157/320 |
| (c) floor@Bfinal + retrieved sentence + literal mask | 154/320 |
| (d) full_cache (nothing ever evicted) + retrieved sentence + mask | 150/320 |
| frozen focused SnapKV (BM25 pointer restricted to eligible/retained sentences) | 107/320 |

Arms (a), (b), (c), and (d) are **statistically indistinguishable from each other** (150-157/320,
overlapping 95% instance-bootstrap CIs) and **all sit roughly 15 points above the frozen focused
SnapKV system**. Arm (a) has **no compressed context cache in it at all** -- the input is just
the question and one retrieved sentence -- and it matches arm (d), which has the **entire
original, uncompressed context**. Between the two extremes of "nothing retained" and "everything
retained," accuracy does not move outside noise. The literal mask barely moves it either
(unmasked (b) 157 vs masked (c) 154).

**The reinterpretation**: the ~28-point gain over plain floor that all four ablation arms share is
a retrieval-accuracy effect, not a cache-management or masking effect. The retrieved sentence is
found by BM25 over *all* sentences of the original document -- full-text search, not the frozen
system's own retention-eligibility-restricted pointer. Once you have the right sentence, *how*
you get it into the model's context (compressed cache, full cache, or no cache and just the
sentence alone) barely matters. The frozen system's own +13.75-point win over plain floor is real
precisely because its own SnapKV-restricted pointer only finds the right sentence some of the
time (181/320 hit in the confirmation prefill audit, vs candidate ceiling constraints); the
ablation's BM25-over-everything pointer finds it far more often, and that difference -- not the
cache, not the mask -- is what separates 107/320 from 150-157/320.

## Supporting exploratory findings, recorded here

- **Single-prefill equivalence holds.** On 10 fresh instances/20 queries per model from the
  frozen confirmation manifest: floor_pos keep-hash matched the frozen two-pass system's stored
  hash **20/20 on both models**, once the reconstruction was corrected to NOT protect the
  pointer-selected sentence (matching the frozen system's own, unprotective, second-pass
  behavior exactly). SnapKV's keep-hash never matched exactly, but this is fully characterized
  as **boundary ties, not a flawed reconstruction**: the dropped-vs-kept score boundary was
  verified directly against captured scores (invariant held in every printed case, frequently an
  *exact* tie), and the symmetric difference against a freshly re-run real second prefill was
  0-4 tokens out of a 500+ token budget (11/40 cases had symdiff exactly 0). Answer match was
  38/40 (M2) and 40/40 (M3); score match was 40/40 on both models. If this holds at confirmation
  scale, it implies most of the measured 2.43x/1.72x latency penalty (`REALTEXT_TIGHT_KV_LATENCY_
  FINAL_RESULT.md`) could be eliminated by skipping the second real prefill entirely -- diagnostic
  only, not a frozen result.

- **The retention-for-pointer track is closed.** Neither a sentence-level aggregated-SnapKV
  policy (u_snapkv) nor a relevance-free farthest-first whole-sentence diversity policy beat
  SnapKV's own token-level ranking on held-whole rate or pointer-hit rate, on either model,
  prefill-only (no generation). SnapKV remains the best retention policy found for this
  mechanism. Combined with the point above: no retention policy tested, including the frozen one,
  closes the gap to full-text search.

## SQuAD vs needle split: not applicable to this ablation

The manifest (`out/realtext_tight_confirm_160.jsonl`) carries 4 queries per instance (2 SQuAD, 2
needle), but the frozen tight-KV generator (`_realtext_tight_generate.py`) and every script in
this exploratory thread process only `case["queries"][:2]` -- which are **always the two SQuAD
queries** by construction of that manifest. The needle queries (indices 2-3) were never part of
the tight-KV confirmation or this ablation; they belong to the earlier, separate "5070 extra-KV"
mixed study (`REALTEXT_5070_M2_RESULT.md`), which used a different generator and a different
(80-instance) manifest. **There is no needle-vs-SQuAD split to report for the tight-KV ablation
because it is SQuAD-only by design** -- not an oversight in this analysis. All 320 compared
queries above are SQuAD.

## LEDGER-C in the same light

LEDGER-C's own query literally names the target record ID: *"Reproduce record R003 exactly, in
full, in the format shown..."* (verified directly against the instance builder,
`p4.common.CM.build`). A trivial exact-substring search for "R003" over the 40 records in the
document would find the correct record with 100% precision -- no attention mechanism, BM25, or
compression heuristic required. **LEDGER-C's results therefore apply only to the cache-only
setting** where the model has no ability to search or grep the original text and must rely
entirely on what survived compression -- the moment any form of full-text lookup is available
(as it always is outside this constrained setting, since the query contains the answer key by
construction), the retrieval half of the task becomes trivial by design, exactly paralleling the
real-text finding above.

**Check performed** (M2, the same 80-instance/320-query anchor cell as the frozen LEDGER-C
primary confirmation, `out/_step5_primary_M2.jsonl`): reused the ALREADY-STORED top-attended
record (`max_rid`, no attention recomputation) from the frozen confirmatory run, inserted that
record's full text next to the question, and decoded **without** the literal-copy mask.

| arm | accuracy |
|---|---|
| floor_pos (frozen) | 0.1938 (62/320) |
| floor + literal mask (frozen) | 0.2812 (90/320) |
| **floor + proximity insertion, no mask (this check)** | **0.30 (96/320)** |

- Proximity-unmasked minus floor_pos: **+0.106** [+0.075, +0.141], 34 converted, **0 broken**
- Proximity-unmasked minus floor+mask: **+0.019** [+0.006, +0.034] -- CI excludes zero on the
  **positive** side, i.e. unmasked proximity insertion is not just competitive with the mask, it
  is measurably slightly *better*, with strictly fewer failure modes (0 broken vs the mask's own
  reported breakage on the harder subset).

**Proximity insertion without the mask matches or slightly exceeds the mask's gain; whether they
share a mechanism is untested pending the combination arm, and the proximity arm used ~40 extra
tokens beyond the budget** (the frozen mask arm decodes from the same, unmodified floor_pos
cache; this check's proximity arm appends the full record text into the query, which is extra
context outside the retained budget, not a like-for-like comparison at equal KV -- see the
follow-up equal-KV check below). This is directionally consistent with the real-text ablation's
finding -- in both cases, an unmasked arm with the right evidence placed near the query performs
comparably to the masked arm -- but here it is confounded by the extra out-of-budget tokens, so
it is not yet established that retention/masking is doing no work; the equal-KV proximity check
below (item 2) is designed to remove that confound. For LEDGER-C the "finding the right evidence"
step is even more trivially solved than real text's BM25 pointer, since the query names the
answer's own key directly.

## Scope

M2 only (the ablation and equivalence check); the pointer-hit retention screen covered both
models. Diagnostic exploratory work, not a preregistered confirmatory test. Does not retract or
amend `REALTEXT_TIGHT_KV_M2_RESULT.md`, `REALTEXT_TIGHT_KV_M3_RESULT.md`, or the latency results,
which remain the frozen, authoritative numbers for the two-pass system as specified and measured.
