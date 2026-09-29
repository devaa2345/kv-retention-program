# Post-session record: everything after REALTEXT_SESSION_FULL_RECORD.md

This picks up exactly where `REALTEXT_SESSION_FULL_RECORD.md` (§6, "Forward plan: exploratory
custom retention policy") left off. That document froze the state of the real-text tight-KV
thread as of its own writing and handed off an open custom-retention-for-pointer direction. Every
finding below was produced after it, entirely under `out/_retpolicy_dev/`. **No file that
`REALTEXT_SESSION_FULL_RECORD.md` marks as frozen was edited, thawed, or reinterpreted in place.**
This record is itself new/uncommitted, exactly like its predecessor.

Read this alongside `REALTEXT_SESSION_FULL_RECORD.md` §5 ("Claims the record supports, and their
limits") — the reframing below updates the STATUS of several of those claims, but does not alter
the numbers behind them.

## 0. What this thread actually did, in one paragraph

Three phases, executed as separate instructed rounds: (1) a custom-retention-policy screen
(u_snapkv, farthest-first) that came back negative against SnapKV; (2) a **component ablation** of
the frozen real-text tight-KV system's own headline win, which found the entire measured gain is a
retrieval/proximity effect, not a cache-management or masking effect — verified independently on
both the real-text task and on LEDGER-C; (3) a bridge two-hop competence-gate pilot, which failed
on both 3B models, closing the "where would the cache actually matter" question for this hardware
tier and pointing the next real test at an 8B/48GB card. A parallel single-prefill-equivalence
check found the frozen system's expensive second prefill is very likely replaceable by direct
cache surgery — a genuine, actionable latency finding, separate from the reinterpretation.

## 1. Chronological narrative (continuing from REALTEXT_SESSION_FULL_RECORD.md §1.13)

### 1.14 Independent audit of the LEDGER-C confirmatory numbers

Before any new work, the LEDGER-C confirmatory numbers themselves were independently re-audited
end to end (`out/_audit_full.py`, two passes — the first had a context-cache keyed only by
instance number, which silently reused M2's context for M3's invented-value check when both
models were audited in the same process; fixed to key by `(tag, instance)`, after which every
invented count matched the historical record's zero). One historical table did mix a same-arm
comparison (`u_floor+mask` own-baseline +0.1281) with a different-comparator column heading
(`oracle_causal`'s +0.7562 vs `floor_pos`) under one "delta" column — a labelling error, not a
recomputation error; both individual numbers were independently reproduced exactly. This audit
did not change any reported number; it re-verified them and found the two issues above, both
process/presentation issues rather than data-integrity issues.

### 1.15 Single-prefill equivalence check (Item 1, two rounds)

**Question**: can the frozen tight-KV system's expensive second full-context prefill be replaced
by directly dropping `e` tokens from the already-compressed scout cache, skipping the second pass
entirely?

**Round 1** (10 instances/20 queries per model) initially protected the pointer-selected sentence
when choosing which tokens to drop. This was WRONG — the frozen system's own second-pass floor
recompression does not protect the selected sentence at all (confirmed directly:
`selected_final_held: false` in the frozen results for the one instance where this mattered).
Corrected to drop only sink/window as protected, matching the frozen system exactly.

**Round 2 (corrected)**: floor_pos keep-hash matched the frozen two-pass system's own stored hash
**20/20 on both models** (M2, M3). SnapKV's keep-hash never matched exactly (0/20 both models),
but this was characterized precisely rather than dismissed: the drop-selection boundary was
verified directly against actual captured scores (dropped-vs-kept score invariant held in every
printed case, frequently an *exact tie*), and the symmetric difference against a freshly re-run
real second prefill was **0-4 tokens out of a 500+ token budget**, with **11/40 cases having
symdiff exactly 0** — meaning two independent real second-prefills of the FROZEN system don't
always agree with each other either. This is inherent bf16 score-recomputation nondeterminism at
near-tied cutoff boundaries, not a flaw in the reconstruction. Answer match: 38/40 (M2), 40/40
(M3). Score match: 40/40 both models.

**Implication, not yet confirmed at scale**: if this holds on the full 160-instance manifest, it
implies most of the measured 2.43×/1.72× latency penalty (`REALTEXT_TIGHT_KV_LATENCY_FINAL_
RESULT.md`) could be eliminated by skipping the second prefill — a genuine, actionable engineering
finding, independent of the reinterpretation below. **Not run at confirmatory scale; diagnostic
only.**

Artifacts: `single_prefill_equivalence.py`, `equiv_M2/M3.jsonl` (round 1, superseded),
`equiv_v2_M2/M3.jsonl` (round 2, corrected — authoritative).

### 1.16 Component ablation of the real-text tight-KV headline win (Item 2)

**Question**: what is the compressed cache actually contributing to the frozen system's own
+13.75-point win over plain floor?

Four arms, M2, 160-instance/320-query confirmation manifest, BM25-over-all-sentences (not the
frozen eligibility-restricted pointer) picking the evidence sentence for all four:

| arm | accuracy |
|---|---|
| plain floor (frozen) | 63/320 (19.7%) |
| (a) question + retrieved sentence, **no context cache at all** | **151/320 (47.2%)** |
| (b) floor@Bfinal + retrieved sentence, unmasked | 157/320 (49.1%) |
| (c) floor@Bfinal + retrieved sentence + literal mask | 154/320 (48.1%) |
| (d) full_cache (nothing evicted) + retrieved sentence + mask | 150/320 (46.9%) |
| frozen focused SnapKV (own eligibility-restricted pointer) | 107/320 (33.4%) |

**All four ablation arms are statistically indistinguishable from each other** (150-157/320,
overlapping 95% instance-bootstrap CIs), all ~15 points above the frozen system, and arm (a) —
**zero cache, just the sentence and the question** — matches arm (d) — **the maximal possible
cache, nothing ever evicted**. Between the two extremes of "nothing retained" and "everything
retained," accuracy does not move outside noise, and the literal mask barely moves it either
(unmasked 157 vs masked 154). SQuAD-vs-needle split was requested and found **not applicable**:
the tight-KV generator only ever processes the manifest's first two (always-SQuAD) queries per
instance; needle queries belong to a separate, earlier study.

**First framing was overclaimed and was corrected on instruction**: "the compressed cache
contributes essentially nothing" was walked back to the precise, defensible claim — *arm (a)
matches or slightly exceeds the mask's gain; whether the mechanisms are shared is untested
pending a combination arm; arm (a) also used ~40 extra tokens beyond the nominal budget, so it is
not a like-for-like equal-KV comparison with the frozen system.* This correction is recorded in
`REALTEXT_REINTERPRETATION.md` itself, not just here.

Artifacts: `component_ablation.py`, `ablation_M2_full.jsonl`; also reuses `bm25_M2_full.jsonl`
(arm c, generated slightly earlier as its own check) and `text_available_bm25_baseline.py`.

### 1.17 Text-access fidelity check (Item 2, CPU-only side-check)

For every selected sentence in both frozen confirmation sets, decoding its retained token IDs
reproduced the source text exactly in **640/640 (M2)** and **455/640 (M3)** cases; ALL M2
mismatches and most M3 mismatches were pure tokenizer boundary whitespace bleed (verified via
`.strip()` equality). M3 additionally showed a punctuation-spacing quirk (wikitext's raw " ,"
formatting vs Llama's decode collapsing it) explaining most, not all, of its residual mismatches;
~60/497 M3 mismatches remain unexplained by either rule (spot-checked: looks like a mix of the
same punctuation issue plus a `split_sentences()` boundary quirk around en-dashes). Not a
correctness issue for any reported accuracy number (those use normalized comparison against gold,
not raw decoded text) — a fidelity note only. Artifacts: `text_access_check.py`,
`text_access_mismatches_M2/M3.jsonl`.

### 1.18 Retention-for-pointer screen: closed negative (Item 3, prefill-only)

Four retention policies (floor_pos, snapkv, a new whole-sentence u_snapkv, a new relevance-free
farthest-first whole-sentence diversity policy — Jaccard distance, deterministic earliest-position
start, greedy max-min-distance selection, stated before coding) at the exact floor budget, both
models, full 160-instance manifest, prefill-only (no generation), scored with the frozen lexical
pointer applied uniformly.

**SnapKV wins on both held-whole rate and pointer-hit rate, on both models, against both new
policies, by wide, CI-excluding-zero margins** (e.g. M3: SnapKV held 93.8% vs u_snapkv 32.8% vs
farthest_first 57.5%). Farthest-first is the closer of the two new policies but never wins. **This
track is closed**: no retention policy tested, including the two new ones, beats SnapKV's own
token-level ranking at finding the gold sentence. Artifacts: `retention_screen.py`,
`screen_M2/M3.jsonl`.

### 1.19 LEDGER-C in the same light: the mask's gain is also proximity, at extra KV (Item 2 cont'd)

LEDGER-C's own query literally names the target record ID ("Reproduce record R003 exactly...") —
verified directly against the instance builder. A trivial exact-substring search over the 40
records would solve retrieval with 100% precision; **LEDGER-C's results therefore apply only to
the cache-only setting**, where no such lookup is available.

Check performed (M2, the same 80-instance/320-query anchor cell as the frozen LEDGER-C primary
confirmation): reused the already-stored top-attended record (`max_rid`, no attention
recomputation), inserted its full text next to the question, decoded **without** the mask, at
**extra KV** (the record text is appended beyond the nominal budget, same confound as the
real-text ablation's arm (a)):

| arm | accuracy |
|---|---|
| floor_pos (frozen) | 0.1938 (62/320) |
| floor + literal mask (frozen) | 0.2812 (90/320) |
| floor + proximity insertion, no mask, **extra KV** | **0.30 (96/320)** |

Proximity-unmasked minus floor+mask: **+0.019 [+0.006, +0.034]**, CI excludes zero on the positive
side. **Corrected framing (per instruction, applied identically to the real-text case)**:
proximity insertion without the mask matches or slightly exceeds the mask's own gain; whether they
share a mechanism is untested pending a combination arm; this check used extra out-of-budget
tokens, so it is not yet a like-for-like equal-KV comparison. Artifacts: `ledger_proximity_check.py`,
`ledger_proximity_M2.jsonl`.

### 1.20 LEDGER-C equal-KV tightening: attempted, blocked (Item 2, tightened)

The natural next step — repeat the LEDGER-C proximity check at EQUAL KV (drop `e` tokens from the
already-compressed cache via the same verified single-pass eviction method from §1.15, rather than
appending extra tokens) — was attempted and **could not be completed**. A new script
(`ledger_equalkv_check.py`) reproducibly OOM'd on the second LEDGER instance processed, every
time, across six distinct fix attempts:

1. Direct `.compress()` override (mirroring the SnapKV path in `single_prefill_equivalence.py`) —
   failed with allocated memory climbing from ~5.9 "GB" (this environment's own inflated reporting
   units, not literal — the physical card is 11.94GB) to ~30GB after one instance, ~36GB before
   the second instance's prefill, then OOM.
2. Added per-query `gc.collect()`/`empty_cache()` — identical failure, byte-for-byte identical
   error message, ruling out garbage-collection timing as the cause (a live reference, not
   uncollected garbage).
3. Added post-scout-prefill cleanup before the query loop — identical failure again.
4. Switched to the PROVEN pattern from `_step5_primary_run.py` (patch `.score()` via
   `methods.make_capturing` + instance-bound `types.MethodType`, letting kvpress's own
   `.compress()` gather logic run untouched, exactly as every other LEDGER script tonight has done
   safely across 80 instances) — **identical failure, same exact numbers**, ruling out the
   specific hook-patching mechanism as the sole cause.
5. Tried `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the fix the error message itself
   suggests) — **made it worse**: crashed even earlier, with a different error class (`CUDA driver
   error: out of memory` rather than PyTorch's own `OutOfMemoryError`).
6. Reverted the environment variable; stopped after 6 attempts, reporting the blocker plainly
   rather than continuing to guess.

**Best current hypothesis, not verified**: genuine memory fragmentation specific to this script's
per-head `torch.cat`/`gather` cache-surgery pattern (many small tensor operations per query,
across all layers and KV heads), which no other script tonight performs at this frequency. The
frozen tight-KV system's own real second prefill (a full fresh forward pass, not cache surgery) is
the alternative it exists to avoid — ironically, the mechanism proposed to make the system faster
is itself harder to run stably than the system it would replace, at least in this environment.
**Not resolved. The LEDGER-C equal-KV comparison, and its M3 replication, were never run.**

### 1.21 Prior art on the mechanism (Item 3/4)

**Verified real**: "Attention Sorting Combats Recency Bias In Long Context Language Models"
(Peysakhovich & Lerer, arXiv 2310.01427, 2023-10) — sorts context by attention received, places
highest-attention content closest to the query, iterates, then generates. This is the closest
existing precedent to the proximity-insertion finding in §1.16/§1.19: both move
already-identified-as-important content physically near the query and report this alone recovers
most of another mechanism's measured gain. Adjacent, more recent work in the same family:
In-Context Re-ranking (attention as an implicit relevance signal for reordering) and ReContext
(recursive evidence replay). **Plain assessment**: the general mechanism is not new (2023 paper,
active surrounding literature); applying it specifically to reinterpret a confirmed KV-cache-
compression benchmark result, and pairing it with a full-text-retrieval-baseline contrast, looks
narrower and was not found in this search pass. Separately: the general critique that long-context
benchmarks can be dominated by retrieval/lookup rather than the mechanism under test is also
established (NIAH's lexical-matching critique → NoLiMa; a very recent query-visibility/matched-
budget audit of KV-cache compression specifically, arXiv 2607.11942). Artifacts:
`PRIOR_ART_RETRIEVAL_VS_COMPRESSION.md`.

### 1.22 Bridge two-hop pilot: competence gate FAILS on both 3B models (Item 4)

Per `PREREG_BRIDGE_TWOHOP_DESIGN.md` (design written in this thread, §1.23 below), a 40-instance
gate-only pilot (HotpotQA bridge questions, distractor setting, full document context, no
compression) was run on both models before any retention-arm comparison:

| model | full_cache exact | full_cache "contains" (permissive) | top1_sentence exact |
|---|---|---|---|
| M2 | 16/40 (40.0%) | 21/40 (52.5%) | 7/40 (17.5%) |
| M3 | 18/40 (45.0%)* | 22/40 (55.0%) | 6/40 (15.0%) |

*M3's raw strict-match score was initially 0/40 due to a scoring bug on this session's part
(trailing-punctuation sensitivity in the normalization function, e.g. "Animorphs." vs "Animorphs"
counted as a mismatch) — inspected, confirmed as a scoring artifact not a reasoning failure, fixed,
and rerun/recomputed honestly to 18/40.

**Both models fail the preregistered [0.55, 0.97] competence gate on strict exact match, and sit
at or just below it even on the most permissive reasonable measure.** top1_sentence is worse still
on both models, confirming single-sentence retrieval cannot solve bridge questions either (as
designed — this was the point of including that arm). **Per instruction, this is a hard stop**:
no retention-arm comparison (floor, SnapKV, top-k, full_cache-vs-compressed) was run on this task,
because a failure to beat floor here would be indistinguishable from "the model can't do two-hop
reasoning at all," matching Paper 2's own documented precedent (one-hop 0.4375 vs two-hop
0.100-0.1263 for this model class) cited in the design doc before any GPU time was spent. **The
bridge experiment, if pursued further, needs an 8B-class model, not another 3B pilot.** Artifacts:
`bridge_pilot_gate.py`, `bridge_M2.jsonl`, `bridge_M3.jsonl`.

### 1.23 Design work (no GPU): the bridge task itself

`PREREG_BRIDGE_TWOHOP_DESIGN.md` — full task construction (HotpotQA bridge subset, gold paragraphs
placed far apart, target 2000-3000 tokens), 5 planned arms (top-1 sentence, top-k sentences, floor,
SnapKV, full_cache), the full-cache competence gate (now exercised and failed, §1.22), expected
costs, and four named failure risks (competence gate failure — the one that materialized;
non-extractive answers; distractor dilution; SnapKV's own known cross-head-diffusion failure mode
from tonight's LEDGER work, requiring the same kind of component ablation before trusting any
future positive SnapKV result on this task).

## 2. Reframing: what got stronger, what got weaker, what's now closed vs. open

Read this against `REALTEXT_SESSION_FULL_RECORD.md` §5's claims table — each row below either
sharpens, weakens, or closes something that table left open.

| Prior claim (from the full record) | Status now | What changed |
|---|---|---|
| "Focused evidence system beats both floor controls on SQuAD with extra KV" — established, extra-KV framing | **Weaker as a mechanism claim, unchanged as a measured result.** | The +13.75-point win over plain floor is real and unmoved. But the component ablation (§1.16) shows the win is not specifically about the cache or the mask: a zero-cache, sentence-only arm matches it. The frozen system's *accuracy number* stands; its *causal story* ("compression + masking + reinsertion together produce this") does not survive the ablation as originally framed. |
| "Tight per-query focus beats both floor controls at equal pre-answer KV" — established on both 3B models | **Unchanged as a measured result; its interpretation now needs the same caveat as above**, though this specific equal-KV claim was NOT itself re-ablated (the ablation used the extra-KV `focus_query` insertion, not the tight-KV equal-budget mechanism). This is an open gap, not a resolved weakening — see §3. |
| "Tight focus has a measurable latency cost" — established, 2.43×/1.72× | **Possibly much smaller than measured, pending confirmation.** | The single-prefill equivalence check (§1.15) suggests the expensive second prefill may be replaceable by direct cache surgery with near-perfect fidelity (20/20 floor keep-hash match; SnapKV differences are bf16 boundary ties). Not yet run at the scale needed to actually revise the latency number — this is a lead, not a result. |
| LEDGER-C floor copy-mask improves exact structured extraction — established on M2 and M3 | **Unchanged as a measured result; same proximity caveat as the real-text case now applies.** | §1.19: unmasked proximity insertion (at extra KV) matches or slightly exceeds the mask's own gain on the SAME anchor cell. The mask's contribution beyond "having the record nearby" is not established, mirroring the real-text finding almost exactly. The attempt to check this at EQUAL KV (removing the extra-KV confound entirely) is the one significant loose end this thread leaves open — see §1.20 and §3. |
| Custom retention raises frozen-pointer hit rate over floor — open, no result yet | **Closed, negative.** | §1.18: neither new policy beats SnapKV on either metric on either model. This specific direction is done; it does not need revisiting without a new policy idea. |
| Real-text and tight-KV effects generalize to 8B/48GB — open, not started | **Still open, but now with a concrete reason to expect the interesting result is elsewhere.** | The 3B two-hop competence gate failure (§1.22) means an 8B model is now the *minimum* requirement for the one task type (multi-span bridging) where the cache's own management could plausibly matter, rather than an optional scaling check. |
| (New, not in the prior record) Does a full-text retrieval baseline with zero compression beat the frozen system? | **Established, on the real-text task.** BM25-over-all-sentences + one plain floor prefill (no second pass, no scout) scores 154/320 vs the frozen focused SnapKV's 107/320 — the frozen, previously-confirmed compression+mask system is beaten by ~15 points by a retrieval method with NO compression management at all. This is the thread's single strongest, most concrete finding. |

## 3. What is still genuinely open (the honest gap list)

1. **The LEDGER-C equal-KV proximity check never ran** (§1.20, blocked by an unresolved OOM). This
   is the single most important unresolved question from this whole thread: does proximity
   insertion still match/beat the mask once the extra-KV confound is removed, on LEDGER-C? Without
   it, the LEDGER-C proximity finding (§1.19) remains confounded by extra tokens, exactly as
   flagged.
2. **The real-text tight-KV system's own equal-KV mechanism was never directly re-ablated.** The
   component ablation (§1.16) used the EXTRA-KV `focus_query` insertion mechanism to test "does the
   cache matter," not the tight-KV system's own equal-budget eviction. It is very plausible the
   answer is the same (the frozen tight-KV numbers and the extra-KV numbers tell a consistent
   story throughout `REALTEXT_SESSION_FULL_RECORD.md`), but it was never checked directly at equal
   KV on the real-text task either. A clean combination arm — equal-KV eviction (§1.15's verified
   method) + proximity insertion + mask, vs the same without the mask, vs plain floor — would
   close both gaps 1 and 2 at once, on both tasks, if the OOM in §1.20 can be fixed.
3. **The single-prefill equivalence result (§1.15) was only checked on 10 instances per model.**
   If confirmed at the full 160-instance scale, it's an actionable latency fix; until then it's a
   strong lead, not a result to act on.
4. **~60/497 M3 text-access mismatches remain unexplained** (§1.17) — almost certainly harmless
   (tokenizer/formatting noise), but not run to ground.
5. **The bridge two-hop task itself was only checked for competence, never for whether ANY
   retention/pointer arm would help** — that question is now explicitly deferred to an 8B model,
   not answered as negative for the mechanism itself.

## 4. What to do on 8B — concrete plan

The 3B competence gate failure (§1.22) is the load-bearing fact here: it means the two-hop bridge
task, the one place in this whole program where the cache's OWN management plausibly matters
(because no single retrieved span suffices), cannot be tested on the current hardware tier at all
— not "tested and found not to help," but genuinely untestable, because the base model can't do
the task even with perfect, complete evidence in front of it.

**Step 0 — reproduction gate, before anything else.** Per `REALTEXT_SESSION_FULL_RECORD.md` §7's
own standing note: any new hardware/model migration must first log driver/CUDA/torch/VRAM and
reproduce a frozen M2 output slice exactly on the new setup, explaining any mismatch, before
scaling up. Do this before touching the 8B model at all. This is cheap and catches environment
drift early — exactly the kind of check this whole session has repeatedly needed (recall tonight's
own multiple OOM/nondeterminism investigations).

**Step 1 — re-run the bridge competence gate on the 8B model, same 40 instances, same scoring
(with the punctuation-normalization fix from §1.22 applied from the start this time).** This is the
actual gate: if 8B also fails `[0.55, 0.97]`, the bridge task needs a smaller-hop or more scaffolded
design (e.g. explicit "first find X, then find Y" prompting) before any cache experiment is
meaningful, on ANY hardware. If 8B passes, proceed to Step 2.

**Step 2 — the same component-ablation logic, applied to bridge, BEFORE trusting any positive
retention-arm result.** Given the two-hop task structurally requires combining evidence from two
places (unlike every task in this whole thread so far), a naive top-k-sentences-no-cache arm is
the right first check — exactly as designed in `PREREG_BRIDGE_TWOHOP_DESIGN.md`'s arm list. If
top-k-sentences alone (no managed cache) already solves it, the cache still isn't the interesting
variable, even on a task designed to need two spans. Only if top-k-sentences under-performs a
genuinely managed cache (floor vs SnapKV vs full_cache) does this task actually demonstrate a
cache-management effect — which is the thing this entire thread has been unable to find anywhere
else.

**Step 3 — if and only if Step 2 shows a real, ablation-surviving cache-management effect, THEN
run the frozen tight-KV two-pass mechanism (or its single-prefill replacement, once §1.15/§3 item 3
is resolved) on bridge at 8B scale**, with the same rigor as the rest of this program: prereg
before generation, dual-CI gate, paired instance bootstrap, full context-integrity checks
(keep-hash, literal-mask fidelity, corrected full-context invented-value check).

**Do NOT skip Step 2.** The single biggest lesson of this entire thread is that every claimed
cache/mask win so far (real-text extra-KV, real-text tight-KV, LEDGER-C mask) has turned out to be
substantially or entirely explainable by "the right evidence was placed near the query," not by
anything about how the cache itself was managed. Moving straight to a tight-KV 8B run without first
checking whether a zero-cache pointer-plus-insertion baseline already solves bridge questions would
repeat the exact mistake this thread spent its whole session finding and correcting.

## 5. File index (this thread only; paths relative to `out/_retpolicy_dev/`)

| File | Contents | Section |
|---|---|---|
| `single_prefill_equivalence.py` | Equal-KV single-prefill reconstruction + verification | 1.15 |
| `equiv_M2.jsonl` / `equiv_M3.jsonl` | Round 1 (superseded — protected selected sentence, wrong) | 1.15 |
| `equiv_v2_M2.jsonl` / `equiv_v2_M3.jsonl` | Round 2 (corrected, authoritative) | 1.15 |
| `component_ablation.py` | 4-arm real-text ablation (sentence-only/unmasked/masked/full-cache) | 1.16 |
| `ablation_M2_full.jsonl` | Ablation raw results, M2, 160 instances | 1.16 |
| `text_available_bm25_baseline.py` | BM25-over-all-sentences reference baseline generator | 1.16 |
| `bm25_M2_full.jsonl` | BM25 baseline raw results (154/320) | 1.16 |
| `text_access_check.py` | CPU-only token-decode-vs-source-text fidelity check | 1.17 |
| `text_access_mismatches_M2.jsonl` / `_M3.jsonl` | Mismatch details | 1.17 |
| `retention_screen.py` | u_snapkv + farthest_first policies, prefill-only screen | 1.18 |
| `screen_M2.jsonl` / `screen_M3.jsonl` | Screen raw results | 1.18 |
| `ledger_proximity_check.py` | LEDGER-C extra-KV proximity insertion, unmasked | 1.19 |
| `ledger_proximity_M2.jsonl` | Proximity check raw results (96/320) | 1.19 |
| `ledger_equalkv_check.py` | LEDGER-C equal-KV attempt — **blocked, does not run to completion past instance 1** | 1.20 |
| `PRIOR_ART_RETRIEVAL_VS_COMPRESSION.md` | Web-search prior art, both rounds | 1.21 |
| `bridge_pilot_gate.py` | Bridge two-hop competence-gate generator | 1.22 |
| `bridge_M2.jsonl` / `bridge_M3.jsonl` | Gate pilot raw results (both fail band) | 1.22 |
| `PREREG_BRIDGE_TWOHOP_DESIGN.md` | Full bridge task design, arms, gate, risks | 1.23 |
| `REALTEXT_REINTERPRETATION.md` | Consolidated reinterpretation narrative (real-text + LEDGER-C sections) | 1.16, 1.19 |
| `POST_SESSION_RECORD_RETPOLICY.md` | This file | all |

## 6. Standing rules this thread followed (carried forward, not new)

Every frozen `PREREG`/`RESULT`/manifest file listed in `REALTEXT_SESSION_FULL_RECORD.md` §2 stayed
read-only throughout. New material went only under `out/_retpolicy_dev/`. Corrections issued
mid-thread (the selected-sentence protection bug in §1.15; the overclaimed "proximity effect"
wording in §1.16/§1.19; the M3 punctuation-normalization scoring bug in §1.22) are recorded here
and in their respective documents rather than silently fixed and re-presented as if they were
right the first time. The one unresolved item (§1.20) is reported as blocked, not worked around
or quietly dropped.
