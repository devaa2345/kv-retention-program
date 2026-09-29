# Contamination — unexplained residual, full context (snapshot, not a new test)

Standalone record of the contamination investigation for whoever picks this thread up next. No
new synthesis or interpretation beyond what each cited commit already established.

## 1. Origin

Contamination was identified via a Track 2 transcript read-through as M2's dominant failure mode
among complete-but-wrong disagreement cases (floor-correct, U-wrong): 6 of 12 cases in the initial
read (`a500cb9`, "Track 2 failure read"), M3 dominated instead by refusal (7/12). A full-population
selection check (`c122dab`, "Track 2 selection check") confirmed this was not a sampling artifact
of the initial 12-case read: **14/82 (~17%)** of M2's full disagreement population is contamination,
against 0% refusal on M2 — the mirror image of M3 (31% refusal, 5% contamination, the latter
initially hidden by an overlap filter and corrected in the same commit).

## 2. Mechanism — two layers, both verified

**Layer 1 — generic recency bias in the base scorer.** Confirmed at `6d65f79` ("Step 3: full-rank
check"): pooled within-instance position-vs-score Spearman rho, computed across ALL ~41 records
per instance (not just the two records involved in a given contamination case), mean = +0.608,
median = +0.645, n=20 instances. Both the queried record and the record the model substitutes are
in the top half of the per-instance score ranking in 20/20 cases — recency makes both records
plausible candidates; it does not by itself explain which one wins.

**Layer 2 — a residual, position-independent preference for one record over the other.** The
substituted record scores higher than the queried record on the RAW base score in 18/20 cases
(`d41ccf6`, "Step 2 data pull" — after fixing a floor-region-token contamination bug in the score
average, see that commit's message). The same 18/20 count survives unchanged after per-instance
linear position-detrending (`6d65f79`): removing the generic recency trend does not remove the
preference. **This residual is real, replicated under two different measurement passes, and its
cause was unidentified as of this writing** (see §3-§4 below for what has since been ruled out).

## 3. Five candidate explanations tested and eliminated for the residual

All five run and reported separately, per instruction, at `f45f855` ("Step 4: five candidate
correlates") unless noted:

1. **Token length** of the substituted vs. queried record: null. Pearson r=+0.192 (p=0.417),
   Spearman rho=+0.095 (p=0.692).
2. **Numeric field magnitude**: null/inconclusive. Pearson r=+0.014 (p=0.953), Spearman
   rho=-0.095 (p=0.690); a weak binary "larger max wins" match (14/20) was noted but not formally
   significant.
3. **Earlier-query compounding** (was the substituted record itself queried earlier in the same
   instance, biasing later attention toward it): **structurally impossible by harness design**,
   not merely untested — each query variant runs generation from an independent clone of the same
   prefilled KV cache (`p3/runner.py:generate_with` / `p4/common.py:generate_checked`, `cache =
   _clone(base)` per variant), so there is no mechanism for one query's generation to leave any
   trace visible to another query's generation. Only 2/20 cases even had the structural
   precondition (co-occurrence), consistent with this being a non-effect.
4. **Query-vocabulary lexical overlap** between the substituted record's neighboring filler text
   and the query wording: null, point estimate in the opposite direction from the naive
   expectation. Pearson r=-0.361 (p=0.118); binary match 9/20 with 5 ties.
5. **Record-ID content stickiness** (does this specific record ID score generically higher across
   OTHER instances too, not just this one): **initially significant** on raw score, r=+0.709,
   p<0.001, 20/20 cases with a comparable bystander sample (mean 4.9 other instances per case)
   (`f45f855`). Re-tested on position-detrended residuals instead of raw score — **still
   significant**, r=+0.668 (p=0.0013), Spearman rho=+0.625 (p=0.0032), same sample comparability
   (`b7a86a5`, "Step 5"). But only **9 distinct record-content indices** back all 20 cases, and
   every one of them sits at record index i in {26,29,30,31,32,33,34,36,38} — the corpus's
   deterministic slot formula in `paper3/p3/tasks/ledger_c.py:170-187`
   (`pos = round((i+0.5)*total/n_records)`) means these 9 indices have **never** been observed
   anywhere but the document's last third, in any instance generated for this project. A
   preregistered probe (`SLOT_IDENTITY_PROBE.md`, committed `12997c1`) generated 20 fresh
   instances with content-to-slot assignment randomly permuted per instance
   (`out/_step6_permuted_build.py`), confirmed the precondition (9/9 flagged indices spanned a
   39-49 slot range, comfortably clearing the >=15 bar), and re-ran the same residual correlation
   restricted to slot-decoupled instance pairs. Result: **NOT-IT** — Pearson r=-0.099 (p=0.189),
   Spearman rho=-0.323 (p<0.0001), 177 usable pairs (`52095cc`, "Step 6"). The apparent stickiness
   was an artifact of imperfect linear detrending near the document tail, not a genuine per-record
   content property; once decoupled from slot, the preference did not just disappear, it weakly
   reversed.

**All five candidates generated from the 20-case transcript read are now closed. None explain the
residual.**

## 4. Fix attempted and rejected: spacing intervention

Preregistered at `9c181c9` ("Prereg the spacing hypothesis"), before any spacing-constrained row
was generated: enforce a minimum token-distance gap (K=24) between kept units competing for the
same query, with instance-level relaxation when the budget can't otherwise be met, scored via
contamination-tagged accuracy as the primary readout. Scope: M2 dominant mode plus M3's minority
contamination; M3 refusal explicitly out of scope from the start.

- Implementation at `73628d3` (fixed a budget-eviction bug in CPU testing before any GPU row).
- First scored pass (`28530e2`): both models NOT SCORED against the contamination-tagged-query
  precondition (<20 for SnapKV alone).
- Amendment S1 (`814464e`, digest re-recorded `f0e7f35`): widened scope to pool SnapKV +
  ExpectedAttention; AdaKV blocked on a cross-head allocator mismatch, not silently approximated.
  Pooled S1 result (`fbbaf7e`): still NOT SCORED (M2 9 pooled queries, M3 1; floor is 20).
- Amendment S2 (`2434918`, digest `7e065dd`): cross-head-aware AdaKV spacing allocator built and
  unit-tested before any AdaKV row was generated; wired in at `c8992eb`, a real crash at instance
  50 found and fixed at `53a0ea7`.
- n-raise scale-up infrastructure at `dabc141`; scored at `1da3fe4`: M2 (n=120) reached 16/20
  pooled contamination queries, M3 (n=150) only 3/20 — both still short of the 20-query floor,
  both reported NOT SCOREABLE per instruction, not chased further at that pass.
- **Three-arm pooled final result** (`3d06852`): M2 finally clears the floor (26/20 pooled
  queries). Scored contamination-tagged accuracy: **0.000, NOT-IT, CI [0.000, 0.000] — 0 of 26
  correct.**
- Instrumentation follow-up (`172db42`, "Track 3", no new hypothesis/prereg): a prefill-only GPU
  re-run recovered per-slot keep-sets for the 26 still-wrong cases. Achieved separation gap was
  dramatically different between two sub-groups: 6 cases had a median 2-token gap (relaxation
  silently failed to enforce the spacing constraint — excluded as test artifacts) versus 20 cases
  with a median 44-98 token gap (genuine, substantial separation achieved, model still wrong on
  every one).

**Mechanistic reading, on the 20 genuinely-separated cases: the model shows no distance-sensitive
record disambiguation.** Denied its nearest wrong answer by forced separation, it does not recover
the correct answer — it produces a more diffuse wrong answer instead. Spacing does not fix
contamination; the failure is not (solely) about physical proximity of the competing record.

## 5. Accuracy ceiling, computed before the final (slot-identity) test

From `PILOT_C40_POSTFIX.md` (post-A4-wrapper pilot cell, commit `68dc570`): current M2 U-X
accuracy at c=40, C=512 is snapkv 0.080, adakv_snapkv 0.090, expected_attn 0.035, against
`floor_pos` 0.170 at the same cell. Using the 9 flagged records' ~20-case share of the
contamination population (an upper-bound estimate relative to the original 14/82 Track 2 count,
since the 20-case set was drawn from the later, larger Track 3 gap-recovery sample — this mismatch
was flagged explicitly, not treated as an exact subset ratio) spread across the 3 arms where these
records actually cause contamination (not keydiff): a **100%-effective fix to exactly this
mechanism** would move those three arms to approximately **0.213 / 0.223 / 0.168** — near-parity
with `floor_pos` on two of three arms, still short on the third, at a theoretical maximum that was
never going to be fully achievable even before the slot-identity test came back NOT-IT. This
ceiling was reported in full before the slot-identity probe ran, specifically so a REAL or NOT-IT
verdict would be read against what it could actually be worth rather than treated as decisive
regardless of scale.

## 6. Current state

**Unsolved.** The cause of the Layer-2 residual preference (§2) is unidentified. Every feature-
based hypothesis generated from reading the 20-case transcript sample has been tested under a
preregistered threshold and closed as null, structurally impossible, or (for stickiness)
reclassified as a measurement artifact once properly decoupled from its confound. The spacing fix
built on the "nearby competing record" framing was preregistered, built correctly for every arm,
and returned a clean 0/26 NOT-IT on genuinely-separated cases — ruling out proximity as the fixable
lever, not just failing to find a fix for it.

**Remaining open avenues, explicitly for future work, not attempted:**
- (a) Direct inspection of attention weights at the divergence point, rather than derived/
  aggregate score features — nothing in this thread has looked at attention directly.
- (b) A larger or differently-constructed sample, from which new hypotheses (beyond the five
  already closed) could be formed; the current hypothesis set was exhausted by what a 20-case
  transcript read could suggest.
- (c) Whatever the parallel hallucination and refusal investigations turn up, checked afterward
  for whether it generalizes back to contamination — no assumption made either way about whether
  the mechanisms are related.
