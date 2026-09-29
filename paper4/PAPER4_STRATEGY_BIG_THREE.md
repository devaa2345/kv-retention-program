# Paper 4 — Strategy for a Big-Three Submission

Written after reviewing the full Paper 4 record. The goal is a method that at least
matches `floor_pos` and a paper that can compete at ICML or NeurIPS.

---

## Part 1 — Why trial-and-error stalled

This is not bad luck. Four structural reasons, each fixable.

### 1.1 The screens could not see the effects that exist in this system

Every real effect Paper 4 has measured is small: fragmentation +0.035 to +0.045,
ExpectedAttention's order component +0.025, the hallucination attention deficit −0.013.
The A1–A9 GPU screen ran at n=20 with a CI half-width of about **±0.10**. It could only
detect effects **two to three times larger than any effect ever observed in this
program.** Six nulls were close to guaranteed before the screen ran.

The consequence matters: **A1–A9 were not rejected. They were unmeasured.** In particular,
**A3 floor-hybrid tied the floor exactly (0.237 vs 0.237)** — the only arm in the whole
search that reached the target, and it was filed as a null because the CI was wide.

### 1.2 The search fought floor on floor's home ground

At C=512 and L≈2048 the floor keeps about 28% of the context as one contiguous recent
block. At that ratio a recency window is genuinely a good policy, and uniform record
interleaving guarantees it catches a fixed share of records. Deployment ratios are 1–10%
at 8k–128k context. Floor's completion scales roughly as `(B−c)/L`; a scorer's does not
depend on `L` in the same way. **The length sweep (T5) was planned twice and never run**,
and it is the single experiment most likely to produce "beats floor."

### 1.3 The search optimised the smaller of two levers

Paper 2 measured the information share `I` = 0.73–0.75 at binding budgets: three quarters
of tight-budget headroom is the cost of not knowing the query, and only a quarter is
ranking. Everything in Paper 4 — allocation, spacing, gather order, head consensus —
tuned **ranking and arrangement**, the 25% lever.

### 1.4 Diagnosis kept outrunning intervention

Contamination received five hypothesis checks, a spacing prereg with four escalations, a
gap recovery, and a slot probe. The mechanism narrowed; nothing was recovered. Meanwhile
the one confirmed and actionable mechanism — hallucination's attention deficit — has had
no intervention built against it. Contamination's full-fix ceiling is near-parity on two
of three arms; it was never large enough to carry the paper by itself.

---

## Part 2 — The reframe

Stop asking "which tokens should a scorer keep instead of the floor." Ask:

> **The floor is a strong prior. What can be added to it?**

This changes the claim from "our selector beats position" (which three papers of evidence
say is false at realistic fact sizes) to "position plus X beats position alone." It also
makes the minimum guarantee structural: a hybrid that reserves a fraction ρ of budget for
X reduces to the floor at ρ=0, so a validated ρ can never be worse than floor by
construction. Reviewers value that property.

Four levers, grounded in results already in hand.

### Lever 1 — Floor-anchored hybrid (ranking, done carefully)

Keep the floor's contiguous sink + recent block. Spend a reserved fraction ρ on whole
units selected by a scorer from the far context. Evidence: A3 tied the floor at n=20;
Paper 3's c=1 result shows scorers win 3× on small units; fragmentation is confirmed, so
whole units are the right granularity. **Tune ρ on a held-out split, never on test.**

### Lever 2 — Coverage-aware selection (the anticipatory lever)

The query is unknown at compression and uniform over candidates. The optimal causal
policy maximises the number of *distinct* complete facts retained — that is literally
what `oracle_causal` does. Contamination showed the base scorer ranks near-identical
neighbouring records both high (recency makes them rivals). A redundancy-penalised
selection (MMR-style or submodular coverage over units) spends budget on distinct facts
rather than on clusters of similar ones. This targets the 75% share, not the 25%.

### Lever 3 — Grounded decoding (the hallucination lever)

Test A is the strongest mechanism in the program: at the divergence point (token 0 in
23/25 cases) the model under-attends to the queried record's content, independent of
position. The query *is* known at decode time, so query-conditioned intervention there is
legitimate. Candidates: attention bias toward retained spans matching entities named in
the query; or abstention when attention mass on any candidate record falls below a
threshold. Target population: 25/141 of M2 disagreements, the largest single slice found.

### Lever 4 — Change the battlefield (length)

Run every lever at L ∈ {2048, 4096, 8192} and deployment budget ratios. If the floor
collapses as predicted while a hybrid holds, the paper's central figure is the crossover
length, predicted in advance from `(B−c)/L`.

**Deliberately deprioritised:** further contamination archaeology (ceiling capped, all
visible hypotheses closed), position remapping (the invariant exists because it once
zeroed every arm; revisit only with a separate prereg), and new allocator variants
screened at n=20.

---

## Part 3 — What a big-three submission needs

Stated plainly so the plan is judged against it.

| Requirement | Status |
|---|---|
| A method that beats strong baselines | Not yet |
| Baselines: StreamingLLM/floor, SnapKV, PyramidKV, AdaKV, **ChunkKV** | Partial; ChunkKV and PyramidKV missing |
| Standard benchmarks (RULER, LongBench, NIAH) | None |
| Scale: 7–8B+ | Only 3B for Paper 4 |
| Efficiency analysis (latency, memory) | None |
| A principled "why" | **Strong** — this is the program's real differentiator |

The last row is the asset. Most eviction papers show a method wins; few can say why the
field's methods fail. Papers 2–3 plus Paper 4's mechanism findings are a motivation
section almost no competing submission will have.

**The RTX Pro 5000 48 GB changes the scale row.** Llama-3.1-8B at bf16 with 16k context
fits, which makes RULER and LongBench feasible at a scale reviewers accept — the thing the
5070 could never do.

**Honest odds.** As it stands, a big-three submission would not be competitive. With a
method that robustly matches or beats the floor, beats the other baselines at long
context on RULER/LongBench at 8B, and carries the mechanism story, odds become roughly
the venue base rate (~20–30%). The mechanism depth is what could push above base rate.

**Target: ICML 2027** (deadline typically late January — verify on the official site).
About four months. MLSys (30 Oct 2026) conflicts with that timeline; treat it as optional.

---

## Part 4 — The plan

Every experiment: threshold committed before data, paired CIs, per-model verdicts,
confound check before scoring — the same discipline as before. Two new rules:

- **Minimum screen power.** No candidate screen below the n needed to detect ±0.04 (the
  size of real effects here). Compute n before running; if it is too expensive, don't
  screen that candidate.
- **Ceiling before test.** Every hypothesis states its accuracy ceiling before running,
  as contamination's did.

### Phase 0 — Consolidate and decide (days 1–2, mostly CPU)

1. Read `PAPER4_COMPLETE_RECORD.md`. Commit it.
2. **Overlap check** (CPU, from existing transcripts): of the 25 plausible-fabrication
   cases, how many are contamination-shaped (a specific substituted record) versus
   genuinely unsourced? Decides whether Test A's mechanism covers contamination too.
3. **Power-correct A3.** Compute the n needed to resolve A3's tie at ±0.04.

### Phase 1 — Three decisive experiments (week 1)

Each can change the strategy. Run in this order.

**E1 — Length sweep.** Anchors first: `full_cache` at L ∈ {2048, 4096, 8192} on M2/M3,
c≈8 and c≈19 (not c=40, where the oracle collapsed at long context on 14B). Gate: anchor
in [0.55, 0.97] **and** floor > 0.05 at each planned budget. Then floor vs X vs U-X vs
hybrid at fixed B, crossover length predicted and committed in advance.
- Floor collapses and U-X/hybrid holds → the paper's central result.
- Floor holds at all lengths → the floor is genuinely strong; pivot weight to Levers 2–3.
- Competence dies at 8k on 3B → move E1 to the 48 GB card at 8B.

**E2 — Hybrid at proper n.** A3 floor-hybrid, ρ ∈ {0.1, 0.25, 0.5}, n sized for ±0.04,
M2 and M3, ρ selected on a held-out split. Question: does any ρ > 0 exceed floor?

**E3 — Grounded decoding pilot.** One intervention against Test A's mechanism, prereg'd,
evaluated on the 25 fabrication cases and a matched correct set (to check it does not
break correct answers). Ceiling stated first.

### Phase 2 — Build the method (weeks 2–4)

Combine what survives Phase 1: floor-anchored, coverage-aware, grounded-decoding. Ablate
each component; each must earn its place with a CI excluding zero. Freeze a single method
definition with no per-task tuning.

### Phase 3 — Scale and standard benchmarks (weeks 5–9, 48 GB card)

- Llama-3.1-8B and Qwen2.5-7B, bf16 (no quantisation — it reintroduces Paper 1's
  confound).
- RULER, LongBench, NIAH, at deployment budget ratios.
- Baselines: StreamingLLM, SnapKV, PyramidKV, AdaKV, ChunkKV, LU-KV if curves available.
- Report the fact-cost of each benchmark (T3) — it explains where each baseline wins.
- Efficiency: prefill latency, decode latency, memory, vs each baseline.

### Phase 4 — Write (weeks 10–14)

Story: the field's methods fail at realistic fact sizes for identifiable reasons
(fragmentation, anticipation, grounding); here is a method built on those reasons; it
wins where deployment actually happens.

---

## Part 5 — Decision points and kill criteria

| Point | If... | Then |
|---|---|---|
| Phase 0 overlap | fabrication and contamination share structure | One mechanism; drop the contamination lane |
| E1 | floor collapses at long L | Lead with length; hybrid is the method |
| E1 | floor holds everywhere | Floor is strong; lean on Levers 2–3 |
| E2 | no ρ > 0 beats floor at proper n | Hybrid lever is dead at 2k; test only at long L |
| E3 | intervention recovers fabrications without hurting correct cases | Keep as a component |
| End of Phase 2 | nothing ≥ floor anywhere | Paper becomes "floor is Pareto-optimal under measurable conditions, and here is why" — publishable, but not as a big-three method paper |

The last row is the honest fallback, and it should be named now rather than discovered
in January.

---

## Part 6 — What not to do

- No more contamination feature-mining on the same 20 cases.
- No n=20 screens for effects that are known to be ~0.04.
- No method tuned on the evaluation cell.
- No quantised large models.
- No cross-device pooling; the 48 GB card is its own stratum.
