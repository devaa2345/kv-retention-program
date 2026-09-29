# Candidate mechanisms for a method that beats `floor_pos` — committed BEFORE any test runs

Scope: every candidate grounded in a cited finding from Papers 1–4. No cap, no pre-selection for
plausibility. Rules below are written so a script cannot implement half of one — each REAL rule
lists **every** conjunct it requires, and the scoring script must print each conjunct's value
separately. (This project already shipped a script that implemented half of C1's rule and printed
the wrong verdict; `MECHANISM_RESULTS.md` records it.)

**Replication rule, applies to every candidate:** a candidate that clears REAL on one model is
**NOT a lead** until it clears the same threshold on the second model. Discovery-model-only results
are reported as "unreplicated", never as survivors. With this many candidates, threshold-clearing
by chance is expected; replication is the filter.

**Tier-1 screen power, stated up front:** tier-1 generation probes use n = 20 instances (80
queries). The paired 95% CI half-width at that n is roughly ±0.10, so tier 1 detects only **large**
effects. A tier-1 NOT-IT means "no large effect", never "no effect".

**Shared comparator cell** for tier-1 generation probes: **M2, LEDGER-C c=8, C=512** — admissible
(floor 0.200), with large measured headroom (`oracle_causal` 0.890) and a known `U-snapkv` effect
(+0.035). All arms of a probe run in ONE process (no session boundary inside a compared set), with
the Stage 3 invariants asserted per row (position_ids from uncompressed length, parity from
captured keep-sets, floors retained, generation required).

**Closed, not re-opened:** positional dispersion (`LIMITATIONS_P4.md` L1) and distractor-load /
whole-record interference (Paper 3). Any candidate that reduces to either is excluded and said so.

---

## Tier 1 — under 30 GPU-minutes each

### A1 — head-consensus unit allocation
- **Evidence.** Paper 2 `Δ_head` is **negative at every squeezed budget on all three models** —
  per-head allocation in isolation hurts. Paper 4 C1: slot agreement predicts accuracy at matched
  completion on both models (M2 +0.054 [+0.042,+0.067]; M3 +0.048 [+0.029,+0.067]). `floor_pos` is
  head-agnostic (agreement 1.000) and is the best converter.
- **Prediction.** Allocating one unit set from head-averaged scores and replicating it to every
  head beats per-head `U-snapkv` at matched budget, and closes part of the gap to `floor_pos`.
- **REAL** (all three conjuncts): (i) `A(consensus) − A(U-snapkv)` paired CI excludes zero,
  positive, on M2; (ii) the same on M3; (iii) `A(consensus) ≥ A(floor_pos)` point estimate on at
  least one model.
- **NOT-IT**: (i) fails, or (i) holds and (ii) fails (then: unreplicated, reported as such).
- **Test.** Generation, n=20, arms {floor_pos, snapkv, U-snapkv, consensus-U-snapkv}. **~8 min.**

### A2 — early-position unit prior
- **Evidence.** Paper 4 C3 came back **contradicted on both models**: gold *further* from the query
  converts better (M2 +0.032 [+0.012,+0.052]; M3 +0.065 [+0.038,+0.090]). C4 found exemplar
  retention negative on M2, so this is not "keep the head of the context" — it is a within-body
  early-vs-late ordering effect.
- **Prediction.** Breaking unit ties toward earlier units beats breaking them toward later ones at
  matched budget and matched score function.
- **REAL** (both): (i) `A(early-prior U-snapkv) − A(U-snapkv)` paired CI excludes zero, positive,
  on M2; (ii) same on M3.
- **NOT-IT**: (i) fails. **AMBIGUOUS**: (i) holds, (ii) null.
- **Test.** Generation, n=20, arms {U-snapkv, early-U-snapkv}. **~6 min.**

### A3 — floor-hybrid (split budget: contiguous tail + whole units)
- **Evidence.** `floor_pos` converts best in every admissible cell (Papers 2–4); Paper 3 probe 2.2
  showed the *same* gold-token count arranged coherently raises accuracy **2.6×** (0.125 → 0.325);
  Paper 4 shows `U-X` reaches near-floor completion but converts worse. A split budget keeps the
  floor's contiguous tail (which converts well) and buys whole units with the remainder.
- **Prediction.** At matched `B`, a hybrid spending half of `C` on the floor tail and half on whole
  units beats **both** `floor_pos` and `U-snapkv`.
- **REAL** (all three): (i) `A(hybrid) − A(floor_pos)` paired CI excludes zero, positive, on M2;
  (ii) same on M3; (iii) `A(hybrid) ≥ A(U-snapkv)` point estimate on both models.
- **NOT-IT**: (i) fails, or (i) holds and (ii) fails.
- **Test.** Generation, n=20, arms {floor_pos, U-snapkv, hybrid50}. **~8 min.**

### A4 — unit granularity (sub-line vs supra-line)
- **Evidence.** Paper 3 Evidence 4: an oracle holding every queried record **whole** still loses
  half its M3 accuracy when the record is split into two adjacent labelled parts — larger coherent
  units matter. Paper 3 also **excluded ChunkKV** because fixed-size chunks consume the whole
  budget at tight C (G2 permutation failure), i.e. granularity interacts with budget.
- **Prediction.** Supra-line units (record + its adjacent filler line) beat line units; sub-line
  units (one unit per field) do worse than line units.
- **REAL** (both): (i) `A(supra) − A(U-snapkv)` paired CI excludes zero, positive, on M2; (ii) same
  on M3. Sub-line is reported alongside as the directional check (expected worse).
- **NOT-IT**: (i) fails.
- **Test.** Generation, n=20, arms {U-snapkv, supra-U-snapkv, sub-U-snapkv}. **~9 min.**

### A5 — sink/anchor expansion
- **Evidence.** Paper 2's blind reimplementation: **dropping the BOS/sink token collapses every
  arm, including both oracles, to 0.0000.** Anchors are load-bearing far out of proportion to their
  token count; the harness pins `n_sink = 8` everywhere and has never varied it.
- **Prediction.** At matched `B`, raising `n_sink` 8 → 32 (taking the 24 tokens out of `C`)
  improves accuracy.
- **REAL** (both): (i) `A(sink32) − A(floor_pos)` paired CI excludes zero, positive, on M2;
  (ii) same on M3.
- **NOT-IT**: (i) fails.
- **Test.** Generation, n=20, arms {floor_pos, floor_pos-sink32}. **~6 min.**

### A6 — gather-order hygiene (falsification, not a method)
- **Evidence.** `ScorerPress.compress` gathers keys in **top-k score order**; `p4/unitwrap.py`
  gathers in **ascending position order**; `_ExactSetPress` (floor, oracles) gathers in yet another
  order under KEEP/DROP ties. Attention is permutation-invariant once RoPE is baked in, so this
  should be inert — but it is an uncontrolled difference between compared arms.
- **Prediction.** Identical keep-set, two gather orders ⇒ identical generations.
- **REAL (i.e. a hygiene violation, which would contaminate every X-vs-U-X comparison)**: paired
  accuracy difference CI excludes zero, or byte-identical generation rate < 100%.
- **NOT-IT (the expected, reassuring outcome)**: 100% byte-identical generations.
- **Test.** Generation, n=20, one arm, two orders. **~5 min.**

### A7 — budget-gated unit policy (CPU only, 0 GPU)
- **Evidence.** Paper 3's F5 failed its own kill criterion because its residual correlates with
  **log B at +0.917–0.948** — budget dependence is real and unmodelled. Paper 4's own grid shows
  the sign of `Δ_U` flipping with budget (M2 c=8: −0.030 at C=128, +0.035 at C=512).
- **Prediction.** A rule "use whole units only when `C` suffices for ≥ k complete units, else use
  the floor" dominates both fixed policies across the cells already collected.
- **REAL** (both): (i) the gated policy's implied accuracy exceeds both `floor_pos` and `U-snapkv`
  in a majority of admissible cells on M2; (ii) same on M3. Implied accuracy uses only
  already-collected per-cell arm accuracies — no new data, no fitting of `k` per cell (k is fixed
  at the smallest value that makes the rule non-trivial, stated before scoring: **k = 4**).
- **NOT-IT**: (i) fails.
- **Test.** CPU, existing rows. **0 GPU, ~1 min.**

### A8 — unit content composition (CPU only, 0 GPU)
- **Evidence.** The ExpectedAttention confound (`DIAGNOSIS_EA.md`): its value-norm rescaling makes
  allocation shift *which kinds* of tokens are kept (Δ p_g +0.084 at M2 c=8 C=512), while KeyDiff
  moves the other way (Δ p_g −0.080). Composition of a unit demonstrably drives retention.
- **Prediction.** Within an arm, the fraction of a retained unit's tokens that are numeric fields
  predicts whether its record is answered, at matched completion.
- **REAL** (both): coefficient on z(numeric-fraction of retained gold units) positive with CI
  excluding zero on both models, with |r| ≤ 0.8 against `q_slot`.
- **NOT-IT**: CI includes zero on both, or sign negative on either.
- **Test.** CPU, saved keep-sets. **0 GPU, ~3 min.**

### A9 — layer-selective unit awareness (prefill only)
- **Evidence.** Paper 1: promotion signals are interchangeable while *retention* carries a 0.52
  accuracy gap — where a token is kept matters more than how it is scored. Paper 4 C1 shows slot
  agreement varies by arm and predicts accuracy.
- **Prediction.** Slot agreement and per-slot completion are not uniform across layers; if unit
  packing helps only in some layer band, applying it everywhere is not optimal.
- **REAL**: per-layer completion under `U-snapkv` varies by more than 2× between the best and worst
  layer quartile on **both** models (a precondition for a layer-selective method to exist).
- **NOT-IT**: variation under 2× on either model.
- **Test.** Prefill only, existing keep-sets (no new GPU). **0 GPU, ~2 min.**

## Tier 2 — 30 min to 2 GPU-hours (run only after tier 1 reports in full)

### B1 — consensus × hybrid at full n, both models
Whichever of A1/A3 survives, confirmed at n=50 on one admissible cell per model. **≈ 1 GPU-h.**

### B2 — query-aware allocation
- **Evidence.** Paper 2's `I` = **0.73–0.75 at the tightest binding budget**: knowing which fact is
  queried is worth most of the oracle gap exactly where budget binds. Paper 4's `oracle_causal`
  reaches 0.890 against floor 0.200 at M2 c=8 C=512.
- **Prediction.** Scoring units by similarity to the query (a deployable proxy for the oracle's
  information) beats `floor_pos` at matched budget.
- **REAL** (both): CI-excluding-zero gain over `floor_pos` on both models.
- **Cost.** Protocol change (Paper 2 pinned query-agnostic) — must be reported as a different
  protocol, not as a comparable arm. **≈ 1.5 GPU-h.**

## Tier 3 — new task construction or > 2 GPU-hours (NOT run; prereg first, one at a time)

### C1' — surface-form integrity
Paper 3 Evidence 4 (oracle holds records whole, M3 accuracy halves when split) and Stage 2.3
(`full_cache` 0.910 → 0.400 under scattering, no compression). Needs a task where retained facts
vary in surface form at fixed content. **≈ 1–2 GPU-h + task construction.**

### C2' — retention-oracle gap transfer from Paper 1
Paper 1: protection-alone 0.423 vs retention-oracle 0.943 at retention 0.25. The same
oracle-vs-deployable gap measured in Paper 4's units. **≈ 2 GPU-h.**

## Dropped, with reasons (not silently omitted)

- **id+value minimal-set retention.** In LEDGER-C the query asks the model to *reproduce the whole
  record line*, so the minimal answerable set **is** the whole line (`ledger_c.py` docstring). The
  candidate is structurally void for this task family.
- **Whole-record interference / distractor suppression.** Closed by Paper 3's within-cell null.
- **Re-testing dispersion.** Closed by `LIMITATIONS_P4.md` L1.
