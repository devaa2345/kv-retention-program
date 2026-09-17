# PREREG_P2 — Pre-registration, Paper 2

> ## ✅ STATUS: **FROZEN AND HASHED — 2026-09-06**
>
> This document **supersedes `PREREG_P2.md`** (frozen 2026-09-06, sha256 `8ac3895709be0721656cef0c705cf843302245ab6d841373ee1676032db203e9`).
> It is a **new pre-registration, not an amendment**: a run against a modified instrument is a
> new pre-registration. v1 is retained unaltered as the superseded record.
>
> **No records were ever generated under v1**, and none have been generated under v2 either:
> `runs/` is empty, no method has been admitted, no arm has entered a ratio.
>
> **Frozen from this moment. Neither session edits this document again.** If either believes it
> needs changing, it **stops and writes to `STATUS.md`**.
>
> Hashed only after Stage 5 ladder validation passed on both admitted models at every admitted
> budget: checks A, B, C and D all pass in 11/11 cells (`gates/nvidia/stage5_ladder_*.json`).
>
> The finding-1 slot in §11 remains **NOT APPLIED — UNRESOLVED**.

**Governs:** Paper 2 — *Two Ceilings: How Much of the KV-Retention Headroom Is Capturable, and
How Much Is Irreducibly Predictive?*
**Derived from:** `PAPER2_DESIGN_AND_EXECUTION_PLAN.md` (v1) and
`PAPER2_PLAN_V2_TWO_MACHINE.md` (device allocation and timings only).
**Predecessor:** Paper 1, closed. `../PREREG.md`, `../ADDENDUM_2026-09-03.md`, `../ANALYSIS.md`.
**Version:** v2. **Draft date:** 2026-09-06.
**Supersedes:** `PREREG_P2.md`, sha256 `8ac3895709be0721656cef0c705cf843302245ab6d841373ee1676032db203e9`.

---

## 1. What this document does

It fixes, in advance of any comparison run, every choice that could otherwise be made after
seeing results: the reference ladder, the two metrics and their refusal rules, the budget
ladder, the arm roster, the protocol, the admission gates, the outcome framings, and the
analysis script's behaviour.

Paper 1's practice applies unchanged: **a run against a modified instrument is a new
pre-registration, not an amendment to this one.**

## 2. Inputs already measured (Stage 1, complete, gate PASS)

These are facts, not choices. Recorded here because the decisions in §3 depend on them.
Source: `gates/nvidia/stage1_budget_arithmetic.json`, 24 probe instances per model, every
length measured in that model's own tokenizer.

**Re-derived 2026-09-06 on the ONE-HOP task at N=40, across the six-cell ladder.** Payable
figures govern: tokens inside the mandatory floors are retained by every arm for free.

| model | L median (range) | `k_gold` payable | `K_all` payable med / max |
|---|---|---|---|
| M1 `Qwen2.5-1.5B` *(EXCLUDED, §3.10)* | 2045.0 [2025, 2064] | 19.0 | 75.0 / 79 |
| M2 `Qwen2.5-3B` | 2046.0 [2033, 2064] | 19.0 | 75.5 / 79 |
| M3 `Llama-3.2-3B` | 2048.0 [2022, 2064] | 13.0 | 52.0 / 55 |

Cell classification, `zone vs k_gold / zone vs K_all` (`B = C + 72`):

| cell | C | B | M2 Qwen2.5-3B | M3 Llama-3.2-3B |
|---|---|---|---|---|
| `b-2` | 16 | 88 | VOID / VOID | PARTIAL / VOID |
| `b-1` | 32 | 104 | PARTIAL / VOID | COMPETITIVE / VOID |
| `b0` | 64 | 136 | COMPETITIVE / VOID | COMPETITIVE / PARTIAL |
| `b1` | 128 | 200 | COMPETITIVE / PARTIAL | COMPETITIVE / COMPETITIVE |
| `b2` | 256 | 328 | COMPETITIVE / COMPETITIVE | COMPETITIVE / COMPETITIVE |
| `b3` | 512 | 584 | COMPETITIVE / COMPETITIVE | COMPETITIVE / COMPETITIVE |

**`b-2` is VOID on the Qwens** (`k_gold` 19 > C 16) and is excluded from their grid before any
run, per v1 §3.2. Every other admitted cell clears `k_gold`. PARTIAL cells are run and reported
separately, never pooled.

### Throughput, re-measured under the pinned WSL toolchain at `max_new_tokens = 32`

200 timed records per model, 2048 prefill + **32** decode, bf16, `sdpa`, batch 1.

| model | @12 decode | **@32 decode (adopted)** | peak VRAM |
|---|---|---|---|
| M1 Qwen2.5-1.5B | 0.2666 s | **0.5249 s** | 3,648 MiB |
| M2 Qwen2.5-3B | 0.4249 s | **0.7705 s** | 6,696 MiB |
| M3 Llama-3.2-3B | 0.4186 s | **0.7517 s** | 6,874 MiB |

(Windows, for reference, was 0.855 / 1.274 s at 12 decode; the WSL move remains a ~3x win at
matched decode length.)

**Two decisions raise the cost and must be stated together with the win.** B9's move to 32
decode tokens roughly doubles per-generation time, and decision 7's mean-over-`H` scoring
requires `H = 4` generations per instance per arm. Under the query-agnostic protocol the
2048-token prefill is paid **once** per instance per arm and all `H` queries reuse that
compressed cache, so the marginal cost is `H` short query prefills plus `H x 32` decode tokens,
not `H` full prefills. Measured per instance-arm: **M2 2.429 s, M3 2.351 s.**

**Batching the H variants was tested and rejected** (`gates/nvidia/batch_identity_*.json`):
92/100, 94/100 and 83/100 generated-token-id matches against batch 1, and a divergent generation
returns a wholly different value rather than a tie-break. `batch_size` stays in the dedup key.

Machine N, two admitted models across their admitted budgets:

| package | instance-arms | hours |
|---|---|---|
| N6 main grid M2 (5 budgets) | 24,000 | 16.20 |
| N7 main grid M3 (6 budgets) | 28,800 | 18.80 |
| N8 aware sub-grid | 7,200 | 4.78 |
| N1–N4, N9, N10 | 18,400 | 12.21 |
| **total** | **78,400** | **52.0** |

**~52 h raw / ~68 h buffered (~5.6 days at 12 h/day).** Essentially unchanged from the
three-model four-budget estimate: dropping M1 and adding two budgets roughly cancel.

---

## 3. DECISIONS

| # | Decision | Resolution | Status |
|---|---|---|---|
| 1 | Context length | `L = 2048` | ☑ ADOPTED |
| 2 | Floor arm | `floor_pos`: `n_sink=8` + last `C−8`, fixed split | ☑ ADOPTED |
| 3 | Ceiling arms | triple: `oracle_causal` + `oracle_prescient` | ☑ ADOPTED |
| 4 | Budget ladder | `C ∈ {16,32,64,128,256,512}`, `B = C + 72` | ☑ ADOPTED (extended) |
| 5 | Method roster | 5 kvpress Tier-A + LU-KV + ForesightKV(SFT-only) | ☑ ADOPTED |
| 6 | Protocol & backend | query-agnostic primary; `sdpa` pinned everywhere | ☑ ADOPTED |
| 7 | **`oracle_causal` packing rule** | optimal complete-pair knapsack; mean over all H variants | ☑ **RESOLVED (human)** |
| 8 | M3 identity | rule applied → **Llama-3.2-3B-Instruct** | ☑ ADOPTED |
| 9 | Δ_head scope, LU-KV tier, press checks, H2O | see §3.9 | ☑ **RESOLVED (human)** |

Decisions 1–6 and 8 were delegated and are resolved below, each with the concrete check that
was run to look for a blocker. **Decisions 7 and 9 were settled by the human on 2026-09-06.**

### 3.1 Context length — `L = 2048`. ADOPTED.

Adopted as recommended, with no blocker found. Measured realised length is 2049–2054 median
across all three tokenizers, inside the 2048 ± 16 tolerance, and the range [2031, 2064] never
strays further. The reasons hold empirically: every budget clears the floors (§2, zero VOID
cells against `k_gold`), and the VRAM headroom is real — worst measured case is 11,323 MiB of
12,227 under `eager`, and under the `sdpa` pinned by decision 6 no attention matrices are
materialised at all. The one cost is that 2048 is short enough that a reviewer will ask about
external validity; Task C on natural LongBench text (v1 §4.3) is the answer to that and stays
in scope.

### 3.2 Floor arm — `floor_pos`, `n_sink = 8`, fixed split. ADOPTED.

Adopted as recommended. The decisive argument is Paper 1's own data: its `random` floor was
**absorbing**, scoring 0.010–0.013 at *every* budget including 514, which made
`G_m ≈ 1.07 · A_m` — a relabelling, not a normalisation. A position-only floor is graded,
genuinely zero-cost, and is the strongest trivial baseline in the published matched-budget
audit. Implemented at `harness/ladder.py:floor_pos` and unit-tested to retain exactly `B`
tokens including the sink and the full recency window. `random` and `null` are retained as
**tripwires** with the registered ordering `null ≤ random ≤ floor_pos` asserted per cell
(`harness/stats.py:check_ladder_ordering`), never as denominators.

### 3.3 Ceiling arms — the causal/prescient pair. ADOPTED.

Adopted as recommended, and §3.7's evidence makes the case stronger than v1 stated. Measured at
b0, `oracle_prescient` completes the queried fact on 100% of instances while `oracle_causal`
completes it on 47.5–70% depending on the degradation rule — so the two ceilings are separated
by a large, real margin exactly where the paper claims a separation exists, rather than by
assertion. Without the split, a single "oracle" number at b0 would silently mix "a better
ranker could close this" with "no causal method could ever close this", which is the
misdirection v1 §1.2 exists to prevent.

### 3.4 Budget ladder — `C ∈ {64,128,256,512}`, `B = C + 72`. ADOPTED.

Adopted as recommended; the Stage 1 gate passes with zero VOID cells against `k_gold` on all
three models. Inverting v1's derivation (fix `C`, derive `B`) is what avoids Paper 1's trap,
where 3 of 4 percent-of-context budgets were void before a GPU cycle was spent. One property
of the adopted ladder is worth stating in advance rather than discovering: **b0 is the only
cell where the causal oracle degrades**, so b0 carries all of decision 7's consequences and all
of the informative `I`, while b1–b3 carry the `G_m` table. b1–b3 have `I ≈ 0` structurally.

### 3.5 Method roster. ADOPTED, with LU-KV reclassified.

Adopted as recommended. Six methods spanning the objective space, one per family, five at zero
implementation cost. The one change is provenance rather than membership: **kvpress v0.5.4
ships `lukv_press`**, so LU-KV moves from "our port" to "upstream press + our profiling"
(§4.3). This lowers implementation risk materially — the convex-hull relaxation and greedy
solver, the part most likely to be subtly wrong in a negative-result paper, is now upstream and
Apache-2.0. It does **not** remove the offline profiling pass, because no budget curve exists
for any of our models. `ObservedAttention`/H2O is excluded as a consequence of decision 6, not
on merit. All exclusions carry written reasons into the paper's Limitations verbatim.

### 3.6 Protocol and backend — agnostic primary, `sdpa` pinned. ADOPTED.

Adopted as recommended. Query-agnostic is the only protocol under which a *causal* oracle means
anything, and it is the deployment that makes a compressed cache economically interesting.
`sdpa` is pinned in every cell. **Measured cost of that pin: `sdpa` is slower than `eager` on
this box** — 0.855 vs 0.767 s/record (M1) and 1.274 vs 1.037 (M2), an 11–23% throughput
penalty. Adopted anyway and without hesitation: v1 §5.3 records a 0.221 accuracy swing from a
backend change on mathematically equivalent kernels, which is larger than most method-vs-
baseline gaps in this literature. Throughput is not the currency the backend decision is
denominated in. The aware calibration sub-grid runs as specified, and its published reference
ordering (SnapKV ≫ AdaKV > TOVA > ExpectedAttention > KeyDiff) is the harness's own tripwire.

### 3.8 M3 identity — rule applied. ADOPTED: `meta-llama/Llama-3.2-3B-Instruct`.

The frozen rule was: Llama-3.2-3B if the license has landed by the time it is checked, else
`microsoft/Phi-3-mini-4k-instruct`. **Checked 2026-09-06: the Llama-3.2-3B-Instruct weights
resolve** (commit `0cb88a4f764b7a12671c53f0838cd831a0843b95`), so the primary branch of the
rule fires and M3 is Llama-3.2-3B-Instruct. The substitution does not trigger, and §2's M3
row therefore stands as measured and needs no re-derivation. Phi-3-mini remains the documented
fallback if access is later withdrawn. Note that Llama-3.2 remains the model with the tightest
causal-oracle budget (107 payable candidate tokens against Qwen's 90, and 0 free in the floors
against Qwen's 44), so it is the model where decision 7 bites hardest.

### 3.7 — Decision 7: `oracle_causal` packing. **RESOLVED (human, 2026-09-06).**

> **`oracle_causal` retains complete candidate span-pairs only, selected to maximise the number
> of complete pairs fitting within `C` — greedy knapsack, pairs packed in ascending token cost.
> No partial spans. Remainder filled from `floor_pos`. Each instance is scored as the mean over
> all `H` query variants.**

**Rationale (human).** A causal oracle must be the *optimal policy given its information*. It
knows content but not the query; the query is uniform over `H`; a fact scores only if both its
spans survive. Expected accuracy is therefore `(complete pairs) / H` and the argmax is forced.
There is no convention left to choose: the ceiling is determined by its own definition.

**All five previously-tabled candidates are STRUCK**, not chosen between. Document-order,
shortest-first, earliest-position, round-robin and seeded-random each complete strictly fewer
pairs than achievable, which makes every one of them a *policy wearing a ceiling's label*
rather than a ceiling. The measured `I` values they produced (0.200 → 1.000, §3.7 of the
previous draft) were artifacts of that mislabelling and are withdrawn.

**Assertion, per instance, on live data.** `harness/ladder.py:oracle_causal` verifies by
exhaustive search over pair subsets (`_optimal_pair_count`, `2**H` subsets, H = 4) that **no
admissible packing completes more pairs**, and raises `AssertionError` if the packing is
beaten. It additionally asserts that the number of complete facts equals the number of packed
pairs, so a partial pair can never be retained silently. `n_pairs_taken`, `n_facts_complete`
and `expected_accuracy` are logged on every record.

This closes the D-04 class for this arm: there is no unordered set, no arbitrary subset, and
the selection is checked against the optimum rather than trusted.

**Scoring change this forces.** Every arm — not only the oracle — is now scored as the mean
over all `H` query variants of an instance, because a single-variant score would make the
oracle's accuracy a lottery on *which* fact its packing happened to complete. `LEDGER` emits
all `H` variants (`harness/tasks/ledger.py:Variant`), and `score_instance()` takes their mean.
Under the query-agnostic protocol the KV cache is compressed **once per instance per arm** and
all `H` queries are asked against that same compressed cache, so the marginal cost is `H` short
query prefills plus `H × 32` decode tokens, not `H` full 2048-token prefills.

### 3.8 — Decision 8: M3 identity

**Freeze the rule, not the outcome:**

> **M3 is `meta-llama/Llama-3.2-3B-Instruct` if the license is accepted on this box before this
> document is hashed. Otherwise M3 is `microsoft/Phi-3-mini-4k-instruct`.**

Status here: Llama-3.2 tokenizer configs resolve unauthenticated (which is why M3 appears in
§2's table); **weights do not**. If the substitution fires, §2's M3 row must be re-measured
before hashing, because `k_gold`/`K_all` are tokenizer-dependent and Phi-3 uses a different
tokenizer — the substitution is **not** free of consequences for the budget arithmetic.

Phi-3-mini rationale (yours): ungated, 3.8 B (~7.6 GB bf16, fits 12 GB at 2048),
`Phi3ForCausalLM` upstream-tested in kvpress, preserves the second-lineage requirement M3
exists to satisfy.

### 3.9 — Decision 9. **RESOLVED (human, 2026-09-06).**

Four rulings, all adopted:

**(a) `Δ_head` is measured on M3/M4 only.** M1 and M2 have **2 KV heads** each, so per-head
budget allocation has almost no room to express itself and `Δ_head` on them would measure the
architecture rather than the methods. M3 (Llama-3.2-3B, 8 KV heads) and M4 (Qwen2.5-7B) carry
the head-allocation question. v1 §3.4's `oracle_causal_global` vs `oracle_causal_perhead`
contrast is therefore reported for M3/M4 and **not** for M1/M2, which is stated in the results
rather than left as a silent absence.

**(b) LU-KV moves to Tier A.** `LUKVPress` ships in kvpress 0.5.4 (verified: imports and
constructs). **v1 §6.3's port specification is deleted** — we no longer write the convex-hull
relaxation or the marginal-utility greedy solver. **Its G1 reproduction gate is kept**: LU-KV
must still reproduce a published number on its own reported setting before admission. The
offline profiling pass remains ours, because `BUDGET_CURVE_URLS` ships exactly one curve
(Llama-3.1-8B × ExpectedAttention) and none of our models have one; that curve must be
downloaded once, hashed, and committed rather than fetched at runtime (B7).

**(c) Every press check must generate tokens.** Constructing a press, or checking that it
returns a score vector, is not evidence that it works: kvpress presses act through forward
hooks, and a press that silently no-ops still constructs cleanly and still returns scores. So
every admission gate (§7 G1/G2/G3) and every smoke test runs a real generation and compares
output, not just internal state. This is Paper 1's transferable practice — assert the invariant
on live data, not on a reimplementation — and it is the check that would have caught D-06 (a
factor under test being inert) immediately rather than after a phase had run.

**(d) ObservedAttention / H2O is dropped**, consistent with decision 6's `sdpa` pin. It
requires `eager`, and the published audit had to withdraw its H2O ranking for exactly this
reason. Its exclusion is on backend-stratum grounds and is stated as such in Limitations, not
on merit.

---

### 3.11 — `oracle_prescient` redesigned and `I` restated (v2)

**Why.** Stage 5 check B failed twice — `oracle_causal` 0.9988 > `oracle_prescient` 0.9950 at
M3/C=128, n=200 (3 generations in 800), after 1 in 192 at n=48. It replicated and widened, so it
was not noise. Root cause: the two ceiling arms differed in **two faculties at once**.

* *information* — prescient knows which candidate is queried, causal does not;
* *content* — causal retained all H packed candidates while prescient retained **one** and spent
  the rest of its budget on filler.

Both arms always contain the queried gold, so the information advantage bought prescient nothing,
while causal's extra clean records displaced mildly harmful filler. `I` is defined as if only
information differed, so it went **negative**: `(0.9950 − 0.9988)/(0.9950 − 0.0862) = −0.0042`.

**The fix.** `oracle_prescient` now holds **the same candidates `oracle_causal` holds, with the
queried one guaranteed to be among them** — it starts from causal's packing and, if the queried
candidate was not chosen, swaps out the most expensive chosen candidate for it. Budget, filler
order and floors are identical. The arms now differ in exactly one faculty: **when the budget
cannot hold all H candidates, causal must choose without knowing the query while prescient always
includes the queried one.**

*A strict superset was not used because it is unreachable inside a matched budget:*
`oracle_causal` packs the maximum number of candidates that fit in `C`, so if the queried one was
not chosen then by maximality no packing of that many plus one fits, and adding it would break the
equal-`B` invariant every arm depends on. The swap preserves both the candidate count and the
budget.

*Verified on real instances rather than argued* (`bench/prescient_swap_check.py`, 800 cases on M2
and 960 on M3): budget equality holds in every case; **candidate-count mismatches 0/800 and
0/960**, confirming the cost-uniformity the swap relies on (LEDGER record lines are
format-identical, payable cost 17–21 tokens on M2 and 11–16 on M3); and the queried candidate is
retained whole in every case. In **660/800 (M2) and 772/960 (M3)** the two keep-sets are
*identical* — the queried candidate was already in causal's packing, so knowing the query is worth
exactly nothing there and those instances contribute 0 to `I` by construction.

**`I` restated (this replaces the §1.2 framing).**

> `I` is **the accuracy cost of not knowing the query, measured between two otherwise-identical
> retention policies.** It is a measured gap between two retention policies, not a proof of
> irreducibility.
>
> `oracle_prescient` bounds **what a perfect predictor could retain — not what the model then
> does with the retained set.** A large `I` says that choosing candidates without knowing the
> query costs accuracy at this budget; it does not establish that no method could ever close
> that gap by other means, and it is not an upper bound on achievable accuracy.

### 3.12 — Two measurements from Stage 5 validation that belong in the results

**(a) The ceiling denoises across most of the grid — report it, do not footnote it.**
`oracle_causal > full_cache` fires in **8 of 11 validation cells** (M2 at C ≥ 128, M3 at C ≥ 64);
e.g. M2/C=512 causal 0.9896 against full_cache 0.9479. Cause: compression deletes 36 distractor
records and the model reads a clean ~200-token context better than 2049 tokens. Per §4.1 this is
an **audit trigger, not a validation failure**. Its consequence must be stated in the results:
**across most of the grid, methods are normalised against a ceiling above no-compression**, so a
`G_m` near 1 does not mean "as good as keeping everything" — it means "as good as a policy that
beats keeping everything".

**(b) The attention sink carries the floor — eight tokens.** `null` (last `B`, no sink) scores
**0.0000 at every budget on both models**, while `floor_pos` (sink + last `B−8`) reaches
**0.2708 on M2 and 0.2969 on M3 at C=512**. The arms differ by 8 tokens. This is an internal
reproduction of StreamingLLM's central claim, obtained inside our own ladder rather than cited,
and it substantiates decision 2's choice of a sink-bearing position-only floor: without the sink
the same positional policy is not merely weaker but completely inert.

### 3.10 — Budget ladder extended; M1 excluded (2026-09-06)

**Two tighter cells added: `b-2` (C=16, B=88) and `b-1` (C=32, B=104).** Reason: on the one-hop
task the causal oracle's candidates are cheap, so it fitted all H at nearly every budget and the
information share `I` collapsed to a single non-zero cell. The tighter cells restore range —
`I` is now 0.750 / 0.250 on M2 at b-1 / b0, and 0.750 / 0.500 on M3 at b-2 / b-1.

**`b-2` is VOID on the Qwens and is excluded from their grid** (v1 §3.2: `C < k_gold` ⇒ excluded
before any run). `k_gold` is 19 on Qwen and 13 on Llama, so C=16 cannot carry the gold span on a
Qwen at all. `harness/ladder.py:oracle_prescient` **raises** rather than silently truncating the
gold, which is how this was caught; `decision7_analysis.py` now records such cells as VOID
instead of running them.

Admitted budgets after exclusion, with PARTIAL cells reported separately and never pooled:

| model | admitted budgets |
|---|---|
| M2 Qwen2.5-3B | `b-1` (PARTIAL), `b0`, `b1`, `b2`, `b3` — 5 |
| M3 Llama-3.2-3B | `b-2` (PARTIAL), `b-1`, `b0`, `b1`, `b2`, `b3` — 6 |

### M1 Qwen2.5-1.5B — **EXCLUDED from the main grid** (v1 §4.5)

**Measured competence anchor: 0.4375** (n=200, mean over H=4, `max_new_tokens=32`, one-hop
LEDGER at N=40, `gates/nvidia/stage4_competence_Qwen2_5-1_5B-Instruct_mn32.json`). The band is
[0.55, 0.97]; 0.4375 is below it.

v1 §4.5 is explicit that a model which cannot be moved into band "is dropped from the main grid
and reported as excluded with the anchor value; it is not 'fixed' by post-hoc tuning after seeing
method results." **No further tuning was attempted.** The exclusion is recorded here, before any
method has been run, so it cannot be mistaken for a post-hoc choice.

Its well-formedness is 1.000, so this is a retrieval-under-compression capability limit, not a
formatting artifact. For the record, its anchor across the task variants tried: 0.100 (two-hop
surname-link N=96), 0.1225 (N=80), 0.1263 (two-hop id-link N=40), **0.4375 (one-hop N=40)**.

**Consequence for the paper.** The main grid is now two models, both ~3B, from two lineages
(Qwen2.5 and Llama-3.2).

**Scale generality — amendment to v1 §3.1.** v1 called M4 (Qwen2.5-7B) a "robustness replicate"
whose removal the claims survive. That framing no longer holds: with M1 excluded, **M4 is the
only scale variation in the project**, and the main grid spans no scale at all — both admitted
models are ~3B.

M4 is **not** promoted into the main grid. It lives on Machine A (24 GB; 15.2 GB of bf16 weights
do not fit the 5070), so promoting it would put a main-grid model on the other device and force
exactly the cross-device comparison §5.3 forbids: "a cell is executed entirely on one device...
no arm's number is ever compared to an arm run on the other GPU." A 0.221 backend swing on
mathematically equivalent kernels is larger than most method-vs-baseline gaps in this
literature, and it would land directly inside `G_m`.

**So the ruling is: Paper 2 makes no scale-generality claim.** Every `G_m` and `I` in the paper
is stated for ~3B models on one task family at `L = 2048`, and the abstract says so. M4 is
reported as a **separate single-device replicate on Machine A** — its own table, its own device
column, never pooled with and never differenced against a Machine-N number. It can corroborate
a *direction*; it cannot extend the paper's scope, and it must not be written as if it does.

**Paper 1 continuity now rests on Task A alone.** M1 was the model-side continuity anchor (v1
§3.1: "continuity with Paper 1") and it is gone. What remains is `SECRET-2048` (§4.4), Paper 1's
own credential task re-tokenised — a task-side anchor, not a model-side one. Paper 1's headline
numbers were measured on Qwen2.5-1.5B, so **no Paper 2 number is directly commensurable with a
Paper 1 number**, and comparisons to Paper 1 are qualitative from here on. Stated rather than
quietly dropped.

## 4. Design — carried from v1, unchanged

### 4.1 Reference ladder (6 arms)

| arm | definition | role |
|---|---|---|
| `full_cache` | no compression | anchor / competence gate |
| `null` | keep last `B` only | tripwire |
| `random` | uniform random `C` from the compressible region | tripwire |
| `floor_pos` | `n_sink=8` + last `C−8` (StreamingLLM at this budget) | **denominator** |
| `oracle_causal` | all candidate gold spans + filler from `floor_pos`; degrades per decision 7 | **numerator ceiling** |
| `oracle_prescient` | **the same candidates `oracle_causal` holds, with the queried one guaranteed among them**; remainder from `floor_pos` | the query-known counterpart; gives `I` |

Head-allocation variants (v1 §3.4), both run: `oracle_causal_global` (one token set replicated
across heads) and `oracle_causal_perhead` (per-head sets, same total budget).
`Δ_head = A(perhead) − A(global)` is measurable before any method is admitted.

**Registered ordering invariants, asserted per cell:** `null ≤ random ≤ floor_pos` and
`oracle_prescient ≥ oracle_causal ≥ floor_pos`. Any violation halts the cell and is
investigated to root cause before proceeding — it is not smoothed. `oracle_causal > full_cache`
additionally triggers the ceiling-validity audit (the arm is denoising, not retaining; this
fired in Paper 1 at budget 514).

### 4.2 Methods

| arm | implementation | family | query-visible | tier |
|---|---|---|---|---|
| `snapkv` | `kvpress.SnapKVPress` | attention over last-64 window | yes, heavily | A |
| `tova` | `kvpress.TOVAPress` | attention of last query | yes, 1 token | A |
| `expected_attn` | `kvpress.ExpectedAttentionPress` | analytic future-query distribution | no | A |
| `keydiff` | `kvpress.KeyDiffPress` | `−cos(k_i, k̄)`, query-free | no | A |
| `adakv_snapkv` | `kvpress.AdaKVPress(SnapKVPress)` | head-adaptive budget | yes | A |
| `lukv` | `kvpress.LUKVPress` — **Tier A per decision 9(b)** | head-wise budget by long-horizon utility | no | **A** |
| `foresightkv` | RUCAIBox reference + our SFT run | learned long-term contribution | no | B |

Exclusions carried verbatim from v1 §3.3 into the paper's Limitations: **LKV** (custom backward
CUDA kernel, will not build on gfx1100 without a HIP port we cannot validate), **KVzip /
FastKVzip** (multi-pass, does not fit a single-pass matched-budget harness), **LookaheadKV**
(per-model LoRA training; two trained methods is the compute ceiling), **PyramidKV** (layer
axis, orthogonal to the head-allocation question), **ObservedAttention / H2O** (requires `eager`;
struck by decisions 6 and 9(d) on backend-stratum grounds, not on merit).

### 4.3 LU-KV is Tier A (decision 9(b) — resolved)

v1 §6.3 planned LU-KV as one of the two ports we write: offline profiling, convex-hull
relaxation, marginal-utility greedy solver, wrapped as a `BasePress`. Estimated 4–7 h on
Machine A (plan v2 package A2).

**kvpress v0.5.4 ships `lukv_press`.** It is genuinely LU-KV — it loads budget curves from the
authors' own repository (`baidu-baige/LU-KV`). So the solver and allocator are upstream,
Apache-2.0, and audited, and **the riskiest part of the port disappears.**

**But it is not free.** Two carried risks:

1. **No budget curve exists for any of our models.** `BUDGET_CURVE_URLS` in v0.5.4 contains
   exactly one entry: `("meta-llama/Llama-3.1-8B-Instruct", "ExpectedAttentionPress")`. M1, M2
   and M3 have none. **v1 §6.3 step 1 — the offline profiling pass on a frozen, hashed
   calibration split disjoint from eval — is still ours to run**, and it is still the part that
   determines what the method does.
2. **It fetches the curve over the network at runtime** (`requests.get` on a raw GitHub URL).
   That is a reproducibility hazard of the dedup-key class: the arm's behaviour could change
   without any version, SHA, or key changing. **If admitted, the curve must be downloaded once,
   hashed, committed, and loaded from disk**, with the hash recorded in this document.

**Resolved:** `lukv` is Tier A. v1 §6.3's port specification is deleted; its **G1 reproduction
gate is kept**. Provenance is "upstream press + our profiling", which removes ~half of plan v2's
A2 package and, more importantly, removes the solver from our implementation-risk surface.

### 4.4 Tasks

- **Task A `SECRET-2048`** — Paper 1's credential task re-tokenised to 2048. **Now the project's
  only link to Paper 1**, since M1 (the model-side continuity anchor) is excluded (§3.10); the
  link is task-side only and no number is directly commensurable across the two papers.
  **Labelled shortcut-admitting in every table.** No headline claim rests on it alone. Registered
  control: the two-line regex ceiling, reported beside `oracle_causal`.
- **Task B `LEDGER`** — carries the main claim. Implemented at `harness/tasks/ledger.py`.
  96 format-identical records, 4 binding sentences, two disjoint gold spans, candidate set
  (2H = 8 spans) enumerated at construction time. Chance = 1/96 ≈ 0.0104.
  **Five shortcut probes are a Stage 4 gate** — BM25 with query hidden, regex value-extractor,
  position prior, `full_cache` with bindings deleted, `full_cache` with target record deleted.
  Task B is not admitted until all pass.
- **Task B Stage 4 status (2026-09-06):** the three model-free shortcut probes **PASS**
  (`gates/nvidia/stage4_shortcut_probes.json`, n=200, chance 1/96 = 0.0104, gate 0.0304):
  BM25-query-hidden **0.0250**, regex value-extractor **0.0050**, position prior **0.0050**.
  The two ablation probes (`full_cache` with bindings deleted; with the target record deleted)
  require a forward pass and are **deferred to the pinned WSL toolchain**, because they produce
  admission-gating accuracy numbers and every such number must come from the frozen environment.

  **The first run FAILED and the task was tuned, which is what Stage 4 is for.** BM25 with the
  query hidden scored **0.140**, 4.6x the gate. Diagnosis: a bound surname appeared exactly twice
  (its binding sentence and its record line) while an unbound one appeared once, so occurrence
  count alone identified the bound records — the hidden-query retriever landed on a bound record
  0.445 of the time against 0.0417 chance. Fix: every filler sentence now mentions a surname in a
  **non-binding** frame, drawn from the unbound names without replacement so that decoys also
  reach exactly two occurrences and the count carries no information. Two iterations:
  0.140 (dept-only filler) → 0.055 (half-mention) → **0.0250** (coverage-maximal decoys).
  Stage 1 was re-run on the tuned task and is unchanged (gate still PASS); the decision-7
  evidence in §3.7 is measured on the tuned task.

  **Residual, reported rather than tuned away:** the hidden-query retriever still lands on *some*
  bound record 0.110 of the time against 0.0417 chance (2.6x, down from 10.7x). This is intrinsic
  to two-hop construction — any link between two spans repeats a token — and it narrows to the
  candidate set without identifying the queried member, which is `oracle_causal`'s information,
  not `oracle_prescient`'s. It is a non-gating diagnostic and belongs in the paper's task section.

- **Task B difficulty — B8 RESOLVED: two-hop structure kept, 2-shot exemplars added.**
  At N=96/H=4 with no exemplars the competence anchor was 0.100 / 0.200 / 0.150 against v1
  §4.5's [0.55, 0.97] band, and **N was not the knob** — sweeping M1 at 96/48/24/12 plateaued at
  0.275 while well-formedness *fell* 0.975 -> 0.500, because the model answered with the surname
  and stopped after hop 1. So the binding constraint was the two-hop chain itself, not the
  distractor count. Rather than weaken the chain — which is the property the task exists to
  test (v1 §4.2: retaining `s2` without `s1` is useless) — the preamble now carries **two worked
  exemplars** showing the full chain (note -> responsible party -> that party's record -> value)
  and a bare-value answer. The exemplars use roles, surnames and record ids drawn from outside
  every generated pool (`warden`/`steward`; `Aldermill`/`Brackenhold`; `R900`/`R901`), so they
  cannot collide with or leak any instance — asserted in `tests/test_harness.py`.

- **Task B layout — THE ONE CHANGE IN v2. Records are INTERLEAVED UNIFORMLY through the
  filler**, rather than sitting as a contiguous block followed by governance notes. Content is
  byte-for-byte the same generator output: the same N record lines, the same value
  distribution, the same record-id decoy mentions, the same exemplars. **Only line ordering
  changes.**

  *Why.* Under the v1 layout gold occupied token positions 206–955 while `floor_pos` retains
  the sinks plus the most recent `B − n_sink` — `[1487, 2063)` even at the largest admitted
  budget. Gold was unreachable by the floor at every budget, so the registered denominator was
  absorbing at exactly 0.0000 and `G_m` degenerated to `A_m / A_causal`. Decision 2 adopted a
  position-only floor precisely because it is "non-absorbing — it scores > 0 whenever gold
  happens to sit early or late"; a contiguous mid-context block defeats that property.

  *Effect.* Gold is now positionally uniform (measured: token positions 156–2060 against a
  median context of 2048), so a recency-shaped floor retains gold at roughly its budget
  fraction. Measured on the new layout at n=8: `floor_pos` = 0.0312 at C=128 and 0.2188 at
  C=512 — **graded and non-absorbing**, which is what decision 2 assumed and what makes `G_m`
  measurable.

  *What this does NOT change.* The floor is still position-only and still zero-cost; the
  candidate set, the oracles, decision 7's packing, the budget ladder and every metric are
  untouched. The change is confined to the order in which lines are emitted.

  **Interleaving is uniform and deterministic**, not random: record `i` is placed at slot
  `round((i + 0.5) · total / N)` with deterministic collision resolution, so no seed controls
  where gold lands and position cannot become a hidden nuisance parameter.

- **Task C** — three LongBench English tasks truncated to 2048 in-tokenizer, `b1` and `b3` only,
  agnostic protocol only, ladder + Tier-A only. **No oracle, therefore raw accuracy only, never
  `G_m`.** Runs entirely on Machine A.

### 4.5 Competence gate

A model enters a cell only if its `full_cache` anchor on that task sits in **[0.55, 0.97]**.
Above 0.97 the ceiling cannot be distinguished from the floor; below 0.55 the floor effect
dominates. Task difficulty (`N` records, `H` bindings, hop count) is tuned in Stage 4 until every
admitted model sits in band, and **frozen before Stage 6**. A model that cannot be moved into
band is dropped and reported as excluded with its anchor value — never "fixed" post hoc after
method results are visible.

---

## 5. Metrics and analysis rules — the analysis script's contract

```
Capture ratio      G_m = (A_m     − A_floor)  / (A_causal  − A_floor)
Information share  I   = (A_presc − A_causal) / (A_presc   − A_floor)
```

**Refusal-to-normalise rule.** If `A_causal − A_floor < 0.15` in a cell, that cell is reported
**raw only** and `G_m` is emitted as `n/a — degenerate headroom`. Written into the analysis
script before any data exists.

**Clipping rule.** `G_m` is reported **unclipped**. `G_m > 1` is diagnostic, not an error, and
triggers the ceiling-validity audit.

**Estimator.** Every contrast is **paired per instance**. Bootstrap 50,000 resamples, paired,
percentile CIs, resampling unit = instance. `G_m` and `I` CIs are obtained by propagating the
bootstrap **through the ratio** — resample instances, recompute numerator and denominator on the
*same* resample. Never divide two independently bootstrapped means.

**Carried from Paper 1 §I.7:** the median is degenerate on this data shape and is not used. Every
contrast additionally reports **the count of instances on which the two arms differ at all**,
which is estimator-independent. The two kinds of null — "same behaviour" vs "different behaviour,
same outcome" — are reported distinctly and never described in the same words.

**Multiplicity.** One pre-registered primary contrast per outcome in §6. Everything else is
labelled exploratory.

**Anticipated reporting asymmetry (stated in advance, not discovered at analysis).** `I` is
informative only where the causal oracle is squeezed — b0/b1 — which is where the
refusal-to-normalise rule is most likely to suppress `G_m`. The `I(C)` figure will therefore rest
on b0/b1 and the `G_m` table on b2/b3. This is a property of the design, and the paper states it
rather than presenting whichever half looks better.

**Cross-model comparability caveat (stated in advance).** Llama-3.2 needs 26.5 gold tokens
against Qwen's 33.5, and 107 vs 134 for the candidate set, so **cells are not comparable across
models at equal `C`.** Cross-model statements are made at equal `C/k_gold`, or explicitly
qualified. If decision 8 substitutes Phi-3, this must be re-derived.

---

## 6. Outcomes — all four pre-declared publishable

| outcome | signature | claim |
|---|---|---|
| **A — saturated** | max `G_m` ≥ 0.85 in ≥ 2/3 of competitive cells | Existing methods approach the attainable causal ceiling; residual retention headroom is not a productive Paper 3 target |
| **B — residual** | max `G_m` ≤ 0.65 in ≥ 2/3 of competitive cells, `I` ≤ 0.4 | A large *capturable* headroom remains; §8 localises it |
| **C — regime map** | rank order flips with budget or task; no dominant method in ≥ 1/3 of cells | No universally dominant retention objective |
| **D — predictive** | `I` ≥ 0.5 | Half or more of the apparent headroom is anticipatory, not rankable |

A fifth possibility — methods fail admission — is reported in the **unadmitted table**, a real
deliverable, not a footnote.

## 7. Admission gates (v1 §6.4)

```
per method:
  [G1] reproduction     |ours − published| ≤ max(0.02, published_variance)
  [G2] permutation      accuracy(permuted score) ≈ accuracy(random arm)
                        and ρ(retained_set, permuted_retained_set) ≈ 0
  [G3] oracle overlap   |retained ∩ gold| increases with C
  all three pass → enters the grid;  any fail → unadmitted table, with the failing gate
```

**ForesightKV hard stop:** if it has not passed by the end of Stage 6 it is dropped and the paper
reports five methods. No claim depends on it. SFT/pairwise-ranking stage only; **GRPO explicitly
skipped and reported as skipped.**

## 8. Environment — pinned into the dedup key

```
kvpress        0.5.4  @ git 6d965557a5b9f0201a2301b23c454473dd681d0d   (released 2026-07-02)
transformers   >=4.56.0, <5.3     ← CONSTRAINED BY kvpress 0.5.4, not a free choice
python         3.12.3             (kvpress requires >=3.10)
attn_impl      sdpa   (decision 6; never mixed within a cell)
dtype          bfloat16
decoding       greedy, do_sample=False, max_new_tokens=32 (Task B)   <- B9 RESOLVED
seeds          CRC32 of the record key (harness/keys.py). Never Python hash().
```

**B9 RESOLVED — `max_new_tokens` raised 12 -> 32.** v1 §5.4 pinned 12 before anyone measured
whether models comply with a terse-answer instruction. They do not comply uniformly: at 12,
Llama-3.2-3B scored **0.000 with well-formedness 0.000** because it spends every token on a
restatement preamble, while both Qwens were unaffected (identical at 12 and 32). The pin was
therefore a **model-dependent format confound** — it measured instruction-following on M3 and
retrieval on M1/M2 — of exactly the kind v1 §3.2 warns about for tokenizers. At 32, Llama's
well-formedness is 1.000. The cost is decode time, which the WSL move has already more than
paid for.

**The transformers version is not ours to pick.** kvpress 0.5.4 declares
`transformers<5.3,>=4.56.0`. The Windows box that ran Stage 1 has **transformers 5.10.2 —
outside that range.** The exact in-range version is selected and pinned at Stage 5 when kvpress is
installed, and it enters the dedup key. This also retires the earlier concern about porting Paper 1
code written against `transformers>=4.44`: the target is now a specific pinned version in
[4.56, 5.3), and any ported cache code is validated against *that*, not assumed.

**Host platform — resolved, per your amendment.** The box moves to **WSL2 Ubuntu 24.04**, because a
Windows CUDA stack has different kernel selection and Triton availability from the Linux stack v1
was costed on, and v1 §5.3 makes backend numerics a stratum rather than an implementation detail.

Installed and verified 2026-09-05:

| | value |
|---|---|
| distro | Ubuntu 24.04.4 LTS on WSL2 |
| python | **3.12.3** (the distro default — so the 3.11 question is moot, we get 3.12 free) |
| GPU passthrough | **works** — `nvidia-smi` inside WSL reports RTX 5070, 12227 MiB |

**Every number in §2 is re-measured under this platform before hashing.** Throughput is certainly
host-dependent; the tokenizer-length figures should be invariant but are re-run rather than
assumed, since the transformers version is also changing.

**Known consequence to check, not assume:** the repo and `HF_HOME` live on `D:`, reached from WSL
via `/mnt/d` (9p), which is materially slower than native ext4. Model load and JSONL append cost
should be re-measured, and moving `HF_HOME` inside the WSL filesystem considered.

Devices are **strata, not resources** (v1 §5.3): a cell executes entirely on one device; devices
are assigned by cell, never by queue depth; no arm's number is ever compared to an arm run on the
other GPU.

## 9. Coverage and determinism

Coverage is **asserted, not assumed**: after every stage a checker compares the exact instance set
per arm per cell and fails loudly. Holes are refilled **on the owning device only** (repo rule 2)
and every refill is logged. All raw per-instance records to JSONL, one line per record. Nothing is
deleted; invalid rows are quarantined with checksums.

## 10. Stage order

Per v1 §7. Stages 0–2 are paperwork and arithmetic; **no GPU cycles until Stage 3.** Stage 1 is
complete and passed. **Stage 2 is this document, and it is not finished.** Stage 3 is N1, the
random-span control, and it is a hard gate: if random-span protection ≈ gold-span protection, every
`G_m` is measured against a misdescribed reference and the design is rebuilt before proceeding.

## 11. Deviations register

### finding-1 slot — **NOT APPLIED — UNRESOLVED**

Reserved and deliberately left empty. Nothing is recorded here and nothing downstream depends
on it. It is not a synonym for any resolved decision, and it must not be quietly closed by
attaching an existing finding to it.

### Deviations

Empty. Every deviation from this document once hashed is recorded here rather than absorbed —
Paper 1's practice, which is why its median→mean substitution is legible today.

## 12. Hash

Computed over the file's **LF-normalised UTF-8 bytes** with this section's `sha256:` line held
at the placeholder `<pending>`. Normalisation is load-bearing: the repo is checked out on
Windows where git rewrites LF to CRLF, so hashing raw bytes would make the value depend on the
checkout platform rather than the content. Verify with:

```python
import io, re, hashlib
s = io.open("PREREG_P2_v2.md", encoding="utf-8", newline="").read()
s = re.sub(r"sha256: [0-9a-f]{64}", "sha256: f25b842a1fed9c52d92a04018af35fc1f79bc30f562adb3711ec3a5f165a9ffe", s).replace("
", "
")
print(hashlib.sha256(s.encode("utf-8")).hexdigest())
```

```
sha256: b3f5fb3c1e949c94e0785ba7a9843cbcc2548103215100fe0ec1d42b47ba886e
frozen: 2026-09-06
supersedes: 8ac3895709be0721656cef0c705cf843302245ab6d841373ee1676032db203e9
```
