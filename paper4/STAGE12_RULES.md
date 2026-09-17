# Paper 4 — Stages 1–2: design, gates and deviations

**Committed before any Stage 2 record is produced.** All results are produced on the
RTX 5070 (Machine N), single process, pinned toolchain (WSL `/opt/p2venv`: torch 2.11.0+cu128,
transformers 5.2.0, kvpress 0.5.4, sdpa, bf16, batch 1).

## 0. Missing input, stated

The instruction was to read `PAPER4_CONTEXT_AND_PLAN.md` first. **No such file exists** on this
machine (searched all of `D:\`, the user profile's Downloads/Desktop/Documents, and the WSL
filesystem). Stages 1–2 are therefore built from (i) the instruction's own restatement of the
harness invariants, (ii) Paper 3's frozen harness and `LIMITATIONS_P3.md`, which hands this
question to Paper 4. Any invariant the context file adds beyond those is not implemented.

## 1. The wrapper (Stage 1) — `p4/unitwrap.py`

U-X = X's own `score()` (including the mandatory-floor pinning), called once and not modified;
per-token scores averaged over each unit's payable tokens (score-per-token); budget C filled
greedily by whole units in descending score-per-token, all-or-nothing (a unit that does not fit is
skipped, never truncated). Region tokens belonging to no unit are singleton units. Floors are free.
Budget per (layer, KV head) is exactly B = C + 72 (per-layer total B·h for AdaKV, the invariant
AdaKV itself satisfies). AdaKV's safeguard and global top-k are both allocation, so both are made
unit-wise. Any residual budget no whole unit can fill is filled by top score and **counted**
(`fallback`), never hidden. Arms produced: `U-snapkv`, `U-adakv_snapkv`, `U-expected_attn`,
`U-keydiff`.

## 2. Units — oracle, from the generators (`p4/units.py`)

- LEDGER-C: every record line (40 records + the 2 worked-example record lines = 42 units, Paper 3's
  `units_complete` accounting). **All records, never only the queried four** — that would leak the
  query.
- MARK-1 (c = 1): every entry word (the generator's own fact span). The four queried words are one
  token each; the 36 surnames are multi-token. So at c = 1 U-X cannot make a queried fact more
  complete, but it does reallocate among distractors: the c = 1 cell is a **control** (any gain
  there is not from completing the fact). The strict identity (all units singletons ⇒ U-X ≡ X) is
  checked separately and exactly, on real scores.

## 3. Stage 1 verification — deviation, stated

The instruction was to verify on Paper 3's existing captures with no GPU. **Those captures store
per-instance aggregates, not keep-sets or per-token scores**, and unit re-allocation needs the
scores. So verification is split:

- `stage1_verify_cpu.py` (no GPU): rebuilds Stage 4's instances, checks geometry against every
  stored capture row, and checks the predicted direction on the real token geometry under several
  synthetic score families including an adversarial one.
- `stage2_run.py verify` (prefill only, **no generation**, ~minutes of GPU): the real-score check,
  **blocking** before any generation:
  - (a) X's keep-set aggregates (p_g, q_complete, units_complete, units_touched) reproduce Paper 3
    Stage 4's stored captures exactly (|Δ| ≤ 1e-9) on all 50 instances × 4 methods × c∈{40,1};
  - (b) U-X with all-singleton units equals X on the same real scores up to exact ties (kept-score
    multisets equal), 2 instances × c∈{40,1} × 4 methods × 2 models;
  - (c) at c = 40, for every method on both models: mean distinct units touched U-X < X **and**
    mean units complete U-X > X, at matched budget.
  Any failure: stop and report; no generation is run.

## 4. Stage 2 pilot and gate

Cells: LEDGER-C c≈40 and MARK-1 c=1, C = 512, n = 50 (instances `s4_00000`–`s4_00049`, Stage 4's
own seeds, so X arms can be compared byte-for-byte with Stage 4's stored generations as a session
check), both models (M2 Qwen2.5-3B, M3 Llama-3.2-3B). Arms: `floor_pos`, `snapkv`, `adakv_snapkv`,
`U-snapkv`, `U-adakv_snapkv`. All arms of an instance run in one pass.

Invariants, asserted per row: position_ids continue from the uncompressed length; every arm
generates tokens and the cache is checked compressed (length B, or a mask present for AdaKV);
floors retained in every slot; realised budget parity from the keep-sets **captured during the
generating prefill**; score called once per layer for U arms; degenerate cell (floor_pos accuracy
< 0.05) excluded; CRC32 seeds (instance and bootstrap); device in the dedup key.

Readings, reported separately: [1] U-X − X accuracy, paired, 95% percentile bootstrap (10,000,
CRC32-seeded); [2] each arm vs floor_pos (ratio and difference); [3] completion counts
(units touched / complete, q_complete) from the same keep-sets, with paired differences.

**Gate:** PASS iff at c = 40 at least one (model, method) pair has U-X − X with a bootstrap CI
whose lower bound > 0, **and** at c = 1 no admissible pair has a CI excluding zero (either sign).
If c = 1 shows an effect: STOP AND DIAGNOSE, whatever c = 40 shows. If no c = 40 pair passes: FAIL,
reported as a negative result about unit-awareness. Four comparisons at c = 40, no multiplicity
correction — stated, not adjusted. **The wrapper is not tuned after seeing results.** Stage 3 is
not started.

## 5. Amendment 1 — verify condition (a) on M3 (made after verify, before ANY generation)

**What happened.** The verify gate failed on one condition only: on M3, X's keep-set aggregates
matched Paper 3 Stage 4's stored captures on 0/400 rows (M2: 400/400 exact). Conditions (b) and (c)
passed on both models: identity 16/16 per model; at c = 40 every method on both models touched far
fewer units and completed more, CIs far from zero.

**Diagnosis (`stage2_det_check.py`, `out/stage2_det_check_{M2,M3}.json`).** The M3 differences are
zero-mean and small (p_g |Δ| ≤ 0.013; units counts |Δ| ≤ 0.15 against means of 3–40, i.e. a few of
224 slots flipping a boundary token; n_ctx and slot counts identical on 400/400; q_any identical at
c = 1). A fresh process re-capturing instances 0–4 × c∈{40,1} × 4 methods gave:

| | rerun same process | fresh == this session | fresh == Paper 3 stored |
|---|---|---|---|
| M3 | 40/40 | 40/40 | 0/40 |
| M2 | 40/40 | 40/40 | 40/40 |

The harness is deterministic within and across processes today on both models. The stored M3
captures were written **before Paper 3's power loss**; Paper 3 already established that M3 (and not
M2) differs across that reboot, for generations. Condition (a) as written therefore tested the
reboot boundary, not the harness.

**Amendment.** Condition (a) is satisfied where exact reproduction fails if fresh-process
determinism holds on every re-captured row (rerun identical and fresh == this session). The M3
mismatch with the pre-power-loss captures is reported, not hidden. **Nothing else changes**: the
wrapper, units, arms, cells, n, conditions (b)/(c), the pilot gate and its readings are as
committed. Every Paper 4 contrast is within one session, so the amendment does not touch any
compared quantity. Consequence for the pilot's session check: M3 X-arm generations are expected NOT
to be byte-identical to Paper 3's stored ones for instances 0–50 (pre-power-loss); M2 should be.
