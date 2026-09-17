# Paper 2 — repo rules (inherited by every agent session in this repo)

Two agent sessions work in this repo on separate branches that never touch the same paths.
Session **N** owns the NVIDIA RTX 5070 box. Session **A** owns the AMD RX 7900 XTX box.

`PREREG_P2.md` is **frozen and hashed at Stage 2** and is **read-only to both sessions**.
Neither session may edit it. If either believes it needs changing, **stop and ask the human**.

---

## The five rules

1. **Never write outside your device's `runs/` directory.**
   Session N writes `runs/nvidia/`, session A writes `runs/amd/`. Both may *read* either.
   A record in the wrong directory is a corrupted cell.

2. **Never refill a hole on the other machine.**
   If a coverage assertion finds a missing record in `runs/nvidia/`, **only session N** may
   regenerate it. Scheduling a refill onto whichever box is free silently reintroduces the exact
   device confound the design exists to avoid. **This is the single most likely way to ruin the
   paper by accident.**

3. **The dedup key is `harness/keys.py` and nothing else.**
   It includes: `task, instance_id, model, model_revision, arm, B, protocol, device, backend,
   torch_version, transformers_version, kvpress_version, dtype, seed, batch_size`.
   Assert it in code. **Never reconstruct the key inline.**

4. **Gates are blocking.**
   If `gates/` does not contain a PASS for the prerequisite stage, do not start the next stage —
   write to `STATUS.md` and stop. **Do not "proceed provisionally".**

5. **Seeds by CRC32 of the record key. Never Python `hash()`** — it is salted per process and is
   irreproducible across runs and machines. Use `harness.keys.seed_for()`.

---

## Ownership map

| path | write | read |
|---|---|---|
| `PREREG_P2.md` | **nobody** (frozen, hashed) | both |
| `harness/` | **session N** | both — A proposes changes by PR only |
| `runs/nvidia/` | session N | both |
| `runs/amd/` | session A | both |
| `gates/nvidia/` | session N | both |
| `gates/amd/` | session A | both |
| `env/nvidia.lock` | session N | both |
| `env/amd.lock` | session A | both |
| `STATUS.md` | both — **append only, never rewrite** | both |

## Blind-reimplementation firewall (A6)

Session A must **not** read `harness/ladder.py`. A6 is only worth running if session A has
genuinely not seen the primary implementation; two agents on one repo makes it very easy to
destroy that independence by accident. A6 implements from the `PREREG_P2.md` specification alone,
and by a **different mechanism**: primary uses kvpress forward hooks, secondary uses explicit
attention-mask construction with the full cache resident. Compare pooled within ±0.01.

## Environment pinning

Each machine keeps its own lockfile (`env/nvidia.lock`, `env/amd.lock`) with exact versions of
torch, transformers, kvpress (plus git SHA), and the CUDA/ROCm runtime. Both are committed.
**A version drift on either box mid-run invalidates every record produced after it**, and the
dedup key is what lets you find them. Re-run `python -m harness.envlock` after any package change
and commit the result before producing another record.

## Allocation principle (why these rules exist)

> The RTX 5070 owns every number that appears inside a ratio. The 7900 XTX runs only self-contained
> blocks where every arm in the comparison also lives on the XTX. **No contrast ever crosses a
> device boundary.**
