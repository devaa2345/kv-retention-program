"""Check 3 — same instances, both code paths, retained tokens compared directly.

The decisive question behind the negative G_m: methods and `floor_pos` retain the SAME NUMBER
of tokens (verified), so if methods score lower they must be retaining *different* tokens. Two
rival explanations:

  (a) harness fault — the method arms are somehow handicapped;
  (b) fragmentation — a record line is ~19 tokens and is only usable if retained WHOLE.
      `floor_pos` keeps one contiguous block, so every record inside it is complete. A top-k
      scorer scatters its budget by per-token score, so it can retain 60% of many records and
      100% of none. Retaining more candidate TOKENS while completing fewer candidate RECORDS
      would be worth less than nothing.

This measures both, per arm, per budget, on identical instances:
    tokens_in_region   how much of the compressible region each arm keeps  (parity check)
    cand_tokens        candidate (record) tokens retained
    complete_records   candidate records retained WHOLE  <-- the one that decides the answer
"""
from __future__ import annotations
import sys, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import ladder, methods, press
from harness.keys import seed_key_without_seed
from harness.tasks import ledger
from stage5_ladder_validation import facts_and_ctx, templated_parts, seq_len_of, _null_ctx

M = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-3B-Instruct"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
BUDGETS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "64,128,512").split(",")]
N_SINK, N_WINDOW = 8, 64

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
rev = getattr(model.config, "_commit_hash", None) or "x"

insts = []
for i in range(N):
    sd = seed_key_without_seed(
        task="ledger", instance_id=f"grid_{i:05d}", model=M, model_revision=rev, arm="grid",
        B=1, protocol="agnostic", device="nvidia", backend="cuda-12.8",
        torch_version=torch.__version__, transformers_version="5.2.0",
        kvpress_version="0.5.4", dtype="bfloat16")
    insts.append((ledger.build(sd, f"grid_{i:05d}", target_tokens=2048, tokenizer=tok), sd))


N_CTX_HOLD=[0]


@torch.inference_mode()
def capture_for(name, ratio, pre):
    cap = methods.Capture()
    base = methods.make_floor_constrained(methods.build_method(name, ratio), N_CTX_HOLD[0], N_SINK, N_WINDOW)
    p = methods.make_capturing(base, cap)
    p.compression_ratio = ratio
    ids = tok(pre, add_special_tokens=False, return_tensors="pt").to(model.device)
    with p(model):
        model(**ids, use_cache=True)
    return cap


print(f"{M}  N={N}")
print(f"{'C':>5} {'arm':16s} {'region_kept':>12s} {'cand_tok':>9s} {'complete_recs':>14s} {'sink_kept':>10s} {'window_kept':>12s}")
for C in BUDGETS:
    agg = {}
    for inst, sd in insts:
        pre, _ = templated_parts(tok, inst.context, "")
        facts, n_ctx = facts_and_ctx(inst, tok, pre)
        region = set(range(N_SINK, n_ctx - N_WINDOW))
        ratio = press._ratio_for(C + N_SINK + N_WINDOW, n_ctx)
        N_CTX_HOLD[0] = n_ctx

        # floor_pos, via the ladder path
        fp = ladder.floor_pos(n_ctx, C, N_SINK, N_WINDOW, facts)
        k = set(fp.kept)
        sink = set(range(N_SINK)); win = set(range(n_ctx - N_WINDOW, n_ctx))
        agg.setdefault("floor_pos", []).append(
            (len(k & region), len({t for f in facts for t in f.tokens} & k),
             sum(1 for f in facts if f.is_complete_in(k)),
             len(k & sink) / N_SINK, len(k & win) / N_WINDOW))

        for name in ("snapkv", "expected_attn", "keydiff", "adakv_snapkv"):
            cap = capture_for(name, ratio, pre)
            # average over heads: each head has its own retained set
            rk, ct, cr = [], [], []
            for key in cap.heads():
                kk = cap.per_head[key]
                rk.append(len(kk & region))
                ct.append(len({t for f in facts for t in f.tokens} & kk))
                cr.append(sum(1 for f in facts if f.is_complete_in(kk)))
            sk = [len(cap.per_head[key] & sink) / N_SINK for key in cap.heads()]
            wn = [len(cap.per_head[key] & win) / N_WINDOW for key in cap.heads()]
            agg.setdefault(name, []).append(
                (statistics.fmean(rk), statistics.fmean(ct), statistics.fmean(cr),
                 statistics.fmean(sk), statistics.fmean(wn)))

    for name, vals in agg.items():
        a = statistics.fmean(v[0] for v in vals)
        b = statistics.fmean(v[1] for v in vals)
        c = statistics.fmean(v[2] for v in vals)
        d = statistics.fmean(v[3] for v in vals)
        e = statistics.fmean(v[4] for v in vals)
        print(f"{C:5d} {name:16s} {a:12.1f} {b:9.2f} {c:14.3f} {d:10.3f} {e:12.3f}")
    print()
