"""Audit the method arms against the ladder arms. Three checks, in order.

Raised because every admitted method scored BELOW `floor_pos` at every budget — a uniform
signature across six independent methods is a harness property, not six independent failures.

  1. position_ids — the question must continue from the UNCOMPRESSED context length. The ladder
     was fixed for this at Stage 5; confirm the method path inherits it. Diagnostic: constant or
     content-ignoring generations.
  2. realised budget parity — each method arm must retain exactly as many tokens as `floor_pos`
     at the same cell. A press that reserves its own sink/window ON TOP of B competes at a
     smaller effective budget.
  3. same instances, both code paths, retained-token counts compared directly.

AdaKV is accounted separately throughout: it masks rather than gathering, so its cache length
stays at n_ctx and its realised budget must be read per head from `module.masked_key_indices`.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import ladder, methods, press
from harness.keys import seed_key_without_seed
from harness.tasks import ledger
from stage5_ladder_validation import (facts_and_ctx, generate_with, seq_len_of,
                                      templated_parts, _clone, _null_ctx)

M = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-3B-Instruct"
C = int(sys.argv[2]) if len(sys.argv) > 2 else 512
N_SINK, N_WINDOW = 8, 64
B = C + N_SINK + N_WINDOW

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
rev = getattr(model.config, "_commit_hash", None) or "x"

sd = seed_key_without_seed(
    task="ledger", instance_id="audit0", model=M, model_revision=rev, arm="audit", B=1,
    protocol="agnostic", device="nvidia", backend="cuda-12.8",
    torch_version=torch.__version__, transformers_version="5.2.0",
    kvpress_version="0.5.4", dtype="bfloat16")
inst = ledger.build(sd, "audit0", target_tokens=2048, tokenizer=tok)
pre, _ = templated_parts(tok, inst.context, "")
facts, n_ctx = facts_and_ctx(inst, tok, pre)
posts = [templated_parts(tok, inst.context, v.query)[1] for v in inst.variants]

print(f"model={M}  C={C}  B={B}  n_ctx={n_ctx}")
print(f"target ratio = {press._ratio_for(B, n_ctx):.6f}\n")


@torch.inference_mode()
def cache_len_under(p):
    ids = tok(pre, add_special_tokens=False, return_tensors="pt").to(model.device)
    with (p(model) if p is not None else _null_ctx()):
        o = model(**ids, use_cache=True)
    return seq_len_of(o.past_key_values), o.past_key_values


rows = []

# ladder reference
pf, _ = press.build_arm("floor_pos", n_ctx=n_ctx, C=C, n_sink=N_SINK, n_window=N_WINDOW,
                        facts=facts, seed=sd)
n_floor, _ = cache_len_under(pf)
out_floor = generate_with(model, tok, pre, posts, pf)
rows.append(("floor_pos (ladder path)", n_floor, n_floor == B, out_floor[0][:28]))

for name in ("snapkv", "expected_attn", "keydiff", "tova", "adakv_snapkv"):
    ratio = press._ratio_for(B, n_ctx)
    p = methods.build_method(name, ratio)
    p.compression_ratio = ratio
    try:
        n_cache, _ = cache_len_under(p)
    except Exception as e:
        rows.append((name, f"ERR {type(e).__name__}", False, "")); continue
    outs = generate_with(model, tok, pre, posts, p)
    # per-head realised budget for the masking press
    extra = ""
    if name == "adakv_snapkv":
        cap = methods.Capture()
        pc = methods.make_capturing(methods.build_method(name, ratio), cap)
        pc.compression_ratio = ratio
        cache_len_under(pc)
        kept = [cap.n_kept[k] for k in cap.heads()]
        extra = f" | per-head kept min={min(kept)} max={max(kept)} mean={sum(kept)/len(kept):.1f}"
    rows.append((name, n_cache, n_cache == B, outs[0][:28] + extra))

print(f"{'arm':26s} {'cache_len':>10s} {'== B':>6s}  first generation")
for a, n, ok, s in rows:
    print(f"{a:26s} {str(n):>10s} {str(ok):>6s}  {s!r}")

# check 1 diagnostic: are method generations constant across instances?
print("\n[1] position_ids inheritance — generations across 3 instances, snapkv:")
for i in range(3):
    s2 = seed_key_without_seed(
        task="ledger", instance_id=f"aud{i}", model=M, model_revision=rev, arm="audit", B=1,
        protocol="agnostic", device="nvidia", backend="cuda-12.8",
        torch_version=torch.__version__, transformers_version="5.2.0",
        kvpress_version="0.5.4", dtype="bfloat16")
    it = ledger.build(s2, f"aud{i}", target_tokens=2048, tokenizer=tok)
    pr, _ = templated_parts(tok, it.context, "")
    _, nc = facts_and_ctx(it, tok, pr)
    po = [templated_parts(tok, it.context, v.query)[1] for v in it.variants]
    pp = methods.build_method("snapkv", press._ratio_for(C + 72, nc))
    pp.compression_ratio = press._ratio_for(C + 72, nc)
    o = generate_with(model, tok, pr, po, pp)
    print(f"   inst {i}: gold={it.variants[0].answer}  gen={o[0][:30]!r}")
