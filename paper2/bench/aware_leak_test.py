"""Decisive test: under the AWARE protocol, can an arm answer when the gold record is NOT retained?

If yes, the answer is not coming from the retained record, and the aware numbers do not measure
retention quality. The suspected mechanism: kvpress compresses in a forward hook that runs AFTER
the prefill attention is computed, so in the aware protocol the question tokens have already
attended to the FULL uncompressed context. Their own KV entries can carry the answer, and the
floor pins the last n_window tokens — which is exactly where the question sits.

Conditioning accuracy on whether the gold record survived separates the two:
    acc | gold retained      should be high in either protocol
    acc | gold NOT retained  should be ~chance if the retained cache is doing the work
"""
from __future__ import annotations
import statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import methods, press
from harness.keys import seed_key_without_seed
from harness.tasks import ledger
from stage5_ladder_validation import facts_and_ctx, generate_with, templated_parts
from run_grid_aware import aware_parts, spans_over

M = "Qwen/Qwen2.5-3B-Instruct"
C = int(sys.argv[1]) if len(sys.argv) > 1 else 512
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
N_SINK, N_WINDOW = 8, 64
ARMS = ["keydiff", "snapkv", "floor_pos"]

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
rev = getattr(model.config, "_commit_hash", None) or "x"

res = {a: {p: {"ret": [], "not": []} for p in ("agnostic", "aware")} for a in ARMS}

for i in range(N):
    iid = f"grid_{i:05d}"
    sd = seed_key_without_seed(
        task="ledger", instance_id=iid, model=M, model_revision=rev, arm="grid", B=1,
        protocol="agnostic", device="nvidia", backend="cuda-12.8",
        torch_version=torch.__version__, transformers_version="5.2.0",
        kvpress_version="0.5.4", dtype="bfloat16")
    inst = ledger.build(sd, iid, target_tokens=2048, tokenizer=tok)

    for v in inst.variants[:2]:
        for proto in ("agnostic", "aware"):
            if proto == "agnostic":
                pre, _ = templated_parts(tok, inst.context, "")
                facts, n_ctx = facts_and_ctx(inst, tok, pre)
                post = templated_parts(tok, inst.context, v.query)[1]
            else:
                pre, post = aware_parts(tok, inst.context, v.query)
                facts, n_ctx = spans_over(inst, tok, pre)
            gold = {t for f in facts if f.fact_id == v.rec_id for t in f.tokens}
            ratio = press._ratio_for(C + N_SINK + N_WINDOW, n_ctx)

            for a in ARMS:
                cap = methods.Capture()
                if a == "floor_pos":
                    p, _ = press.build_arm("floor_pos", n_ctx=n_ctx, C=C, n_sink=N_SINK,
                                           n_window=N_WINDOW, facts=facts, seed=sd)
                    keep_frac = None
                else:
                    base = methods.make_floor_constrained(
                        methods.build_method(a, ratio), n_ctx, N_SINK, N_WINDOW)
                    p = methods.make_capturing(base, cap)
                    p.compression_ratio = ratio
                o = generate_with(model, tok, pre, [post], p)
                ok = 1.0 if v.answer in o[0] else 0.0
                if a == "floor_pos":
                    fp, _ = press.build_arm("floor_pos", n_ctx=n_ctx, C=C, n_sink=N_SINK,
                                            n_window=N_WINDOW, facts=facts, seed=sd)
                    from harness import ladder
                    kept = set(ladder.floor_pos(n_ctx, C, N_SINK, N_WINDOW, facts).kept)
                    retained = gold <= kept
                else:
                    fr = [1.0 if gold <= cap.per_head[k] else 0.0 for k in cap.heads()]
                    retained = statistics.fmean(fr) > 0.5 if fr else False
                res[a][proto]["ret" if retained else "not"].append(ok)

print(f"{M}  C={C}  N={N} instances x 2 variants")
print(f"{'arm':12s} {'protocol':10s} {'n(gold ret)':>12s} {'acc|ret':>9s} "
      f"{'n(gold NOT)':>12s} {'acc|NOT ret':>12s}")
for a in ARMS:
    for p in ("agnostic", "aware"):
        r, nr = res[a][p]["ret"], res[a][p]["not"]
        ar = statistics.fmean(r) if r else float("nan")
        an = statistics.fmean(nr) if nr else float("nan")
        print(f"{a:12s} {p:10s} {len(r):12d} {ar:9.4f} {len(nr):12d} {an:12.4f}")
