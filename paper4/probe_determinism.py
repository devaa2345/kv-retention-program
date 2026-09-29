"""A6 diagnosis: is the gather-order difference nondeterminism, or a deterministic order effect?

Same process, same instance, same arm, run TWICE -> tests nondeterminism.
Compared against the snapkv/ascend-snapkv pair -> tests the order effect.
"""
import json, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from harness import methods, press
from p4 import common as CM
from p4 import probes as PR

HERE = Path(__file__).resolve().parent
C, N = 512, 10
model_name = sys.argv[1]
tag = CM.TAGS[model_name]
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16,
                                             attn_implementation="sdpa").to("cuda").eval()
rev = model.config._commit_hash
B = C + CM.N_SINK + CM.N_WINDOW

def gen(arm, b):
    cap, stats = methods.Capture(), {}
    ratio = press._ratio_for(B, b["n_ctx"])
    if arm == "floor_pos":
        p, _ = press.build_arm("floor_pos", n_ctx=b["n_ctx"], C=C, n_sink=CM.N_SINK, n_window=CM.N_WINDOW)
        p = methods.make_capturing(p, cap)
    else:
        base = methods.make_floor_constrained(methods.build_method("snapkv", ratio), b["n_ctx"], CM.N_SINK, CM.N_WINDOW)
        if arm == "snapkv":
            p = methods.make_capturing(base, cap)
        elif arm == "ascend-snapkv":
            p = PR.make_ascending_gather(base, cap, stats)
        else:
            from p4 import unitwrap as UW
            p = UW.make_unit_aware(base, b["ui"], C, capture=cap, stats=stats)
    p.compression_ratio = ratio
    outs, _ = CM.generate_checked(model, tok, b["pre"], b["posts"], p, b["mns"], b["n_ctx"], B, headwise=False)
    return outs, {k: sorted(v) for k, v in cap.per_head.items()}

res = {}
for arm in ("floor_pos", "snapkv", "U-snapkv", "ascend-snapkv"):
    same_gen = same_keep = 0
    for i in range(N):
        b = CM.build(tag, model_name, rev, tok, 8, "s4_%05d" % i)
        g1, k1 = gen(arm, b)
        g2, k2 = gen(arm, b)
        same_gen += (g1 == g2)
        same_keep += (k1 == k2)
    res[arm] = (same_gen, same_keep)
    print(f"  {tag} {arm}: repeat-run generations identical {same_gen}/{N}, keep-sets identical {same_keep}/{N}", flush=True)

# cross-arm: identical keep-set, different gather order
sg = sk = 0
for i in range(N):
    b = CM.build(tag, model_name, rev, tok, 8, "s4_%05d" % i)
    g1, k1 = gen("snapkv", b)
    g2, k2 = gen("ascend-snapkv", b)
    sg += (g1 == g2); sk += (k1 == k2)
print(f"  {tag} snapkv vs ascend-snapkv: generations identical {sg}/{N}, keep-sets identical {sk}/{N}")
(HERE / "out" / f"a6_determinism_{tag}.json").write_text(json.dumps(
    dict(repeat=res, cross=dict(gen=sg, keep=sk), n=N), indent=1), encoding="utf-8")
