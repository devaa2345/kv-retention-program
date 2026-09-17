"""Why does the aware protocol not reproduce the literature's ordering?

Observed at M2/C=512: SnapKV delta -0.002 (reference ~+0.20), KeyDiff +0.367 (reference ~+0.01),
ordering inverted. Checks, in order:

  1. Is the question actually inside the compressed prefix? If it is not, no press can see it and
     "aware" is agnostic under another name.
  2. Does each method's retained set actually CHANGE between protocols? A query-aware method
     whose retained set is identical in both is not reading the question.
  3. Where does SnapKV's observation window fall? SnapKV scores context by attention from the
     last `window_size` queries. If the floor constraint or the template puts something other
     than the question there, SnapKV is scoring against the wrong thing.
  4. Does the retained set overlap the GOLD more under aware than agnostic? That is the only
     overlap that should improve if the query is being used.
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
from stage5_ladder_validation import facts_and_ctx, templated_parts
from run_grid_aware import aware_parts, spans_over

M = "Qwen/Qwen2.5-3B-Instruct"
C = int(sys.argv[1]) if len(sys.argv) > 1 else 512
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
N_SINK, N_WINDOW = 8, 64
ARMS = ["snapkv", "expected_attn", "keydiff", "adakv_snapkv"]

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
rev = getattr(model.config, "_commit_hash", None) or "x"


@torch.inference_mode()
def cap_for(name, pre, n_ctx):
    ratio = press._ratio_for(C + N_SINK + N_WINDOW, n_ctx)
    cap = methods.Capture()
    p = methods.make_capturing(
        methods.make_floor_constrained(methods.build_method(name, ratio),
                                       n_ctx, N_SINK, N_WINDOW), cap)
    p.compression_ratio = ratio
    ids = tok(pre, add_special_tokens=False, return_tensors="pt").to(model.device)
    with p(model):
        model(**ids, use_cache=True)
    return cap


rows = {a: {"jac": [], "gold_ag": [], "gold_aw": []} for a in ARMS}
qpos = []
for i in range(N):
    iid = f"grid_{i:05d}"
    sd = seed_key_without_seed(
        task="ledger", instance_id=iid, model=M, model_revision=rev, arm="grid", B=1,
        protocol="agnostic", device="nvidia", backend="cuda-12.8",
        torch_version=torch.__version__, transformers_version="5.2.0",
        kvpress_version="0.5.4", dtype="bfloat16")
    inst = ledger.build(sd, iid, target_tokens=2048, tokenizer=tok)
    v = inst.variants[0]

    ag_pre, _ = templated_parts(tok, inst.context, "")
    ag_facts, ag_n = facts_and_ctx(inst, tok, ag_pre)
    aw_pre, _ = aware_parts(tok, inst.context, v.query)
    aw_facts, aw_n = spans_over(inst, tok, aw_pre)

    if i == 0:
        print("[1] question inside the compressed prefix?")
        print(f"    agnostic prefix ends: {ag_pre[-70:]!r}")
        print(f"    aware    prefix ends: {aw_pre[-90:]!r}")
        print(f"    query text in aware prefix: {v.query in aw_pre}")
        print(f"    n_ctx agnostic {ag_n} -> aware {aw_n} (+{aw_n-ag_n} tokens)")
    # where does the question sit relative to the last window_size tokens?
    qtok = len(tok(v.query, add_special_tokens=False)["input_ids"])
    qpos.append(qtok)

    ag_gold = {t for f in ag_facts if f.fact_id == v.rec_id for t in f.tokens}
    aw_gold = {t for f in aw_facts if f.fact_id == v.rec_id for t in f.tokens}

    for a in ARMS:
        ca, cw = cap_for(a, ag_pre, ag_n), cap_for(a, aw_pre, aw_n)
        ja, jw = [], []
        for k in ca.heads():
            if k not in cw.per_head:
                continue
            A, B_ = ca.per_head[k], cw.per_head[k]
            ja.append(len(A & B_) / max(1, len(A | B_)))
        rows[a]["jac"].append(statistics.fmean(ja) if ja else float("nan"))
        rows[a]["gold_ag"].append(statistics.fmean(
            [len(ca.per_head[k] & ag_gold) / max(1, len(ag_gold)) for k in ca.heads()]))
        rows[a]["gold_aw"].append(statistics.fmean(
            [len(cw.per_head[k] & aw_gold) / max(1, len(aw_gold)) for k in cw.heads()]))

print(f"\n    question length: {statistics.fmean(qpos):.1f} tokens "
      f"(SnapKV observation window = 64; floors pin the last {N_WINDOW})")
print("\n[2/4] retained-set change between protocols, and gold recall")
print(f"    {'method':16s} {'jaccard(ag,aw)':>15s} {'gold recall ag':>15s} {'gold recall aw':>15s} {'delta':>8s}")
for a in ARMS:
    j = statistics.fmean(rows[a]["jac"])
    g1 = statistics.fmean(rows[a]["gold_ag"])
    g2 = statistics.fmean(rows[a]["gold_aw"])
    print(f"    {a:16s} {j:15.3f} {g1:15.3f} {g2:15.3f} {g2-g1:+8.3f}")
