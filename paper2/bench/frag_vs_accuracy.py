"""Follow-up 1 — fragmentation at N=200 on the SAME instances as the M2 grid.

Fragmentation and accuracy then come from one sample, so they can be correlated per instance
rather than compared across samples. Accuracy is read from the completed M2 grid JSONL; the
retained sets are captured here on the identical instance ids (grid_00000..).

Reports the same table plus, per arm per budget, Spearman rho between an instance's
fragmentation ratio and its accuracy.
"""
from __future__ import annotations
import json, re, statistics, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import ladder, methods, press
from harness.keys import seed_key_without_seed
from harness.tasks import ledger
from stage5_ladder_validation import facts_and_ctx, templated_parts

M = "Qwen/Qwen2.5-3B-Instruct"
GRID = Path("runs/nvidia/grid_M2_ledger_agnostic.jsonl")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
BUDGETS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "32,128,512").split(",")]
ARMS = ["snapkv", "expected_attn", "keydiff", "adakv_snapkv"]
N_SINK, N_WINDOW = 8, 64
REC = re.compile(r"^R(\d{3}) \| ")

acc = defaultdict(dict)                       # (C, arm) -> {instance_id: score}
with GRID.open(encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        acc[(r["C"], r["key"]["arm"])][r["key"]["instance_id"]] = r["score"]

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
rev = getattr(model.config, "_commit_hash", None) or "x"


def all_record_spans(inst, pre):
    enc = tok(pre, add_special_tokens=False, return_offsets_mapping=True)
    off = enc["offset_mapping"]; base = pre.index(inst.context)
    out = {}
    for line in inst.context.split("\n"):
        m = REC.match(line)
        if not m or not (1 <= int(m.group(1)) <= ledger.N_RECORDS):
            continue
        a = inst.context.index(line) + base; b = a + len(line)
        out[m.group(0)] = {ti for ti, (x, y) in enumerate(off) if y > x and x < b and y > a}
    return out


@torch.inference_mode()
def capture(name, ratio, pre, n_ctx):
    cap = methods.Capture()
    p = methods.make_capturing(
        methods.make_floor_constrained(methods.build_method(name, ratio),
                                       n_ctx, N_SINK, N_WINDOW), cap)
    p.compression_ratio = ratio
    ids = tok(pre, add_special_tokens=False, return_tensors="pt").to(model.device)
    with p(model):
        model(**ids, use_cache=True)
    return cap


print(f"{M}  N={N}  instances shared with the M2 grid")
print(f"{'C':>5} {'arm':15s} {'gold_tok':>9s} {'complete':>9s} {'partial':>8s} {'none':>7s} "
      f"{'touched':>8s} {'recs_cpl':>9s} {'frag':>6s} {'acc':>7s} {'rho(frag,acc)':>14s}")

for C in BUDGETS:
    per_arm = defaultdict(list)               # arm -> list of (metrics..., frag, acc)
    for i in range(N):
        iid = f"grid_{i:05d}"
        sd = seed_key_without_seed(
            task="ledger", instance_id=iid, model=M, model_revision=rev, arm="grid", B=1,
            protocol="agnostic", device="nvidia", backend="cuda-12.8",
            torch_version=torch.__version__, transformers_version="5.2.0",
            kvpress_version="0.5.4", dtype="bfloat16")
        inst = ledger.build(sd, iid, target_tokens=2048, tokenizer=tok)
        pre, _ = templated_parts(tok, inst.context, "")
        facts, n_ctx = facts_and_ctx(inst, tok, pre)
        recs = all_record_spans(inst, pre)
        qids = [v.rec_id + " | " for v in inst.variants]
        ratio = press._ratio_for(C + N_SINK + N_WINDOW, n_ctx)

        def tally(keepsets):
            gt, cpl, par, non, tch, rcp = [], [], [], [], [], []
            for kk in keepsets:
                gt.append(sum(len(recs[q] & kk) for q in qids if q in recs))
                c = p_ = z = 0
                for q in qids:
                    s = recs.get(q, set()); n_in = len(s & kk)
                    if s and n_in == len(s): c += 1
                    elif n_in > 0: p_ += 1
                    else: z += 1
                h = len(qids)
                cpl.append(c / h); par.append(p_ / h); non.append(z / h)
                tch.append(sum(1 for s in recs.values() if s & kk))
                rcp.append(sum(1 for s in recs.values() if s and s <= kk))
            return [statistics.fmean(x) for x in (gt, cpl, par, non, tch, rcp)]

        fp = set(ladder.floor_pos(n_ctx, C, N_SINK, N_WINDOW, facts).kept)
        m = tally([fp]); m.append(m[4] / max(1e-9, m[5]))
        m.append(acc[(C, "floor_pos")].get(iid, float("nan")))
        per_arm["floor_pos"].append(m)

        for name in ARMS:
            cap = capture(name, ratio, pre, n_ctx)
            m = tally([cap.per_head[k] for k in cap.heads()])
            m.append(m[4] / max(1e-9, m[5]))
            m.append(acc[(C, name)].get(iid, float("nan")))
            per_arm[name].append(m)

    for name in ["floor_pos"] + ARMS:
        v = per_arm[name]
        cols = [statistics.fmean(r[i] for r in v) for i in range(8)]
        fr = [r[6] for r in v]; ac = [r[7] for r in v]
        ok = [(a, b) for a, b in zip(fr, ac) if b == b]
        rho = spearmanr([a for a, _ in ok], [b for _, b in ok]).statistic if len(ok) > 2 else float("nan")
        rho = rho if rho == rho else float("nan")
        print(f"{C:5d} {name:15s} {cols[0]:9.2f} {cols[1]:9.3f} {cols[2]:8.3f} {cols[3]:7.3f} "
              f"{cols[4]:8.2f} {cols[5]:9.2f} {cols[6]:6.1f} {cols[7]:7.4f} {rho:14.3f}")
    print()
