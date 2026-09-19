"""Natural-text ladder: anchor gate (GPU). Single process. Resumable.

Per model, n instances, per instance:
  * full_cache                                       -> condition 1 (band [0.55, 0.97]) per level
  * floor_pos at every planned budget                -> condition 2 (> 0.05) per (level, budget)
  * full_cache with the QUERIED fact sentence DELETED -> shortcut probe 4 (must fall to chance)
Every (arm, budget) is scored at each cost level (1, 3, 5 elements) as the mean over the H
query variants. Rows are appended as they are produced and keyed, so a power loss resumes.
"""
import argparse
import json
import statistics as st
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from p3 import runner
from p3.natural import facts as F
from p3.natural import score as S
from harness import press  # path added by p3.runner

MODELS = {"M2": "Qwen/Qwen2.5-3B-Instruct", "M3": "meta-llama/Llama-3.2-3B-Instruct"}
BAND = (0.55, 0.97)
FLOOR_MIN = 0.05
N_SINK, N_WINDOW = 8, 64
RUNS = Path("runs/nvidia")


def variants_at(rec, level):
    return [v for v in rec["variants"] if v["level"] == level]


SLACK = 16


def max_new(tok, v):
    return len(tok(" | ".join(v["elements"]) + " $", add_special_tokens=False)["input_ids"]) + SLACK


def run_arm(model, tok, ctx, vs, press_obj):
    pre, _ = runner.templated_parts(tok, ctx, "")
    posts = [runner.templated_parts(tok, ctx, v["query"])[1] for v in vs]
    return runner.generate_with(model, tok, pre, posts, press_obj, [max_new(tok, v) for v in vs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=list(MODELS))
    ap.add_argument("--data", default="data/natural/nat_v1.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--budgets", default="256,512,1024")
    ap.add_argument("--out", default=None)
    ap.add_argument("--slack", type=int, default=16)
    ap.add_argument("--only_full", action="store_true")
    ap.add_argument("--levels", default="1,3,5")
    a = ap.parse_args()
    global SLACK
    SLACK = a.slack
    LV = [int(x) for x in a.levels.split(",")]
    budgets = [int(x) for x in a.budgets.split(",")]
    tok = AutoTokenizer.from_pretrained(MODELS[a.tag])
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[a.tag], dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    out = Path(a.out or RUNS / f"nat_gate_{Path(a.data).stem}_{a.tag}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["iid"], r["arm"], r["C"]))
            except Exception:
                pass
    recs = [json.loads(l) for l in open(a.data, encoding="utf-8")][:a.n]
    t0 = time.time()
    for i, rec in enumerate(recs):
        rd = rec["renderings"][a.tag]
        ctx, iid = rd["context"], rec["instance_id"]
        pre, _ = runner.templated_parts(tok, ctx, "")
        n_ctx = len(tok(pre, add_special_tokens=False)["input_ids"])
        persons = rec["persons"]
        plans = [("full_cache", 0)] + ([] if a.only_full else [("floor_pos", C) for C in budgets])
        for arm, C in plans:
            if (iid, arm, C) in done:
                continue
            p = None
            if arm == "floor_pos":
                p, _ = press.build_arm("floor_pos", n_ctx=n_ctx, C=C, n_sink=N_SINK,
                                       n_window=N_WINDOW, facts=None, seed=0)
            row = dict(iid=iid, tag=a.tag, arm=arm, C=C, n_ctx=n_ctx, levels={})
            for lv in LV:
                vs = variants_at(rec, lv)
                outs, flags = run_arm(model, tok, ctx, vs, p)
                row["levels"][str(lv)] = dict(score=S.score_instance(outs, vs, persons),
                                              gen=outs, answers=[v["elements"] for v in vs],
                                              flags=flags)
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        # probe 4: delete each queried fact from the context, ask about exactly that fact
        if not a.only_full and (iid, "full_cache_deleted", 0) not in done:
            row = dict(iid=iid, tag=a.tag, arm="full_cache_deleted", C=0, n_ctx=n_ctx, levels={})
            per = {lv: ([], []) for lv in F.LEVELS}
            for fr in [f for f in rd["facts"] if f["role"] == "queried"]:
                s, e = fr["char"]
                s0 = s - 1 if ctx[s - 1] == " " else s
                ctx_del = ctx[:s0] + ctx[e:]
                for lv in F.LEVELS:
                    v = next(x for x in variants_at(rec, lv) if x["fact_id"] == fr["fact_id"])
                    outs, _ = run_arm(model, tok, ctx_del, [v], None)
                    per[lv][0].append(outs[0])
                    per[lv][1].append(v)
            for lv in F.LEVELS:
                row["levels"][str(lv)] = dict(
                    score=S.score_instance(per[lv][0], per[lv][1], persons),
                    gen=per[lv][0], answers=[v["elements"] for v in per[lv][1]])
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        if (i + 1) % 5 == 0:
            el = (time.time() - t0) / 60
            print(f"  [{a.tag}] {i + 1}/{len(recs)}  {el:.1f} min  eta {el / (i + 1) * (len(recs) - i - 1):.1f} min",
                  flush=True)
    print("DONE", a.tag, flush=True)


if __name__ == "__main__":
    main()
