"""Experiment C: full ladder + four admitted methods on the natural-text dataset (GPU).

One process, one (model, budget) per invocation, resumable. Every row stores the raw generations,
so scoring rules can change without regenerating.

Arms
  full_cache, null, random, floor_pos, snapkv, adakv_snapkv, expected_attn, keydiff
      -- independent of cost level, so ONE prefill answers all 12 query variants.
  oracle_causal   -- keeps all H candidate facts' gold tokens; gold differs per cost level,
                     so one prefill PER LEVEL answers that level's 4 variants.
  oracle_prescient -- the queried fact only; one prefill PER VARIANT.
Gold at level L = sentence start .. end of the L-th required element (contiguous, tokens from the
templated prefix, the same mapping the runner uses for every other task).
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from p3 import runner
from p3.natural import facts as F
from p3.natural import gen
from p3.natural import score as S
from harness import ladder, methods, press  # path added by p3.runner

MODELS = {"M2": "Qwen/Qwen2.5-3B-Instruct", "M3": "meta-llama/Llama-3.2-3B-Instruct"}
N_SINK, N_WINDOW = 8, 64
SLACK = 128
NOISE_SLACK = 32     # null/random are chance baselines that ramble to the cap; 32 keeps them cheap
GENERIC = ("full_cache", "null", "random", "floor_pos", "snapkv", "adakv_snapkv",
           "expected_attn", "keydiff")
ORACLES = ("oracle_causal", "oracle_prescient")


def max_new(tok, v):
    return len(tok(" | ".join(v["elements"]) + " $", add_special_tokens=False)["input_ids"]) + SLACK


def token_facts(rd, pre, tok, level):
    """FactSpans for the H queried facts at `level`, over the TEMPLATED prefix."""
    enc = tok(pre, add_special_tokens=False, return_offsets_mapping=True)
    off = enc["offset_mapping"]
    base = pre.index(rd["context"])
    out = []
    for fr in rd["facts"]:
        if fr["role"] != "queried":
            continue
        a, b = fr["gold_char"][str(level)]
        a += base
        b += base
        idx = tuple(ti for ti, (x, y) in enumerate(off) if y > x and x < b and y > a)
        out.append(ladder.FactSpans(fr["fact_id"], (idx,)))
    return out, len(off)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=list(MODELS))
    ap.add_argument("--C", type=int, required=True)
    ap.add_argument("--data", default="data/natural/nat_v1.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODELS[a.tag])
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[a.tag], dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    out = Path(a.out or f"runs/nvidia/natC_{a.tag}_C{a.C}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["iid"], r["arm"], r["C"], r["level"], r["vkey"]))
            except Exception:
                pass
    recs = [json.loads(l) for l in open(a.data, encoding="utf-8")][:a.n]
    C, B = a.C, a.C + N_SINK + N_WINDOW
    t0, ncell = time.time(), 0
    last = [time.time()]

    def emit(rec, arm, level, vkey, vs, outs, flags, n_ctx, extra=None):
        row = dict(iid=rec["instance_id"], tag=a.tag, arm=arm, C=C if arm != "full_cache" else 0,
                   level=level, vkey=vkey, n_ctx=n_ctx,
                   variants=[dict(fact_id=v["fact_id"], level=v["level"]) for v in vs],
                   answers=[v["elements"] for v in vs], gen=outs, flags=flags,
                   score=[S.score_one(o, v["elements"], rec["persons"]) for o, v in zip(outs, vs)])
        now = time.time()
        row['wall_s'] = now - last[0]
        last[0] = now
        row.update(extra or {})
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    for i, rec in enumerate(recs):
        rd = rec["renderings"][a.tag]
        ctx = rd["context"]
        pre, _ = runner.templated_parts(tok, ctx, "")
        n_ctx = len(tok(pre, add_special_tokens=False)["input_ids"])
        vs_all = rec["variants"]
        posts_all = [runner.templated_parts(tok, ctx, v["query"])[1] for v in vs_all]
        mn_all = [max_new(tok, v) for v in vs_all]
        iid = rec["instance_id"]

        for arm in GENERIC:
            arm_C = 0 if arm == "full_cache" else C
            if arm == "full_cache" and C != 256:
                continue                                   # budget-free; produced once
            if (iid, arm, arm_C, None, "all") in done:
                continue
            if arm == "full_cache":
                p = None
            elif arm in ("null", "random", "floor_pos"):
                p, _ = press.build_arm(arm, n_ctx=n_ctx, C=C, n_sink=N_SINK, n_window=N_WINDOW,
                                       facts=None, seed=rec["seed"])
            else:
                ratio = press._ratio_for(B, n_ctx)
                p = methods.make_floor_constrained(methods.build_method(arm, ratio),
                                                   n_ctx, N_SINK, N_WINDOW)
                p.compression_ratio = ratio
                if i < 2:                                  # budget parity, from realised keep-sets
                    cap = runner.capture_keepsets(model, tok, pre, arm, C, n_ctx)
                    runner.assert_budget_parity(cap, C, n_ctx, arm)
            mns_arm = [m - SLACK + NOISE_SLACK for m in mn_all] if arm in ("null", "random") else mn_all
            outs, flags = gen.generate_stop(model, tok, pre, posts_all, p, mns_arm)
            emit(rec, arm, None, "all", vs_all, outs, flags, n_ctx)
            ncell += 1

        for lv in F.LEVELS:
            facts, _ = token_facts(rd, pre, tok, lv)
            vs = [v for v in vs_all if v["level"] == lv]
            posts = [posts_all[vs_all.index(v)] for v in vs]
            mns = [mn_all[vs_all.index(v)] for v in vs]
            if (iid, "oracle_causal", C, lv, "all") not in done:
                p, _ = press.build_arm("oracle_causal", n_ctx=n_ctx, C=C, n_sink=N_SINK,
                                       n_window=N_WINDOW, facts=facts, seed=rec["seed"])
                outs, flags = gen.generate_stop(model, tok, pre, posts, p, mns)
                emit(rec, "oracle_causal", lv, "all", vs, outs, flags, n_ctx)
            for v, post, mn in zip(vs, posts, mns):
                if (iid, "oracle_prescient", C, lv, v["fact_id"]) in done:
                    continue
                gf = next(f for f in facts if f.fact_id == v["fact_id"])
                p, _ = press.build_arm("oracle_prescient", n_ctx=n_ctx, C=C, n_sink=N_SINK,
                                       n_window=N_WINDOW, facts=facts, gold=gf, seed=rec["seed"])
                outs, flags = gen.generate_stop(model, tok, pre, [post], p, [mn])
                emit(rec, "oracle_prescient", lv, v["fact_id"], [v], outs, flags, n_ctx)
        el = (time.time() - t0) / 60
        print(f"  [{a.tag} C={C}] {i + 1}/{len(recs)}  {el:.1f} min  eta "
              f"{el / (i + 1) * (len(recs) - i - 1):.1f} min", flush=True)
    print("DONE", a.tag, C, flush=True)


if __name__ == "__main__":
    main()
