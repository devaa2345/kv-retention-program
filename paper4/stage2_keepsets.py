"""Prefill-only recapture of the c=40 pilot keep-sets (DISPERSION_RULE.md). No generation.

Every row must reproduce its pilot row's keep-set summaries exactly, so the saved keep-sets are the
ones that generated. Saves per-row slot bitmaps and per-query fact token lists.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import methods
from p4 import common as CM

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "nvidia"
C = 512
ARMS = ("floor_pos", "snapkv", "adakv_snapkv", "U-snapkv", "U-adakv_snapkv")
F = ("p_g", "q_complete", "q_any", "units_complete", "units_touched")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    tag = CM.TAGS[a.model]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to("cuda").eval()
    rev = model.config._commit_hash
    pilot = {(r["arm"], r["instance"]): r for r in map(
        json.loads, open(RUNS / f"p4_pilot_{tag}.jsonl", encoding="utf-8")) if r["label"] == "c=40"}
    bitmaps, meta, bad, worst = {}, {}, 0, 0.0
    for i in range(a.n):
        b = CM.build(tag, a.model, rev, tok, 40, "s4_%05d" % i)
        by_id = {f.fact_id: sorted({t for s in f.spans for t in s}) for f in b["facts"]}
        meta[i] = dict(n_ctx=b["n_ctx"],
                       facts=[by_id[v.rec_id] for v in b["inst"].variants],
                       per_variant={arm: pilot[(arm, i)]["per_variant"] for arm in ARMS})
        for arm in ARMS:
            cap = methods.Capture()
            CM.prefill(model, tok, b["pre"], CM.build_press(arm, b["n_ctx"], C, b["ui"], cap, {}))
            keeps = [cap.per_head[h] for h in cap.heads()]
            m = CM.keep_metrics(keeps, b["facts"], b["line_units"], b["n_ctx"])
            d = max(abs(m[f] - pilot[(arm, i)][f]) for f in F)
            worst = max(worst, d)
            bad += d > 1e-9
            bm = np.zeros((len(keeps), b["n_ctx"]), dtype=bool)
            for s, kk in enumerate(keeps):
                bm[s, sorted(kk)] = True
            bitmaps[f"{i}|{arm}"] = np.packbits(bm, axis=1)
        if (i + 1) % 10 == 0:
            print(f"  [{tag}] inst {i + 1}/{a.n}  mismatches {bad}  max|d| {worst:.2e}", flush=True)
    np.savez_compressed(RUNS / f"p4_keepsets_c40_{tag}.npz", **bitmaps)
    (RUNS / f"p4_keepsets_c40_{tag}.meta.json").write_text(json.dumps(dict(
        produced_on=CM.PRODUCED_ON, rows=a.n * len(ARMS), mismatches=bad, max_abs_diff=worst,
        instances=meta)), encoding="utf-8")
    print(f"  DONE {tag}: {a.n * len(ARMS)} rows, mismatches vs pilot {bad}, max|d| {worst:.2e}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
