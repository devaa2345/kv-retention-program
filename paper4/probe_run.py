"""Tier-1 probe runner (CANDIDATES.md A1-A6). One process per invocation; all arms of an
instance generated in that one session, invariants asserted per row.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import methods, press
from p3 import keys3, runner
from p4 import common as CM
from p4 import probes as PR
from p4.floorcheck import assert_floor_pos

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "nvidia"
ENV3 = dict(CM.ENV, torch_version="2.11.0+cu128")
NL = "\n"
ARMS = ("floor_pos", "snapkv", "U-snapkv", "consensus-snapkv", "early-snapkv", "hybrid-snapkv",
        "supra-snapkv", "sub-snapkv", "floor_sink32", "ascend-snapkv")


def build(arm, b, C, tok, cap, stats):
    """Returns (press, headwise, n_sink, n_window, unit_index_or_None)."""
    NS, NW = CM.N_SINK, CM.N_WINDOW
    B = C + NS + NW
    ratio = press._ratio_for(B, b["n_ctx"])
    if arm == "floor_pos":
        p, _ = press.build_arm("floor_pos", n_ctx=b["n_ctx"], C=C, n_sink=NS, n_window=NW)
        p = methods.make_capturing(p, cap)
        p.compression_ratio = ratio
        return p, False, NS, NW
    if arm == "floor_sink32":                       # A5: matched B, 24 tokens moved into the sink
        p, _ = press.build_arm("floor_pos", n_ctx=b["n_ctx"], C=C - 24, n_sink=32, n_window=NW)
        p = methods.make_capturing(p, cap)
        p.compression_ratio = ratio
        return p, False, 32, NW
    base = methods.make_floor_constrained(methods.build_method("snapkv", ratio), b["n_ctx"], NS, NW)
    if arm == "snapkv":
        p = methods.make_capturing(base, cap)
    elif arm == "ascend-snapkv":                    # A6
        p = PR.make_ascending_gather(base, cap, stats)
    elif arm == "U-snapkv":
        from p4 import unitwrap as UW
        p = UW.make_unit_aware(base, b["ui"], C, capture=cap, stats=stats)
    elif arm in ("consensus-snapkv", "early-snapkv", "hybrid-snapkv"):
        mode = {"consensus-snapkv": "consensus", "early-snapkv": "early",
                "hybrid-snapkv": "hybrid"}[arm]
        p = PR.make_probe_press(base, b["ui"], C, mode, cap, stats)
    elif arm in ("supra-snapkv", "sub-snapkv"):     # A4
        from p4 import unitwrap as UW
        gran = "supra" if arm.startswith("supra") else "sub"
        ui = PR.unit_index(b["inst"], b["sp"]["task"], tok, b["pre"], NS, NW, gran)
        p = UW.make_unit_aware(base, ui, C, capture=cap, stats=stats)
    else:
        raise ValueError(arm)
    p.compression_ratio = ratio
    return p, False, NS, NW


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--c", type=int, default=8)
    ap.add_argument("--C", type=int, default=512)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    a = ap.parse_args()
    session_id = uuid.uuid4().hex
    tag = CM.TAGS[a.model]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    rev = getattr(model.config, "_commit_hash", None) or "unresolved"
    out = RUNS / f"p4_probe_{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    B = a.C + CM.N_SINK + CM.N_WINDOW
    print(f"  probes {tag} c={a.c} C={a.C} n={a.n} session {session_id[:8]}: {a.arms}", flush=True)
    t0 = time.time()
    n_fail = 0
    for i in range(a.n):
        b = CM.build(tag, a.model, rev, tok, a.c, "s4_%05d" % i)
        for arm in a.arms:
            t1 = time.time()
            try:
                cap, stats = methods.Capture(), {}
                p, headwise, ns, nw = build(arm, b, a.C, tok, cap, stats)
                outs, flags = CM.generate_checked(model, tok, b["pre"], b["posts"], p, b["mns"],
                                                  b["n_ctx"], B, headwise=headwise)
                parity = runner.assert_budget_parity(cap, a.C, b["n_ctx"], "snapkv")
                keeps = [cap.per_head[h] for h in cap.heads()]
                floor = set(range(ns)) | set(range(b["n_ctx"] - nw, b["n_ctx"]))
                if not all(floor <= kk for kk in keeps):
                    raise AssertionError(f"{arm}: mandatory floors not retained")
                if arm == "floor_pos":
                    for kk in keeps:
                        assert_floor_pos(kk, b["n_ctx"], a.C)
                met = CM.keep_metrics(keeps, b["facts"], b["line_units"], b["n_ctx"])
                kd = dict(task=b["sp"]["task"], instance_id="s4_%05d" % i, model=a.model,
                          model_revision=rev, arm=arm, B=B, C=a.C, seed=b["sd"],
                          n_fields=b["sp"]["n_fields"], n_records=b["sp"]["k"],
                          layout="p4_probe_tier1", matched_to=None, max_new=max(b["mns"]), **ENV3)
                row = dict(key_digest=keys3.digest(kd), key=kd, session_id=session_id,
                           gather="score",
                           produced_on=CM.PRODUCED_ON, model_tag=tag, c_tag=a.c, C=a.C, B=B,
                           arm=arm, instance=i, n_ctx=b["n_ctx"], B_asserted=parity,
                           units_taken=stats.get("units_taken"), fallback=stats.get("fallback"),
                           score=CM.S4.SCORERS[b["sp"]["task"]](outs, b["inst"]),
                           per_variant=[CM.S4.SCORE_ONE[b["sp"]["task"]](o, v.answer)
                                        for v, o in zip(b["inst"].variants, outs)],
                           gen=outs, answers=[v.answer for v in b["inst"].variants],
                           stop_flags=flags, wall_s=time.time() - t1, **met)
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + NL)
            except Exception as e:
                n_fail += 1
                print(f"  FAIL {arm} inst {i}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 5 == 0:
            print(f"    inst {i+1}/{a.n}  {(time.time()-t0)/60:.1f} min  fails {n_fail}",
                  flush=True)
    print(f"  DONE probes {tag}: {(time.time()-t0)/60:.1f} min, {n_fail} failures", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
