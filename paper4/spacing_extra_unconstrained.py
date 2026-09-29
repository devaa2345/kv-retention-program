"""Additional unconstrained comparator rows for instances >= 50, c=40 C=512, for the SPACING
test's own n-raise. Written to a SEPARATE file from the frozen Stage 3 grid
(`runs/nvidia/p4_s3_{tag}.jsonl`, PREREG_P4_S3.md's own n=50 parameter), so raising n here does
NOT touch or retroactively violate that frozen grid. Generates floor_pos, U-snapkv,
U-expected_attn only -- the three rows the spacing test's contamination tagging and confound
check need (no oracle_causal, no other method arms).
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

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "nvidia"
C_TAG, C_BUDGET = 40, 512
NL = "\n"
BASES = ("snapkv", "expected_attn", "adakv_snapkv")


def load_done(tag):
    p = RUNS / f"p4_spacing_extra_unconstrained_{tag}.jsonl"
    done = set()
    if p.exists():
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            done.add((r["arm"], r["instance"]))
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    a = ap.parse_args()
    session_id = uuid.uuid4().hex
    tag = CM.TAGS[a.model]
    out = RUNS / f"p4_spacing_extra_unconstrained_{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(tag)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    rev = getattr(model.config, "_commit_hash", None) or "unresolved"
    B = C_BUDGET + CM.N_SINK + CM.N_WINDOW
    bitmaps = {}
    npz_path = RUNS / f"p4_spacing_extra_keepsets_{tag}.npz"
    if npz_path.exists():
        with np.load(npz_path) as z:
            bitmaps = {k: z[k] for k in z.files}

    t0 = time.time()
    n_fail = 0
    for i in range(a.start, a.end):
        iid = "s4_%05d" % i
        b = CM.build(tag, a.model, rev, tok, C_TAG, iid)
        arms = ["floor_pos"] + ["U-" + x for x in BASES]
        for arm in arms:
            if (arm, i) in done:
                continue
            t1 = time.time()
            try:
                cap, stats = methods.Capture(), {}
                if arm == "floor_pos":
                    p, _ = press.build_arm("floor_pos", n_ctx=b["n_ctx"], C=C_BUDGET,
                                           n_sink=CM.N_SINK, n_window=CM.N_WINDOW)
                    p = methods.make_capturing(p, cap)
                    p.compression_ratio = press._ratio_for(B, b["n_ctx"])
                else:
                    p = CM.build_press(arm, b["n_ctx"], C_BUDGET, b["ui"], cap, stats)
                headwise = arm.startswith("U-adakv")
                outs, flags = CM.generate_checked(model, tok, b["pre"], b["posts"], p, b["mns"],
                                                  b["n_ctx"], B, headwise=headwise)
                parity = runner.assert_budget_parity(cap, C_BUDGET, b["n_ctx"], CM.base_arm(arm))
                keeps = [cap.per_head[h] for h in cap.heads()]
                met = CM.keep_metrics(keeps, b["facts"], b["line_units"], b["n_ctx"])
                if not met["floor_ok"]:
                    raise AssertionError(f"{arm}: mandatory floors not retained")
                bm = np.zeros((len(keeps), b["n_ctx"]), dtype=bool)
                for s, kk in enumerate(keeps):
                    bm[s, sorted(kk)] = True
                bitmaps[f"{i}|{arm}"] = np.packbits(bm, axis=1)
                row = dict(key=dict(task=b["sp"]["task"], instance_id=iid, model=a.model,
                                    model_revision=rev, arm=arm, B=B, C=C_BUDGET, seed=b["sd"]),
                           session_id=session_id, produced_on=CM.PRODUCED_ON, model_tag=tag,
                           c_tag=C_TAG, C=C_BUDGET, B=B, arm=arm, instance=i, n_ctx=b["n_ctx"],
                           B_asserted=parity,
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
        if (i + 1) % 10 == 0:
            print(f"  [{tag}] inst {i+1}/{a.end}  {(time.time()-t0)/60:.1f} min  fails {n_fail}",
                  flush=True)
    np.savez_compressed(npz_path, **bitmaps)
    print(f"  DONE extra-unconstrained {tag} [{a.start},{a.end}): "
          f"{(time.time()-t0)/60:.1f} min, {n_fail} failures", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
