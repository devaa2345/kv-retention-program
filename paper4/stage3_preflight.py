"""Stage 3 preflight. Blocking; runs before any generation row.

  seeds     (CPU)  amendment A2: instance seeds must equal Paper 3 Stage 4's / Paper 4 Stage 2's
                   for the same instance ids, i.e. adding torch_version to the dedup key did NOT
                   change the instances.
  identity  (GPU, prefill only) prereg section 6 hygiene: with every unit forced to a single
                   token, U-X must reproduce X's keep-sets up to exact score ties, once per
                   (model, budget, method).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import methods
from p4 import common as CM
from p4 import unitwrap as UW

HERE = Path(__file__).resolve().parent
P3RUNS = HERE.parent / "paper3" / "runs" / "nvidia"
RUNS = HERE / "runs" / "nvidia"
BUDGETS = (32, 64, 128, 256, 512)
C_TAGS = (1, 8, 19, 40)
N = 50


def stored_seeds(tag):
    """(task, n_fields, instance_id) -> seed, from Paper 3 Stage 4 and Paper 4 Stage 2 rows."""
    out = {}
    for f in (P3RUNS / f"stage4_plane_{tag}.jsonl", P3RUNS / f"stage4_capture_{tag}.jsonl",
              RUNS / f"p4_pilot_{tag}.jsonl"):
        if not f.exists():
            continue
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                k = json.loads(line)["key"]
                i = int(k["instance_id"][3:])
                if i < N:
                    out[(k["task"], k["n_fields"], k["instance_id"])] = k["seed"]
    return out


def check_seeds():
    ok = True
    for model, tag in CM.TAGS.items():
        tok = AutoTokenizer.from_pretrained(model)
        stored = stored_seeds(tag)
        rev = None
        for f in (P3RUNS / f"stage4_plane_{tag}.jsonl",):
            with f.open(encoding="utf-8") as fh:
                rev = json.loads(fh.readline())["key"]["model_revision"]
        n_cmp = n_bad = 0
        for c_tag in C_TAGS:
            sp = CM.S4.spec(tag, c_tag=c_tag)
            for i in range(N):
                iid = "s4_%05d" % i
                key = (sp["task"], sp["n_fields"], iid)
                if key not in stored:
                    continue
                sd = CM.S4.keys3.instance_seed(
                    task=sp["task"], instance_id=iid, model=model, model_revision=rev,
                    n_fields=sp["n_fields"], layout=None, n_records=sp["k"], **CM.ENV)
                n_cmp += 1
                n_bad += sd != stored[key]
        print(f"  seeds {tag}: {n_cmp - n_bad}/{n_cmp} match stored Paper 3/Stage 2 seeds")
        ok &= n_bad == 0 and n_cmp > 0
    return ok


def check_identity():
    ok = True
    rows = []
    for model, tag in CM.TAGS.items():
        tok = AutoTokenizer.from_pretrained(model)
        mdl = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        rev = mdl.config._commit_hash
        b = CM.build(tag, model, rev, tok, 40, "s4_00000")
        for C in BUDGETS:
            for m in CM.X_METHODS:
                capx = methods.Capture()
                CM.prefill(mdl, tok, b["pre"], CM.build_press(m, b["n_ctx"], C, b["ui"], capx, {}))
                ui0 = UW.UnitIndex(b["n_ctx"], [], CM.N_SINK, CM.N_WINDOW)
                capu, st = methods.Capture(), {"store_scores": True}
                CM.prefill(mdl, tok, b["pre"], CM.build_press("U-" + m, b["n_ctx"], C, ui0, capu, st))
                r = CM.identity_check(capx, capu, st["scores"], "U-" + m)
                rows.append(dict(model_tag=tag, C=C, method=m, **r))
                ok &= r["identity_ok"]
                print(f"  identity {tag} C={C:3d} {m:14s} ok={r['identity_ok']} "
                      f"slots_exact={r['slots_exact']}/{r['slots']}", flush=True)
        del mdl
        torch.cuda.empty_cache()
    (HERE / "out" / "stage3_preflight_identity.json").write_text(
        json.dumps(dict(produced_on=CM.PRODUCED_ON, rows=rows), indent=1), encoding="utf-8")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=("seeds", "identity"))
    a = ap.parse_args()
    ok = check_seeds() if a.check == "seeds" else check_identity()
    print(f"PREFLIGHT {a.check}: {'PASS' if ok else 'FAIL -- STOP'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
