"""Diagnosis of verify condition (a) failing on M3 only. Prefill only, no generation.

Fresh process, re-captures the X arms on instances 0-4 at c in {40, 1} and compares keep-set
aggregates with (i) this session's verify rows and (ii) Paper 3 Stage 4's stored (pre-power-loss)
captures. Run for M3 and, as a control, M2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import methods
from p4 import common as CM

HERE = Path(__file__).resolve().parent
F = ("p_g", "q_complete", "q_any", "units_complete", "units_touched")
C = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    tag = CM.TAGS[a.model]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to("cuda").eval()
    rev = model.config._commit_hash
    mine = {(r["label"], r["arm"], r["instance"]): r for r in map(
        json.loads, open(HERE / "runs" / "nvidia" / f"p4_verify_{tag}.jsonl", encoding="utf-8"))}
    stored = {}
    for line in open(HERE.parent / "paper3" / "runs" / "nvidia" / f"stage4_capture_{tag}.jsonl",
                     encoding="utf-8"):
        r = json.loads(line)
        if r["C"] == C and r["label"] in ("c=1", "c=40"):
            i = int(r["key"]["instance_id"][3:])
            if i < a.n:
                stored[(r["label"], r["arm"], i)] = r
    rows = []
    for c_tag in (40, 1):
        for i in range(a.n):
            b = CM.build(tag, a.model, rev, tok, c_tag, "s4_%05d" % i)
            for arm in CM.X_METHODS:
                reps = []
                for _ in range(2):
                    cap = methods.Capture()
                    CM.prefill(model, tok, b["pre"], CM.build_press(arm, b["n_ctx"], C, b["ui"], cap, {}))
                    reps.append((cap, CM.keep_metrics([cap.per_head[h] for h in cap.heads()],
                                                      b["facts"], b["line_units"], b["n_ctx"])))
                k = ("c=%d" % c_tag, arm, i)
                (c1, m1), (c2, m2) = reps
                rows.append(dict(
                    model_tag=tag, label=k[0], arm=arm, instance=i,
                    rerun_keepsets_identical=c1.per_head == c2.per_head,
                    fresh_vs_session=max(abs(m1[f] - mine[k][f]) for f in F),
                    fresh_vs_stored=max(abs(m1[f] - stored[k][f]) for f in F)))
                print(json.dumps(rows[-1]), flush=True)
    out = HERE / "out" / f"stage2_det_check_{tag}.json"
    out.write_text(json.dumps(dict(produced_on=CM.PRODUCED_ON, rows=rows), indent=1), encoding="utf-8")
    s = lambda key: sum(1 for r in rows if (r[key] <= 1e-9 if key != "rerun_keepsets_identical" else r[key]))
    print("SUMMARY %s: rerun identical %d/%d; fresh==session %d/%d; fresh==stored %d/%d" % (
        tag, s("rerun_keepsets_identical"), len(rows), s("fresh_vs_session"), len(rows),
        s("fresh_vs_stored"), len(rows)))


if __name__ == "__main__":
    main()
