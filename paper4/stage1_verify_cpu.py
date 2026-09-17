"""Stage 1 verification, NO GPU. Real instances and real token geometry; synthetic scores.

What Paper 3's captures CAN and CANNOT support.  `runs/nvidia/stage4_capture_*.jsonl` stores per
(instance, arm, cell) AGGREGATES -- p_g, q_complete, units_complete, units_touched -- not the
per-(layer, head) keep-sets and not the per-token scores. Unit-aware re-allocation needs the
scores, so it cannot be recomputed from the stored captures without a model forward pass.

So this script does the two things that are possible on CPU:

  1. GEOMETRY. Rebuilds Stage 4's instances (same seeds, same ids, own tokenizer) and checks that
     n_ctx, the gold-token count and the unit count match every stored capture row exactly, so
     the units the wrapper allocates over are Paper 3's units.
  2. DIRECTION ON REAL GEOMETRY. Over several score families -- iid, heavy-tailed, F5's
     within-unit correlation, a recency trend, and an adversarial family that prefers filler to
     records -- X (top-C) vs U-X at matched budget: distinct units touched, units complete.

The REAL-score check (does it hold for SnapKV's actual scores?) needs a prefill and is run as the
first, blocking package of Stage 2 (`stage2_run.py verify`), before any generation.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
from p4 import common as CM                              # noqa: E402
from p4 import unitwrap as UW                            # noqa: E402

P3RUNS = HERE.parent / "paper3" / "runs" / "nvidia"
C = 512
N = 50


def stored_caps(tag):
    out = {}
    with (P3RUNS / f"stage4_capture_{tag}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["C"] != C or r["label"] not in ("c=1", "c=40") or r["arm"] != "floor_pos":
                continue
            i = int(r["key"]["instance_id"][3:])
            if i < N:
                out[(r["label"], i)] = r
    return out


def families(rng, ui, n, h):
    unit_of = -np.ones(n, dtype=np.int64)
    for j, u in enumerate(ui.units):
        unit_of[u] = j
    in_unit = unit_of >= 0
    pos = np.arange(n) / n
    fam = {}
    fam["iid_normal"] = rng.standard_normal((h, n))
    fam["heavy_t2"] = rng.standard_t(2, (h, n))
    for rho in (0.3, 0.7):
        mu = rng.standard_normal((h, len(ui.units))) * np.sqrt(rho)
        e = rng.standard_normal((h, n)) * np.sqrt(1 - rho)
        s = e.copy()
        s[:, in_unit] += mu[:, unit_of[in_unit]]
        fam[f"f5_rho{rho}"] = s
    fam["recency"] = 2.0 * pos[None, :] + rng.standard_normal((h, n))
    adv = rng.standard_normal((h, n))
    adv[:, in_unit] -= 1.0
    fam["adversarial_filler"] = adv
    return fam


def main():
    report = ["Paper 4 Stage 1 CPU verification -- produced on %s, no GPU." % CM.PRODUCED_ON, ""]
    ok_all = True
    summary = {}
    for model_name, tag in CM.TAGS.items():
        tok = AutoTokenizer.from_pretrained(model_name)
        caps = stored_caps(tag)
        rev = next(iter(caps.values()))["key"]["model_revision"]
        geo_bad = 0
        agg = defaultdict(lambda: defaultdict(list))
        for c_tag, label in ((40, "c=40"), (1, "c=1")):
            for i in range(N):
                b = CM.build(tag, model_name, rev, tok, c_tag, "s4_%05d" % i)
                r = caps[(label, i)]
                gold = len({t for f in b["facts"] for s in f.spans for t in s})
                if (b["n_ctx"], gold, len(b["line_units"])) != (r["n_ctx"], r["gold_tokens"], r["n_units"]):
                    geo_bad += 1
                ui = b["ui"]
                rng = np.random.default_rng(1000 + i)
                region = np.flatnonzero(~ui.floor)
                queried = [set(t for s in f.spans for t in s) for f in b["facts"]]
                lines = [s for s in b["line_units"].values() if s]
                for fam, S in families(rng, ui, b["n_ctx"], 8).items():
                    keepsU, stU = UW.unit_keep_per_head(S, ui, C)
                    for h in range(S.shape[0]):
                        kx = set(region[np.lexsort((region, -S[h, region]))[:C]].tolist()) | set(ui.floor_idx.tolist())
                        ku = set(keepsU[h].tolist())
                        for arm, kk in (("X", kx), ("U", ku)):
                            agg[(label, fam)][arm + "_touched"].append(sum(1 for s in lines if s & kk))
                            agg[(label, fam)][arm + "_complete"].append(sum(1 for s in lines if s <= kk))
                            agg[(label, fam)][arm + "_qc"].append(sum(1 for q in queried if q <= kk) / len(queried))
                    agg[(label, fam)]["fallback"].append(stU["fallback"])
        report.append(f"== {tag} ({model_name})")
        report.append(f"  geometry vs stored Stage 4 captures (n_ctx, gold tokens, units): "
                      f"{2 * N - geo_bad}/{2 * N} instances match")
        ok_all &= geo_bad == 0
        report.append("  %-6s %-20s %9s %9s %9s %9s %8s %8s %5s %s" % (
            "cell", "score family", "X touch", "U touch", "X compl", "U compl", "X q_c", "U q_c", "fb", "direction"))
        for (label, fam), a in sorted(agg.items()):
            m = {k: st.fmean(v) for k, v in a.items()}
            dir_ok = m["U_touched"] < m["X_touched"] and m["U_complete"] > m["X_complete"]
            if label == "c=40":
                ok_all &= dir_ok
            summary[f"{tag}|{label}|{fam}"] = m
            report.append("  %-6s %-20s %9.2f %9.2f %9.2f %9.2f %8.3f %8.3f %5.0f %s" % (
                label, fam, m["X_touched"], m["U_touched"], m["X_complete"], m["U_complete"],
                m["X_qc"], m["U_qc"], sum(a["fallback"]),
                ("as predicted" if dir_ok else "NOT as predicted") if label == "c=40" else "(no prediction)"))
        report.append("")
    report.append("Stage 1 CPU verification: %s" % ("PASS" if ok_all else "FAIL"))
    report.append("Scope: synthetic scores on real geometry. The real-score check is Stage 2's "
                  "blocking `verify` package.")
    (HERE / "out" / "stage1_verify_cpu.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    (HERE / "out" / "stage1_verify_cpu.json").write_text(json.dumps(dict(pass_=ok_all, cells=summary), indent=1), encoding="utf-8")
    print("\n".join(report))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
