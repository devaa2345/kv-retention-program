"""Grid analysis — G_m and I, per the frozen analysis contract. NO GPU.

The contract is `harness/stats.py`, frozen at Stage 2 and run rather than rewritten:

    G_m = (A_m     − A_floor)  / (A_causal  − A_floor)      capture ratio
    I   = (A_presc − A_causal) / (A_presc   − A_floor)      information share

  * every contrast is PAIRED PER INSTANCE;
  * CIs propagate the bootstrap THROUGH the ratio — numerator and denominator recomputed on
    the same resample, never two independently bootstrapped means;
  * refusal-to-normalise: if `A_causal − A_floor < 0.15` the cell is raw-only and `G_m` is
    reported as `n/a — degenerate headroom`;
  * `G_m` is reported UNCLIPPED. `G_m > 1` is diagnostic, not an error to hide: it means the
    method beat the ceiling arm, and it triggers the ceiling-validity audit (§4.1).

`I` is reported under its v2 restatement: the accuracy cost of not knowing the query, measured
between two otherwise-identical retention policies. It is a measured gap, not a proof of
irreducibility.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from harness import stats

LADDER = ("full_cache", "null", "random", "floor_pos", "oracle_causal", "oracle_prescient")


def load(path: Path):
    """rows -> {(C, arm): {instance_id: score}}"""
    by = defaultdict(dict)
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            k = r["key"]
            by[(r["C"], k["arm"])][k["instance_id"]] = r["score"]
    return by


def aligned(by, C, arms):
    """Instances present for EVERY arm at this budget — paired analysis needs alignment."""
    sets = [set(by.get((C, a), {})) for a in arms]
    if not sets or any(not s for s in sets):
        return []
    return sorted(set.intersection(*sets))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="gates/nvidia/grid_analysis.json")
    args = ap.parse_args()

    report = {"files": {}, "gm_gt_1": [], "refused_cells": []}

    for path in args.runs:
        p = Path(path)
        if not p.exists():
            continue
        by = load(p)
        budgets = sorted({c for c, _ in by})
        arms = sorted({a for _, a in by})
        methods = [a for a in arms if a not in LADDER]
        out = {"budgets": budgets, "arms": arms, "methods": methods, "cells": {}}
        print(f"\n=== {p.name} ===")
        print(f"  budgets {budgets}")
        print(f"  methods {methods}")

        for C in budgets:
            need = ["floor_pos", "oracle_causal", "oracle_prescient"]
            ids = aligned(by, C, need)
            if not ids:
                continue
            f = [by[(C, "floor_pos")][i] for i in ids]
            oc = [by[(C, "oracle_causal")][i] for i in ids]
            op = [by[(C, "oracle_prescient")][i] for i in ids]

            head = round(sum(oc) / len(oc) - sum(f) / len(f), 4)
            I = stats.information_share(op, oc, f, seed=C, n_boot=20000)
            cell = {"C": C, "n_aligned": len(ids),
                    "means": {a: round(sum(by[(C, a)][i] for i in ids) / len(ids), 4)
                              for a in arms if all(i in by.get((C, a), {}) for i in ids)},
                    "headroom_causal_minus_floor": head,
                    "I": {"value": I.value, "ci": [I.ci_low, I.ci_high],
                          "refused": I.refused, "reason": I.reason},
                    "G_m": {}}
            if head < stats.REFUSAL_THRESHOLD:
                report["refused_cells"].append(f"{p.stem}/C{C}")

            for m in methods:
                if not all(i in by.get((C, m), {}) for i in ids):
                    continue
                am = [by[(C, m)][i] for i in ids]
                g = stats.capture_ratio(am, f, oc, name=f"{m}@C{C}", seed=C, n_boot=20000)
                cell["G_m"][m] = {"value": g.value, "ci": [g.ci_low, g.ci_high],
                                  "refused": g.refused, "reason": g.reason,
                                  "exceeds_one": g.exceeds_one}
                if g.exceeds_one:
                    report["gm_gt_1"].append(
                        {"file": p.stem, "C": C, "method": m, "G_m": g.value,
                         "ci": [g.ci_low, g.ci_high],
                         "note": "reported unclipped; triggers the ceiling-validity audit"})
            out["cells"][f"C{C}"] = cell

            gm = "  ".join(f"{m}={cell['G_m'][m]['value']:.3f}"
                           if cell["G_m"].get(m) and cell["G_m"][m]["value"] is not None
                           else f"{m}=n/a" for m in methods if m in cell["G_m"])
            iv = "n/a" if I.refused else f"{I.value:+.3f}"
            print(f"  C={C:4d} n={len(ids):3d} headroom={head:+.3f} I={iv}  {gm}")

        report["files"][p.stem] = out

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n  G_m > 1 cells: {len(report['gm_gt_1'])}")
    for r in report["gm_gt_1"]:
        print(f"    {r['file']} C={r['C']} {r['method']}: G_m={r['G_m']:.4f} {r['ci']}")
    print(f"  refused (degenerate headroom): {report['refused_cells']}")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
