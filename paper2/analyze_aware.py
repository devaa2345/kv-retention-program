"""N8 analysis — protocol deltas, and whether the fragmentation deficit survives query-awareness.

Two questions:

  1. **Does the harness reproduce the field's own result?** PREREG §5.2 records the reference
     picture: SnapKV's aware-arm gain ≈ +0.20, KeyDiff's ≈ +0.01, ordering
     SnapKV >> AdaKV > TOVA > ExpectedAttention > KeyDiff. If our deltas reproduce that ordering,
     the harness measures these methods the way the literature does, and the agnostic finding
     stands on a harness that is known to work. If they do not, that is a harness problem and is
     reported as one rather than explained away.

  2. **Does the fragmentation deficit persist when the method can see the query?** If methods
     beat `floor_pos` when aware and lose when agnostic, the result sharpens considerably:
     pointwise scoring works when it knows what to look for and fragments when it does not.

Contrasts are paired per instance across protocols — the same instance ids are used in both
grids, so `aware − agnostic` is a within-instance difference.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from harness import stats

LADDER = ("full_cache", "null", "random", "floor_pos", "oracle_causal", "oracle_prescient")
REFERENCE = {"snapkv": 0.20, "keydiff": 0.01}
REF_ORDER = ["snapkv", "adakv_snapkv", "tova", "expected_attn", "keydiff"]


def load(path):
    by = defaultdict(dict)
    if not Path(path).exists():
        return by
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            by[(r["C"], r["key"]["arm"])][r["key"]["instance_id"]] = r["score"]
    return by


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agnostic", required=True)
    ap.add_argument("--aware", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ag, aw = load(args.agnostic), load(args.aware)
    budgets = sorted({c for c, _ in aw})
    arms = sorted({a for _, a in aw})
    methods = [a for a in arms if a not in LADDER]
    rep = {"label": args.label, "budgets": budgets, "methods": methods, "cells": {}}

    print(f"\n=== {args.label} ===")
    for C in budgets:
        ids = sorted(set(aw.get((C, "floor_pos"), {})) & set(ag.get((C, "floor_pos"), {})))
        if not ids:
            continue
        fl_aw = [aw[(C, "floor_pos")][i] for i in ids]
        cell = {"C": C, "n": len(ids), "floor_aware": round(sum(fl_aw) / len(ids), 4),
                "methods": {}}
        print(f"\n  C={C}  n={len(ids)}   floor_pos aware = {cell['floor_aware']:.4f}"
              f"   (agnostic {sum(ag[(C,'floor_pos')][i] for i in ids)/len(ids):.4f})")
        print(f"    {'method':16s} {'agnostic':>9s} {'aware':>8s} {'delta':>8s} {'95% CI':>20s} "
              f"{'vs floor(aware)':>16s}")
        for m in methods:
            if not all(i in aw.get((C, m), {}) for i in ids):
                continue
            a_ag = [ag[(C, m)][i] for i in ids] if all(i in ag.get((C, m), {}) for i in ids) else None
            a_aw = [aw[(C, m)][i] for i in ids]
            d = stats.paired_contrast(a_aw, a_ag, seed=C, n_boot=20000) if a_ag else None
            vf = stats.paired_contrast(a_aw, fl_aw, seed=C + 1, n_boot=20000)
            cell["methods"][m] = {
                "agnostic": round(sum(a_ag) / len(ids), 4) if a_ag else None,
                "aware": round(sum(a_aw) / len(ids), 4),
                "protocol_delta": {"mean": round(d.mean_diff, 4),
                                   "ci": [round(d.ci_low, 4), round(d.ci_high, 4)]} if d else None,
                "vs_floor_aware": {"mean": round(vf.mean_diff, 4),
                                   "ci": [round(vf.ci_low, 4), round(vf.ci_high, 4)],
                                   "beats_floor": bool(vf.ci_low > 0)},
            }
            c = cell["methods"][m]
            print(f"    {m:16s} {str(c['agnostic']):>9s} {c['aware']:8.4f} "
                  f"{c['protocol_delta']['mean']:+8.4f} "
                  f"{str(c['protocol_delta']['ci']):>20s} "
                  f"{c['vs_floor_aware']['mean']:+9.4f} "
                  f"{'BEATS' if c['vs_floor_aware']['beats_floor'] else 'loses':>6s}")
        rep["cells"][f"C{C}"] = cell

    # reference check on the largest budget available
    if budgets:
        C = budgets[-1]
        got = {m: rep["cells"][f"C{C}"]["methods"][m]["protocol_delta"]["mean"]
               for m in rep["cells"][f"C{C}"]["methods"]
               if rep["cells"][f"C{C}"]["methods"][m]["protocol_delta"]}
        order = [m for m in REF_ORDER if m in got]
        obs = sorted(got, key=lambda m: -got[m])
        rep["reference_check"] = {
            "C": C, "observed_deltas": got,
            "expected_order": order, "observed_order": obs,
            "order_matches": obs == order,
            "snapkv_delta": got.get("snapkv"), "snapkv_reference": REFERENCE["snapkv"],
            "keydiff_delta": got.get("keydiff"), "keydiff_reference": REFERENCE["keydiff"],
        }
        print(f"\n  REFERENCE CHECK @C={C}")
        print(f"    expected order: {order}")
        print(f"    observed order: {obs}")
        print(f"    ordering reproduces the literature: {obs == order}")
        print(f"    snapkv delta {got.get('snapkv')} (ref ~+0.20); "
              f"keydiff delta {got.get('keydiff')} (ref ~+0.01)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
