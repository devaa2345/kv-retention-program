"""Items 1 and 2, pure re-analysis of the slot-aware capture.

Item 1: the matched-gold-token contrast under MEAN (current) vs ANY (>=1 slot) completeness,
        both models, all budgets, side by side.
Item 2: the M3 C=512 fragmentation-accuracy anti-correlation recomputed slot-aware.
"""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ARMS = ["snapkv", "expected_attn", "keydiff", "adakv_snapkv"]


def pearson(x, y):
    if len(x) < 3:
        return float("nan")
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def grid_acc(tag):
    acc = defaultdict(list)
    p = Path("runs/nvidia/grid_%s_ledger_agnostic.jsonl" % tag)
    for line in p.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        acc[(r["C"], r["key"]["arm"])].append(r["score"])
    return {k: statistics.fmean(v) for k, v in acc.items()}


for tag in ("M2", "M3"):
    dump = Path("runs/nvidia/slotaware_%s.jsonl" % tag)
    if not dump.exists():
        continue
    ACC = grid_acc(tag)
    cells = defaultdict(list)
    for line in dump.open(encoding="utf-8"):
        r = json.loads(line)
        cells[(r["C"], r["arm"])].append(r)

    n_slots = cells[(512, "snapkv")][0]["n_slots"] if (512, "snapkv") in cells else "?"
    print("\n" + "=" * 104)
    print("### %s   (%s slots per method capture; floor_pos has 1)" % (tag, n_slots))
    print("=" * 104)

    for C in sorted({c for c, _ in cells}):
        m = {}
        for arm in ["floor_pos"] + ARMS:
            v = cells.get((C, arm))
            if not v:
                continue
            f = statistics.fmean
            m[arm] = dict(gt=f(r["gold_tok_mean"] for r in v),
                          mean=f(r["q_mean_line"] for r in v),
                          any=f(r["q_any_line"] for r in v),
                          maj=f(r["q_maj_line"] for r in v),
                          acc=ACC.get((C, arm)))
        if "floor_pos" not in m:
            continue
        fp = m["floor_pos"]
        if fp["acc"] is None:
            continue
        print("\n  C=%d   floor_pos: gold_tok %.2f  acc %.4f  completeness MEAN %.4f / ANY %.4f"
              % (C, fp["gt"], fp["acc"], fp["mean"], fp["any"]))
        print("    %-15s %8s %7s %7s | %8s %8s %8s | %s"
              % ("method", "gold_tok", "d_gold", "acc", "MEAN", "ANY", "floorANY", "verdict"))
        for arm in ARMS:
            if arm not in m or m[arm]["acc"] is None:
                continue
            a = m[arm]
            dg = a["gt"] - fp["gt"]
            gap_mean = fp["mean"] - a["mean"]
            gap_any = fp["any"] - a["any"]
            if gap_mean <= 1e-9:
                verd = "n/a"
            elif gap_any <= 0:
                verd = "REVERSES under ANY"
            elif gap_any < 0.25 * gap_mean:
                verd = "gap mostly closes"
            elif gap_any < 0.75 * gap_mean:
                verd = "gap shrinks"
            else:
                verd = "gap survives"
            flag = "  <-- MATCHED" if abs(dg) <= 0.5 else ""
            print("    %-15s %8.2f %+7.2f %7.4f | %8.4f %8.4f %8.4f | %s%s"
                  % (arm, a["gt"], dg, a["acc"], a["mean"], a["any"], fp["any"], verd, flag))

    # ---- item 2: fragmentation-accuracy across arms, MEAN vs ANY -------------
    print("\n  fragmentation-accuracy across arms (5 arm means per cell)")
    print("    %-6s %14s %14s" % ("C", "r using MEAN", "r using ANY"))
    for C in sorted({c for c, _ in cells}):
        xs_m, xs_a, ys = [], [], []
        for arm in ["floor_pos"] + ARMS:
            v = cells.get((C, arm))
            if not v or ACC.get((C, arm)) is None:
                continue
            f = statistics.fmean
            t = f(r["recs_touched_mean"] for r in v)
            ta = f(r["recs_touched_any"] for r in v)
            xs_m.append(f(r["recs_mean_line"] for r in v) / max(1e-9, t))
            xs_a.append(f(r["recs_any_line"] for r in v) / max(1e-9, ta))
            ys.append(ACC[(C, arm)])
        if len(ys) >= 3:
            print("    %-6d %14.4f %14.4f" % (C, pearson(xs_m, ys), pearson(xs_a, ys)))
