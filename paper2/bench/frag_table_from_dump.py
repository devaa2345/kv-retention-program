"""Rebuild the fragmentation table from the per-instance dump.

Same numbers the run prints, but recomputed from `frag_perinstance_*.jsonl` so the table does
not depend on catching stdout, and so it can be regenerated at any time.

  LINE  = every token of `R011 | Surname | Dept | 280294` retained   (old completeness unit)
  IDVAL = every token of the id AND of the 6-digit value retained    (new unit)
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

MODEL = sys.argv[1] if len(sys.argv) > 1 else "M2"
DUMP = Path("runs/nvidia/frag_perinstance_%s.jsonl" % MODEL)
GRID = Path("runs/nvidia/grid_%s_ledger_agnostic.jsonl" % MODEL)
ARMS = ["snapkv", "expected_attn", "keydiff", "adakv_snapkv"]

acc = defaultdict(list)
if GRID.exists():
    for line in GRID.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        acc[(r["C"], r["key"]["arm"])].append(r["score"])
ACC = {k: statistics.fmean(v) for k, v in acc.items()}

rows = defaultdict(list)
for line in DUMP.open(encoding="utf-8"):
    r = json.loads(line)
    rows[(r["C"], r["arm"])].append(r)

budgets = sorted({c for c, _ in rows})
print("%s  n=%d instances  (accuracy joined from %s where the grid has that budget)"
      % (MODEL, len({r['instance_id'] for v in rows.values() for r in v}), GRID.name))

for C in budgets:
    print("\n%s C=%d %s" % ("=" * 30, C, "=" * 30))
    print("%-15s %9s %8s | %10s %11s %8s | %10s %11s | %8s"
          % ("arm", "gold_tok", "acc", "qCPL_LINE", "qCPL_IDVAL", "delta",
             "rCPL_LINE", "rCPL_IDVAL", "touched"))
    means = {}
    for arm in ["floor_pos"] + ARMS:
        v = rows.get((C, arm))
        if not v:
            continue
        f = statistics.fmean
        m = dict(gt=f(r["gold_tok_line"] for r in v),
                 qL=f(r["qcpl_line"] for r in v),
                 qV=f(r["qcpl_idval"] for r in v),
                 rL=f(r["recs_complete_line"] for r in v),
                 rV=f(r["recs_complete_idval"] for r in v),
                 t=f(r["recs_touched"] for r in v),
                 acc=ACC.get((C, arm)))
        means[arm] = m
        a = "%.4f" % m["acc"] if m["acc"] is not None else "n/a"
        print("%-15s %9.2f %8s | %10.4f %11.4f %+8.4f | %10.2f %11.2f | %8.2f"
              % (arm, m["gt"], a, m["qL"], m["qV"], m["qV"] - m["qL"],
                 m["rL"], m["rV"], m["t"]))

    if "floor_pos" not in means:
        continue
    fp = means["floor_pos"]
    fa = "%.4f" % fp["acc"] if fp["acc"] is not None else "n/a"
    print("\n  MATCHED-GOLD-TOKEN COMPARISON  (floor_pos holds %.2f gold tokens, acc %s)"
          % (fp["gt"], fa))
    print("  %-15s %9s %8s %8s %10s %10s %11s  %s"
          % ("method", "gold_tok", "d_gold", "acc", "acc ratio", "qCPL_LINE",
             "qCPL_IDVAL", "IDVAL closes gap?"))
    for arm in ARMS:
        if arm not in means:
            continue
        m = means[arm]
        dg = m["gt"] - fp["gt"]
        ratio = (m["acc"] / fp["acc"]) if (m["acc"] and fp["acc"]) else float("nan")
        gapL = fp["qL"] - m["qL"]
        gapV = fp["qV"] - m["qV"]
        if abs(gapL) < 1e-9:
            verdict = "n/a (no gap)"
        elif gapV < 0.25 * gapL:
            verdict = "YES -- gap mostly closes"
        elif gapV < 0.75 * gapL:
            verdict = "partly"
        else:
            verdict = "NO -- gap survives"
        flag = "   <-- MATCHED" if abs(dg) <= 0.5 else ""
        a = "%.4f" % m["acc"] if m["acc"] is not None else "n/a"
        print("  %-15s %9.2f %+8.2f %8s %10.2f %10.4f %11.4f  %s%s"
              % (arm, m["gt"], dg, a, ratio, m["qL"], m["qV"], verdict, flag))
