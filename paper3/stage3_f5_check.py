"""F5 against Stage 2 — calibrate rho on ONE cell, predict the other eleven.

The mechanism and every boundary condition were fixed in `p3/theory5.py` before this ran.
F5 gets exactly one free parameter, `rho`, like F2 and F3, and it is fixed on a single
declared cell: **M2, c ~ 19, C = 64**. Every other (model, c, C) cell is out of sample,
including all of M3 -- a cross-model extrapolation, as in Stage 1.

Scored on `c_eff` directly rather than on completion, because `c_eff` is where Stage 2
refuted F2 and F3 and is the quantity F5 was derived to explain.
"""
from __future__ import annotations

import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

from p3 import theory5 as T5

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "nvidia"
CALIB = json.loads((HERE / "out" / "c_calibration.json").read_text(encoding="utf-8"))
C_TARGETS = (8, 19, 40)
ARMS = ("snapkv", "adakv_snapkv")
CAL_MODEL, CAL_C_TAG, CAL_BUDGET = "M2", 19, 64
NL = chr(10)


def cells():
    """(model, arm, c_tag) -> dict(c, p_g, q, c_eff_measured) per budget."""
    out = {}
    for tag in ("M2", "M3"):
        agg = defaultdict(list)
        p = RUNS / ("stage2_capture_%s.jsonl" % tag)
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            if r["task"] != "ledger_c":
                continue
            agg[(r["arm"], r["c_tag"], r["C"])].append(r)
        for (arm, ct, C), rs in agg.items():
            pg = st.fmean(r["p_g"] for r in rs)
            q = st.fmean(r["q_complete"] for r in rs)
            ce = math.log(q) / math.log(pg) if 0 < q < 1 and 0 < pg < 1 else float("nan")
            out[(tag, arm, ct, C)] = dict(
                c=CALIB[tag]["chosen"][str(ct)]["c"], p_g=pg, q=q, c_eff=ce, n=len(rs))
    return out


def main() -> int:
    tab = cells()
    print("=" * 98)
    print("F5 vs STAGE 2 -- one calibration cell (%s, c~%d, C=%d), everything else held out"
          % (CAL_MODEL, CAL_C_TAG, CAL_BUDGET))
    print("=" * 98)

    rho = {}
    for arm in ARMS:
        d = tab[(CAL_MODEL, arm, CAL_C_TAG, CAL_BUDGET)]
        rho[arm] = T5.rho_from_cell(d["c"], d["p_g"], d["q"])
        print("  calibrated rho[%-13s] = %.4f   (from c=%.2f, p_g=%.4f, q=%.4f)"
              % (arm, rho[arm], d["c"], d["p_g"], d["q"]))

    print(NL + "%-6s %-14s %5s %7s %8s %9s %9s %9s"
          % ("model", "arm", "C", "c", "p_g", "c_eff obs", "F5 pred", "err"))
    errs, errs_held = [], []
    rows = []
    for tag in ("M2", "M3"):
        for arm in ARMS:
            for C in (64, 512):
                for ct in C_TARGETS:
                    d = tab.get((tag, arm, ct, C))
                    if not d:
                        continue
                    pred = T5.c_eff(d["c"], rho[arm], d["p_g"])
                    e = pred - d["c_eff"]
                    held = not (tag == CAL_MODEL and ct == CAL_C_TAG and C == CAL_BUDGET)
                    errs.append(abs(e))
                    if held:
                        errs_held.append(abs(e))
                    rows.append((tag, arm, C, d["c"], d["p_g"], d["c_eff"], pred, e, held))
                    print("%-6s %-14s %5d %7.2f %8.4f %9.4f %9.4f %+9.4f%s"
                          % (tag, arm, C, d["c"], d["p_g"], d["c_eff"], pred, e,
                             "" if held else "   <- CAL"))

    print(NL + "  mean |error| over the %d held-out cells: %.4f   (all %d cells: %.4f)"
          % (len(errs_held), st.fmean(errs_held), len(errs), st.fmean(errs)))

    print(NL + "  DOES F5 REPRODUCE THE TWO FACTS THAT REFUTED F2 AND F3?")
    for tag in ("M2", "M3"):
        for arm in ARMS:
            obs_b, pred_b = [], []
            for ct in C_TARGETS:
                a = tab.get((tag, arm, ct, 512))
                b = tab.get((tag, arm, ct, 64))
                if not a or not b:
                    continue
                obs_b.append(a["c_eff"] - b["c_eff"])
                pred_b.append(T5.c_eff(a["c"], rho[arm], a["p_g"])
                              - T5.c_eff(b["c"], rho[arm], b["p_g"]))
            # cost slope at each budget
            slopes = {}
            for C in (64, 512):
                pts = [(tab[(tag, arm, ct, C)]["c"], tab[(tag, arm, ct, C)]["c_eff"],
                        T5.c_eff(tab[(tag, arm, ct, C)]["c"], rho[arm],
                                 tab[(tag, arm, ct, C)]["p_g"]))
                       for ct in C_TARGETS if (tag, arm, ct, C) in tab]
                if len(pts) >= 2:
                    xs = [p[0] for p in pts]
                    mx = st.fmean(xs)
                    sxx = sum((x - mx) ** 2 for x in xs)
                    for j, lab in ((1, "obs"), (2, "pred")):
                        ys = [p[j] for p in pts]
                        my = st.fmean(ys)
                        slopes[(C, lab)] = sum((x - mx) * (y - my)
                                               for x, y in zip(xs, ys)) / sxx
            print("    %-3s %-14s budget shift obs %s  pred %s"
                  % (tag, arm, ["%+.3f" % v for v in obs_b],
                     ["%+.3f" % v for v in pred_b]))
            print("        %-14s cost slope C=64  obs %+.4f pred %+.4f | C=512 obs %+.4f "
                  "pred %+.4f" % ("", slopes.get((64, "obs"), float("nan")),
                                  slopes.get((64, "pred"), float("nan")),
                                  slopes.get((512, "obs"), float("nan")),
                                  slopes.get((512, "pred"), float("nan"))))

    print(NL + "  floor_pos control -- the model REQUIRES c_eff = 1 for a contiguous policy:")
    for tag in ("M2", "M3"):
        v = [tab[(tag, "floor_pos", ct, C)]["c_eff"]
             for C in (64, 512) for ct in C_TARGETS if (tag, "floor_pos", ct, C) in tab]
        print("    %s  measured c_eff %s" % (tag, ["%.3f" % x for x in v]))

    json.dump(dict(rho=rho, mae_held=st.fmean(errs_held), mae_all=st.fmean(errs),
                   rows=[list(r) for r in rows]),
              open(HERE / "out" / "f5_check.json", "w", encoding="utf-8"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
