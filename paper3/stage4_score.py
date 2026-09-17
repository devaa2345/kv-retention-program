"""Stage 4 — score the five forms against PREREG_P3.md, frozen at c920c404.

Procedure is section 5.4 of the prereg and is not re-decided here:

  Mode A (PRIMARY)  p_g measured per cell, fed to every form. Scores the form alone.
  Mode B            p_g from the registered log-linear interpolation. Scores form + p_g model.
  hit               |log10(pred) - log10(obs)| <= 0.30, both floored at 1/(200*4) = 0.00125
  winner            lowest mean |log10 error| on held-out cells, ties broken by hit rate,
                    reported whichever it is
  held out          every method cell except the single calibration cell (M2, c~19, C=64)
  secondary         residual correlation with log B and with c, reported always -- Stage 1
                    passed a hit-rate threshold while carrying r = +0.879 and that will not be
                    reported as success again

Two scoring decisions the prereg left underdetermined, both settled here and flagged:

  * F4's base cost. The prereg says "the smallest cost on the plane", but c = 1 is MARK-1, a
    different task family with a different unit. Extrapolating across families would not be the
    non-parametric baseline it is meant to be, so F4 uses c = 8 as its base WITHIN LEDGER-C and
    is scored only where it is defined (c = 19, 40).
  * Arms beyond snapkv and adakv_snapkv. The prereg tabulates those two. expected_attn and
    keydiff are calibrated on the SAME declared cell by the same rule and reported as an
    extension, separately from the registered primary.
"""
from __future__ import annotations

import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

from p3 import theory as T
from p3 import theory5 as T5

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "nvidia"
PRED = json.loads((HERE / "out" / "stage4_predictions.json").read_text(encoding="utf-8"))
FORMS = ("F1", "F2", "F3", "F4", "F5")
PRIMARY_ARMS = ("snapkv", "adakv_snapkv")
EXT_ARMS = ("expected_attn", "keydiff")
CAL = ("M2", 19, 64)
TOL, FLOOR = 0.30, 1.0 / 800.0
NL = chr(10)
C_TAG_OF = {}


def load():
    out = defaultdict(list)
    for tag in ("M2", "M3"):
        p = RUNS / ("stage4_capture_%s.jsonl" % tag)
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            if r["task"] == "multispan":
                continue                      # k axis scored separately
            out[(tag, r["arm"], round(r["c"], 2), r["C"])].append(r)
    cells = {}
    for k, rs in out.items():
        cells[k] = dict(c=k[2], C=k[3], arm=k[1], model=k[0], n=len(rs),
                        p_g=st.fmean(r["p_g"] for r in rs),
                        q=st.fmean(r["q_complete"] for r in rs),
                        q_any=st.fmean(r["q_any"] for r in rs))
    return cells


def rho_for(c, p, q):
    ce = math.log(q) / math.log(p)
    return dict(F2=1.0 - (ce - 1.0) / (c - 1.0),
                F3=1.0 - math.log(ce) / math.log(c) if ce > 0 else float("nan"),
                F5=T5.rho_from_cell(c, p, q))


def predict(form, c, p_g, rho, base=None):
    if form == "F1":
        return T.completion_pointwise(p_g, c, 0.0, "F1_naive")
    if form == "F2":
        return T.completion_pointwise(p_g, c, rho["F2"], "F2_linear")
    if form == "F3":
        return T.completion_pointwise(p_g, c, rho["F3"], "F3_power")
    if form == "F4":
        return T.nonparam_completion(base[0], c, base[1]) if base else float("nan")
    return T5.q_complete(c, rho["F5"], p_g)


def lg(x):
    return math.log10(max(x, FLOOR))


def pearson(x, y):
    pts = [(a, b) for a, b in zip(x, y) if not (math.isnan(a) or math.isnan(b))]
    if len(pts) < 3:
        return float("nan")
    xs, ys = zip(*pts)
    mx, my = st.fmean(xs), st.fmean(ys)
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy)


def score(cells, arms, label):
    cal_c = [c for (t, a, c, C) in cells if t == "M2" and C == 64
             and a == arms[0] and 18 < c < 20]
    ccal = cal_c[0]
    rho = {}
    for a in arms:
        d = cells[("M2", a, ccal, 64)]
        rho[a] = rho_for(d["c"], d["p_g"], d["q"])

    rows = []
    for (tag, arm, c, C), d in sorted(cells.items()):
        if arm not in arms:
            continue
        base = None
        cmin = min(cc for (t2, a2, cc, C2) in cells
                   if t2 == tag and a2 == arm and C2 == C and cc > 2)
        if c > cmin + 0.01:
            b = cells[(tag, arm, cmin, C)]
            base = (b["q"], b["c"])
        held = not (tag == "M2" and arm in arms and abs(c - ccal) < 0.01 and C == 64)
        preds = {f: predict(f, c, d["p_g"], rho[arm], base) for f in FORMS}
        rows.append(dict(tag=tag, arm=arm, c=c, C=C, p_g=d["p_g"], obs=d["q"],
                         preds=preds, held=held))

    print(NL + "=" * 100)
    print("MODE A (PRIMARY) -- %s   p_g measured, form scored alone" % label)
    print("=" * 100)
    for a in arms:
        print("  rho[%-14s] F2 %.4f  F3 %.4f  F5 %.4f"
              % (a, rho[a]["F2"], rho[a]["F3"], rho[a]["F5"]))
    held = [r for r in rows if r["held"]]
    print(NL + "  %d cells, %d held out" % (len(rows), len(held)))
    print(NL + "  %-12s %16s %9s %10s %12s %12s"
          % ("form", "mean|log10 err|", "median", "hit rate", "r vs logB", "r vs c"))
    summ = {}
    for f in FORMS:
        sub = [r for r in held if not math.isnan(r["preds"][f])]
        if not sub:
            print("  %-12s %16s" % (f, "not defined"))
            continue
        errs = [lg(r["preds"][f]) - lg(r["obs"]) for r in sub]
        hit = sum(1 for e in errs if abs(e) <= TOL) / len(errs)
        rb = pearson([math.log(r["C"] + 72) for r in sub], errs)
        rc = pearson([r["c"] for r in sub], errs)
        summ[f] = dict(mae=st.fmean(abs(e) for e in errs), hit=hit, n=len(sub),
                       r_logB=rb, r_c=rc)
        print("  %-12s %16.4f %9.4f %9.1f%% %12s %12s"
              % (f, summ[f]["mae"], st.median(abs(e) for e in errs), 100 * hit,
                 "%+.4f" % rb, "%+.4f" % rc))
    win = min(summ, key=lambda f: (summ[f]["mae"], -summ[f]["hit"]))
    print(NL + "  WINNER: %s  (mean |log10 err| %.4f, hit %.1f%%, n=%d)"
          % (win, summ[win]["mae"], 100 * summ[win]["hit"], summ[win]["n"]))

    # c = 1 is a BOUNDARY CONDITION every parametric form satisfies by construction
    # (q = p_g identically), so those cells are free marks that inflate every hit rate.
    # F1's entire hit rate is exactly these cells. Scored again with them removed.
    nz = [r for r in held if r["c"] > 2]
    print(NL + "  EXCLUDING the c = 1 boundary cells (%d of %d held out) -- these are free"
          % (len(held) - len(nz), len(held)))
    print("  marks: every form predicts q = p_g there by construction, and F1's whole hit")
    print("  rate is exactly those cells.")
    print(NL + "  %-12s %16s %10s %12s %12s"
          % ("form", "mean|log10 err|", "hit rate", "r vs logB", "r vs c"))
    summ_nz = {}
    for f in FORMS:
        sub = [r for r in nz if not math.isnan(r["preds"][f])]
        if not sub:
            continue
        errs = [lg(r["preds"][f]) - lg(r["obs"]) for r in sub]
        summ_nz[f] = dict(mae=st.fmean(abs(e) for e in errs),
                          hit=sum(1 for e in errs if abs(e) <= TOL) / len(errs),
                          n=len(sub),
                          r_logB=pearson([math.log(r["C"] + 72) for r in sub], errs),
                          r_c=pearson([r["c"] for r in sub], errs))
        print("  %-12s %16.4f %9.1f%% %12s %12s"
              % (f, summ_nz[f]["mae"], 100 * summ_nz[f]["hit"],
                 "%+.4f" % summ_nz[f]["r_logB"], "%+.4f" % summ_nz[f]["r_c"]))
    win_nz = min(summ_nz, key=lambda f: (summ_nz[f]["mae"], -summ_nz[f]["hit"]))
    print(NL + "  WINNER excluding c=1: %s  (mean |log10 err| %.4f, hit %.1f%%, n=%d)"
          % (win_nz, summ_nz[win_nz]["mae"], 100 * summ_nz[win_nz]["hit"],
             summ_nz[win_nz]["n"]))

    print(NL + "  REGISTERED KILL CRITERION (prereg section 10): 'if F5's residual budget")
    print("  trend is as large as F2's was (|r| with log B above 0.5), the mechanism is not")
    print("  the operative one and section 5.3's regularity is the honest stopping point.'")
    r5 = summ_nz.get("F5", summ.get("F5", {})).get("r_logB", float("nan"))
    fired = abs(r5) > 0.5
    print("    F5 residual r vs log B = %+.4f   -> criterion %s"
          % (r5, "FIRES" if fired else "does not fire"))
    return rows, summ, win, rho, summ_nz, win_nz, fired


def main() -> int:
    cells = load()
    print("loaded %d (model, arm, c, C) capture cells" % len(cells))
    rows, summ, win, rho, summ_nz, win_nz, fired = score(
        cells, PRIMARY_ARMS, "registered arms (snapkv, adakv_snapkv)")

    print(NL + "  per-cell detail, held-out only:")
    print("  %-4s %-14s %6s %5s %8s %9s %10s %10s %10s %10s"
          % ("mdl", "arm", "c", "C", "p_g", "obs q", "F2", "F3", "F5", "F4"))
    for r in sorted(rows, key=lambda r: (r["tag"], r["arm"], r["c"], r["C"])):
        if not r["held"]:
            continue
        p = r["preds"]
        print("  %-4s %-14s %6.2f %5d %8.4f %9.5f %10.5f %10.5f %10.5f %10s"
              % (r["tag"], r["arm"], r["c"], r["C"], r["p_g"], r["obs"],
                 p["F2"], p["F3"], p["F5"],
                 "-" if math.isnan(p["F4"]) else "%.5f" % p["F4"]))

    score(cells, EXT_ARMS, "EXTENSION (expected_attn, keydiff) -- not the registered primary")

    json.dump(dict(all_cells=summ, excl_c1=summ_nz, winner=win,
                   winner_excl_c1=win_nz, kill_criterion_fires=fired),
              open(HERE / "out" / "stage4_score.json", "w", encoding="utf-8"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
