"""Distractor load — does what ELSE was kept explain the completion -> accuracy step?

Exploratory, not registered in PREREG_P3. The hypothesis and the decision rule below were
written and committed BEFORE this ran on the complete Stage 4 data.

BACKGROUND. At Stage 4 the crossover test agrees on 20/23 admissible cells under F5 and 19/23
under the MEASURED per-slot completion, so no completion-based predictor can fix the misses: in
all three, the method's measured completion is below the floor's while its accuracy is higher.
The floor's failures are 68-87% "no gold words" -- the model copies a DIFFERENT record rather
than declining. So accuracy per unit of completion (conversion) differs across arms at equal
completion, and nothing in the retention framework models that step.

HYPOTHESIS (H-D). Conversion is governed by retrieval interference from the other records that
were retained WHOLE. At fixed completion of the queried record, more complete non-queried
records -> lower accuracy.

    D  = complete NON-QUERIED records retained
       = units_complete - q_complete * n_facts          (per instance, mean over KV slots)

DESIGN -- chosen to avoid the obvious confound. D rises with budget, and so does nearly
everything else (RoPE gaps shrink, exemplars and preamble survive), so a cross-cell correlation
of conversion with D would mostly measure budget. The primary test therefore uses only
WITHIN-CELL variation: inside one (model, task, c, C, arm) cell, instances differ in how many
non-queried records happen to be retained whole, while budget, cost and policy are fixed.

  PRIMARY    Per-instance accuracy regressed on per-instance q_complete and D, both centred
             within their cell (cell fixed effects), pooled per (model, arm class). The
             coefficient on D is the interference effect at fixed completion. 95% CI by
             bootstrap over cells (cells, not instances, are the independent units of design).
  SECONDARY  Cell-level: conversion = acc / q_complete against mean D, partialling out log B
             and c. Reported because it is the question as first posed; confounded, and
             labelled as such.
  DESCRIPTIVE Mean D per arm per cell, so the claim "floor and methods retain different
             distractor loads at the same budget" is checked directly rather than assumed.

DECISION RULE (fixed before running on the final data):
  H-D SUPPORTED   primary coefficient on D is negative with the 95% CI excluding zero, on
                  BOTH models, for the floor AND for the method class.
  H-D PARTIAL     negative with CI excluding zero on at least one (model, class), and no
                  (model, class) positive with CI excluding zero.
  H-D NOT SUPPORTED  otherwise. Reported whichever it is.

LIMITATIONS, stated in advance:
  * D for methods is a MEAN over KV slots. The model reads across all slots, so the effective
    distractor set may be closer to the union, which the captures do not record. If H-D fails
    for methods but holds for the floor (one global keep-set, no such ambiguity), that is the
    first place to look, and it would need a recapture to settle.
  * c = 1 (MARK-1) units are single-word entries; "complete non-queried" means other entries
    kept. Analysed as its own group because the KeyDiff c=1 over-call is the second
    independent place completeness fails to index usability.
  * Accuracy is n=100 per cell (generation); captures n=200. The join uses the first 100
    instances, verified to be the same instances in both packages (identical seed and n_ctx).
"""
from __future__ import annotations

import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "nvidia"
MODELS = ("M2", "M3")
METHODS = ("snapkv", "adakv_snapkv", "expected_attn", "keydiff")
FLOOR_MIN = 0.05
NL = chr(10)


def jl(path):
    out = []
    if not path.exists():
        return out
    for line in path.open(encoding="utf-8"):
        if line.strip():
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def ols2(y, x1, x2):
    """y ~ b1 x1 + b2 x2 on already-centred data (no intercept needed)."""
    s11 = sum(a * a for a in x1)
    s22 = sum(b * b for b in x2)
    s12 = sum(a * b for a, b in zip(x1, x2))
    s1y = sum(a * c for a, c in zip(x1, y))
    s2y = sum(b * c for b, c in zip(x2, y))
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-12:
        return float("nan"), float("nan")
    return (s22 * s1y - s12 * s2y) / det, (s11 * s2y - s12 * s1y) / det


def pearson(x, y):
    if len(x) < 3:
        return float("nan")
    mx, my = st.fmean(x), st.fmean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def residualise(v, covs):
    """Residual of v after OLS on the covariate columns (with intercept)."""
    n = len(v)
    cols = [[1.0] * n] + covs
    k = len(cols)
    XtX = [[sum(cols[i][r] * cols[j][r] for r in range(n)) for j in range(k)] for i in range(k)]
    Xty = [sum(cols[i][r] * v[r] for r in range(n)) for i in range(k)]
    A = [row[:] + [Xty[i]] for i, row in enumerate(XtX)]
    for i in range(k):
        piv = max(range(i, k), key=lambda r: abs(A[r][i]))
        A[i], A[piv] = A[piv], A[i]
        if abs(A[i][i]) < 1e-12:
            return [float("nan")] * n
        for r in range(k):
            if r != i:
                f = A[r][i] / A[i][i]
                A[r] = [a - f * b for a, b in zip(A[r], A[i])]
    beta = [A[i][k] / A[i][i] for i in range(k)]
    return [v[r] - sum(beta[i] * cols[i][r] for i in range(k)) for r in range(n)]


def main() -> int:
    groups = defaultdict(list)       # (model, class, family) -> list of cells
    cell_level = []
    desc = defaultdict(dict)
    for tag in MODELS:
        caps = {}
        for r in jl(RUNS / ("stage4_capture_%s.jsonl" % tag)):
            if r["task"] == "multispan":
                continue
            caps[(r["task"], round(r["c"], 2), r["C"], r["arm"], r["key"]["instance_id"])] = r
        acc = defaultdict(dict)
        for r in jl(RUNS / ("stage4_plane_%s.jsonl" % tag)):
            acc[(r["task"], round(r["c"], 2), r["C"], r["arm"])][r["key"]["instance_id"]] = r

        # floor accuracy per cell, for the section 9.1 admissibility rule
        floor_acc = {}
        for (task, c, C, arm), d in acc.items():
            if arm == "floor_pos":
                floor_acc[(task, c, C)] = st.fmean(x["score"] for x in d.values())

        for (task, c, C, arm), d in sorted(acc.items()):
            if arm not in ("floor_pos",) + METHODS:
                continue
            if floor_acc.get((task, c, C), 0.0) < FLOOR_MIN:
                continue
            ys, qs, ds = [], [], []
            for iid, g in d.items():
                cp = caps.get((task, c, C, arm, iid))
                if cp is None:
                    continue
                if cp["n_ctx"] != g["n_ctx"] or cp["key"]["seed"] != g["key"]["seed"]:
                    raise SystemExit("instance mismatch between capture and plane: %s %s"
                                     % (tag, iid))
                D = cp["units_complete"] - cp["q_complete"] * cp["n_facts"]
                ys.append(g["score"])
                qs.append(cp["q_complete"])
                ds.append(max(0.0, D))
            if len(ys) < 10:
                continue
            fam = "c=1" if task == "mark1" else "c>1"
            cls = "floor" if arm == "floor_pos" else "method"
            groups[(tag, cls, fam)].append(dict(c=c, C=C, arm=arm, y=ys, q=qs, d=ds))
            mq, macc, md = st.fmean(qs), st.fmean(ys), st.fmean(ds)
            desc[(tag, fam)][(c, C, arm)] = (md, mq, macc)
            if mq > 0.02:
                cell_level.append(dict(tag=tag, fam=fam, cls=cls, arm=arm, c=c, C=C,
                                       conv=macc / mq, D=md))

    # ---------------- descriptive ----------------
    print("=" * 100)
    print("DESCRIPTIVE -- mean complete NON-QUERIED records retained (D), per arm per cell")
    print("=" * 100)
    for (tag, fam), cells in sorted(desc.items()):
        print(NL + "### %s  %s   (D / q_complete / accuracy)" % (tag, fam))
        keys = sorted({(c, C) for (c, C, a) in cells})
        print("  %-10s" % "c, C" + "".join("%22s" % a for a in ("floor_pos",) + METHODS))
        for c, C in keys:
            row = []
            for a in ("floor_pos",) + METHODS:
                v = cells.get((c, C, a))
                row.append("%22s" % ("%.2f / %.3f / %.3f" % v if v else "-"))
            print("  %-10s" % ("%.0f, %d" % (c, C)) + "".join(row))

    # ---------------- primary ----------------
    print(NL + "=" * 100)
    print("PRIMARY -- within-cell: accuracy ~ q_complete + D, cell fixed effects")
    print("=" * 100)
    print("  %-4s %-7s %-4s %6s %7s %12s %22s %12s"
          % ("mdl", "class", "fam", "cells", "inst", "b_D", "95% CI (cell bootstrap)", "b_q"))
    verdict = {}
    for key in sorted(groups):
        cells = groups[key]

        def fit(cs):
            Y, Q, Dd = [], [], []
            for cl in cs:
                my, mq, md = st.fmean(cl["y"]), st.fmean(cl["q"]), st.fmean(cl["d"])
                Y += [v - my for v in cl["y"]]
                Q += [v - mq for v in cl["q"]]
                Dd += [v - md for v in cl["d"]]
            return ols2(Y, Q, Dd)

        bq, bd = fit(cells)
        rng = random.Random(11)
        boots = []
        for _ in range(1000):
            samp = [cells[rng.randrange(len(cells))] for _ in cells]
            b = fit(samp)[1]
            if not math.isnan(b):
                boots.append(b)
        boots.sort()
        lo = boots[int(0.025 * len(boots))] if boots else float("nan")
        hi = boots[int(0.975 * len(boots))] if boots else float("nan")
        n_inst = sum(len(cl["y"]) for cl in cells)
        sig_neg = hi < 0
        sig_pos = lo > 0
        verdict[key] = "neg" if sig_neg else ("pos" if sig_pos else "ns")
        print("  %-4s %-7s %-4s %6d %7d %+12.4f %22s %+12.4f   %s"
              % (key[0], key[1], key[2], len(cells), n_inst, bd,
                 "[%+.4f, %+.4f]" % (lo, hi), bq,
                 {"neg": "NEGATIVE, CI excludes 0", "pos": "POSITIVE, CI excludes 0",
                  "ns": "CI includes 0"}[verdict[key]]))

    # ---------------- secondary ----------------
    print(NL + "=" * 100)
    print("SECONDARY -- cell-level conversion (acc / q) vs mean D, partialling out log B and c")
    print("CONFOUNDED BY DESIGN: D and conversion both move with budget. Reported, not weighed.")
    print("=" * 100)
    for tag in MODELS:
        for fam in ("c>1", "c=1"):
            sub = [r for r in cell_level if r["tag"] == tag and r["fam"] == fam]
            if len(sub) < 5:
                continue
            conv = [r["conv"] for r in sub]
            D = [r["D"] for r in sub]
            covs = [[math.log(r["C"] + 72) for r in sub]]
            if fam == "c>1":
                covs.append([r["c"] for r in sub])
            print("  %-3s %-4s cells %2d   raw r(conv, D) = %+.3f   partial r | logB%s = %+.3f"
                  % (tag, fam, len(sub), pearson(D, conv), ", c" if fam == "c>1" else "",
                     pearson(residualise(D, covs), residualise(conv, covs))))

    # ---------------- decision ----------------
    print(NL + "=" * 100)
    print("DECISION (rule fixed before the final run) -- primary test, c>1")
    print("=" * 100)
    need = [(t, cl, "c>1") for t in MODELS for cl in ("floor", "method")]
    have = {k: verdict.get(k, "missing") for k in need}
    for k, v in have.items():
        print("  %-3s %-7s %s" % (k[0], k[1], v))
    if all(v == "neg" for v in have.values()):
        d = "H-D SUPPORTED"
    elif any(v == "neg" for v in have.values()) and not any(v == "pos" for v in have.values()):
        d = "H-D PARTIAL"
    else:
        d = "H-D NOT SUPPORTED"
    print(NL + "  %s" % d)
    c1 = {k: v for k, v in verdict.items() if k[2] == "c=1"}
    if c1:
        print("  c=1 group, reported separately: " +
              ", ".join("%s %s %s" % (k[0], k[1], v) for k, v in sorted(c1.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
