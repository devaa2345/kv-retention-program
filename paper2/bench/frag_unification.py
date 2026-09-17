"""Are token-level and head-level fragmentation the same mechanism, or two?

Token level asks, of the records an arm TOUCHES, how many does it COMPLETE.
Head level asks the same question across KV heads for the QUERIED record: of the heads
holding any of it, how many hold all of it.

Both are computed on the SAME capture, per instance, so the comparison is not across runs.

Direction is normalised so higher = less fragmented for both:
    token_completion = recs_complete  / recs_touched         in [0, 1]
    head_completion  = heads_complete / heads_touched        in [0, 1]

Both condition on TOUCHED. An earlier version divided the head measure by n_heads instead;
that quantity is algebraically identical to per-record completeness (the same indicator matrix
averaged in the other order -- verified equal to 0.0 to machine precision), so regressing
accuracy on it was circular and its "SEPARABLE" verdict was void. Conditioning on touched is
what makes the head measure a genuine analogue of the token one rather than a restatement of
completeness.

`n_heads` here counts (layer, KV-head) slots -- 72 for M2 (36 layers x 2 heads), 224 for M3
(28 x 8). So the head measure averages over depth as well as across heads, and cannot separate
"scattered across heads" from "scattered across layers". M3 carries the head question; M2 has
2 KV heads and is near-degenerate for it, exactly as in N9.
(the first is the reciprocal of the touched/complete ratio, which is unbounded and goes
infinite whenever an arm completes nothing -- the reciprocal is the same quantity without that
pathology, and the raw ratio is still printed in the cell table.)

Three questions, answered separately:
  1. Do both predict accuracy, and in the same direction?
  2. Comparable magnitude?  -> standardised betas from the two-predictor regression
  3. One mechanism or two?  -> if they are one, neither should survive controlling for the
     other; if separable, at least one partial correlation stays well away from zero.

**N9's Delta_head is NOT the correlate here.** It is measured on ORACLE arms with token
completeness held perfect, so it isolates the head mechanism; the per-arm head measure below is
computed from method captures and confounds both. N9 is the calibration anchor, printed
alongside, not a column in the regression.
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

DUMPS = {
    "M2": ("runs/nvidia/frag_perinstance_M2.jsonl",
           "runs/nvidia/grid_M2_ledger_agnostic.jsonl",
           "runs/nvidia/n9_M2_ledger.jsonl.analysis.json"),
    "M3": ("runs/nvidia/frag_perinstance_M3.jsonl",
           "runs/nvidia/grid_M3_ledger_agnostic.jsonl",
           "runs/nvidia/n9_M3_ledger.jsonl.analysis.json"),
}
UNIT = sys.argv[1] if len(sys.argv) > 1 else "line"       # "line" or "idval"


def pearson(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def partial(rxy, rxz, ryz):
    d = (1 - rxz ** 2) * (1 - ryz ** 2)
    if d <= 0:
        return float("nan")
    return (rxy - rxz * ryz) / math.sqrt(d)


def betas(rxy, rzy, rxz):
    """Standardised OLS coefficients for y ~ x + z."""
    d = 1 - rxz ** 2
    if abs(d) < 1e-12:
        return float("nan"), float("nan")
    return (rxy - rxz * rzy) / d, (rzy - rxz * rxy) / d


def boot_r(x, y, seed=0, n=4000):
    rng = random.Random(seed)
    m = len(x)
    rs = []
    for _ in range(n):
        idx = [rng.randrange(m) for _ in range(m)]
        r = pearson([x[i] for i in idx], [y[i] for i in idx])
        if not math.isnan(r):
            rs.append(r)
    if not rs:
        return float("nan"), float("nan")
    rs.sort()
    return rs[int(0.025 * len(rs))], rs[int(0.975 * len(rs))]


def block(label, rs):
    y = [r["acc"] for r in rs]
    x = [r["tok"] for r in rs]
    z = [r["head"] for r in rs]
    rxy, rzy, rxz = pearson(x, y), pearson(z, y), pearson(x, z)
    if any(math.isnan(v) for v in (rxy, rzy, rxz)):
        print("\n  %s: degenerate (no variance in one measure) -- skipped" % label)
        return
    pxy, pzy = partial(rxy, rxz, rzy), partial(rzy, rxz, rxy)
    bx, bz = betas(rxy, rzy, rxz)
    cx = boot_r(x, y, 1)
    cz = boot_r(z, y, 2)
    print("\n  %s  (n=%d)" % (label, len(rs)))
    print("    r(token_completion, acc) = %+.4f  95%% CI [%+.4f, %+.4f]" % (rxy, cx[0], cx[1]))
    print("    r(head_completion,  acc) = %+.4f  95%% CI [%+.4f, %+.4f]" % (rzy, cz[0], cz[1]))
    print("    collinearity r(token, head) = %+.4f" % rxz)
    print("    partial r(token | head) = %+.4f     partial r(head | token) = %+.4f" % (pxy, pzy))
    print("    standardised beta: token %+.4f   head %+.4f" % (bx, bz))
    same_dir = (rxy > 0) == (rzy > 0)
    hi = max(abs(rxy), abs(rzy))
    mag = (min(abs(rxy), abs(rzy)) / hi) if hi > 0 else float("nan")
    if abs(pxy) < 0.10 and abs(pzy) < 0.10:
        verdict = "UNIFIED -- neither survives controlling for the other"
    elif abs(pxy) >= 0.10 and abs(pzy) >= 0.10:
        verdict = "SEPARABLE -- both survive controlling for the other"
    elif abs(pxy) >= 0.10:
        verdict = "TOKEN-DOMINANT -- head effect vanishes when token is controlled"
    else:
        verdict = "HEAD-DOMINANT -- token effect vanishes when head is controlled"
    print("    same direction: %s    magnitude ratio (min/max |r|): %.3f" % (same_dir, mag))
    print("    -> %s" % verdict)


def main() -> int:
    for model, (dump, grid, n9) in DUMPS.items():
        if not Path(dump).exists():
            print("\n### %s: no per-instance dump yet (%s) -- skipped" % (model, dump))
            continue

        acc = {}
        if Path(grid).exists():
            for line in Path(grid).open(encoding="utf-8"):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                acc[(r["C"], r["key"]["arm"], r["key"]["instance_id"])] = r["score"]

        rows = []
        for line in Path(dump).open(encoding="utf-8"):
            r = json.loads(line)
            a = acc.get((r["C"], r["arm"], r["instance_id"]))
            if a is None:
                continue
            rc = r["recs_complete_%s" % UNIT]
            # NOT head_complete_frac: that is ALGEBRAICALLY IDENTICAL to qcpl (the completeness
            # measure re-averaged over the same indicator matrix -- verified equal to 0.0 to
            # machine precision), so regressing accuracy on it is circular. The genuine
            # head-level analogue of the token ratio conditions on TOUCHED, exactly as the
            # token measure does.
            hc = r["heads_complete_%s" % UNIT] / max(1e-9, r["heads_touched"])
            rows.append(dict(
                C=r["C"], arm=r["arm"], iid=r["instance_id"], acc=a,
                tok=rc / max(1e-9, r["recs_touched"]),
                head=hc,
                raw_frag=(r["recs_touched"] / rc) if rc else float("inf"),
                n_heads=r["n_heads"]))

        if not rows:
            print("\n### %s: dump present but nothing joined to accuracy -- skipped" % model)
            continue

        bar = "=" * 100
        print("\n%s\n### %s   unit=%s   n=%d instance-arm-budget rows (%d KV heads)\n%s"
              % (bar, model, UNIT, len(rows), rows[0]["n_heads"], bar))

        cells = defaultdict(list)
        for r in rows:
            cells[(r["C"], r["arm"])].append(r)
        print("%5s %-15s %7s %10s %11s %9s"
              % ("C", "arm", "acc", "tok_compl", "head_compl", "raw_frag"))
        for key in sorted(cells):
            v = cells[key]
            fr = [r["raw_frag"] for r in v if math.isfinite(r["raw_frag"])]
            print("%5d %-15s %7.4f %10.4f %11.4f %9.2f"
                  % (key[0], key[1],
                     statistics.fmean(r["acc"] for r in v),
                     statistics.fmean(r["tok"] for r in v),
                     statistics.fmean(r["head"] for r in v),
                     statistics.fmean(fr) if fr else float("inf")))

        block("POOLED across all cells", rows)
        block("METHOD ARMS only (floor_pos excluded)",
              [r for r in rows if r["arm"] != "floor_pos"])
        for C in sorted({r["C"] for r in rows}):
            block("WITHIN budget C=%d" % C, [r for r in rows if r["C"] == C])

        if Path(n9).exists():
            d = json.load(open(n9, encoding="utf-8"))
            print("\n  N9 ANCHOR (%s, %d KV heads) -- head mechanism in ISOLATION, token "
                  "completeness held perfect:" % (model, d["n_kv_heads"]))
            for c in d["cells"].values():
                deg = ("   (structurally degenerate: all candidates fit)"
                       if c["delta_head"]["mean"] == 0 else "")
                print("    C=%-4d Delta_head = %+.4f %s%s"
                      % (c["C"], c["delta_head"]["mean"], c["delta_head"]["ci"], deg))
        else:
            print("\n  N9 anchor not available yet (%s)" % n9)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
