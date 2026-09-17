"""Two follow-ups.

(1) Is the M3 / C=512 ordering real, or one noisy cell?
    Paired bootstrap CI on (method - floor_pos) per instance, n=200.

(2) Does the fragmentation-accuracy relationship hold where the budget CONSTRAINS the choice
    and dissolve where it does not?

    N9 fixed the boundary structurally: the candidate set costs ~54 payable tokens, so at
    C=16 the optimal packing completes 1 of 4 candidates and at C=32 it completes 2 of 4 --
    the budget forces a choice. At C>=64 all four fit, `oracle_causal` already holds every
    answerable record, and Delta_head = Delta_temporal = I = 0 by construction. Cutting by
    regime rather than by budget is therefore the cut the mechanism implies.

    BINDING cells available: M3 C=16, M3 C=32, M2 C=32  (the M2 grid has no C=16 cell).
    NON-BINDING cells: C in {64,128,256,512} on both models.

    Two specifications, because they answer different questions:
      (A) across-ARM, within cell -- the claim "the more fragmented arm scores worse" is a
          statement about arms, so correlate the 5 arm means within each cell, then pool by
          regime. Small n per cell (5), so per-cell values are printed, not just the mean.
      (B) instance-level, DEMEANED BY (model, C) -- removes model and budget level differences
          while keeping the arm variation that carries the claim. Bootstrap CI over instances.
"""
from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

BINDING = {("M3", 16), ("M3", 32), ("M2", 32), ("M2", 64)}   # corrected from N9 M2
ARMS = ["floor_pos", "snapkv", "expected_attn", "keydiff", "adakv_snapkv"]


def pearson(x, y):
    if len(x) < 3:
        return float("nan")
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def boot_ci(x, y, seed=0, n=10000):
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


def paired_boot(a, b, seed=0, n=20000):
    """CI on mean(a - b), paired."""
    d = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    m = len(d)
    ms = sorted(statistics.fmean(d[rng.randrange(m)] for _ in range(m)) for _ in range(n))
    return statistics.fmean(d), ms[int(0.025 * n)], ms[int(0.975 * n)]


def load_grid(model):
    by = defaultdict(dict)
    p = Path("runs/nvidia/grid_%s_ledger_agnostic.jsonl" % model)
    for line in p.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        by[(r["C"], r["key"]["arm"])][r["key"]["instance_id"]] = r["score"]
    return by


# ============================ (1) M3 C=512 ordering ============================
print("=" * 90)
print("(1) M3 / C=512 -- is the ordering real? paired bootstrap on (method - floor_pos)")
print("=" * 90)
g3 = load_grid("M3")
fl = g3[(512, "floor_pos")]
print("  %-16s %8s %10s %22s %s"
      % ("method", "acc", "d vs floor", "95% CI", "verdict"))
print("  %-16s %8.4f" % ("floor_pos", statistics.fmean(fl.values())))
for arm in ["adakv_snapkv", "expected_attn", "keydiff", "snapkv"]:
    cur = g3.get((512, arm), {})
    ids = sorted(set(cur) & set(fl))
    if not ids:
        continue
    a = [cur[i] for i in ids]
    b = [fl[i] for i in ids]
    mean, lo, hi = paired_boot(a, b, seed=512)
    if lo > 0:
        v = "BEATS floor (CI excludes 0)"
    elif hi < 0:
        v = "loses to floor (CI excludes 0)"
    else:
        v = "indistinguishable (CI spans 0)"
    print("  %-16s %8.4f %+10.4f  [%+8.4f, %+8.4f]  %s"
          % (arm, statistics.fmean(a), mean, lo, hi, v))
print("  (n=%d paired instances)" % len(ids))

# ============================ (2) regime re-cut ===============================
print("\n" + "=" * 90)
print("(2) fragmentation-accuracy by REGIME (N9 boundary), pooled across models")
print("=" * 90)

rows = []
for model in ("M2", "M3"):
    grid = load_grid(model)
    dump = Path("runs/nvidia/frag_perinstance_%s.jsonl" % model)
    if not dump.exists():
        continue
    for line in dump.open(encoding="utf-8"):
        r = json.loads(line)
        a = grid.get((r["C"], r["arm"]), {}).get(r["instance_id"])
        if a is None:
            continue
        rows.append(dict(
            model=model, C=r["C"], arm=r["arm"], iid=r["instance_id"], acc=a,
            tok=r["recs_complete_line"] / max(1e-9, r["recs_touched"]),
            binding=(model, r["C"]) in BINDING))

print("  joined %d rows   binding=%d  non-binding=%d"
      % (len(rows), sum(r["binding"] for r in rows), sum(not r["binding"] for r in rows)))
cells = sorted({(r["model"], r["C"]) for r in rows})
print("  binding cells:     %s" % [c for c in cells if c in BINDING])
print("  non-binding cells: %s" % [c for c in cells if c not in BINDING])

# ---- (A) across-arm, within cell ----
print("\n  (A) ACROSS-ARM within each cell: r(token_completion, accuracy) over %d arm means"
      % len(ARMS))
print("      %-8s %-6s %10s   %s" % ("model", "C", "r", "regime"))
per_regime = {True: [], False: []}
for (model, C) in cells:
    xs, ys = [], []
    for arm in ARMS:
        v = [r for r in rows if r["model"] == model and r["C"] == C and r["arm"] == arm]
        if not v:
            continue
        xs.append(statistics.fmean(r["tok"] for r in v))
        ys.append(statistics.fmean(r["acc"] for r in v))
    r_ = pearson(xs, ys)
    binding = (model, C) in BINDING
    per_regime[binding].append(r_)
    print("      %-8s %-6d %+10.4f   %s"
          % (model, C, r_, "BINDING" if binding else "non-binding"))
for b in (True, False):
    v = [x for x in per_regime[b] if not math.isnan(x)]
    if v:
        print("      mean r, %-12s = %+.4f   (cells: %d)"
              % ("BINDING" if b else "non-binding", statistics.fmean(v), len(v)))

# ---- (B) instance-level, demeaned by (model, C) ----
print("\n  (B) INSTANCE-LEVEL, demeaned by (model, C) -- keeps arm variation, removes "
      "model/budget level")
mu = defaultdict(lambda: [0.0, 0.0, 0])
for r in rows:
    k = (r["model"], r["C"])
    mu[k][0] += r["tok"]
    mu[k][1] += r["acc"]
    mu[k][2] += 1
for r in rows:
    k = (r["model"], r["C"])
    r["tok_d"] = r["tok"] - mu[k][0] / mu[k][2]
    r["acc_d"] = r["acc"] - mu[k][1] / mu[k][2]

for b in (True, False):
    v = [r for r in rows if r["binding"] is b]
    if len(v) < 10:
        print("      %-12s n=%d -- too few rows" % ("BINDING" if b else "non-binding", len(v)))
        continue
    x = [r["tok_d"] for r in v]
    y = [r["acc_d"] for r in v]
    r_ = pearson(x, y)
    lo, hi = boot_ci(x, y, seed=1 if b else 2, n=4000)
    print("      %-12s n=%-5d r = %+.4f   95%% CI [%+.4f, %+.4f]"
          % ("BINDING" if b else "non-binding", len(v), r_, lo, hi))
