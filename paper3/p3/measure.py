"""Measured quantities, read from Paper 2's captures. No theory here.

Sources (all already on disk, no GPU):
    runs/nvidia/frag_perinstance_{M2,M3}.jsonl   per (instance, C, arm) fragmentation dump
    runs/nvidia/slotaware_{M2,M3}.jsonl          per (instance, C, arm) slot-level dump
    runs/nvidia/grid_{M2,M3}_ledger_agnostic.jsonl   the agnostic accuracy grid
    runs/nvidia/n9_{M2,M3}_ledger.jsonl(.analysis.json)  the head decomposition

Field semantics, taken from paper2/bench/fragmentation_units.py and
paper2/bench/slot_aware_completeness.py:

    gold_tok_line   mean over (layer, KV-head) slots of the number of tokens of the H=4
                    queried record LINES retained.  A COUNT, not a rate.
    qcpl_line       mean over slots of the fraction of the H queried records retained WHOLE.
                    This is the per-fact completion rate the theory predicts.
    q_mean_line     the same quantity in the slot dump (MEAN over slots)
    q_any_line      fraction of queried records complete in AT LEAST ONE slot (the UNION)
    recs_touched    distinct records (of N=40) with >= 1 token retained, mean over slots
    recs_complete_* distinct records retained whole, mean over slots
"""
from __future__ import annotations

import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

from . import facts as F

P2 = F.P2
RUNS = P2 / "runs" / "nvidia"
N_SINK, N_WINDOW = 8, 64
H = 4
N_RECORDS = 40

FILES = {
    "M2": dict(frag=RUNS / "frag_perinstance_M2.jsonl",
               slot=RUNS / "slotaware_M2.jsonl",
               grid=RUNS / "grid_M2_ledger_agnostic.jsonl",
               n9=RUNS / "n9_M2_ledger.jsonl.analysis.json"),
    "M3": dict(frag=RUNS / "frag_perinstance_M3.jsonl",
               slot=RUNS / "slotaware_M3.jsonl",
               grid=RUNS / "grid_M3_ledger_agnostic.jsonl",
               n9=RUNS / "n9_M3_ledger.jsonl.analysis.json"),
}

METHOD_ARMS = ["snapkv", "expected_attn", "keydiff", "adakv_snapkv"]
ALL_ARMS = ["floor_pos"] + METHOD_ARMS


def _jsonl(p: Path):
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def accuracy_table():
    """mean accuracy and n per (model, C, arm) from the canonical agnostic grid."""
    out = {}
    for tag, f in FILES.items():
        acc = defaultdict(list)
        for r in _jsonl(f["grid"]):
            acc[(r["C"], r["key"]["arm"])].append(r["score"])
        out[tag] = {k: (st.fmean(v), len(v)) for k, v in acc.items()}
    return out


def cells(unit: str = "line"):
    """Per (model, C, arm) measured record. `unit` is 'line' or 'idval'.

    Returns a dict keyed (model, C, arm) with:
        p_g           per-gold-token keep rate, all gold tokens
        p_g_payable   same, with tokens already inside the mandatory floors removed from
                      both numerator and denominator
        q_complete    measured per-fact completion, MEAN over slots  (the target of T2)
        q_any         measured per-fact completion, UNION over slots (Paper 2's accounting)
        q_maj         complete in > 50% of slots
        q_complete_other   completion measured in the OTHER unit (feeds the F4 baseline)
        c, c_other, L      fact costs and context length, instance-averaged
        recs_complete, recs_touched   all-40-record accounting, mean over slots
        heads_touched, heads_complete, n_slots
        acc, n
    """
    other = "idval" if unit == "line" else "line"
    geo = F.build()
    accs = accuracy_table()
    out = {}

    for tag, f in FILES.items():
        ins = geo[tag]["instances"]
        # per-instance denominators
        den, den_free, den_o, Ls = {}, {}, {}, {}
        for iid, v in ins.items():
            den[iid] = float(sum(v[f"q_{unit}"]))
            den_free[iid] = float(sum(v[f"q_{unit}_floor"]))
            den_o[iid] = float(sum(v[f"q_{other}"]))
            Ls[iid] = float(v["L"])

        slot = defaultdict(dict)
        for r in _jsonl(f["slot"]):
            slot[(r["C"], r["arm"])][r["instance_id"]] = r

        agg = defaultdict(lambda: defaultdict(list))
        for r in _jsonl(f["frag"]):
            iid = r["instance_id"]
            k = (r["C"], r["arm"])
            g = r[f"gold_tok_{unit}"]
            go = r[f"gold_tok_{other}"]
            a = agg[k]
            a["p_g"].append(g / den[iid])
            free = den_free[iid]
            a["p_g_payable"].append(
                (g - free) / (den[iid] - free) if den[iid] - free > 0 else float("nan"))
            a["q_complete"].append(r[f"qcpl_{unit}"])
            a["q_complete_other"].append(r[f"qcpl_{other}"])
            a["p_g_other"].append(go / den_o[iid])
            a["recs_complete"].append(r[f"recs_complete_{unit}"])
            a["recs_touched"].append(r["recs_touched"])
            a["heads_touched"].append(r["heads_touched"])
            a["heads_complete"].append(r[f"heads_complete_{unit}"])
            a["n_slots"].append(r["n_heads"])
            a["c"].append(den[iid] / H)
            a["c_other"].append(den_o[iid] / H)
            a["L"].append(Ls[iid])
            s = slot[k].get(iid)
            if s is not None:
                a["q_any"].append(s[f"q_any_{unit}"])
                a["q_maj"].append(s[f"q_maj_{unit}"])
                a["q_mean"].append(s[f"q_mean_{unit}"])
                a["recs_touched_any"].append(s["recs_touched_any"])
                a["recs_touched_mean"].append(s["recs_touched_mean"])

        for (C, arm), a in agg.items():
            acc, n = accs[tag].get((C, arm), (float("nan"), 0))
            rec = {kk: (st.fmean(vv) if vv else float("nan")) for kk, vv in a.items()}
            rec.update(model=tag, C=C, arm=arm, acc=acc, n_acc=n, n_inst=len(a["p_g"]),
                       unit=unit,
                       q_complete_sd=st.pstdev(a["q_complete"]) if len(a["q_complete"]) > 1 else 0.0,
                       p_g_sd=st.pstdev(a["p_g"]) if len(a["p_g"]) > 1 else 0.0,
                       per_instance={"p_g": a["p_g"], "q_complete": a["q_complete"]})
            out[(tag, C, arm)] = rec
    return out


def n_free_causal():
    """Expected number of the H queried records already wholly inside the mandatory floors.
    Feeds the floor-corrected T1 ceiling."""
    geo = F.build()
    out = {}
    for tag, m in geo.items():
        vals = []
        for v in m["instances"].values():
            vals.append(sum(1 for a, b in zip(v["q_line"], v["q_line_floor"]) if a == b))
        out[tag] = st.fmean(vals)
    return out


def n9_cells():
    out = {}
    for tag, f in FILES.items():
        d = json.loads(f["n9"].read_text(encoding="utf-8"))
        out[tag] = d
    return out


def p_g_interpolator(tag, arm, unit="line", table=None):
    """Log-linear interpolation/extrapolation of measured p_g in total budget B.

    p_g is measured only at the ladder budgets. Solving for B* needs it as a function, so
    log p_g is fitted linearly in log B by ordinary least squares over the measured cells.
    The fit is reported (slope, intercept, R^2) wherever a B* is quoted.
    """
    table = table or cells(unit)
    pts = [(C + N_SINK + N_WINDOW, table[(tag, C, arm)]["p_g"])
           for (t, C, a) in table if t == tag and a == arm]
    pts = [(b, p) for b, p in pts if p > 0]
    pts.sort()
    xs = [math.log(b) for b, _ in pts]
    ys = [math.log(p) for _, p in pts]
    mx, my = st.fmean(xs), st.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    k = sxy / sxx if sxx else 0.0
    b0 = my - k * mx
    ss_res = sum((y - (b0 + k * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")

    def f(B):
        return min(1.0, math.exp(b0 + k * math.log(max(B, 1.0))))

    f.slope, f.intercept, f.r2, f.points = k, b0, r2, pts
    return f
