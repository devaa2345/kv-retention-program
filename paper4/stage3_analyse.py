"""Paper 4 Stage 3 analysis, scored exactly as PREREG_P4_S3.md (131278...838c) registers it.

    python stage3_analyse.py control   -> c=1 control only (prereg 6). exit 1 = STOP AND DIAGNOSE
    python stage3_analyse.py report    -> full readings, K1-K5, P1-P5 (partial grid tolerated)
"""
from __future__ import annotations

import json
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "nvidia"
OUT = HERE / "out"
TAGS = ("M2", "M3")
BUDGETS = (32, 64, 128, 256, 512)
C_TAGS = (1, 8, 19, 40)
PAIRS = (("snapkv", "U-snapkv"), ("adakv_snapkv", "U-adakv_snapkv"),
         ("expected_attn", "U-expected_attn"), ("keydiff", "U-keydiff"))
ARMS = ("floor_pos", "oracle_causal") + tuple(a for p in PAIRS for a in p)
N = 50
R_BOOT = 10000
DEGEN = 0.05
HDR = ("Produced on RTX 5070 (Machine N). Paper 4 Stage 3, oracle units, n=50, "
       "prereg 34a20a7ef35c1510067359a76d05de10f32d912b9058a1c94ac9187e9d8e7a3f (A1-A3); "
       "c=1 rows were collected under the superseded 131278...838c and stand as collected (A3.2)")


def load(tag):
    """Only cells that are complete AND single-session are scored (prereg 5.12)."""
    p = RUNS / f"p4_s3_{tag}.jsonl"
    if not p.exists():
        return {}
    by_cell = defaultdict(list)
    with p.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_cell[(r["c_tag"], r["C"])].append(r)
    cells = {}
    for cell, rr in by_cell.items():
        sessions = {r["session_id"] for r in rr}
        idx = {(r["arm"], r["instance"]): r for r in rr}
        if len(sessions) == 1 and len(idx) == len(ARMS) * N:
            cells[cell] = idx
    return cells


def boot(v, key):
    v = np.asarray(v, dtype=float)
    rng = np.random.default_rng(zlib.crc32(("p4|s3|" + key).encode()) & 0xFFFFFFFF)
    m = v[rng.integers(0, len(v), size=(R_BOOT, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fmt(t, p=3):
    return "%+.*f [%+.*f, %+.*f]" % (p, t[0], p, t[1], p, t[2])


def cell_acc(idx, arm):
    return [idx[(arm, i)]["score"] for i in range(N)]


def delta_u(idx, x, u, key):
    return boot([idx[(u, i)]["score"] - idx[(x, i)]["score"] for i in range(N)], key)


def control():
    """MARK-1 c=1: a REPORTED DIAGNOSTIC, not a gate (amendment A3.2). Always exits 0.

    The registered inertness test is now the singleton-unit identity (A3.1), verified in
    preflight. Both A3.2 channels are reported per method: d p_g and d units_complete.
    """
    L = [HDR, "",
         "MARK-1 c = 1 DIAGNOSTIC (amendment A3.2 -- NOT a pass/fail gate).",
         "Registered inertness test is the singleton-unit identity (A3.1), see "
         "out/stage3_preflight_identity.json.", ""]
    ok, seen = True, 0
    for tag in TAGS:
        cells = load(tag)
        L.append(f"== {tag}")
        for C in BUDGETS:
            idx = cells.get((1, C))
            if idx is None:
                L.append(f"  C={C:3d}  (no complete single-session cell yet)")
                continue
            seen += 1
            fl = float(np.mean(cell_acc(idx, "floor_pos")))
            if fl < DEGEN:
                L.append(f"  C={C:3d}  floor {fl:.3f} -> DEGENERATE, excluded from the control")
                continue
            for x, u in PAIRS:
                d = delta_u(idx, x, u, f"ctrl|{tag}|{C}|{x}")
                dpg = float(np.mean([idx[(u, i)]["p_g"] - idx[(x, i)]["p_g"] for i in range(N)]))
                duc = float(np.mean([idx[(u, i)]["units_complete"] - idx[(x, i)]["units_complete"]
                                     for i in range(N)]))
                excl = d[1] > 0 or d[2] < 0
                L.append(f"  C={C:3d}  floor {fl:.3f}  {u:16s} - {x:14s} dU {fmt(d)}"
                         f"  d p_g {dpg:+.4f}  d units_complete {duc:+.2f}"
                         f"{'  [CI excludes 0]' if excl else ''}")
        L.append("")
    L.append("Channels (A3.2): d p_g is the score-variance gold-retention shift -- material for "
             "KeyDiff/ExpectedAttention, ~0 for SnapKV/AdaKV; d units_complete is context-structure "
             "reallocation. Neither is a gate; the gate is the singleton-unit identity (A3.1).")
    verdict_ok = True
    if seen == 0:
        L.append("NOTE: no c=1 cells scored yet.")
    OUT.mkdir(exist_ok=True)
    (OUT / "stage3_control.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0 if verdict_ok else 1


def report():
    L = [HDR, ""]
    res = {"cells": {}}
    dU_by_c = defaultdict(lambda: defaultdict(list))
    rho_rows, conv_rows, k5_bad, k5_tot, k3_ge, k3_tot = [], [], 0, 0, 0, 0
    p1 = []
    for tag in TAGS:
        cells = load(tag)
        L.append(f"==================== {tag}: {len(cells)} complete single-session cells")
        for c_tag in C_TAGS:
            for C in BUDGETS:
                idx = cells.get((c_tag, C))
                if idx is None:
                    continue
                A = {a: float(np.mean(cell_acc(idx, a))) for a in ARMS}
                fl = A["floor_pos"]
                degen = fl < DEGEN
                q = {a: {k: float(np.mean([idx[(a, i)][k] for i in range(N)]))
                         for k in ("q_slot", "q_any", "q_maj", "units_complete", "units_touched")}
                     for a in ARMS}
                cell = dict(acc=A, completion=q, floor=fl, degenerate=degen)
                L.append(f"  ---- c={c_tag} C={C}  floor {fl:.3f}"
                         f"{'  DEGENERATE (excluded from ratios/criteria)' if degen else ''}")
                L.append("      " + "  ".join(f"{a}={A[a]:.3f}" for a in ARMS))
                for x, u in PAIRS:
                    d = delta_u(idx, x, u, f"{tag}|{c_tag}|{C}|{x}")
                    cell[f"dU|{x}"] = d
                    L.append(f"      dU {u:16s} - {x:14s} {fmt(d)}"
                             f"{'' if not degen else '  (degenerate)'}")
                    if not degen:
                        dU_by_c[tag][c_tag].append(d[0])
                        if c_tag == 40 and tag == "M2" and C in (128, 256, 512) and x in (
                                "snapkv", "adakv_snapkv"):
                            p1.append((C, x, d))
                if not degen:
                    for x, u in PAIRS:
                        rho_x, rho_u = A[x] / fl, A[u] / fl
                        rho_rows.append((tag, c_tag, C, x, rho_x, rho_u))
                        if c_tag >= 19:
                            k3_tot += 1
                            k3_ge += A[u] >= fl
                    k5_tot += 1
                    k5_bad += (A["oracle_causal"] < fl) and c_tag >= 8
                    for a in ARMS:
                        v = {k: (A[a] / q[a][k] if q[a][k] > 0 else None)
                             for k in ("q_slot", "q_any", "q_maj")}
                        conv_rows.append((tag, c_tag, C, a, v))
                        cell[f"conv|{a}"] = v
                    L.append("      conversion (V_slot / V_any / V_maj), no default measure:")
                    for a in ARMS:
                        v = cell[f"conv|{a}"]
                        L.append("        %-16s %s  %s  %s" % (
                            a, *["n/a" if v[k] is None else "%.2f" % v[k]
                                 for k in ("q_slot", "q_any", "q_maj")]))
                res["cells"][f"{tag}|{c_tag}|{C}"] = cell
        L.append("")

    L.append("==================== KILL CRITERIA (prereg section 7, K1 per amendment A3.1)")
    idp = OUT / "stage3_preflight_identity.json"
    if idp.exists():
        ir = json.loads(idp.read_text(encoding="utf-8"))["rows"]
        n_ok = sum(1 for r in ir if r["identity_ok"])
        L.append(f"  K1 singleton-unit identity (A3.1): {n_ok}/{len(ir)} checks pass"
                 + ("" if n_ok == len(ir) else "  -> STOP AND DIAGNOSE"))
    else:
        L.append("  K1 singleton-unit identity (A3.1): preflight evidence missing -- not scored")
    watch = []
    for tag in TAGS:
        for (c_tag, C), idx in load(tag).items():
            if c_tag < 8 or float(np.mean(cell_acc(idx, "floor_pos"))) < DEGEN:
                continue
            for x, u in PAIRS:
                if x in ("expected_attn", "keydiff"):
                    d = delta_u(idx, x, u, f"{tag}|{c_tag}|{C}|{x}")
                    if d[1] > 0:
                        watch.append(f"{tag} c={c_tag} C={C} {u}-{x} {fmt(d)}")
    L.append("  A3.4 confound watch (KeyDiff/ExpectedAttention, c>=8, CI excluding zero positive): "
             + ("; ".join(watch) if watch else "none"))
    k2 = {}
    for tag in TAGS:
        d40 = dU_by_c[tag].get(40, [])
        d1 = dU_by_c[tag].get(1, [])
        k2[tag] = (float(np.mean(d40)) if d40 else None, float(np.mean(d1)) if d1 else None)
        L.append(f"  K2 {tag}: mean dU at c=40 {k2[tag][0]} vs c=1 {k2[tag][1]}"
                 + ("" if None in k2[tag] else
                    ("  -> grows with c" if k2[tag][0] > k2[tag][1] else "  -> KILL FIRES")))
    L.append(f"  K3 sufficiency: U-X >= floor in {k3_ge}/{k3_tot} admissible c>=19 cells"
             + ("" if k3_tot == 0 else
                ("  -> KILL FIRES (majority)" if k3_ge > k3_tot / 2 else "  -> holds")))
    L.append(f"  K5 ceiling sanity: oracle_causal < floor in {k5_bad}/{k5_tot} admissible c>=8 "
             f"cells" + ("  -> KILL FIRES (>25%)" if k5_tot and k5_bad > 0.25 * k5_tot else "  -> ok"))

    L.append("")
    L.append("==================== PREDICTIONS (prereg section 8)")
    n_p1 = sum(1 for _, _, d in p1 if d[1] > 0)
    L.append(f"  P1 (M2 c=40, C in 128/256/512, snapkv+adakv, >=4 of 6 CIs > 0): {n_p1}/{len(p1)}"
             f" -> {'MET' if n_p1 >= 4 else 'NOT MET' if len(p1) == 6 else 'pending'}")
    for tag in TAGS:
        seq = {c: (float(np.mean(v)) if v else None) for c, v in sorted(dU_by_c[tag].items())}
        L.append(f"  P2 {tag} mean dU by c: " + "  ".join(
            f"c={c}:{'n/a' if v is None else '%+.3f' % v}" for c, v in seq.items()))
    if rho_rows:
        hi = [(r[4], r[5]) for r in rho_rows if r[1] >= 19]
        if hi:
            better = sum(1 for x, u in hi if u > x)
            under = sum(1 for _, u in hi if u < 1)
            L.append(f"  P3 rho_floor(U-X) > rho_floor(X) in {better}/{len(hi)}; "
                     f"rho_floor(U-X) < 1 in {under}/{len(hi)} (admissible c>=19)")
    hiconv = [(t, c, C, a, v) for t, c, C, a, v in conv_rows if c >= 19]
    if hiconv:
        byc = defaultdict(dict)
        for t, c, C, a, v in hiconv:
            byc[(t, c, C)][a] = v
        worse = tot = 0
        for k, d in byc.items():
            f_v = d.get("floor_pos", {}).get("q_slot")
            for _, u in PAIRS:
                uv = d.get(u, {}).get("q_slot")
                if f_v and uv:
                    tot += 1
                    worse += uv < f_v
        L.append(f"  P4 V_slot(U-X) < V_slot(floor) in {worse}/{tot} admissible c>=19 arm-cells "
                 "(V_any/V_maj reported per cell above; no measure designated 'the' usability one)")
    res["K2"] = k2
    (OUT / "stage3_report.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "stage3_report.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\n".join(L))
    return 0


def control_quiet():
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = control()
    return rc


if __name__ == "__main__":
    raise SystemExit(control() if sys.argv[1] == "control" else report())
