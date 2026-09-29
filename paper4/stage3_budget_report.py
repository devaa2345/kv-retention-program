"""Per-cost-level report, split by what the keep-set evidence supports (see DIAGNOSIS_EA.md).

    python stage3_budget_report.py --c 19

PRIMARY (SnapKV, AdaKV): Delta_U with the keep-set evidence that content is held fixed.
REPORTED-WITH-CONFOUND (ExpectedAttention, KeyDiff): Delta_U is NEVER shown alone -- always with
d p_g and d units_complete, because gold retention shifts alongside packing for these two.
"""
from __future__ import annotations

import argparse
import json
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from p4 import common as CM

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "nvidia"
N, R_BOOT = 50, 10000
BUDGETS = (32, 64, 128, 256, 512)
CLEAN = (("snapkv", "U-snapkv"), ("adakv_snapkv", "U-adakv_snapkv"))
CONFOUNDED = (("expected_attn", "U-expected_attn"), ("keydiff", "U-keydiff"))
NS, NW = CM.N_SINK, CM.N_WINDOW
KEEPSET_INSTANCES = 5


def boot(d, key):
    d = np.asarray(d, float)
    rng = np.random.default_rng(zlib.crc32(key.encode()) & 0xFFFFFFFF)
    m = d[rng.integers(0, len(d), size=(R_BOOT, len(d)))].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def fmt(t):
    return "%+.3f [%+.3f, %+.3f]" % t


def load(tag):
    p = RUNS / f"p4_s3_{tag}.jsonl"
    rows = defaultdict(dict)
    if p.exists():
        with p.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows[(r["c_tag"], r["C"], r["arm"])][r["instance"]] = r
    return rows


def keepset_evidence(tag, model, rev, tok, c_tag, C, x, u):
    """Mean tokens per slot by destination -- is content held fixed, or shifted?"""
    f = RUNS / f"p4_s3_keepsets_{tag}_c{c_tag}_C{C}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    agg = defaultdict(list)
    for i in range(KEEPSET_INSTANCES):
        b = CM.build(tag, model, rev, tok, c_tag, "s4_%05d" % i)
        n = b["n_ctx"]
        floor = np.zeros(n, bool)
        floor[:NS] = True
        floor[n - NW:] = True
        gold = np.zeros(n, bool)
        for fa in b["facts"]:
            for s in fa.spans:
                gold[list(s)] = True
        rec = np.zeros(n, bool)
        for s in b["line_units"].values():
            rec[list(s)] = True
        bx = np.unpackbits(z[f"{i}|{x}"], axis=1, count=n).astype(bool)
        bu = np.unpackbits(z[f"{i}|{u}"], axis=1, count=n).astype(bool)
        reg = ~floor
        for s in range(bx.shape[0]):
            kx, ku = bx[s] & reg, bu[s] & reg
            agg["ov"].append((kx & ku).sum() / max(1, kx.sum()))
            agg["gx"].append((kx & gold).sum())
            agg["gu"].append((ku & gold).sum())
            agg["rx"].append((kx & rec).sum())
            agg["ru"].append((ku & rec).sum())
            agg["ox"].append((kx & ~rec).sum())
            agg["ou"].append((ku & ~rec).sum())
    return {k: float(np.mean(v)) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=int, required=True)
    ap.add_argument("--keepsets", action="store_true", default=True)
    a = ap.parse_args()
    L = [f"Paper 4 Stage 3 -- cost level c={a.c}. Produced on {CM.PRODUCED_ON}.",
         "Split follows DIAGNOSIS_EA.md: SnapKV/AdaKV carry the fragmentation claim; "
         "ExpectedAttention/KeyDiff are reported with their confound, never Delta_U alone.", ""]
    corr = defaultdict(list)
    for model, tag in CM.TAGS.items():
        rows = load(tag)
        cells = [C for C in BUDGETS if (a.c, C, "floor_pos") in rows
                 and len(rows[(a.c, C, "floor_pos")]) == N]
        if not cells:
            continue
        tok = AutoTokenizer.from_pretrained(model)
        rev = rows[(a.c, cells[0], "floor_pos")][0]["key"]["model_revision"]
        adm, deg = [], []
        for C in cells:
            fl = float(np.mean([rows[(a.c, C, "floor_pos")][i]["score"] for i in range(N)]))
            (deg if fl < 0.05 else adm).append((C, fl))
        L.append(f"== {tag}")
        L.append("  admissible: " + (", ".join(f"C={C} (floor {f:.3f})" for C, f in adm) or "none"))
        L.append("  degenerate: " + (", ".join(f"C={C} (floor {f:.3f})" for C, f in deg) or "none"))

        L.append("  -- PRIMARY: fragmentation claim (SnapKV, AdaKV). Keep-set evidence per slot, "
                 "instances 0-4:")
        L.append("     %-4s %-14s %22s %8s %14s %14s %14s %12s" % (
            "C", "method", "dU [95% CI]", "overlap", "gold X->U", "record X->U", "other X->U",
            "unitsC X->U"))
        for C, fl in adm:
            for x, u in CLEAN:
                d = boot([rows[(a.c, C, u)][i]["score"] - rows[(a.c, C, x)][i]["score"]
                          for i in range(N)], f"p4|s3|{tag}|{a.c}|{C}|{x}")
                ucx = np.mean([rows[(a.c, C, x)][i]["units_complete"] for i in range(N)])
                ucu = np.mean([rows[(a.c, C, u)][i]["units_complete"] for i in range(N)])
                dpg = np.mean([rows[(a.c, C, u)][i]["p_g"] - rows[(a.c, C, x)][i]["p_g"]
                               for i in range(N)])
                corr[x].append((d[0], dpg))
                k = keepset_evidence(tag, model, rev, tok, a.c, C, x, u) if a.keepsets else None
                ks = ("%8.2f %6.1f->%-6.1f %6.1f->%-6.1f %6.1f->%-6.1f" % (
                    k["ov"], k["gx"], k["gu"], k["rx"], k["ru"], k["ox"], k["ou"])
                    if k else " " * 44)
                L.append("     %-4d %-14s %22s %s %5.1f->%-5.1f" % (
                    C, x, fmt(d), ks, ucx, ucu))
        L.append("  -- REPORTED WITH CONFOUND: ExpectedAttention, KeyDiff. Delta_U is not "
                 "interpretable alone: gold retention shifts alongside packing.")
        L.append("     %-4s %-14s %22s %10s %12s %14s" % (
            "C", "method", "dU [95% CI]", "d p_g", "d unitsC", "gold X->U (slot)"))
        for C, fl in adm:
            for x, u in CONFOUNDED:
                d = boot([rows[(a.c, C, u)][i]["score"] - rows[(a.c, C, x)][i]["score"]
                          for i in range(N)], f"p4|s3|{tag}|{a.c}|{C}|{x}")
                dpg = float(np.mean([rows[(a.c, C, u)][i]["p_g"] - rows[(a.c, C, x)][i]["p_g"]
                                     for i in range(N)]))
                duc = float(np.mean([rows[(a.c, C, u)][i]["units_complete"]
                                     - rows[(a.c, C, x)][i]["units_complete"] for i in range(N)]))
                corr[x].append((d[0], dpg))
                k = keepset_evidence(tag, model, rev, tok, a.c, C, x, u) if a.keepsets else None
                ks = ("%6.1f->%-6.1f" % (k["gx"], k["gu"])) if k else ""
                L.append("     %-4d %-14s %22s %+10.4f %+12.2f %14s" % (C, x, fmt(d), dpg, duc, ks))
        L.append("")

    L.append("== does the c=1/c=8 correlation pattern hold at c=%d?" % a.c)
    L.append("   (Delta_U vs d p_g across this level's admissible cells; c=1/c=8 gave r=+0.944 "
             "for ExpectedAttention, and ~0 d p_g for SnapKV/AdaKV)")
    for m, v in corr.items():
        if len(v) > 2:
            dU = np.array([p[0] for p in v])
            dp = np.array([p[1] for p in v])
            r = float(np.corrcoef(dU, dp)[0, 1]) if dp.std() > 0 else float("nan")
            L.append("   %-16s n=%d  mean d p_g %+0.4f  r(dU, d p_g) %+0.3f" % (
                m, len(v), dp.mean(), r))
        elif v:
            L.append("   %-16s n=%d  mean d p_g %+0.4f  (too few cells for r)" % (
                m, len(v), np.mean([p[1] for p in v])))
    out = HERE / "out" / f"stage3_report_c{a.c}.txt"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
