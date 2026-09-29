"""Cheap mechanism tests C2, C1, C3, C4 -- scored exactly as MECHANISM_RULES.md (commit bf05e1b).

CPU only. Reads rows and saved keep-sets already on disk; runs no model.
"""
from __future__ import annotations

import json
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from p4 import common as CM

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "nvidia"
OUT = HERE / "out"
N, R_BOOT = 50, 10000
NS, NW = CM.N_SINK, CM.N_WINDOW
PAIRS = (("snapkv", "U-snapkv"), ("adakv_snapkv", "U-adakv_snapkv"),
         ("expected_attn", "U-expected_attn"), ("keydiff", "U-keydiff"))
METHOD_ARMS = [a for p in PAIRS for a in p]
JACCARD_SLOTS = 16          # subsample, seeded -- full pairwise is O(S^2) with S up to 224
PREAMBLE_MARK = {"ledger_c": "INTERNAL LEDGER EXTRACT", "mark1": "INVENTORY TAG LIST"}


def boot_coef(y, X, groups, key):
    rng = np.random.default_rng(zlib.crc32(key.encode()) & 0xFFFFFFFF)
    idx_by_g = defaultdict(list)
    for k, g in enumerate(groups):
        idx_by_g[g].append(k)
    gs = sorted(idx_by_g)
    b0 = float(np.linalg.lstsq(X, y, rcond=None)[0][-1])
    out = []
    for _ in range(R_BOOT):
        sel = np.concatenate([idx_by_g[gs[j]] for j in rng.integers(0, len(gs), len(gs))])
        out.append(np.linalg.lstsq(X[sel], y[sel], rcond=None)[0][-1])
    return b0, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def cells_on_disk():
    """(tag, c_tag, C, source) for every complete cell with keep-sets saved."""
    out = []
    for tag in ("M2", "M3"):
        p = RUNS / f"p4_s3_{tag}.jsonl"
        if p.exists():
            seen = defaultdict(int)
            for line in p.open(encoding="utf-8"):
                r = json.loads(line)
                seen[(r["c_tag"], r["C"])] += 1
            for (c_tag, C), n in sorted(seen.items()):
                if (RUNS / f"p4_s3_keepsets_{tag}_c{c_tag}_C{C}.npz").exists():
                    out.append((tag, c_tag, C, "s3"))
        if (RUNS / f"p4_keepsets_c40_{tag}.npz").exists():
            out.append((tag, 40, 512, "s2"))
    return out


def load_rows(tag, source, c_tag, C):
    f = RUNS / (f"p4_s3_{tag}.jsonl" if source == "s3" else f"p4_pilot_{tag}.jsonl")
    rows = {}
    for line in f.open(encoding="utf-8"):
        r = json.loads(line)
        if source == "s3":
            if r["c_tag"] != c_tag or r["C"] != C:
                continue
        else:
            if r["label"] != "c=40":
                continue
            r["q_slot"] = r["q_complete"]
            r["q_maj"] = None          # Stage 2 rows predate q_maj; never imputed
        rows[(r["arm"], r["instance"])] = r
    return rows


def measures(tag, model, rev, tok, c_tag, C, source, arms, rows):
    """Per (arm, instance): agree_jaccard, dist_gold, exemplar_frac."""
    npz = RUNS / (f"p4_s3_keepsets_{tag}_c{c_tag}_C{C}.npz" if source == "s3"
                  else f"p4_keepsets_c40_{tag}.npz")
    z = np.load(npz)
    rng = np.random.default_rng(zlib.crc32(f"jac|{tag}|{c_tag}|{C}".encode()) & 0xFFFFFFFF)
    out = {}
    for i in range(N):
        b = CM.build(tag, model, rev, tok, c_tag, "s4_%05d" % i)
        n = b["n_ctx"]
        gold = np.zeros(n, bool)
        for fa in b["facts"]:
            for s in fa.spans:
                gold[list(s)] = True
        pre_off = b["pre"].index(b["inst"].context) + b["inst"].context.index(
            PREAMBLE_MARK[b["sp"]["task"]])
        enc = tok(b["pre"], add_special_tokens=False, return_offsets_mapping=True)
        ex = np.zeros(n, bool)
        for ti, (x0, y0) in enumerate(enc["offset_mapping"]):
            if y0 > x0 and y0 <= pre_off:
                ex[ti] = True
        ex_tok = np.flatnonzero(ex)
        gold_idx = np.flatnonzero(gold)
        for arm in arms:
            key = f"{i}|{arm}"
            if key not in z:
                continue
            bm = np.unpackbits(z[key], axis=1, count=n).astype(bool)
            S = bm.shape[0]
            sel = rng.choice(S, min(JACCARD_SLOTS, S), replace=False)
            sub = bm[sel]
            jac = []
            for a in range(len(sel)):
                for c in range(a + 1, len(sel)):
                    inter = np.count_nonzero(sub[a] & sub[c])
                    union = np.count_nonzero(sub[a] | sub[c])
                    jac.append(inter / union if union else 1.0)
            kept_gold = bm[:, gold_idx]
            d = ((n - gold_idx)[None, :] * kept_gold).sum(1) / np.maximum(1, kept_gold.sum(1))
            out[(arm, i)] = dict(
                agree_jaccard=float(np.mean(jac)) if jac else 1.0,
                dist_gold=float(np.mean(d)) / n,
                exemplar_frac=float(bm[:, ex_tok].mean()) if ex_tok.size else 0.0)
    return out


def main():
    L = ["Mechanism tests C2, C1, C3, C4 -- rules frozen at MECHANISM_RULES.md (commit bf05e1b).",
         f"CPU only, produced on {CM.PRODUCED_ON}.", ""]
    cells = cells_on_disk()
    data = defaultdict(list)          # tag -> list of per-(cell,arm,instance) dicts
    conv = []                          # C2
    toks = {}
    for tag, c_tag, C, source in cells:
        model = [m for m, t in CM.TAGS.items() if t == tag][0]
        rows = load_rows(tag, source, c_tag, C)
        if not rows:
            continue
        floor = np.mean([rows[("floor_pos", i)]["score"] for i in range(N)])
        arms = sorted({a for a, _ in rows})
        if floor < 0.05:
            L.append(f"  (skip {tag} c={c_tag} C={C}: floor {floor:.3f} degenerate)")
            continue
        # ---- C2: conversion under three denominators
        for x, u in PAIRS:
            if (u, 0) not in rows:
                continue
            e = {}
            for arm in ("floor_pos", x, u):
                A = np.mean([rows[(arm, i)]["score"] for i in range(N)])
                e[arm] = {}
                for q in ("q_slot", "q_any", "q_maj"):
                    vals = [rows[(arm, i)].get(q) for i in range(N)]
                    if any(v is None for v in vals):
                        e[arm][q] = None
                        continue
                    d = float(np.mean(vals))
                    e[arm][q] = (A / d) if d > 0 else None
            conv.append((tag, c_tag, C, x, e))
        # ---- measures for C1/C3/C4
        if tag not in toks:
            toks[tag] = AutoTokenizer.from_pretrained(model)
        rev = rows[("floor_pos", 0)]["key"]["model_revision"]
        want = [a for a in arms if a in METHOD_ARMS] + ["floor_pos"]
        m = measures(tag, model, rev, toks[tag], c_tag, C, source, want, rows)
        for (arm, i), mv in m.items():
            data[tag].append(dict(cell=(c_tag, C), arm=arm, inst=i,
                                  acc=rows[(arm, i)]["score"], q_slot=rows[(arm, i)]["q_slot"],
                                  p_g=rows[(arm, i)]["p_g"], **mv))

    # ================= C2
    L.append("== C2 denominator artifact: conversion gap floor_pos - U-X, per denominator")
    L.append("   %-5s %-4s %-5s %-14s %10s %10s %10s" % ("model", "c", "C", "pair", "V_slot gap",
                                                         "V_any gap", "V_maj gap"))
    gaps = defaultdict(list)
    for tag, c_tag, C, x, e in conv:
        u = "U-" + x
        g = {}
        for q in ("q_slot", "q_any", "q_maj"):
            a, b = e["floor_pos"][q], e[u][q]
            g[q] = (a - b) if (a is not None and b is not None) else None
            if g[q] is not None:
                gaps[(tag, q)].append(g[q])
        L.append("   %-5s %-4d %-5d %-14s %10s %10s %10s" % (
            tag, c_tag, C, x, *["n/a" if g[q] is None else "%+.3f" % g[q]
                                for q in ("q_slot", "q_any", "q_maj")]))
    verdict = {}
    for tag in ("M2", "M3"):
        mj = gaps.get((tag, "q_maj"), [])
        verdict[tag] = float(np.mean(mj)) if mj else None
        L.append(f"   {tag} mean gap: " + "  ".join(
            "%s %s" % (q, "n/a" if not gaps.get((tag, q)) else "%+.3f" % np.mean(gaps[(tag, q)]))
            for q in ("q_slot", "q_any", "q_maj")))
    both = [verdict[t] for t in ("M2", "M3") if verdict[t] is not None]
    c2 = ("REAL" if both and all(abs(v) < 0.05 or v < 0 for v in both) and len(both) == 2
          else "NOT IT" if both and all(v > 0.05 for v in both) else "AMBIGUOUS")
    L.append(f"   C2 VERDICT: {c2}")
    L.append("")

    # ================= C1, C3, C4 regressions
    for cand, meas, sign in (("C1", "agree_jaccard", +1), ("C3", "dist_gold", -1),
                             ("C4", "exemplar_frac", +1)):
        L.append(f"== {cand}: {meas}")
        verd = {}
        for tag in ("M2", "M3"):
            rs = [r for r in data[tag] if r["arm"] in METHOD_ARMS]
            if not rs:
                L.append(f"   {tag}: no data")
                continue
            v = np.array([r[meas] for r in rs])
            ncorrect = sum(r["acc"] * 4 for r in rs)
            if v.std() == 0:
                L.append(f"   {tag}: NOT SCORED (no variation in {meas})")
                verd[tag] = "NOT SCORED"
                continue
            if ncorrect < 20:
                L.append(f"   {tag}: NOT SCORED (underpowered, {ncorrect:.0f} correct)")
                verd[tag] = "NOT SCORED"
                continue
            z = (v - v.mean()) / v.std()
            qs = np.array([r["q_slot"] for r in rs])
            rcol = float(np.corrcoef(z, qs)[0, 1])
            if abs(rcol) > 0.8:
                L.append(f"   {tag}: NOT SEPARABLE (|r(z,q_slot)| = {abs(rcol):.2f})")
                verd[tag] = "NOT SEPARABLE"
                continue
            arms_u = sorted({r["arm"] for r in rs})[1:]
            X = np.column_stack([np.ones(len(rs))]
                                + [[1.0 * (r["arm"] == a) for r in rs] for a in arms_u]
                                + [qs, z])
            y = np.array([r["acc"] for r in rs])
            if cand == "C3":     # rule: must survive controlling p_g too
                X = np.column_stack([X[:, :-1], [r["p_g"] for r in rs], z])
            b, lo, hi = boot_coef(y, X, [r["inst"] for r in rs], f"{cand}|{tag}")
            ok = (lo > 0) if sign > 0 else (hi < 0)
            bad = (hi < 0) if sign > 0 else (lo > 0)
            verd[tag] = "supports" if ok else ("opposite sign" if bad else "null")
            L.append("   %s n=%d rows  beta/SD %+.4f [%+.4f, %+.4f]  r(z,q_slot) %+.2f  -> %s"
                     % (tag, len(rs), b, lo, hi, rcol, verd[tag]))
        # descriptive across-arm ordering
        if cand == "C1":
            for tag in ("M2", "M3"):
                rs = data[tag]
                if not rs:
                    continue
                by = defaultdict(list)
                for r in rs:
                    by[r["arm"]].append(r[meas])
                L.append("   %s agreement by arm: " % tag + "  ".join(
                    "%s %.3f" % (a, np.mean(v)) for a, v in sorted(by.items(),
                                                                   key=lambda kv: -np.mean(kv[1]))))
        allv = [verd.get(t) for t in ("M2", "M3")]
        final = ("REAL" if allv.count("supports") == 2 else
                 "NOT IT" if all(v in ("null", "opposite sign") for v in allv if v) else
                 "AMBIGUOUS/NOT SCORED")
        L.append(f"   {cand} VERDICT: {final}")
        L.append("")

    OUT.mkdir(exist_ok=True)
    (OUT / "mechanism_tests.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
