"""CPU-only tier-1 candidates A7, A8, A9 (CANDIDATES.md, commit 3156e6c). No GPU."""
from __future__ import annotations

import json
import re
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
NUM = re.compile(r"\d")
K_GATE = 4          # fixed before scoring, per the rule


def s3_rows(tag):
    p = RUNS / f"p4_s3_{tag}.jsonl"
    rows = defaultdict(dict)
    if p.exists():
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            rows[(r["c_tag"], r["C"], r["arm"])][r["instance"]] = r
    return rows


def boot_coef(y, X, groups, key):
    rng = np.random.default_rng(zlib.crc32(key.encode()) & 0xFFFFFFFF)
    by = defaultdict(list)
    for k, g in enumerate(groups):
        by[g].append(k)
    gs = sorted(by)
    b0 = float(np.linalg.lstsq(X, y, rcond=None)[0][-1])
    bs = [float(np.linalg.lstsq(X[np.concatenate([by[gs[j]] for j in
                                                  rng.integers(0, len(gs), len(gs))])],
                                y[np.concatenate([by[gs[j]] for j in
                                                  rng.integers(0, len(gs), len(gs))])],
                                rcond=None)[0][-1]) for _ in range(200)]
    rng2 = np.random.default_rng(zlib.crc32((key + "|full").encode()) & 0xFFFFFFFF)
    bs = []
    for _ in range(R_BOOT):
        sel = np.concatenate([by[gs[j]] for j in rng2.integers(0, len(gs), len(gs))])
        bs.append(float(np.linalg.lstsq(X[sel], y[sel], rcond=None)[0][-1]))
    return b0, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def a7(L):
    L.append("== A7 budget-gated unit policy (k = 4, fixed before scoring)")
    L.append("   RULE DEFECT, reported not rewritten: the gated policy selects EITHER floor_pos OR")
    L.append("   U-snapkv in each cell, so its per-cell accuracy EQUALS one of them. 'Exceeds both")
    L.append("   in a majority of cells' is therefore unsatisfiable by construction.")
    L.append("   VERDICT: NOT SCORED (rule defect). Descriptive result below.")
    for tag in ("M2", "M3"):
        rows = s3_rows(tag)
        cells = sorted({(c, C) for (c, C, a) in rows if a == "floor_pos"})
        gate, fl_all, u_all, picked = [], [], [], []
        for c, C in cells:
            if ("U-snapkv") not in {a for (cc, CC, a) in rows if (cc, CC) == (c, C)}:
                continue
            fl = np.mean([rows[(c, C, "floor_pos")][i]["score"] for i in range(N)])
            if fl < 0.05:
                continue
            u = np.mean([rows[(c, C, "U-snapkv")][i]["score"] for i in range(N)])
            uc = np.mean([rows[(c, C, "U-snapkv")][i]["units_complete"] for i in range(N)])
            g = u if uc >= K_GATE else fl
            gate.append(g); fl_all.append(fl); u_all.append(u)
            picked.append("U" if uc >= K_GATE else "floor")
        if gate:
            better = sum(1 for g, f, u in zip(gate, fl_all, u_all) if g >= max(f, u))
            L.append(f"   {tag}: {len(gate)} admissible cells; gated mean {np.mean(gate):.3f} vs "
                     f"floor {np.mean(fl_all):.3f} vs U-snapkv {np.mean(u_all):.3f}; "
                     f"gate picked the better arm in {better}/{len(gate)} cells; picks {picked}")
    L.append("")


def a8_a9(L):
    tok_cache = {}
    data = defaultdict(list)
    layer_var = {}
    for tag in ("M2", "M3"):
        model = [m for m, t in CM.TAGS.items() if t == tag][0]
        rows = s3_rows(tag)
        cells = []
        for (c, C, a) in rows:
            if a == "floor_pos" and (RUNS / f"p4_s3_keepsets_{tag}_c{c}_C{C}.npz").exists():
                fl = np.mean([rows[(c, C, "floor_pos")][i]["score"] for i in range(N)])
                if fl >= 0.05:
                    cells.append((c, C))
        cells = sorted(set(cells))
        if not cells:
            continue
        if tag not in tok_cache:
            tok_cache[tag] = AutoTokenizer.from_pretrained(model)
        tok = tok_cache[tag]
        rev = rows[(cells[0][0], cells[0][1], "floor_pos")][0]["key"]["model_revision"]
        per_layer = []
        for c, C in cells:
            z = np.load(RUNS / f"p4_s3_keepsets_{tag}_c{c}_C{C}.npz")
            arms = [a for (cc, CC, a) in rows if (cc, CC) == (c, C)
                    and a not in ("floor_pos", "oracle_causal")]
            for i in range(N):
                b = CM.build(tag, model, rev, tok, c, "s4_%05d" % i)
                n = b["n_ctx"]
                enc = tok(b["pre"], add_special_tokens=False, return_offsets_mapping=True)
                gold_idx = sorted({t for f in b["facts"] for s in f.spans for t in s})
                isnum = np.array([bool(NUM.search(b["pre"][x:y]))
                                  for (x, y) in [enc["offset_mapping"][t] for t in gold_idx]])
                gi = np.array(gold_idx)
                for arm in arms:
                    key = f"{i}|{arm}"
                    if key not in z:
                        continue
                    bm = np.unpackbits(z[key], axis=1, count=n).astype(bool)
                    kept = bm[:, gi]
                    frac = (kept & isnum[None, :]).sum(1) / np.maximum(1, kept.sum(1))
                    r = rows[(c, C, arm)][i]
                    data[tag].append(dict(acc=r["score"], q_slot=r["q_slot"], arm=arm, inst=i,
                                          num_frac=float(frac.mean())))
                    if arm == "U-snapkv":
                        S = bm.shape[0]
                        heads = max(1, S // 36) if tag == "M2" else max(1, S // 28)
                        nl = S // max(1, heads) if heads else S
                        comp = []
                        for li in range(0, S, max(1, S // (36 if tag == "M2" else 28))):
                            sl = bm[li:li + max(1, S // (36 if tag == "M2" else 28))]
                            comp.append(np.mean([
                                np.mean([all(sl[s, list(sp)].all() for sp in f.spans)
                                         for f in b["facts"]]) for s in range(sl.shape[0])]))
                        per_layer.append(comp)
        if per_layer:
            m = np.array([p for p in per_layer if len(p) == len(per_layer[0])])
            if m.size:
                lay = m.mean(0)
                q = np.array_split(np.argsort(lay), 4)
                best, worst = lay[q[-1]].mean(), lay[q[0]].mean()
                layer_var[tag] = (best, worst, (best / worst) if worst > 0 else float("inf"))

    L.append("== A8 unit content composition (numeric fraction of retained gold)")
    verd8 = {}
    for tag in ("M2", "M3"):
        rs = data.get(tag, [])
        if not rs:
            L.append(f"   {tag}: no data")
            continue
        v = np.array([r["num_frac"] for r in rs])
        if v.std() == 0:
            L.append(f"   {tag}: NOT SCORED (no variation)")
            verd8[tag] = "not scored"
            continue
        zv = (v - v.mean()) / v.std()
        qs = np.array([r["q_slot"] for r in rs])
        rcol = float(np.corrcoef(zv, qs)[0, 1])
        arms_u = sorted({r["arm"] for r in rs})[1:]
        X = np.column_stack([np.ones(len(rs))]
                            + [[1.0 * (r["arm"] == a) for r in rs] for a in arms_u] + [qs, zv])
        y = np.array([r["acc"] for r in rs])
        if abs(rcol) > 0.8:
            L.append(f"   {tag}: NOT SEPARABLE (|r| = {abs(rcol):.2f})")
            verd8[tag] = "not separable"
            continue
        b, lo, hi = boot_coef(y, X, [r["inst"] for r in rs], f"A8|{tag}")
        verd8[tag] = "supports" if lo > 0 else ("opposite" if hi < 0 else "null")
        L.append(f"   {tag}: n={len(rs)} beta/SD {b:+.4f} [{lo:+.4f}, {hi:+.4f}] "
                 f"r(z,q_slot) {rcol:+.2f} -> {verd8[tag]}")
    L.append(f"   A8 VERDICT: {'REAL' if list(verd8.values()).count('supports') == 2 else 'NOT IT'}"
             + (" (both models required)" if len(verd8) == 2 else " -- one model only"))
    L.append("")
    L.append("== A9 layer-selective precondition (U-snapkv per-layer completion spread)")
    for tag, (best, worst, ratio) in layer_var.items():
        L.append(f"   {tag}: best-quartile layer completion {best:.3f}, worst {worst:.3f}, "
                 f"ratio {ratio:.2f}x")
    ok = len(layer_var) == 2 and all(v[2] > 2.0 for v in layer_var.values())
    L.append(f"   A9 VERDICT: {'REAL (a layer-selective method could exist)' if ok else 'NOT IT'}"
             + ("" if len(layer_var) == 2 else " -- fewer than two models available"))
    L.append("")


def main():
    L = ["CPU-only tier-1 candidates A7, A8, A9 -- CANDIDATES.md (commit 3156e6c). No GPU.", ""]
    a7(L)
    a8_a9(L)
    OUT.mkdir(exist_ok=True)
    (OUT / "cpu_probes.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
