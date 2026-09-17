"""Diagnostic D, scored exactly as DISPERSION_RULE.md registers it. CPU only."""
from __future__ import annotations

import json
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "nvidia"
N_SINK, N_WINDOW, W = 8, 64, 32
METHODS = ("snapkv", "adakv_snapkv", "U-snapkv", "U-adakv_snapkv")
ARMS = ("floor_pos",) + METHODS
R_BOOT = 10000


def features(tag):
    meta = json.loads((RUNS / f"p4_keepsets_c40_{tag}.meta.json").read_text(encoding="utf-8"))
    if meta["mismatches"]:
        raise SystemExit(f"{tag}: recaptured keep-sets do not reproduce the pilot -- NOT SCORED")
    z = np.load(RUNS / f"p4_keepsets_c40_{tag}.npz")
    rows = []
    for si, m in meta["instances"].items():
        i, n = int(si), m["n_ctx"]
        lo, hi = N_SINK, n - N_WINDOW
        for arm in ARMS:
            bm = np.unpackbits(z[f"{i}|{arm}"], axis=1, count=n).astype(bool)
            reg = bm[:, lo:hi]
            starts = reg & ~np.concatenate([np.zeros((reg.shape[0], 1), bool), reg[:, :-1]], axis=1)
            d_runs = float(starts.sum(1).mean())
            for q, (toks, corr) in enumerate(zip(m["facts"], m["per_variant"][arm])):
                t = np.asarray(toks)
                complete = bm[:, t].all(1)
                a, b = t.min(), t.max()
                nb = [x for x in list(range(a - W, a)) + list(range(b + 1, b + 1 + W)) if lo <= x < hi]
                d_iso = (float(1 - bm[complete][:, nb].mean()) if complete.any() and nb else None)
                rows.append(dict(instance=i, arm=arm, q=q, correct=float(corr),
                                 q_fact=float(complete.mean()), D_runs=d_runs, D_iso=d_iso))
    return rows


def ols_coef(y, X):
    return np.linalg.lstsq(X, y, rcond=None)[0][-1]


def design(rs, measure, mu, sd):
    X = np.array([[1.0] + [1.0 * (r["arm"] == a) for a in METHODS[1:]] +
                  [r["q_fact"], (r[measure] - mu) / sd] for r in rs])
    return np.array([r["correct"] for r in rs]), X


def test(rows, tag, measure):
    rs = [r for r in rows if r["arm"] in METHODS and r[measure] is not None]
    n_correct = sum(r["correct"] for r in rs)
    vals = np.array([r[measure] for r in rs])
    mu, sd = vals.mean(), vals.std()
    out = dict(measure=measure, n_queries=len(rs), n_correct=n_correct)
    if n_correct < 20:
        return dict(out, verdict="NOT SCORED (underpowered: <20 correct)")
    if sd == 0:
        return dict(out, verdict="NOT SCORED (no variation)")
    r_col = float(np.corrcoef((vals - mu) / sd, [r["q_fact"] for r in rs])[0, 1])
    out["r_D_qfact"] = r_col
    if abs(r_col) > 0.8:
        return dict(out, verdict="NOT SEPARABLE (|r| > 0.8)")
    y, X = design(rs, measure, mu, sd)
    beta = float(ols_coef(y, X))
    by_inst = defaultdict(list)
    for k, r in enumerate(rs):
        by_inst[r["instance"]].append(k)
    insts = sorted(by_inst)
    rng = np.random.default_rng(zlib.crc32(f"p4|disp|{tag}|{measure}".encode()) & 0xFFFFFFFF)
    bs = []
    for _ in range(R_BOOT):
        idx = np.concatenate([by_inst[insts[j]] for j in rng.integers(0, len(insts), len(insts))])
        bs.append(ols_coef(y[idx], X[idx]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    verdict = "SUPPORTED" if hi < 0 else "CONTRADICTED" if lo > 0 else "NOT SUPPORTED"
    return dict(out, beta_per_sd=beta, ci=[float(lo), float(hi)], verdict=verdict)


def main():
    L = ["Diagnostic D (DISPERSION_RULE.md). Produced on RTX 5070 (Machine N). c~40, C=512, n=50.", ""]
    res = {}
    for tag in ("M2", "M3"):
        rows = features(tag)
        L.append(f"== {tag}")
        L.append("  descriptive (confounded across arms, not a test):")
        L.append("    %-16s %8s %8s %8s %8s %8s" % ("arm", "D_runs", "D_iso", "q_fact", "acc", "conv"))
        for arm in ARMS:
            rr = [r for r in rows if r["arm"] == arm]
            iso = [r["D_iso"] for r in rr if r["D_iso"] is not None]
            qf, acc = np.mean([r["q_fact"] for r in rr]), np.mean([r["correct"] for r in rr])
            L.append("    %-16s %8.1f %8s %8.3f %8.3f %8s" % (
                arm, np.mean([r["D_runs"] for r in rr]), "%.3f" % np.mean(iso) if iso else "n/a",
                qf, acc, "%.2f" % (acc / qf) if qf else "n/a"))
        res[tag] = {}
        for measure in ("D_runs", "D_iso"):
            t = test(rows, tag, measure)
            res[tag][measure] = t
            extra = ("  beta/SD %+.4f [%+.4f, %+.4f]  r(D,q_fact) %+.2f" % (
                t["beta_per_sd"], *t["ci"], t["r_D_qfact"]) if "ci" in t else "")
            L.append(f"  {measure:7s} queries {t['n_queries']} correct {t['n_correct']:.0f}{extra}  -> {t['verdict']}")
        L.append("")
    mech = all(res[t]["D_runs"]["verdict"] == "SUPPORTED" for t in res)
    L.append("MECHANISM CLAIM (D_runs SUPPORTED on both models): %s" % (
        "SUPPORTED" if mech else "NOT SUPPORTED -- recorded as a failed hypothesis"))
    (HERE / "out" / "stage2_dispersion.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    (HERE / "out" / "stage2_dispersion.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
