"""Analysis for the H-ORTH factorial, our half (Paper 1, Option B item 3). No GPU.

Cell A = our tier-seating x our distractor format  (results/phase2_4bit, budget 257, existing)
Cell B = our tier-seating x AMD distractor format  (results/phase2_4bit_amdfmt, this run)
Same contrasts in both cells: each signal vs P1 (attention, circular) and vs P3 (random), paired by seed,
mean difference, 10k prompt-resampled bootstrap (rng seed 0), n = 150, seeds 3000-3149, 4-bit tier.
"""
import json
import collections
import numpy as np

R = "results/"
SIGS = ["P1_attention", "P2_epiphany", "P3_random", "P4_roundrobin", "P5_oracle_future"]


def boot(d, n=10000):
    rng = np.random.default_rng(0)
    d = np.asarray(d, float)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), int((d != 0).sum()), len(d)]


def load(path, budget=257):
    t = collections.defaultdict(dict)
    rho = collections.defaultdict(list)
    ctx = []
    for l in open(path):
        r = json.loads(l)
        if r["budget"] != budget:
            continue
        t[r["signal"]][r["seed"]] = r["frac_retrieved"]
        if r.get("mean_spearman_rho_evict_vs_promote") is not None:
            rho[r["signal"]].append(r["mean_spearman_rho_evict_vs_promote"])
        if "context_tokens" in r:
            ctx.append(r["context_tokens"])
    return t, rho, ctx


def cell(path):
    t, rho, ctx = load(path)
    out = {"n": {s: len(t[s]) for s in t}, "mean": {s: float(np.mean(list(t[s].values()))) for s in t},
           "rho": {s: float(np.mean(rho[s])) for s in rho}, "context_tokens_mean": float(np.mean(ctx)) if ctx else None}
    for base in ("P1_attention", "P3_random"):
        out["vs_" + base] = {}
        for s in SIGS:
            if s == base or s not in t:
                continue
            seeds = sorted(set(t[s]) & set(t[base]))
            out["vs_" + base][s] = boot([t[s][k] - t[base][k] for k in seeds])
    for name, a, b in (("band_full_minus_quant", "BAND_full", "BAND_quant"),
                       ("structural_minus_none", "R2_protect_permanent", "R1_no_protect_permanent"),
                       ("oracle_minus_structural", "R5_oracle_static", "R2_protect_permanent")):
        if a in t and b in t:
            seeds = sorted(set(t[a]) & set(t[b]))
            out[name] = boot([t[a][k] - t[b][k] for k in seeds])
    return out


def main():
    A = cell(R + "phase2_4bit/raw_results.jsonl")
    B = cell(R + "phase2_4bit_amdfmt/raw_results.jsonl")
    json.dump({"A_our_format": A, "B_amd_format": B}, open(R + "phase2_4bit_amdfmt/analysis.json", "w"), indent=2)
    for name, c in (("A: our tier seating x OUR distractor format", A), ("B: our tier seating x AMD distractor format", B)):
        print("\n" + name, "| context tokens", c["context_tokens_mean"])
        print("  mean:", {k: round(v, 3) for k, v in c["mean"].items()})
        print("  rho :", {k: round(v, 3) for k, v in c["rho"].items()})
        for base in ("P1_attention", "P3_random"):
            for s, v in c["vs_" + base].items():
                print(f"  {s:18s} vs {base:12s} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}] differ {v[3]}/{v[4]}")
        for k in ("band_full_minus_quant", "structural_minus_none", "oracle_minus_structural"):
            if k in c:
                v = c[k]
                print(f"  {k:26s} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}] differ {v[3]}/{v[4]}")


if __name__ == "__main__":
    main()
