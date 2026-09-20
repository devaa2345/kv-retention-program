"""Analysis for the natural-text validation (Paper 1, Option B item 2). No GPU.
Paired by instance, mean difference, 10k instance-resampled bootstrap (rng seed 0)."""
import json
import collections
import numpy as np

R = "results/natural/"


def boot(d, n=10000):
    rng = np.random.default_rng(0)
    d = np.asarray(d, float)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), int((d != 0).sum()), len(d)]


def main():
    rows = [json.loads(l) for l in open(R + "raw_results.jsonl")]
    t = collections.defaultdict(dict)
    for x in rows:
        t[(x["arm"], x["budget"])][x["iid"]] = x["frac_retrieved"]
    out = {"arm_means": {f"{a}@{b}": [float(np.mean(list(v.values()))), len(v)] for (a, b), v in sorted(t.items())},
           "n_ctx_mean": float(np.mean([x["n_ctx"] for x in rows])),
           "protected_tokens_mean": float(np.mean([x["n_protected_tokens"] for x in rows])),
           "oracle_tokens_mean": float(np.mean([x["n_oracle_tokens"] for x in rows])),
           "pattern_sentences": sorted({x["n_pattern_sentences"] for x in rows}), "contrasts": {}}
    full = t[("6_full_cache_ref", 0)]
    for B in (256, 512):
        for name, a, b in (("structural_minus_none", "2_protect_permanent", "1_no_protect_permanent"),
                           ("recoverable_minus_structural", "4_recoverable_protected", "2_protect_permanent"),
                           ("oracle_minus_structural", "5_oracle_static", "2_protect_permanent"),
                           ("oracle_minus_fullcache", "5_oracle_static", None),
                           ("structural_minus_fullcache", "2_protect_permanent", None)):
            A = t[(a, B)]
            Bm = t[(b, B)] if b else full
            ids = sorted(set(A) & set(Bm))
            out["contrasts"][f"{name}@{B}"] = boot([A[i] - Bm[i] for i in ids])
    json.dump(out, open(R + "analysis.json", "w"), indent=2)
    print({k: [round(v[0], 3), v[1]] for k, v in out["arm_means"].items()})
    print("ctx", round(out["n_ctx_mean"], 1), "protected", round(out["protected_tokens_mean"], 1),
          "oracle", round(out["oracle_tokens_mean"], 1), "patterns", out["pattern_sentences"])
    for k, v in out["contrasts"].items():
        print(f"  {k:34s} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}] differ {v[3]}/{v[4]}")


if __name__ == "__main__":
    main()
