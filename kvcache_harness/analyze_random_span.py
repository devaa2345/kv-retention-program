"""Analysis for the random-span control (Paper 1, Option B item 1). No GPU.

Reference arms come from results/phase1/raw_results.jsonl (iso_token; permanent arms are identical
under both iso conditions). The determinism check (run_random_span --check-determinism) showed a
rerun of arm 2 at budget 257 reproduces Phase 1 exactly on 25/25 seeds, so reusing those rows is a
like-for-like comparison, not just a same-seed one.

Contrasts, per budget, paired by seed, mean difference, 10k prompt-resampled bootstrap (rng seed 0):
  random - none        is random protection distinguishable from no protection at all?
  structural - random  is structural protection distinguishable from random protection?
  structural - none    (Phase 1 protection effect, for scale)
  oracle - structural  (the retention gap that is left)
Equivalence is declared, as in PREREG s3, when the whole 95% CI lies inside +-0.05.
Also reports credential-token coverage of the random protected set and accuracy split by coverage.
"""
import json
import collections
import numpy as np

R = "results/"
BUDGETS = (154, 257, 514)


def boot(d, n=10000):
    rng = np.random.default_rng(0)
    d = np.asarray(d, float)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), int((d != 0).sum()), len(d)


def main():
    ref = collections.defaultdict(dict)
    for l in open(R + "phase1/raw_results.jsonl"):
        r = json.loads(l)
        if r["iso_condition"] == "iso_token":
            ref[(r["arm"][0], r["budget"])][r["seed"]] = r["frac_retrieved"]
    rnd = collections.defaultdict(dict)
    cov = collections.defaultdict(dict)
    for l in open(R + "random_span/raw_results.jsonl"):
        r = json.loads(l)
        rnd[r["budget"]][r["seed"]] = r["frac_retrieved"]
        cov[r["budget"]][r["seed"]] = r["credential_token_coverage"]
    out = {}
    for B in BUDGETS:
        seeds = sorted(set(rnd[B]) & set(ref[("1", B)]) & set(ref[("2", B)]) & set(ref[("5", B)]))
        a1, a2, a5 = ref[("1", B)], ref[("2", B)], ref[("5", B)]
        a7 = rnd[B]
        row = {"n": len(seeds),
               "mean": {"1_none": np.mean([a1[s] for s in seeds]), "2_structural": np.mean([a2[s] for s in seeds]),
                        "7_random_span": np.mean([a7[s] for s in seeds]), "5_oracle": np.mean([a5[s] for s in seeds])}}
        for name, x, y in [("random_minus_none", a7, a1), ("structural_minus_random", a2, a7),
                           ("structural_minus_none", a2, a1), ("oracle_minus_structural", a5, a2)]:
            m, lo, hi, nd, n = boot([x[s] - y[s] for s in seeds])
            row[name] = {"mean": m, "ci": [lo, hi], "differ": nd, "n": n, "inside_pm0.05": lo > -0.05 and hi < 0.05}
        cv = np.array([cov[B][s] for s in seeds])
        acc = np.array([a7[s] for s in seeds])
        row["random_credential_coverage_mean"] = float(cv.mean())
        row["random_corr_coverage_vs_acc"] = float(np.corrcoef(cv, acc)[0, 1]) if acc.std() > 0 else None
        lo_m = cv <= np.median(cv)
        row["random_acc_low_vs_high_coverage"] = [float(acc[lo_m].mean()), float(acc[~lo_m].mean())]
        out[B] = row
    json.dump(out, open(R + "random_span/analysis.json", "w"), indent=2)
    for B, r in out.items():
        print(f"\nbudget {B} (n={r['n']}): " + "  ".join(f"{k}={v:.3f}" for k, v in r["mean"].items()))
        for k in ("random_minus_none", "structural_minus_random", "structural_minus_none", "oracle_minus_structural"):
            v = r[k]
            print(f"  {k:26s} {v['mean']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] differ {v['differ']}/{v['n']}"
                  f"  inside +-0.05: {v['inside_pm0.05']}")
        print(f"  random-protection credential-token coverage {r['random_credential_coverage_mean']:.3f}; "
              f"corr(cov, acc) {r['random_corr_coverage_vs_acc']}; acc low/high coverage half {r['random_acc_low_vs_high_coverage']}")


if __name__ == "__main__":
    main()
