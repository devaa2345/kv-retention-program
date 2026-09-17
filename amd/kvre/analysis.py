"""Analysis: paired bootstrap, permutation tests, BH correction, CSV export.

Spec ref: section 8. The spec explicitly warns that the median bootstrap produces zero-width
intervals because paired differences are discrete at 1/6 granularity and mostly exactly zero.
We therefore report all three of: median CI, mean CI, and the count of prompts that differ at
all -- the spec calls that last number the most informative of the three.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
from dataclasses import dataclass, asdict


@dataclass
class PairedResult:
    label: str
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float
    median_diff: float
    median_ci: tuple[float, float]
    mean_ci: tuple[float, float]
    n_differ: int                 # prompts where the two arms differ at all
    n_a_better: int
    n_b_better: int
    p_perm: float
    median_ci_zero_width: bool
    equivalent_at_005: bool       # 95% CI fully inside +/-0.05


def paired_bootstrap(a: list[float], b: list[float], label: str = "",
                     n_boot: int = 10000, seed: int = 12345,
                     mei: float = 0.05) -> PairedResult:
    """Bootstrap paired per-prompt differences. Resampling unit = prompt (section 8)."""
    assert len(a) == len(b) and len(a) > 0, "paired arms must have identical prompt sets"
    n = len(a)
    d = [a[i] - b[i] for i in range(n)]
    rng = random.Random(seed)

    med_s, mean_s = [], []
    idx_range = range(n)
    for _ in range(n_boot):
        samp = [d[rng.choice(idx_range)] for _ in idx_range]
        med_s.append(statistics.median(samp))
        mean_s.append(sum(samp) / n)
    med_s.sort(); mean_s.sort()
    lo_i, hi_i = int(0.025 * n_boot), int(0.975 * n_boot) - 1
    med_ci = (med_s[lo_i], med_s[hi_i])
    mean_ci = (mean_s[lo_i], mean_s[hi_i])

    # paired sign-flip permutation test on the mean difference
    obs = abs(sum(d) / n)
    rng2 = random.Random(seed + 1)
    hits = 0
    n_perm = 10000
    for _ in range(n_perm):
        v = sum(x if rng2.random() < 0.5 else -x for x in d) / n
        if abs(v) >= obs - 1e-12:
            hits += 1
    p_perm = (hits + 1) / (n_perm + 1)

    return PairedResult(
        label=label, n=n,
        mean_a=sum(a) / n, mean_b=sum(b) / n,
        mean_diff=sum(d) / n, median_diff=statistics.median(d),
        median_ci=med_ci, mean_ci=mean_ci,
        n_differ=sum(1 for x in d if x != 0),
        n_a_better=sum(1 for x in d if x > 0),
        n_b_better=sum(1 for x in d if x < 0),
        p_perm=p_perm,
        median_ci_zero_width=(med_ci[0] == med_ci[1]),
        equivalent_at_005=(mean_ci[0] > -mei and mean_ci[1] < mei),
    )


def bh_correct(pvals: list[float], alpha: float = 0.05) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, order preserved."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = min(prev, pvals[i] * m / (rank + 1))
        adj[i] = val
        prev = val
    return adj


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return rows


CSV_FIELDS = ["model", "context_target", "context_length", "nominal_budget",
              "effective_budget", "effective_tokens", "iso_condition", "quant_bits",
              "quant_byte_cost", "physical_byte_cost", "arm", "arm_label", "protection",
              "eviction", "promotion", "promotion_id", "full_fraction", "recency_window",
              "sink", "seed", "accuracy", "k", "n_turns", "n_full", "n_quant", "bytes",
              "n_protected_lines", "oracle_oversubscribed", "dormancy_events",
              "dormancy_max_run", "quant_calls", "quant_violations", "quant_max_levels"]


def export_csv(rows: list[dict], path: str) -> str:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def export_paired_csv(results: list[PairedResult], path: str, extra: list[dict] | None = None):
    with open(path, "w", newline="") as f:
        base = ["label", "n", "mean_a", "mean_b", "mean_diff", "median_diff",
                "median_ci_lo", "median_ci_hi", "mean_ci_lo", "mean_ci_hi",
                "n_differ", "n_a_better", "n_b_better", "p_perm", "p_bh",
                "median_ci_zero_width", "equivalent_at_005"]
        extra_keys = sorted({k for e in (extra or []) for k in e})
        w = csv.DictWriter(f, fieldnames=extra_keys + base)
        w.writeheader()
        pvals = [r.p_perm for r in results]
        padj = bh_correct(pvals) if pvals else []
        for i, r in enumerate(results):
            d = asdict(r)
            d["median_ci_lo"], d["median_ci_hi"] = r.median_ci
            d["mean_ci_lo"], d["mean_ci_hi"] = r.mean_ci
            d.pop("median_ci"); d.pop("mean_ci")
            d["p_bh"] = padj[i]
            if extra:
                d.update(extra[i])
            w.writerow(d)
    return path


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rho, average ranks for ties. Used for the section 7 manipulation check."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0
