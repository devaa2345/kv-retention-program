"""Paired bootstrap and the two headline metrics (v1 §5.5).

FROZEN AT STAGE 2. This module is the analysis contract: it is written before any data
exists and is *run*, not rewritten, at Stage 10.

Design commitments encoded here, each traceable to a document:

* Every contrast is **paired per instance** (v1 §5.1, §5.5).
* Bootstrap 50,000 resamples, resampling unit = instance, percentile CIs.
* `G_m` and `I` CIs propagate the bootstrap **through the ratio** -- numerator and
  denominator are recomputed on the *same* resample. Never divide two independently
  bootstrapped means (v1 §5.5).
* **Refusal-to-normalise** (v1 §1.1): if `A_causal - A_floor < 0.15` the cell is raw-only
  and `G_m` is `None` with reason `degenerate headroom`. A ratio on a 0.05-wide denominator
  is a noise amplifier.
* **Unclipped** (v1 §1.1): `G_m > 1` is diagnostic, not an error. It means a method beat the
  ceiling arm -- usually because the ceiling is partly denoising, as at budget 514 in Paper 1
  where protection 0.964 > oracle_static 0.958 > full_cache 0.947.
* **No median.** Paper 1 pre-registered a bootstrapped median of paired differences and it
  was degenerate on this data shape: paired differences are overwhelmingly exactly zero, the
  median is zero in essentially every resample, and the interval collapses to zero width.
  The mean paired difference is used, and every contrast additionally reports
  `n_differ` -- the count of instances on which the arms differ at all -- which is
  estimator-independent and was the stronger statement in Paper 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

N_BOOT = 50_000
REFUSAL_THRESHOLD = 0.15  # v1 §1.1
CI_ALPHA = 0.05


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


@dataclass
class PairedContrast:
    """A paired per-instance difference between two arms."""
    name: str
    n: int
    mean_diff: float
    ci_low: float
    ci_high: float
    n_differ: int
    frac_differ: float
    null_kind: str | None = None  # "same behaviour" | "different behaviour, same outcome"

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def paired_contrast(
    a: Sequence[float],
    b: Sequence[float],
    *,
    name: str = "",
    seed: int = 0,
    n_boot: int = N_BOOT,
    equivalence_margin: float | None = None,
) -> PairedContrast:
    """Mean paired difference (a - b) with a paired percentile bootstrap CI.

    `n_differ` counts instances where the two arms give different scores at all. Paper 1's
    two kinds of null are distinguished by it: "same behaviour" (arms agree on nearly every
    instance) vs "different behaviour, same outcome" (arms disagree often and still average
    to the same place). These must never be described in the same words.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"paired arms must align per instance: {x.shape} vs {y.shape}")
    n = x.size
    if n == 0:
        raise ValueError("empty contrast")

    d = x - y
    n_differ = int(np.count_nonzero(d))
    obs = float(d.mean())

    rng = _rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * CI_ALPHA / 2, 100 * (1 - CI_ALPHA / 2)])

    null_kind = None
    if equivalence_margin is not None and lo > -equivalence_margin and hi < equivalence_margin:
        frac = n_differ / n
        null_kind = "same behaviour" if frac < 0.10 else "different behaviour, same outcome"

    return PairedContrast(
        name=name, n=n, mean_diff=obs, ci_low=float(lo), ci_high=float(hi),
        n_differ=n_differ, frac_differ=n_differ / n, null_kind=null_kind,
    )


@dataclass
class RatioResult:
    name: str
    value: float | None
    ci_low: float | None
    ci_high: float | None
    n: int
    numerator_mean: float
    denominator_mean: float
    refused: bool = False
    reason: str | None = None
    exceeds_one: bool = False
    audit_required: bool = False

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def capture_ratio(
    a_method: Sequence[float],
    a_floor: Sequence[float],
    a_causal: Sequence[float],
    *,
    name: str = "G_m",
    seed: int = 0,
    n_boot: int = N_BOOT,
    refusal_threshold: float = REFUSAL_THRESHOLD,
) -> RatioResult:
    """G_m = (A_m - A_floor) / (A_causal - A_floor), bootstrapped through the ratio.

    Every array is per-instance and index-aligned. Each resample draws instance indices ONCE
    and recomputes both numerator and denominator on that draw -- this is the whole point,
    and dividing two independently bootstrapped means would understate the CI.
    """
    m = np.asarray(a_method, float)
    f = np.asarray(a_floor, float)
    c = np.asarray(a_causal, float)
    if not (m.shape == f.shape == c.shape):
        raise ValueError(f"arms must align per instance: {m.shape} {f.shape} {c.shape}")
    n = m.size
    num_mean = float((m - f).mean())
    den_mean = float((c - f).mean())

    if den_mean < refusal_threshold:
        return RatioResult(
            name=name, value=None, ci_low=None, ci_high=None, n=n,
            numerator_mean=num_mean, denominator_mean=den_mean,
            refused=True,
            reason=f"n/a — degenerate headroom (A_causal − A_floor = {den_mean:.4f} "
                   f"< {refusal_threshold})",
        )

    rng = _rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    num = (m - f)[idx].mean(axis=1)
    den = (c - f)[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(den != 0, num / den, np.nan)
    ratio = ratio[np.isfinite(ratio)]
    lo, hi = np.percentile(ratio, [100 * CI_ALPHA / 2, 100 * (1 - CI_ALPHA / 2)])

    val = num_mean / den_mean
    return RatioResult(
        name=name, value=float(val), ci_low=float(lo), ci_high=float(hi), n=n,
        numerator_mean=num_mean, denominator_mean=den_mean,
        exceeds_one=bool(val > 1.0), audit_required=bool(val > 1.0),
        reason="G_m > 1 — ceiling-validity audit required (v1 §7.6); reported unclipped"
        if val > 1.0 else None,
    )


def information_share(
    a_presc: Sequence[float],
    a_causal: Sequence[float],
    a_floor: Sequence[float],
    *,
    name: str = "I",
    seed: int = 0,
    n_boot: int = N_BOOT,
    refusal_threshold: float = REFUSAL_THRESHOLD,
) -> RatioResult:
    """I = (A_presc - A_causal) / (A_presc - A_floor), bootstrapped through the ratio.

    The share of headroom no causal method can capture, because it is a prediction problem
    rather than a ranking problem (v1 §1.2). If I is large, Paper 3 is about query
    prediction, not about a better importance score.
    """
    p = np.asarray(a_presc, float)
    c = np.asarray(a_causal, float)
    f = np.asarray(a_floor, float)
    if not (p.shape == c.shape == f.shape):
        raise ValueError(f"arms must align per instance: {p.shape} {c.shape} {f.shape}")
    n = p.size
    num_mean = float((p - c).mean())
    den_mean = float((p - f).mean())

    if den_mean < refusal_threshold:
        return RatioResult(
            name=name, value=None, ci_low=None, ci_high=None, n=n,
            numerator_mean=num_mean, denominator_mean=den_mean, refused=True,
            reason=f"n/a — degenerate headroom (A_presc − A_floor = {den_mean:.4f} "
                   f"< {refusal_threshold})",
        )

    rng = _rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    num = (p - c)[idx].mean(axis=1)
    den = (p - f)[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(den != 0, num / den, np.nan)
    ratio = ratio[np.isfinite(ratio)]
    lo, hi = np.percentile(ratio, [100 * CI_ALPHA / 2, 100 * (1 - CI_ALPHA / 2)])
    return RatioResult(
        name=name, value=float(num_mean / den_mean), ci_low=float(lo), ci_high=float(hi),
        n=n, numerator_mean=num_mean, denominator_mean=den_mean,
    )


@dataclass
class LadderOrdering:
    """v1 §4.1 registered ordering invariants, asserted per cell."""
    ok: bool
    violations: list[str] = field(default_factory=list)
    denoising: bool = False


def check_ladder_ordering(
    *, null: float, random: float, floor_pos: float,
    oracle_causal: float, oracle_prescient: float, full_cache: float,
    tol: float = 0.0,
) -> LadderOrdering:
    """Tripwire and ceiling ordering. A violation halts the cell; it is not smoothed.

    Paper 1's lesson: three of six defects surfaced from a reference arm behaving impossibly,
    not from any test suite. Reference arms earn their compute as tripwires.
    """
    v: list[str] = []
    if not null <= random + tol:
        v.append(f"null({null:.4f}) > random({random:.4f})")
    if not random <= floor_pos + tol:
        v.append(f"random({random:.4f}) > floor_pos({floor_pos:.4f})")
    if not floor_pos <= oracle_causal + tol:
        v.append(f"floor_pos({floor_pos:.4f}) > oracle_causal({oracle_causal:.4f})")
    if not oracle_causal <= oracle_prescient + tol:
        v.append(f"oracle_causal({oracle_causal:.4f}) > oracle_prescient({oracle_prescient:.4f})")
    denoise = oracle_causal > full_cache + tol
    if denoise:
        v.append(
            f"oracle_causal({oracle_causal:.4f}) > full_cache({full_cache:.4f}) — "
            "ceiling arm is denoising, not retaining; ceiling-validity audit required"
        )
    return LadderOrdering(ok=not v, violations=v, denoising=denoise)


__all__ = [
    "N_BOOT", "REFUSAL_THRESHOLD",
    "PairedContrast", "paired_contrast",
    "RatioResult", "capture_ratio", "information_share",
    "LadderOrdering", "check_ladder_ordering",
]
