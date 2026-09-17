"""Paper 3, Stage 0 — the closed forms.

Nothing in this module reads a measurement. Every function is a formula; the measured inputs
(`p_g`, `rho`, `c`, `L`, `C`) are supplied by the caller. That separation is what makes the
Stage 0 / Stage 1 ordering enforceable rather than merely asserted.

SYMBOLS (fixed by Paper 2's harness, not chosen here)
    L          templated context length in tokens            ~2068 (M2) / ~2077 (M3)
    n_sink     mandatory sink, every arm                     8
    n_window   mandatory recency window, every arm           64
    C          competitive budget above the floors           {16,} 32, 64, 128, 256, 512
    B          total tokens retained  = C + n_sink + n_window
    N          record lines in the context                   40
    H          queried candidate records per instance        4
    c          fact cost in tokens (LINE unit)               18.90 (M2) / 12.91 (M3)
               fact cost in tokens (IDVAL unit)              10.00 (M2) /  4.00 (M3)
    p_g        per-gold-token keep rate (measured)
    rho        within-fact keep correlation (measured)
    S          (layer, KV-head) slots                        72 (M2) / 224 (M3)
"""
from __future__ import annotations

import math

N_SINK, N_WINDOW = 8, 64
H_QUERIED = 4
N_RECORDS = 40


# --------------------------------------------------------------------------- T1

def a_causal(C: float, c: float, H: int = H_QUERIED, n_free: float = 0.0) -> float:
    """T1 — the causal ceiling is a knapsack.

    A query-agnostic oracle that knows the H candidates but not which one is asked keeps
    whole candidates, greedily, until C is spent. Under a uniform query prior:

        facts complete = n_free + min(floor((C) / c), H - n_free)
        A_causal       = facts complete / H

    `n_free` is the number of the H candidates already wholly inside the mandatory floors.
    Those cost nothing, so they are complete before the budget is touched. Setting n_free = 0
    recovers the plain form in the plan.
    """
    payable = max(0.0, H - n_free)
    return (n_free + min(math.floor(C / c), payable)) / H


def a_floor(C: float, c: float, L: float,
            n_sink: int = N_SINK, n_window: int = N_WINDOW) -> float:
    """Contiguous (recency) policy. Exact discrete form.

    `floor_pos` keeps the first `n_sink` tokens and the last `n_window + C`. Records are
    interleaved uniformly through the region [n_sink, L); the sink holds none of them. A
    record of `c` tokens is answerable iff it lies wholly inside the recency block, so with
    start positions uniform over the region:

        A_floor = max(0, W - c + 1) / (L - n_sink - c + 1),      W = n_window + C

    The plan's approximation A_floor ~ (B - c) / L is `a_floor_simple` below; both are
    reported so the difference between them is visible rather than absorbed.
    """
    W = n_window + C
    denom = L - n_sink - c + 1.0
    if denom <= 0:
        return 1.0
    return min(1.0, max(0.0, W - c + 1.0) / denom)


def a_floor_simple(C: float, c: float, L: float,
                   n_sink: int = N_SINK, n_window: int = N_WINDOW) -> float:
    """The plan's stated form: (B - c) / L, B = C + n_sink + n_window."""
    B = C + n_sink + n_window
    return min(1.0, max(0.0, (B - c) / L))


def information_share(C: float, c: float, L: float, H: int = H_QUERIED,
                      n_free: float = 0.0, a_prescient: float = 1.0) -> float:
    """I(C) — the anticipatory share of the headroom above the contiguous floor.

        I(C) = (A_prescient - A_causal) / (A_prescient - A_floor)

    A_prescient = 1 whenever a single fact fits the budget (c <= C + n_sink + n_window),
    which holds at every Paper 2 cell. I is the fraction of what a query-aware chooser gets
    that a perfect *content* model, blind to the query, cannot get.
    """
    ac = a_causal(C, c, H, n_free)
    af = a_floor(C, c, L)
    denom = a_prescient - af
    if denom <= 0:
        return float("nan")
    return (a_prescient - ac) / denom


# --------------------------------------------------------------------------- T2

def c_eff(form: str, c: float, rho: float) -> float:
    """The four registered effective-independence forms (plan section 1.4).

        F1  naive       c_eff = c                       full independence, the naive bound
        F2  linear      c_eff = 1 + (c-1)(1-rho)        the working form
        F3  power       c_eff = c^(1-rho)               power interpolation
        F4  nonparam    -- has no c_eff; see `nonparam_completion`
    """
    if form == "F1_naive":
        return c
    if form == "F2_linear":
        return 1.0 + (c - 1.0) * (1.0 - rho)
    if form == "F3_power":
        return c ** (1.0 - rho)
    raise ValueError(f"{form} has no c_eff (F4 is non-parametric)")


def completion_pointwise(p_g: float, c: float, rho: float, form: str) -> float:
    """E[complete | pointwise] per fact = p_g ** c_eff.

    Multiply by H for an expected count; Paper 2 records the per-fact rate (`qcpl`), which
    is what this returns.
    """
    if p_g <= 0.0:
        return 0.0
    if p_g >= 1.0:
        return 1.0
    return p_g ** c_eff(form, c, rho)


def nonparam_completion(q_at_cprime: float, c: float, c_prime: float) -> float:
    """F4 — the non-parametric baseline.

    Measured completion at fact cost c' extrapolated to cost c, with no p_g, no rho and no
    functional assumption beyond 'completion is multiplicative in tokens':

        q(c) = q(c') ** (c / c')

    On Paper 2 the two costs are the two completeness UNITS measured on the same capture:
    c' = the (id, value) unit, c = the whole record line. Both are on disk for every cell,
    so this baseline is available everywhere and is calibrated by nothing.
    """
    q = min(max(q_at_cprime, 1e-12), 1.0)
    return q ** (c / c_prime)


# --------------------------------------------------------------------------- rho

def _bb_p_all(c: int, p: float, rho: float) -> float:
    """P(all c tokens of a fact kept) under BetaBinomial(c, a, b),
    p = a/(a+b), rho = 1/(a+b+1) (the intra-class keep correlation)."""
    if rho <= 1e-9:
        return p ** c
    if rho >= 1.0 - 1e-9:
        return p
    s = 1.0 / rho - 1.0
    a, b = p * s, (1.0 - p) * s
    out = 1.0
    for j in range(c):
        out *= (a + j) / (a + b + j)
    return out


def _bb_p_none(c: int, p: float, rho: float) -> float:
    """P(no token of a fact kept) under the same BetaBinomial."""
    return _bb_p_all(c, 1.0 - p, rho)


def rho_from_completion(p_g: float, q_complete: float, c: float) -> float:
    """Estimate the within-fact keep correlation by moment-matching a BetaBinomial to the
    pair (mean gold-token keep rate, per-slot completion rate).

    NOTE, and it is a real substitution: the plan defines rho as the within-fact *score*
    correlation. Paper 2's captures store keep SETS, not scorer scores, so the score
    correlation is not recoverable from data on disk. What is recoverable, and what the
    theory actually consumes, is the within-fact correlation of the KEEP INDICATOR --
    the quantity that converts p_g into P(all c kept). That is what this returns.
    """
    ci = max(1, int(round(c)))
    p = min(max(p_g, 1e-9), 1.0 - 1e-9)
    lo_v, hi_v = _bb_p_all(ci, p, 1e-9), _bb_p_all(ci, p, 1.0 - 1e-9)
    if q_complete <= lo_v:
        return 0.0
    if q_complete >= hi_v:
        return 1.0
    lo, hi = 1e-9, 1.0 - 1e-9
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _bb_p_all(ci, p, mid) < q_complete:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def rho_from_touched(p: float, q_complete: float, q_touched: float, c: float) -> float:
    """Independent rho estimator that never touches the prediction target.

    Uses the ALL-RECORD pair (fraction complete, fraction touched) instead of the queried
    records: rho is chosen so the BetaBinomial reproduces the observed ratio
    P(all) / P(at least one). Reported alongside `rho_from_completion` in test 1.1 as the
    stability check; a framework whose rho depends on which statistic estimated it is not
    measuring a property of the scorer.
    """
    ci = max(1, int(round(c)))
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    target = q_complete / max(1e-12, q_touched)

    def f(r):
        return _bb_p_all(ci, p, r) / max(1e-12, 1.0 - _bb_p_none(ci, p, r))

    lo, hi = 1e-9, 1.0 - 1e-9
    if target <= f(lo):
        return 0.0
    if target >= f(hi):
        return 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- T2.3 crossover

def crossover_budget(p_g_of_B, c: float, L: float, rho: float, form: str,
                     lo: int = 73, hi: int | None = None,
                     lift=None) -> float:
    """B* — the smallest budget at which a pointwise scorer overtakes the contiguous floor.

    Solves   lift( p_g(B) ** c_eff )  >  A_floor(C = B - n_sink - n_window).

    Scanned rather than bisected: both branches saturate at 1 as B -> L, so a bisection that
    requires a sign change at the upper end reports `inf` for arms that do in fact cross.
    The scan is over integer B in [lo, hi], hi defaulting to the uncompressed length L (above
    which there is no compression and the question is empty).

    `p_g_of_B` is a callable giving the scorer's measured per-gold-token keep rate at total
    budget B (log-linear interpolation of the measured ladder; see p3.measure).
    `lift` optionally maps the per-slot completion onto the union-over-slots scale.
    Returns +inf if the pointwise branch never overtakes, which is itself a prediction.
    """
    hi = int(hi if hi is not None else L)
    for B in range(int(lo), hi + 1):
        q = completion_pointwise(p_g_of_B(B), c, rho, form)
        if lift is not None:
            q = lift(q)
        if q > a_floor(B - N_SINK - N_WINDOW, c, L):
            return float(B)
    return float("inf")


def union_lift(s_eff: float):
    """Map a per-slot completion probability onto the union-over-slots scale.

    A method allocates independently per (layer, KV-head) slot, so the probability that a
    fact is complete SOMEWHERE is larger than the per-slot rate. With `s_eff` effectively
    independent slots:

        q_any = 1 - (1 - q_mean) ** s_eff

    `s_eff` is not a free parameter of the theory; it is read off the ONE calibration cell as
    ln(1 - q_any) / ln(1 - q_mean) and carried unchanged everywhere else.
    """
    def f(q):
        return 1.0 - (1.0 - min(max(q, 0.0), 1.0)) ** s_eff
    return f


# --------------------------------------------------------------------------- T3

def usable_perhead(p_g: float, c: float, rho: float, form: str, coherence: float) -> float:
    """E[usable | per-head] = p_g ** c_eff  *  P(coherent in the reading heads).

    The coherence term is measured, not modelled: on Paper 2's slot dumps it is

        P(coherent) = q_mean / q_any

    -- the probability that a fact available SOMEWHERE across slots is available in a
    typical slot. T3's claim is that accuracy tracks the product (equivalently q_mean),
    while Paper 2's union accounting tracks q_any alone.
    """
    return completion_pointwise(p_g, c, rho, form) * coherence
