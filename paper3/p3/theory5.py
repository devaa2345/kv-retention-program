"""F5 — the threshold model. Derived from mechanism, before any comparison to Stage 2.

===========================================================================================
THE MECHANISM, STATED FIRST
===========================================================================================

Every admitted method is a SCORER PRESS: it assigns each context position a scalar score and
keeps the top `B` of them. That is not an analogy, it is the implementation -- `kvpress`
computes `n_kept = int(n * (1 - compression_ratio))` and takes `scores.topk(n_kept)`. Keeping
the top `B` of `L` is exactly thresholding the score at the `(1 - B/L)` quantile.

So write, for token `i` of fact `f`:

    S_i = mu_f + eps_i          mu_f ~ N(0, rho),  eps_i ~ N(0, 1 - rho),  independent

`mu_f` is what the tokens of one fact SHARE -- the record is in a salient region, its id was
attended to, its neighbourhood scores highly. `eps_i` is what differs between tokens of the
same fact. Scaled so Var(S_i) = 1, the within-fact correlation is exactly `rho`.

A token is kept iff `S_i > z`, and the budget fixes `z`: the marginal keep rate of gold tokens
is `p_g = Phi_bar(z)`, so `z = Phi_bar^-1(p_g)`. A LOOSER budget lowers the threshold.

A fact is complete iff ALL `c` of its tokens clear the threshold. Conditioning on the shared
component makes the tokens independent, so

    q(c, rho, p_g) = INT phi(u) * Phi_bar( (z - sqrt(rho) u) / sqrt(1 - rho) )^c du
    c_eff          = ln q / ln p_g

That is the whole model. It has ONE free parameter, `rho`, exactly as F2 and F3 do, so it is
compared on equal terms.

===========================================================================================
WHAT THE MECHANISM PREDICTS -- derived here, checked against Stage 2 only afterwards
===========================================================================================

**Three boundary conditions, all forced:**

  rho -> 1   every token of a fact shares one score, so the fact is kept or dropped whole:
             q -> p_g and c_eff -> 1. A CONTIGUOUS policy is the rho = 1 limit -- a recency
             block either contains a record or does not -- so the model REQUIRES
             c_eff = 1 for `floor_pos` at every c and every budget.
  c = 1      q = p_g identically, so c_eff = 1 exactly.
  rho -> 0   tokens independent, q = p_g^c, c_eff = c. This recovers F1, the naive bound,
             as the zero-correlation corner rather than as a rival.

**Two limits in the budget, which is where F2 and F3 have nothing to say:**

  p_g -> 0   (tight). Minimising x^T Sigma^-1 x / 2 subject to every x_i >= u, for the
             equicorrelated Sigma, gives x_i = u and quadratic form c u^2 / (1 + (c-1) rho),
             while ln p_g ~ -u^2 / 2. So

                 c_eff  ->  c / (1 + (c - 1) rho)                          [tight-budget]

             which SATURATES at 1/rho as c grows -- near-flat in c, by construction.

             **This is a LIMIT, not an approximation, and the distinction matters here.**
             The prefactors decay only like 1/u^2 ~ 1/(2 ln(1/p_g)), so convergence is
             LOGARITHMIC. Measured against the exact integral at rho = 0.5, c = 8, the limit
             is 1.778 while the integral gives 2.371 at p_g = 1e-2 and is still 1.990 at
             p_g = 1e-14. Nothing below is ever computed from the closed form; F5 always
             evaluates the integral. The closed form is quoted only because it is what makes
             the near-flatness in c intelligible.
  p_g -> 1   (loose). Dropping a token is now the rare event. With rho < 1 the pairwise
             co-drop probability is o(1 - p_g), so the drops behave like a union of rare
             independent events and P(any drop) -> c (1 - p_g), giving

                 c_eff  ->  c                                              [loose-budget]

So the model says `c_eff` rises monotonically with budget, from `c/(1+(c-1)rho)` to `c`, and
is nearly flat in `c` at tight budget while becoming linear in `c` at loose budget. The
budget dependence is not an add-on: it is what a fixed threshold moving through a correlated
score distribution does.

Note what this costs the framework. `c_eff` is no longer a property of a fact and a scorer
alone; it is a property of a fact, a scorer AND a budget. The clean statement "longer facts
are exponentially harder to keep whole" does not survive.

===========================================================================================
WHAT THE MECHANISM DOES **NOT** EXPLAIN
===========================================================================================

`p_g` is an INPUT here, not an output. The model takes the marginal keep rate as given and
derives only how the within-fact correlation converts it into completion. `p_g`'s own decline
with `c` is a property of the MARGINAL score distribution -- where a fact's tokens sit
relative to everything else in the context -- and the correlation structure says nothing about
it. See `p3/pg_decomposition.py`: it is measured and decomposed, not derived.
"""
from __future__ import annotations

import math

# Gauss-Hermite nodes/weights for INT exp(-x^2) f(x) dx, 64 points. Generated once by
# numpy.polynomial.hermgauss and inlined so the model has no runtime dependency on numpy.
_GH_N = 64


def _hermgauss(n: int):
    """Golub-Welsch for the physicists' Hermite weight, in pure Python."""
    # Symmetric tridiagonal Jacobi matrix: off-diagonal b_k = sqrt(k/2)
    import cmath  # noqa: F401  (kept explicit: everything below is real)
    a = [0.0] * n
    b = [math.sqrt(k / 2.0) for k in range(1, n)]
    # QL with implicit shifts on a symmetric tridiagonal matrix
    d = a[:]
    e = b[:] + [0.0]
    z = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for l in range(n):
        it = 0
        while True:
            m = l
            while m < n - 1 and abs(e[m]) > 1e-16 * (abs(d[m]) + abs(d[m + 1])):
                m += 1
            if m == l:
                break
            it += 1
            if it > 50:
                raise RuntimeError("hermgauss: no convergence")
            g = (d[l + 1] - d[l]) / (2.0 * e[l])
            r = math.hypot(g, 1.0)
            g = d[m] - d[l] + e[l] / (g + (r if g >= 0 else -r))
            s = c_ = 1.0
            p = 0.0
            for i in range(m - 1, l - 1, -1):
                f = s * e[i]
                bb = c_ * e[i]
                r = math.hypot(f, g)
                e[i + 1] = r
                if r == 0.0:
                    d[i + 1] -= p
                    e[m] = 0.0
                    break
                s = f / r
                c_ = g / r
                g = d[i + 1] - p
                r = (d[i] - g) * s + 2.0 * c_ * bb
                p = s * r
                d[i + 1] = g + p
                g = c_ * r - bb
                for k in range(n):
                    f = z[k][i + 1]
                    z[k][i + 1] = s * z[k][i] + c_ * f
                    z[k][i] = c_ * z[k][i] - s * f
            else:
                d[l] -= p
                e[l] = g
                e[m] = 0.0
    mu0 = math.sqrt(math.pi)
    return d, [mu0 * z[0][j] ** 2 for j in range(n)]


_X, _W = _hermgauss(_GH_N)


def _phi_bar(x: float) -> float:
    """Upper-tail standard normal, accurate in both tails."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def _phi_bar_inv(p: float) -> float:
    """z such that Phi_bar(z) = p. Bisection: called once per cell, speed is irrelevant."""
    p = min(max(p, 1e-15), 1.0 - 1e-15)
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _phi_bar(mid) > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def q_complete(c: float, rho: float, p_g: float) -> float:
    """P(all c tokens of a fact clear the threshold), under the threshold model.

    Gauss-Hermite over the shared component: with u = sqrt(2) x the Gaussian integral
    INT phi(u) g(u) du becomes (1/sqrt(pi)) SUM w_j g(sqrt(2) x_j).
    """
    if c <= 0:
        return 1.0
    p_g = min(max(p_g, 1e-12), 1.0 - 1e-12)
    rho = min(max(rho, 0.0), 1.0)
    z = _phi_bar_inv(p_g)
    if rho >= 1.0 - 1e-9:
        return p_g
    if rho <= 1e-9:
        return p_g ** c
    s = math.sqrt(1.0 - rho)
    r = math.sqrt(rho)
    tot = 0.0
    for x, w in zip(_X, _W):
        u = math.sqrt(2.0) * x
        h = _phi_bar((z - r * u) / s)
        tot += w * (h ** c)
    return min(1.0, max(0.0, tot / math.sqrt(math.pi)))


def c_eff(c: float, rho: float, p_g: float) -> float:
    """c_eff = ln q / ln p_g. Undefined at p_g = 1; returns nan there."""
    p_g = min(max(p_g, 1e-12), 1.0 - 1e-12)
    q = q_complete(c, rho, p_g)
    if q <= 0.0:
        return float("inf")
    lp = math.log(p_g)
    if lp == 0.0:
        return float("nan")
    return math.log(q) / lp


def c_eff_tight(c: float, rho: float) -> float:
    """Tight-budget closed form, p_g -> 0: c / (1 + (c-1) rho). Saturates at 1/rho."""
    return c / (1.0 + (c - 1.0) * rho)


def rho_from_cell(c: float, p_g: float, q_obs: float) -> float:
    """Invert the model for rho on ONE cell. Monotone in rho, so bisection is exact.

    q is DECREASING in rho? No -- increasing: more within-fact correlation means a fact is
    more often kept whole at the same marginal rate. Checked by construction below.
    """
    lo, hi = 0.0, 1.0
    q_lo, q_hi = q_complete(c, 0.0, p_g), q_complete(c, 1.0, p_g)
    if q_obs <= q_lo:
        return 0.0
    if q_obs >= q_hi:
        return 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if q_complete(c, mid, p_g) < q_obs:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


__all__ = ["q_complete", "c_eff", "c_eff_tight", "rho_from_cell"]
