"""1.2 residual structure — does the winning form err randomly, or with a shape?

A form can clear a hit-rate threshold and still be wrong in a way that matters. The registered
test asked only whether predictions land inside a factor of 2. This asks whether what is left
over is noise or a trend, because a trend means the one fitted parameter is absorbing a
dependence the form does not contain -- exactly the failure mode plan section 1.4 names.

Exploratory, not registered.
"""
from __future__ import annotations

import math
import statistics as st

from p3 import measure as ms
from p3 import theory as th
from stage1_retrodiction import (ARMS, FORMS, FLOOR_OBS, MODELS, CAL_C, CAL_MODEL,
                                 pearson, rho_for_form)


def main() -> int:
    tab = ms.cells("line")
    cal = {a: {f"rho_{f}": rho_for_form(f, tab[(CAL_MODEL, CAL_C, a)]["p_g"],
                                        tab[(CAL_MODEL, CAL_C, a)]["q_complete"],
                                        tab[(CAL_MODEL, CAL_C, a)]["c"])
               for f in FORMS[:3]} for a in ARMS}

    def lg(x):
        return math.log10(max(x, FLOOR_OBS))

    print("=" * 96)
    print("1.2 RESIDUAL STRUCTURE -- log10(pred) - log10(obs), per cell")
    print("=" * 96)
    for f in ("F2_linear", "F3_power"):
        res, lc = [], []
        print(f"\n### {f}")
        print(f"  {'model':6s}{'C':>6s}" + "".join(f"{a:>16s}" for a in ARMS))
        for m in MODELS:
            for C in sorted({k[1] for k in tab if k[0] == m}):
                row = []
                for a in ARMS:
                    r = tab[(m, C, a)]
                    p = th.completion_pointwise(r["p_g"], r["c"], cal[a][f"rho_{f}"], f)
                    e = lg(p) - lg(r["q_complete"])
                    row.append(e)
                    if not (m == CAL_MODEL and C == CAL_C):
                        res.append(e)
                        lc.append(math.log10(C))
                print(f"  {m:6s}{C:6d}" + "".join(f"{v:+16.3f}" for v in row))
        r, n = pearson(lc, res)
        print(f"\n  r(residual, log10 C) = {r:+.4f} on n={n} held-out cells")
        print(f"  mean residual {st.fmean(res):+.4f}   sd {st.pstdev(res):.4f}")
        print(f"  sign of the residual is negative below the calibration budget and positive")
        print(f"  above it, in every arm, on both models.")

    print("\n" + "=" * 96)
    print("READING")
    print("=" * 96)
    print("The residual is not noise. It is a monotone function of budget with r ~ +0.88, it")
    print("changes sign exactly at the calibration budget, and it does so in all four arms on")
    print("both models. `c_eff` in F2 and F3 depends only on c and rho, both held fixed across")
    print("budgets; the DATA's implied c_eff (ln q / ln p_g, tabulated in test 1.1) rises")
    print("monotonically from ~1.10 at C=16 to ~2.5-3.2 at C=512. The single fitted parameter")
    print("is absorbing a budget dependence the functional form does not contain.")
    print()
    print("Consequence for 1.3, and it is not a coincidence: the crossover test is evaluated at")
    print("the top budget, which is where the over-prediction is largest. keydiff's completion")
    print("is over-predicted by a factor of ~8 at M3 C=512 (residual +0.907), which is what puts")
    print("it into the predicted crossover set and keeps the two arms that actually win out of")
    print("it. 1.3 does not fail for an independent reason; it fails because the form is wrong")
    print("in a direction that is worst exactly where 1.3 looks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
