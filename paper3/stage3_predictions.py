"""Stage 4 predictions for the c x C plane, all five forms, committed before Stage 4 runs.

Calibration is ONE cell for every parametric form, on equal terms: **M2, c ~ 19, C = 64**.
F1 and F4 have no free parameter at all. Everything else in the plane is out of sample,
including all of M3 and all three unmeasured budgets.

`p_g` is an INPUT to every form (Stage 2.1b: p_g is what carries the cost effect, and no form
derives it). Two prediction modes are therefore registered and will BOTH be scored:

  MODE A -- form only, PRIMARY. At Stage 4, `p_g` is measured per cell and fed to each form.
            This scores the form and nothing else, which is the question Stage 1 and 2 asked.
  MODE B -- fully a priori. `p_g` is itself predicted, by log-linear interpolation in B over
            the two Stage 2 budgets (B = 136, 584) per (model, arm, c). C = 128 and 256
            interpolate; C = 32 EXTRAPOLATES below the measured range and is flagged.

Mode B numbers are the ones tabulated here, because Mode A cannot be tabulated until the
budgets are run. The Mode A commitment is the calibrated constants plus `p3/theory*.py`,
which are frozen by the same hash.
"""
from __future__ import annotations

import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

from p3 import theory as T
from p3 import theory5 as T5

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "nvidia"
CALIB = json.loads((HERE / "out" / "c_calibration.json").read_text(encoding="utf-8"))
C_TARGETS = (8, 19, 40)
BUDGETS = (32, 64, 128, 256, 512)
ARMS = ("snapkv", "adakv_snapkv")
CAL = ("M2", 19, 64)
N_SINK, N_WINDOW = 8, 64
NL = chr(10)


def stage2_cells():
    out = {}
    for tag in ("M2", "M3"):
        agg = defaultdict(list)
        for line in (RUNS / ("stage2_capture_%s.jsonl" % tag)).open(encoding="utf-8"):
            r = json.loads(line)
            if r["task"] == "ledger_c":
                agg[(r["arm"], r["c_tag"], r["C"])].append(r)
        for (arm, ct, C), rs in agg.items():
            out[(tag, arm, ct, C)] = dict(
                c=CALIB[tag]["chosen"][str(ct)]["c"],
                p_g=st.fmean(r["p_g"] for r in rs),
                q=st.fmean(r["q_complete"] for r in rs),
                L=st.fmean(r["n_ctx"] for r in rs))
    return out


def rho_f2(c, p, q):
    return 1.0 - (math.log(q) / math.log(p) - 1.0) / (c - 1.0)


def rho_f3(c, p, q):
    ce = math.log(q) / math.log(p)
    return 1.0 - math.log(ce) / math.log(c)


def predict(form, c, p_g, rho, q_ref=None, c_ref=None):
    if form == "F1":
        return T.completion_pointwise(p_g, c, 0.0, "F1_naive")
    if form == "F2":
        return T.completion_pointwise(p_g, c, rho["F2"], "F2_linear")
    if form == "F3":
        return T.completion_pointwise(p_g, c, rho["F3"], "F3_power")
    if form == "F4":
        return T.nonparam_completion(q_ref, c, c_ref) if q_ref else float("nan")
    if form == "F5":
        return T5.q_complete(c, rho["F5"], p_g)
    raise ValueError(form)


def main() -> int:
    tab = stage2_cells()
    tag0, ct0, C0 = CAL
    rho = {}
    for arm in ARMS:
        d = tab[(tag0, arm, ct0, C0)]
        rho[arm] = dict(F2=rho_f2(d["c"], d["p_g"], d["q"]),
                        F3=rho_f3(d["c"], d["p_g"], d["q"]),
                        F5=T5.rho_from_cell(d["c"], d["p_g"], d["q"]))

    lines = []
    W = lines.append
    W("### Calibration, one cell for every parametric form: M2, c = %.2f, C = %d"
      % (tab[(tag0, ARMS[0], ct0, C0)]["c"], C0))
    W("")
    W("| arm | p_g (cal) | q (cal) | rho F2 | rho F3 | rho F5 |")
    W("|---|---|---|---|---|---|")
    for arm in ARMS:
        d = tab[(tag0, arm, ct0, C0)]
        W("| %s | %.4f | %.4f | %.4f | %.4f | %.4f |"
          % (arm, d["p_g"], d["q"], rho[arm]["F2"], rho[arm]["F3"], rho[arm]["F5"]))
    W("")
    W("F1 and F4 have no free parameter. F4 extrapolates the SAME cell's measured completion")
    W("at the smallest cost on the plane, `q(c) = q(c_min)^(c/c_min)`, so it is available only")
    W("once c_min is measured at Stage 4 and is scored in Mode A only.")
    W("")

    # --- registered p_g interpolation ------------------------------------------------
    W("### Registered `p_g` interpolation for Mode B")
    W("")
    W("`ln p_g` linear in `ln B` through the two Stage 2 budgets, per (model, arm, c).")
    W("C = 128, 256 interpolate; **C = 32 extrapolates below the measured range** and its")
    W("predictions are flagged accordingly.")
    W("")
    W("| model | arm | c | slope | p_g C=32* | C=64 | C=128 | C=256 | C=512 |")
    W("|---|---|---|---|---|---|---|---|---|")
    pg_pred = {}
    for tag in ("M2", "M3"):
        for arm in ARMS:
            for ct in C_TARGETS:
                a = tab[(tag, arm, ct, 64)]
                b = tab[(tag, arm, ct, 512)]
                x1, x2 = math.log(64 + 72), math.log(512 + 72)
                y1, y2 = math.log(a["p_g"]), math.log(b["p_g"])
                k = (y2 - y1) / (x2 - x1)
                row = []
                for C in BUDGETS:
                    v = min(1.0, math.exp(y1 + k * (math.log(C + 72) - x1)))
                    pg_pred[(tag, arm, ct, C)] = v
                    row.append(v)
                W("| %s | %s | %.2f | %+.4f | %s |"
                  % (tag, arm, a["c"], k, " | ".join("%.4f" % v for v in row)))
    W("")

    # --- the plane -------------------------------------------------------------------
    W("### Predicted per-fact completion `q` on the c x C plane (Mode B)")
    W("")
    W("Prediction target is `qcpl` measured per (layer, KV-head) slot, the LINE unit, exactly")
    W("as in Stage 1 and Stage 2. `A_floor` is the parameter-free contiguous branch and is")
    W("tabulated as the comparator the crossover test uses.")
    W("")
    for tag in ("M2", "M3"):
        L = st.fmean(tab[(tag, a, ct, C)]["L"]
                     for a in ARMS for ct in C_TARGETS for C in (64, 512))
        W("#### %s   (L = %.0f)" % (tag, L))
        W("")
        W("| arm | c | C | p_g pred | F1 | F2 | F3 | F5 | A_floor |")
        W("|---|---|---|---|---|---|---|---|---|")
        for arm in ARMS:
            for ct in C_TARGETS:
                c = tab[(tag, arm, ct, 64)]["c"]
                for C in BUDGETS:
                    p = pg_pred[(tag, arm, ct, C)]
                    vals = [predict(f, c, p, rho[arm]) for f in ("F1", "F2", "F3", "F5")]
                    star = "*" if C == 32 else ""
                    W("| %s | %.1f | %d%s | %.4f | %s | %.4f |"
                      % (arm, c, C, star, p,
                         " | ".join(("%.2e" % v) if v < 1e-4 else ("%.6f" % v)
                                    for v in vals),
                         T.a_floor(C, c, L)))
        W("")

    # --- k axis ----------------------------------------------------------------------
    W("### The `k` axis — registered as BOUNDS, not as a point prediction")
    W("")
    W("A fact of `k` spans introduces a span-level component between the fact level and the")
    W("token level. Fitting its variance would be a second free parameter, so instead the two")
    W("corners are registered and Stage 4 measures where the truth sits between them:")
    W("")
    W("* **UPPER bound on completion -- `k` inert.** Adjacent spans behave as one span: the")
    W("  between-span correlation equals the within-span correlation, so `c_eff(k) = c_eff(1)`")
    W("  and completion is unchanged by `k` at fixed `c`. This is the most favourable corner.")
    W("* **LOWER bound on completion -- spans independent given the fact.** Each of the `k`")
    W("  spans must survive on its own: `q_k = q(c/k, rho, p_g)^k`. Least favourable corner.")
    W("")
    W("**Registered prediction P-k: completion is non-increasing in `k` at fixed `c` and `C`,")
    W("and lies between these two bounds.** Falling below the LOWER bound would mean `k` costs")
    W("something the correlation model cannot express at all; exceeding the UPPER bound would")
    W("mean extra spans help, which no form here allows. The bounds are wide, so this is a")
    W("weak test by design; it is registered as a containment check, not as a discriminating")
    W("prediction.")
    W("")
    W("**`k` = 4 is DROPPED as infeasible, and the arithmetic is why.** Every part must carry")
    W("the record id and a part label to be identifiable, costing ~6 tokens on M2 and ~4 on M3")
    W("BEFORE any field content. At `k` = 4 that overhead alone forces `c` >= 42.5 on M2 and")
    W(">= 30.5 on M3, against a target of 19 -- so `c` cannot be held fixed while `k` varies,")
    W("and an unmatched `k` axis would confound span count with fact cost, which is the exact")
    W("confound Stage 1 spent a stage disentangling for `c` and `C`. Measured during")
    W("calibration, before any anchor was run. Reaching `k` > 2 at matched cost needs")
    W("unlabelled continuation lines (`+ field`), which makes a part unidentifiable on its own")
    W("-- a different construction, deferred rather than improvised here.")
    W("")
    W("| model | arm | c | C | k | UPPER (k inert) | LOWER (independent spans) |")
    W("|---|---|---|---|---|---|---|")
    for tag in ("M3",):
        for arm in ARMS:
            ct = 19
            c = tab[(tag, arm, ct, 64)]["c"]
            for C in (64, 512):
                p = pg_pred[(tag, arm, ct, C)]
                lo = T5.q_complete(c, rho[arm]["F5"], p)
                for k in (1, 2):
                    hi = T5.q_complete(c / k, rho[arm]["F5"], p) ** k
                    W("| %s | %s | %.1f | %d | %d | %.6f | %.6f |"
                      % (tag, arm, c, C, k, lo, hi))
    W("")

    out = HERE / "out" / "stage4_predictions.md"
    out.write_text(NL.join(lines) + NL, encoding="utf-8", newline="\n")
    json.dump(dict(rho=rho, pg_pred={"|".join(map(str, k)): v for k, v in pg_pred.items()}),
              open(HERE / "out" / "stage4_predictions.json", "w", encoding="utf-8"), indent=2)
    print("wrote %s (%d lines)" % (out, len(lines)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
