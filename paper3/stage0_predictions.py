"""Stage 0 — derive, then COMMIT the numbers, before any comparison is run.

This script writes P3_PREDICTIONS.md. It reads only:
  * fact geometry (c, L, n_free) rebuilt deterministically from instance seeds
  * measured p_g per cell            -- an input to the theory, listed as measurable in the plan
  * the measured completion of the ONE declared calibration cell per arm, to fix rho
  * measured completion in the OTHER unit, which is what the non-parametric baseline F4 is

It never reads the measured completion of a held-out cell. Scoring happens in Stage 1, after
this file is committed and hashed.
"""
from __future__ import annotations

import datetime as dt
import math
import statistics as st
from pathlib import Path

from p3 import measure as ms
from p3 import theory as th

OUT = Path(__file__).resolve().parent / "P3_PREDICTIONS.md"
FORMS = ["F1_naive", "F2_linear", "F3_power", "F4_nonparam"]
CAL_C = 128
CAL_MODEL = "M2"
MODELS = ["M2", "M3"]
ARMS = ms.METHOD_ARMS
TOL_LOG10 = 0.30          # factor of 2
TOL_ABS = 0.05
FLOOR_OBS = 1.0 / (200 * 4)   # 200 instances x H=4 queried records


def _fmt(x: float) -> str:
    """Six decimals where that is informative, scientific where it is not."""
    if x == 0.0:
        return "0"
    if abs(x) < 1e-4:
        return f"{x:.2e}"
    return f"{x:.6f}"


def rho_for_form(form, p_g, q, c):
    """The single free parameter of a parametric form, fixed by inverting it on the
    calibration cell. F1 has none; F4 is not parametric."""
    if form in ("F1_naive", "F4_nonparam"):
        return float("nan")
    if not (0 < p_g < 1) or not (0 < q < 1):
        return float("nan")
    ceff = math.log(q) / math.log(p_g)
    if form == "F2_linear":
        return 1.0 - (ceff - 1.0) / (c - 1.0)
    if form == "F3_power":
        if ceff <= 0:
            return float("nan")
        return 1.0 - math.log(ceff) / math.log(c)
    raise ValueError(form)


def main() -> int:
    tab = ms.cells("line")
    nfree = ms.n_free_causal()
    L = {m: st.fmean(tab[k]["L"] for k in tab if k[0] == m) for m in MODELS}
    C_of = {m: sorted({k[1] for k in tab if k[0] == m}) for m in MODELS}
    c_of = {m: st.fmean(tab[k]["c"] for k in tab if k[0] == m) for m in MODELS}
    co_of = {m: st.fmean(tab[k]["c_other"] for k in tab if k[0] == m) for m in MODELS}

    # ---- calibration: one cell per arm, on M2 at C=128 -------------------------
    cal = {}
    for arm in ARMS:
        r = tab[(CAL_MODEL, CAL_C, arm)]
        c = r["c"]
        cal[arm] = dict(
            p_g=r["p_g"], q=r["q_complete"], c=c,
            rho_bb=th.rho_from_completion(r["p_g"], r["q_complete"], c),
            rho_touched=th.rho_from_touched(
                r["p_g"],
                r["recs_complete"] / ms.N_RECORDS,
                r["recs_touched"] / ms.N_RECORDS, c),
            s_eff=(math.log(1.0 - r["q_any"]) / math.log(1.0 - r["q_complete"])),
            q_any=r["q_any"],
            **{f"rho_{f}": rho_for_form(f, r["p_g"], r["q_complete"], c) for f in FORMS[:3]},
        )

    lines = []
    W = lines.append

    W("# P3_PREDICTIONS — Paper 3, Stage 0")
    W("")
    W(f"Written {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}. "
      "Committed and hashed **before** any comparison against the held-out cells.")
    W("")
    W("Everything below is a number derived from closed forms in `p3/theory.py` plus measured "
      "inputs listed explicitly per section. No quantity that this file predicts was consulted "
      "while writing it, with the single declared exception of the calibration cell.")
    W("")

    # ---------------------------------------------------------------- section 0
    W("## 0. Constants, measured once, not fitted")
    W("")
    W("Fact geometry is a deterministic function of the Paper 2 instance seeds. It was rebuilt "
      "on CPU by `p3/facts.py` and validated by reproducing the `n_ctx` recorded in every Paper 2 "
      "grid row **exactly**, for all 400 instances (200 per model). A mismatch would mean the "
      "rebuilt instance is not the one that was run; the cache refuses to write in that case.")
    W("")
    W("| symbol | M2 (Qwen2.5-3B-Instruct) | M3 (Llama-3.2-3B-Instruct) | source |")
    W("|---|---|---|---|")
    W(f"| `L` context tokens | {L['M2']:.2f} | {L['M3']:.2f} | rebuilt == recorded `n_ctx` |")
    W(f"| `c` fact cost, LINE unit | {c_of['M2']:.4f} | {c_of['M3']:.4f} | rebuilt token spans |")
    W(f"| `c'` fact cost, IDVAL unit | {co_of['M2']:.4f} | {co_of['M3']:.4f} | rebuilt token spans |")
    W(f"| `n_free` candidates inside the floors | {nfree['M2']:.3f} | {nfree['M3']:.3f} | rebuilt |")
    W(f"| `S` (layer, KV-head) slots | 72 | 224 | capture dumps |")
    W("| `N` records / `H` queried | 40 / 4 | 40 / 4 | task definition |")
    W("| `n_sink` / `n_window` | 8 / 64 | 8 / 64 | harness floors, mandatory for every arm |")
    W("| `B` total retained | `C + 72` | `C + 72` | harness budget arithmetic |")
    W("")

    # ---------------------------------------------------------------- section 1
    W("## 1. T1 — the causal ceiling and the information share")
    W("")
    W("```")
    W("A_causal(C, c, H) = ( n_free + min( floor(C/c), H - n_free ) ) / H")
    W("A_floor (C, c, L) = max(0, (n_window + C) - c + 1) / (L - n_sink - c + 1)")
    W("A_presc           = 1                       (a single fact always fits)")
    W("I(C)              = (A_presc - A_causal) / (A_presc - A_floor)")
    W("```")
    W("")
    W("`A_floor` is the exact discrete form; the plan's approximation `(B-c)/L` is tabulated "
      "beside it so the difference is visible rather than absorbed. Both are parameter-free.")
    W("")
    W("| model | C | A_causal | A_floor (exact) | A_floor (B-c)/L | I(C) |")
    W("|---|---|---|---|---|---|")
    for m in MODELS:
        for C in C_of[m]:
            c, l = c_of[m], L[m]
            W(f"| {m} | {C} | {th.a_causal(C, c, 4, nfree[m]):.4f} | "
              f"{th.a_floor(C, c, l):.4f} | {th.a_floor_simple(C, c, l):.4f} | "
              f"{th.information_share(C, c, l, 4, nfree[m]):.4f} |")
    W("")
    W("**Prediction T1.** `oracle_causal` accuracy tracks `A_causal` and `floor_pos` accuracy "
      "tracks `A_floor`, at every cell, with no fitted parameter. T1 is the boundary condition, "
      "not evidence: it is the easy case and is reported as such.")
    W("")

    # ---------------------------------------------------------------- section 2
    W("## 2. Calibration — declared before scoring")
    W("")
    W(f"**Calibration cell: `{CAL_MODEL}`, `C = {CAL_C}`, one cell per arm.** Mid-ladder, away "
      "from both budget extremes, on the model whose fact cost (~18.9 tokens) is the one the "
      "plan's `rho` discussion is written around. Every other cell — all other budgets on M2, "
      "and **every cell on M3** — is held out. M3 predictions are therefore a cross-model "
      "extrapolation, which is the harder and more informative test.")
    W("")
    W("Each parametric form gets **exactly one** free parameter, fixed by inverting that form "
      "on the calibration cell:")
    W("")
    W("```")
    W("c_eff*  = ln(q_measured) / ln(p_g_measured)          on the calibration cell only")
    W("F1  c_eff = c                       -- no free parameter at all")
    W("F2  c_eff = 1 + (c-1)(1-rho)        -- rho = 1 - (c_eff* - 1)/(c - 1)")
    W("F3  c_eff = c^(1-rho)               -- rho = 1 - ln(c_eff*)/ln(c)")
    W("F4  q(c)  = q(c')^(c/c')            -- no free parameter; c' is the IDVAL unit,")
    W("                                       measured in the SAME cell being predicted")
    W("```")
    W("")
    W("`rho_BB` below is an independent estimate of the same quantity by moment-matching a "
      "BetaBinomial to (p_g, q); `rho_touched` estimates it from the all-40-record "
      "touched/complete pair, which never touches the prediction target. Their agreement is "
      "test 1.1's stability question and is not used to make any prediction here.")
    W("")
    W("| arm | p_g (cal) | q (cal) | c_eff* | rho F2 | rho F3 | rho_BB | rho_touched |")
    W("|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        d = cal[arm]
        ce = math.log(d["q"]) / math.log(d["p_g"])
        W(f"| {arm} | {d['p_g']:.4f} | {d['q']:.4f} | {ce:.4f} | {d['rho_F2_linear']:.4f} | "
          f"{d['rho_F3_power']:.4f} | {d['rho_bb']:.4f} | {d['rho_touched']:.4f} |")
    W("")
    W("**A named substitution.** The plan defines `rho` as the within-fact *score* correlation. "
      "Paper 2's captures store keep SETS, not scorer scores, so the score correlation is not "
      "recoverable from anything on disk. What is recoverable — and what the theory actually "
      "consumes — is the within-fact correlation of the KEEP INDICATOR, the quantity that "
      "converts `p_g` into `P(all c kept)`. Every `rho` in this file is that. Recovering the "
      "score correlation would need a fresh capture pass and is out of scope for a no-GPU block.")
    W("")

    # ---------------------------------------------------------------- section 3
    W("## 3. T2 — predicted per-fact completion at every Paper 2 cell")
    W("")
    W("```")
    W("E[complete | pointwise] = p_g ** c_eff        per fact, per (layer, KV-head) slot")
    W("E[complete | contiguous] = A_floor(C, c, L)   parameter-free")
    W("```")
    W("")
    W("The prediction target is `qcpl_line` — the per-fact completion rate averaged over slots, "
      "the LINE unit. `p_g` is the measured per-gold-token keep rate of that same cell.")
    W("")
    W("Cells marked **CAL** are the calibration cells; every other row is out of sample. "
      "F4 is calibrated by nothing.")
    W("")
    W("**A degeneracy in the registered comparison, found while deriving and stated here rather "
      "than discovered at scoring.** `c_eff` depends only on `c` and `rho`, both of which are "
      "constant within a model on Paper 2 data — the fact cost was never varied. Once F2 and F3 "
      "are each calibrated to reproduce the same `c_eff*` on the same calibration cell, they "
      "produce **identical** predictions at every M2 cell, and differ on M3 only through the "
      "change in `c` (18.90 -> 12.91). So Paper 2 can separate F2 from F3 only by cross-model "
      "extrapolation, and weakly. Separating them properly is exactly what the Stage 4 `c` sweep "
      "is for; Stage 1 cannot do it and will not be reported as if it could.")
    W("")
    for m in MODELS:
        W(f"### {m}")
        W("")
        W("| C | arm | p_g | q'(IDVAL) meas | F1 naive | F2 linear | F3 power | F4 nonparam | "
          "A_floor | held out |")
        W("|---|---|---|---|---|---|---|---|---|---|")
        for C in C_of[m]:
            for arm in ARMS:
                r = tab[(m, C, arm)]
                c, cprime = r["c"], r["c_other"]
                p = r["p_g"]
                preds = {}
                for f in FORMS[:3]:
                    rho = cal[arm][f"rho_{f}"]
                    preds[f] = th.completion_pointwise(p, c, rho, f)
                preds["F4_nonparam"] = th.nonparam_completion(r["q_complete_other"], c, cprime)
                held = "CAL" if (m == CAL_MODEL and C == CAL_C) else "yes"
                W(f"| {C} | {arm} | {p:.4f} | {r['q_complete_other']:.4f} | "
                  + " | ".join(_fmt(preds[f]) for f in FORMS)
                  + f" | {th.a_floor(C, r['c'], r['L']):.4f} | {held} |")
        W("")

    # ---------------------------------------------------------------- section 4
    W("## 4. Registered scoring rule for T2 (fixed here, applied in Stage 1)")
    W("")
    W(f"- A held-out cell counts as **predicted** if `|log10(pred) - log10(obs)| <= {TOL_LOG10}` "
      f"(a factor of 2). Observed completion is floored at `1/(200*4) = {FLOOR_OBS:.6f}` so that "
      "exact-zero cells are scorable rather than dropped; predictions are floored the same way.")
    W(f"- A secondary absolute band `|pred - obs| <= {TOL_ABS}` is reported alongside, because a "
      "log band is generous where both numbers are tiny.")
    W("- **The winner is the form with the lowest mean |log10 error| over the held-out cells.** "
      "Ties broken by hit rate inside the tolerance. The winner is reported whichever it is, "
      "including F1 (the naive bound) or F4 (the non-parametric baseline) — in which case the "
      "interpolation is dropped and the theory is stated more weakly, per plan section 1.4.")
    W("- Held-out set = all 60 (model, C, arm) method cells minus the 4 calibration cells = 56.")
    W("- Test 1.2 passes if at least one form predicts >= 70% of held-out cells within the "
      "log tolerance.")
    W("")

    # ---------------------------------------------------------------- section 5
    W("## 5. T2.3 — the crossover budget B*")
    W("")
    W("```")
    W("B* solves   p_g(B) ** c_eff  =  A_floor(C = B - 72)")
    W("```")
    W("")
    W("`p_g(B)` is the measured per-gold-token keep rate interpolated log-linearly in `B` over "
      "the measured ladder; the fit quality is reported so a bad fit cannot hide inside a "
      "confident-looking `B*`. `B*` is scanned over integer `B` up to `L`, above which there is "
      "no compression and the question is empty; `inf` means the pointwise branch never "
      "overtakes the contiguous one at any compressing budget, which is itself a prediction. "
      "Every `B*` beyond the ladder top (B = 584) is an extrapolation past the measured range "
      "and is quoted as such, not as an observation.")
    W("")
    W("Two comparators are registered, because the plan's inequality is ambiguous about which "
      "completeness scale a head-wise method should be compared on, and choosing after the fact "
      "would be the exact vice this ordering exists to prevent:")
    W("")
    W("- **(a) per-slot, PRIMARY** — the literal reading. `p_g ** c_eff` is compared directly "
      "against `A_floor`. `floor_pos` has one global keep-set, so its per-slot and union rates "
      "coincide and the comparison is well posed.")
    W("- **(b) union-lifted, SECONDARY** — a method spread across `S` slots may make a fact "
      "reachable somewhere even when a typical slot lacks it. The per-slot prediction is lifted "
      "by `q_any = 1 - (1 - q_mean) ** s_eff`, with `s_eff` read off the SAME single calibration "
      "cell and carried unchanged. This adds no second fitted parameter to the theory.")
    W("")
    W("| arm | s_eff (from cal cell) |")
    W("|---|---|")
    for arm in ARMS:
        W(f"| {arm} | {cal[arm]['s_eff']:.4f} |")
    W("")
    for m in MODELS:
        top = max(C_of[m]) + 72
        W(f"### {m}   (ladder tops out at C = {max(C_of[m])}, i.e. B = {top}; "
          f"L = {L[m]:.0f}, above which there is no compression)")
        W("")
        W("| arm | p_g(B) slope | R^2 | B* F1 (a) | B* F2 (a) | B* F3 (a) | "
          "B* F1 (b) | B* F2 (b) | B* F3 (b) |")
        W("|---|---|---|---|---|---|---|---|---|")
        for arm in ARMS:
            f = ms.p_g_interpolator(m, arm, table=tab)
            lift = th.union_lift(cal[arm]["s_eff"])
            row = []
            for lf in (None, lift):
                for form in FORMS[:3]:
                    b = th.crossover_budget(f, c_of[m], L[m], cal[arm][f"rho_{form}"], form,
                                            lift=lf)
                    row.append("inf" if math.isinf(b) else f"{b:.0f}")
            W(f"| {arm} | {f.slope:+.4f} | {f.r2:.4f} | " + " | ".join(row) + " |")
        W("")
    W("**Registered decision rule for test 1.3.** The *predicted crossover set* at a cell is")
    W("")
    W("```")
    W("{ arm : predicted completion at that cell  >  A_floor at that cell }")
    W("```")
    W("")
    W("evaluated under the form that wins section 4 — declared now, so the winner cannot be "
      "chosen to suit — and under comparator (a), with (b) reported beside it. Test 1.3 passes "
      "iff, at **M3 C=512**, the predicted set equals the observed winner set "
      "`{adakv_snapkv, expected_attn}`: it must contain both and exclude `snapkv` and `keydiff`. "
      "**M2 C=512** is scored as a second, independent instance of the same rule — no method "
      "beats the floor there, so the predicted set must be **empty**. M3 is the gate; M2 is "
      "corroboration or contradiction and is reported either way.")
    W("")
    W("Observed accuracies at those two cells are already public in Paper 2 and restated in the "
      "Paper 3 plan, so no claim is made that the *winner set* was unknown while this was "
      "written. What was not consulted is every predicted quantity above.")
    W("")
    W("**The predicted sets, written out now so scoring is a lookup and not a judgement:**")
    W("")
    W("| model | C | comparator | F1 | F2 | F3 | F4 |")
    W("|---|---|---|---|---|---|---|")
    for m in MODELS:
        C = max(C_of[m])
        af = th.a_floor(C, c_of[m], L[m])
        for lab, use_lift in (("(a) per-slot", False), ("(b) union-lifted", True)):
            sets = {}
            for form in FORMS:
                s = []
                for arm in ARMS:
                    r = tab[(m, C, arm)]
                    if form == "F4_nonparam":
                        q = th.nonparam_completion(r["q_complete_other"], r["c"], r["c_other"])
                    else:
                        q = th.completion_pointwise(r["p_g"], r["c"],
                                                    cal[arm][f"rho_{form}"], form)
                    if use_lift:
                        q = th.union_lift(cal[arm]["s_eff"])(q)
                    if q > af:
                        s.append(arm)
                sets[form] = ", ".join(s) if s else "(empty)"
            W(f"| {m} | {C} | {lab} | " + " | ".join(sets[f] for f in FORMS) + " |")
    W("")
    W(f"The observed winner set is `adakv_snapkv, expected_attn` at M3 C=512 and `(empty)` at "
      f"M2 C=512. Whether any cell of the table above matches is settled in Stage 1, not here.")
    W("")

    # ---------------------------------------------------------------- section 6
    W("## 6. T3 — the per-head coherence term")
    W("")
    W("```")
    W("E[usable | per-head] = p_g ** c_eff  *  P(coherent in the reading heads)")
    W("P(coherent)          = q_mean / q_any        measured on the slot dumps")
    W("```")
    W("")
    W("`q_mean` is completion in a typical (layer, KV-head) slot; `q_any` is completion in at "
      "least one slot — the union. For `floor_pos` the two coincide by construction (one global "
      "keep-set), which is exactly the asymmetry under test.")
    W("")
    W("**Registered prediction P3.3.** Accuracy tracks `q_mean`, not `q_any`. Formally, across "
      "all 60 (model, C, arm) cells:")
    W("")
    W("1. `r(acc, q_mean) > r(acc, q_any)`, and")
    W("2. at **M3 C=512** the `q_mean` ordering places `snapkv` **below** `floor_pos` while the "
      "`q_any` ordering places it **above** — i.e. the per-head measure resolves the inversion "
      "and the union measure creates it.")
    W("")
    W("Test 1.4 passes iff both hold. Failing (2) while passing (1) is reported as a partial "
      "result and does **not** satisfy the Stage 1 gate.")
    W("")
    W("**Registered prediction P3.2 (test 1.5).** Per-head keep-set divergence")
    W("")
    W("```")
    W("D = recs_touched_any / recs_touched_mean      (union spread over slots, >= 1)")
    W("```")
    W("")
    W("is monotonically associated with `|Delta_head|` from N9, with the sign such that greater "
      "divergence goes with a more negative `Delta_head`. Test 1.5 passes iff the association is "
      "monotone and correctly signed on **both** models.")
    W("")
    W("**Stated before scoring, because it bounds what 1.5 can show:** `Delta_head` is non-zero "
      "at only two budgets per model — M3 C=16 and C=32, M2 C=32 and C=64 — and is exactly 0 "
      "everywhere above, where all candidates fit and the contrast is structurally degenerate. "
      "M2's N9 analysis additionally reports `delta_head_measured: false` and flags every cell "
      "`structurally_degenerate: true` (2 KV heads), and its C=16 `oracle_causal_perhead` arm was "
      "never run (447 of 4800 records absent). So 1.5 has at most **2 usable points per model**. "
      "A correlation on n=2 is not evidence, and 1.5 is reported as a sign-and-ordering check "
      "only. This is one reason the Stage 1 gate rests on 1.3 and 1.4.")
    W("")

    # ---------------------------------------------------------------- section 7
    W("## 7. Gate")
    W("")
    W("Stage 1 proceeds past its gate only if **1.3 or 1.4 passes**. 1.1, 1.2 and 1.5 are "
      "consistency checks a merely descriptive framework would also pass, and are reported "
      "separately with their own numbers. A failure is not pooled against a pass.")
    W("")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
