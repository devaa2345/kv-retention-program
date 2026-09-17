"""Stage 4 scoring with the REGISTERED constants, both modes, as PREREG_P3 sections 5.4 and 7 commit.

WHY THIS FILE EXISTS. `stage4_score.py` -- the scorer whose output was reported first --
re-fitted rho for F2, F3 and F5 on Stage 4's OWN calibration cell (M2, c~19, C=64). The prereg
does not say that. Section 7 tabulates rho fitted on the STAGE 2 captures and states that "the
Mode A commitment is the calibrated constants plus p3/theory*.py, which are frozen by the same
hash." Re-fitting on Stage 4 data is therefore a deviation from the registered procedure. The
difference in rho is small (F5 snapkv 0.9696 registered vs 0.9746 re-fitted), but the registered
number is the one that counts, so this file scores with it and prints the earlier result beside
it. `stage4_score.py` is left unchanged so its reported output stays reproducible.

It also scores MODE B, which the prereg registers as secondary and which had not been run.

  MODE A (PRIMARY)  measured p_g per cell, REGISTERED rho (out/stage4_predictions.json)
  MODE B            REGISTERED interpolated p_g per cell, REGISTERED rho; F1, F2, F3, F5 only
                    (F4 is Mode-A-only by the prereg's own wording)
  hit, winner, held-out set, kill criterion -- exactly as in section 5.4 and section 10

Held-out set. The prereg says every method cell except the single calibration cell (M2, c~19,
C=64). With registered constants that cell was a STAGE 2 cell, so its Stage 4 counterpart is a
new measurement and strictly out of sample. Scored both ways; the prereg wording is primary.
"""
from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

import stage4_score as S

HERE = Path(__file__).resolve().parent
PRED = json.loads((HERE / "out" / "stage4_predictions.json").read_text(encoding="utf-8"))
CALIB_C = json.loads((HERE / "out" / "c_calibration.json").read_text(encoding="utf-8"))
ARMS = S.PRIMARY_ARMS
NL = chr(10)


def summarise(rows, forms, label):
    print(NL + "  " + label)
    print("  %-6s %16s %10s %12s %12s %5s" % ("form", "mean|log10 err|", "hit rate",
                                              "r vs logB", "r vs c", "n"))
    out = {}
    for f in forms:
        sub = [r for r in rows if not math.isnan(r["preds"][f])]
        if not sub:
            continue
        errs = [S.lg(r["preds"][f]) - S.lg(r["obs"]) for r in sub]
        out[f] = dict(mae=st.fmean(abs(e) for e in errs),
                      hit=sum(1 for e in errs if abs(e) <= S.TOL) / len(errs),
                      r_logB=S.pearson([math.log(r["C"] + 72) for r in sub], errs),
                      r_c=S.pearson([r["c"] for r in sub], errs), n=len(sub))
        o = out[f]
        print("  %-6s %16.4f %9.1f%% %+12.4f %+12.4f %5d"
              % (f, o["mae"], 100 * o["hit"], o["r_logB"], o["r_c"], o["n"]))
    win = min(out, key=lambda f: (out[f]["mae"], -out[f]["hit"]))
    print("  WINNER: %s" % win)
    return out, win


def main() -> int:
    cells = S.load()
    rho = {a: PRED["rho"][a] for a in ARMS}
    print("=" * 100)
    print("REGISTERED CONSTANTS (fitted on Stage 2, frozen in PREREG_P3 section 7)")
    print("=" * 100)
    for a in ARMS:
        print("  rho[%-13s] F2 %.4f  F3 %.4f  F5 %.4f" % (a, rho[a]["F2"], rho[a]["F3"], rho[a]["F5"]))

    # ---------------- MODE A, registered constants ----------------
    rows = []
    for (tag, arm, c, C), d in sorted(cells.items()):
        if arm not in ARMS:
            continue
        cmin = min(cc for (t2, a2, cc, C2) in cells if t2 == tag and a2 == arm and C2 == C and cc > 2)
        base = None
        if c > cmin + 0.01:
            b = cells[(tag, arm, cmin, C)]
            base = (b["q"], b["c"])
        cal = tag == "M2" and 18 < c < 20 and C == 64
        rows.append(dict(tag=tag, arm=arm, c=c, C=C, obs=d["q"], cal=cal,
                         preds={f: S.predict(f, c, d["p_g"], rho[arm], base) for f in S.FORMS}))

    print(NL + "=" * 100)
    print("MODE A (PRIMARY) -- measured p_g, REGISTERED rho")
    print("=" * 100)
    held = [r for r in rows if not r["cal"]]
    nz = [r for r in held if r["c"] > 2]
    a_all, w_all = summarise(held, S.FORMS, "held out per prereg wording (%d cells)" % len(held))
    a_nz, w_nz = summarise(nz, S.FORMS, "EXCLUDING the c=1 boundary cells (%d cells)" % len(nz))
    incl = [r for r in rows if r["c"] > 2]
    summarise(incl, S.FORMS, "sensitivity: calibration-cell counterpart INCLUDED, c>1 (%d cells)"
              % len(incl))
    r5 = a_nz["F5"]["r_logB"]
    print(NL + "  KILL CRITERION (section 10): F5 residual r vs log B = %+.4f -> %s"
          % (r5, "FIRES" if abs(r5) > 0.5 else "does not fire"))

    # ---------------- MODE B ----------------
    print(NL + "=" * 100)
    print("MODE B -- REGISTERED interpolated p_g, REGISTERED rho (C=32 is an extrapolation)")
    print("=" * 100)
    rows_b = []
    for key, pg in PRED["pg_pred"].items():
        tag, arm, ct, C = key.split("|")
        ct, C = int(ct), int(C)
        if arm not in ARMS:
            continue
        c = round(CALIB_C[tag]["chosen"][str(ct)]["c"], 2)
        d = cells.get((tag, arm, c, C))
        if d is None:
            continue
        cal = tag == "M2" and ct == 19 and C == 64
        rows_b.append(dict(tag=tag, arm=arm, c=c, C=C, obs=d["q"], cal=cal, pg_pred=pg,
                           pg_obs=d["p_g"],
                           preds={f: S.predict(f, c, pg, rho[arm]) for f in ("F1", "F2", "F3", "F5")}))
    held_b = [r for r in rows_b if not r["cal"]]
    b_out, w_b = summarise(held_b, ("F1", "F2", "F3", "F5"), "held out (%d cells, all c>1)" % len(held_b))
    ex32 = [r for r in held_b if r["C"] != 32]
    summarise(ex32, ("F1", "F2", "F3", "F5"), "excluding the C=32 extrapolation (%d cells)" % len(ex32))
    pgerr = [S.lg(r["pg_pred"]) - S.lg(r["pg_obs"]) for r in held_b]
    print(NL + "  p_g interpolation itself: mean |log10 err| %.4f, r vs log B %+.4f"
          % (st.fmean(abs(e) for e in pgerr),
             S.pearson([math.log(r["C"] + 72) for r in held_b], pgerr)))

    # ---------------- comparison with the first-reported scoring ----------------
    prev = json.loads((HERE / "out" / "stage4_score.json").read_text(encoding="utf-8"))
    print(NL + "=" * 100)
    print("COMPARISON -- registered constants vs the first-reported re-fitted scoring (c>1)")
    print("=" * 100)
    print("  %-6s %22s %22s %16s %16s" % ("form", "MAE registered", "MAE re-fitted",
                                          "r logB reg.", "r logB re-fit"))
    for f in S.FORMS:
        if f in a_nz and f in prev.get("excl_c1", {}):
            p = prev["excl_c1"][f]
            print("  %-6s %22.4f %22.4f %+16.4f %+16.4f"
                  % (f, a_nz[f]["mae"], p["mae"], a_nz[f]["r_logB"], p["r_logB"]))
    print("  winner registered: %s   winner re-fitted: %s" % (w_nz, prev.get("winner_excl_c1")))

    json.dump(dict(modeA_heldout=a_all, modeA_excl_c1=a_nz, modeA_winner=w_all,
                   modeA_winner_excl_c1=w_nz, kill_criterion_r_logB=r5,
                   kill_criterion_fires=abs(r5) > 0.5, modeB=b_out, modeB_winner=w_b),
              open(HERE / "out" / "stage4_score_registered.json", "w", encoding="utf-8"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
