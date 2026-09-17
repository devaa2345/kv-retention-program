"""Does p_g's decline with c follow from the threshold mechanism, or is it separate?

The question is load-bearing because Stage 2 showed 2.1's accuracy effect runs through p_g,
not through the exponent. If p_g(c) fell out of the same within-fact-correlation mechanism
that gives c_eff, the framework would still be one thing. If it does not, the theory has two
inputs and must say so.

**The answer is forced by the structure of the model and is not an empirical question.**
`rho` describes how the tokens of ONE fact co-vary about that fact's own level. `p_g` is the
probability a gold token clears a threshold set by the GLOBAL budget -- a property of where
gold sits in the whole context's score distribution. Conditioning on the fact's shared
component removes rho entirely from the marginal. So `q(c, rho, p_g)` is derived and
`p_g(c)` is an input, full stop.

What CAN be measured is how much of the observed decline is scorer behaviour at all. A
recency floor performs no content selection whatsoever, so whatever decline IT shows with c
is geometric -- a longer record is harder to fit wholly inside a fixed window, and its tokens
spill across the window edge. Any decline beyond the floor's is the scorer's own.

    p_g_method(c) / p_g_floor(c)     the scorer's gold advantage, geometry divided out

If that ratio is flat in c, the whole decline is geometric and no scorer mechanism is needed.
If it falls, the scorer is separately worse at long facts and that is a second effect.
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RUNS = HERE / "runs" / "nvidia"
CALIB = json.loads((HERE / "out" / "c_calibration.json").read_text(encoding="utf-8"))
C_TARGETS = (8, 19, 40)
ARMS = ("snapkv", "adakv_snapkv")
NL = chr(10)


def load():
    out = {}
    for tag in ("M2", "M3"):
        agg = defaultdict(list)
        for line in (RUNS / ("stage2_capture_%s.jsonl" % tag)).open(encoding="utf-8"):
            r = json.loads(line)
            if r["task"] == "ledger_c":
                agg[(r["arm"], r["c_tag"], r["C"])].append(r)
        for k, rs in agg.items():
            out[(tag,) + k] = st.fmean(r["p_g"] for r in rs)
    return out


def main() -> int:
    pg = load()
    print("=" * 98)
    print("p_g DECOMPOSITION -- is the decline with c geometric, or the scorer's own?")
    print("=" * 98)
    for tag in ("M2", "M3"):
        cs = [CALIB[tag]["chosen"][str(ct)]["c"] for ct in C_TARGETS]
        print(NL + "### %s" % tag)
        print("%-16s %5s %s" % ("arm", "C", "".join("%12s" % ("c=%.1f" % c) for c in cs)
                                + "%14s" % "rel. drop"))
        for C in (64, 512):
            for arm in ("floor_pos",) + ARMS:
                v = [pg.get((tag, arm, ct, C), float("nan")) for ct in C_TARGETS]
                drop = (v[-1] - v[0]) / v[0] if v[0] else float("nan")
                print("%-16s %5d %s%13.1f%%"
                      % (arm, C, "".join("%12.4f" % x for x in v), 100 * drop))
            print("  %-14s %5s %s" % ("-> advantage", "",
                                      "".join("%12s" % "" for _ in cs)))
            for arm in ARMS:
                v = [pg.get((tag, arm, ct, C), float("nan"))
                     / pg.get((tag, "floor_pos", ct, C), float("nan")) for ct in C_TARGETS]
                drop = (v[-1] - v[0]) / v[0] if v[0] else float("nan")
                print("  %-14s %5d %s%13.1f%%"
                      % (arm + "/floor", C, "".join("%12.4f" % x for x in v), 100 * drop))
            print()
    print("READING")
    print("  The floor performs no content selection, so its decline is pure geometry: a")
    print("  longer record is less likely to sit wholly inside a fixed recency window and its")
    print("  tokens straddle the edge more often. Dividing it out isolates the scorer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
