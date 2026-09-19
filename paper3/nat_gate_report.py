"""Summarise the natural-text gate: both conditions per model, floor table, probe 4."""
import json
import random
import statistics as st
import sys

F_ = sys.argv[1] if len(sys.argv) > 1 else "nat_v1"
BAND, FLOORMIN, THR = (0.55, 0.97), 0.05, 1 / 36 + 0.02


def ci(x):
    r = random.Random(1)
    n = len(x)
    b = sorted(st.fmean(r.choices(x, k=n)) for _ in range(4000))
    return b[100], b[3899]


ok_all = True
for tag in ("M2", "M3"):
    try:
        rows = [json.loads(l) for l in open(f"runs/nvidia/nat_gate_{F_}_{tag}.jsonl")]
    except FileNotFoundError:
        continue
    by = {}
    for r in rows:
        for lv, d in r["levels"].items():
            by.setdefault((r["arm"], r["C"], lv), []).append(d["score"])
    n = len(by[("full_cache", 0, "1")])
    print(f"\n== {tag}  n={n}")
    c1 = c2 = c4 = True
    print("cond 1  full_cache in", BAND)
    for lv in "135":
        m = st.fmean(by[("full_cache", 0, lv)]); lo, hi = ci(by[("full_cache", 0, lv)])
        g = BAND[0] <= m <= BAND[1]; c1 &= g
        print(f"  level {lv}: {m:.3f} [{lo:.3f},{hi:.3f}] {'PASS' if g else 'FAIL'}")
    print("cond 2  floor_pos > 0.05 at every (level, C)")
    for lv in "135":
        line = []
        for C in (256, 512, 1024):
            v = by.get(("floor_pos", C, lv))
            if v:
                m = st.fmean(v); g = m > FLOORMIN; c2 &= g
                line.append(f"C={C}: {m:.3f} {'ok' if g else 'X'}")
        print(f"  level {lv}: " + "   ".join(line))
    print(f"probe 4  full_cache with queried fact deleted <= {THR:.4f}")
    for lv in "135":
        v = by.get(("full_cache_deleted", 0, lv))
        if v:
            m = st.fmean(v); g = m <= THR; c4 &= g
            print(f"  level {lv}: {m:.4f} {'PASS' if g else 'FAIL'}")
    print(f"  {tag}: cond1 {c1}  cond2 {c2}  probe4 {c4}")
    ok_all &= c1 and c2 and c4
print("\nGATE:", "PASS" if ok_all else "FAIL")
