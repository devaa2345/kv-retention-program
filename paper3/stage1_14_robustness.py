"""1.4 robustness — the gate rests on this test, so it gets stressed rather than accepted.

The registered predicate for 1.4 was binary and one-directional: does the per-head measure
avoid the false 'available' call that the union measure makes for SnapKV at M3 C=512? It does.
That is a narrow question, and a gate should not turn on a narrow question without the wider
one being reported beside it.

Three stresses, none of them registered in advance and all reported as exploratory:

  A. BOTH directions of the inversion, at every cell, not just the false-above one.
     A measure that never says 'available' is never wrong in the false-above direction.
  B. Within-cell rank agreement. Ranking the five arms inside each (model, C) removes the
     budget trend that dominates a pooled correlation.
  C. The (id, value) unit, in case the LINE unit is simply the wrong functional unit for the
     arm that most embarrasses both measures.
"""
from __future__ import annotations

import math
import statistics as st

from p3 import measure as ms
from stage1_retrodiction import pearson, spearman, ARMS, ALL_ARMS, MODELS


def rank_agreement(tab, label):
    print(f"\n### B. WITHIN-CELL rank agreement, {label}")
    print("Ranking the 5 arms inside each (model, C) strips out the budget trend, which is what")
    print("a pooled correlation is mostly measuring.")
    print(f"{'model':6s} {'C':>5s} {'rho_s(acc,q_mean)':>19s} {'rho_s(acc,q_any)':>18s} "
          f"{'winner':>10s}")
    wins = {"q_mean": 0, "q_any": 0, "tie": 0}
    for m in MODELS:
        for C in sorted({k[1] for k in tab if k[0] == m}):
            rows = [tab[(m, C, a)] for a in ALL_ARMS]
            acc = [r["acc"] for r in rows]
            if any(math.isnan(a) for a in acc):
                print(f"{m:6s} {C:5d} {'(no accuracy grid at this budget)':>38s}")
                continue
            sm, _ = spearman([r["q_mean"] for r in rows], acc)
            sa, _ = spearman([r["q_any"] for r in rows], acc)
            w = "q_mean" if sm > sa + 1e-9 else ("q_any" if sa > sm + 1e-9 else "tie")
            wins[w] += 1
            print(f"{m:6s} {C:5d} {sm:19.4f} {sa:18.4f} {w:>10s}")
    print(f"\n  cells won: q_mean {wins['q_mean']}, q_any {wins['q_any']}, tie {wins['tie']}")
    return wins


def inversions(tab, unit_label, key_mean, key_any):
    print(f"\n### A. BOTH directions of the inversion, every cell, unit = {unit_label}")
    print("A cell is an inversion for a measure when that measure's verdict relative to")
    print("floor_pos contradicts the accuracy verdict relative to floor_pos.")
    print(f"{'model':6s} {'C':>5s} {'arm':15s} {'acc vs floor':>13s} "
          f"{'q_mean vs floor':>16s} {'q_any vs floor':>15s} {'mean wrong':>11s} "
          f"{'any wrong':>10s}")
    tally = {"mean": 0, "any": 0, "n": 0}
    for m in MODELS:
        for C in sorted({k[1] for k in tab if k[0] == m}):
            fp = tab[(m, C, "floor_pos")]
            if math.isnan(fp["acc"]):
                continue
            for arm in ARMS:
                r = tab[(m, C, arm)]
                if math.isnan(r["acc"]):
                    continue
                a = r["acc"] > fp["acc"]
                qm = r[key_mean] > fp[key_mean]
                qa = r[key_any] > fp[key_any]
                tally["n"] += 1
                tally["mean"] += (qm != a)
                tally["any"] += (qa != a)
                print(f"{m:6s} {C:5d} {arm:15s} {('above' if a else 'below'):>13s} "
                      f"{('above' if qm else 'below'):>16s} {('above' if qa else 'below'):>15s} "
                      f"{('WRONG' if qm != a else ''):>11s} {('WRONG' if qa != a else ''):>10s}")
    print(f"\n  over {tally['n']} method cells with an accuracy grid:")
    print(f"    per-head (q_mean) contradicts accuracy in {tally['mean']} "
          f"({tally['mean'] / tally['n']:.1%})")
    print(f"    union    (q_any)  contradicts accuracy in {tally['any']} "
          f"({tally['any'] / tally['n']:.1%})")
    return tally


def main() -> int:
    line = ms.cells("line")
    idval = ms.cells("idval")

    print("=" * 96)
    print("1.4 ROBUSTNESS -- exploratory, not registered. The registered predicate already")
    print("returned PASS; this asks whether that verdict survives contact with the rest of")
    print("the data.")
    print("=" * 96)

    t_line = inversions(line, "LINE (registered)", "q_mean", "q_any")
    rank_agreement(line, "LINE unit")

    print("\n" + "-" * 96)
    print("### C. the (id, value) unit -- the functionally sufficient span")
    print("-" * 96)
    print("Answering needs the record id and the 6-digit value; the surname and department are")
    print("dispensable. If the LINE unit is simply too strict, the IDVAL unit should rescue the")
    print("arms that both LINE measures mis-call.")
    t_idval = inversions(idval, "IDVAL", "q_mean", "q_any")
    rank_agreement(idval, "IDVAL unit")

    print("\n" + "=" * 96)
    print("### The cell the gate is about: M3 C=512, both units, all four measures")
    print("=" * 96)
    m, C = "M3", 512
    print(f"{'arm':15s} {'acc':>8s} | {'LINE q_mean':>12s} {'LINE q_any':>11s} | "
          f"{'IDVAL q_mean':>13s} {'IDVAL q_any':>12s}")
    for arm in ALL_ARMS:
        rl, ri = line[(m, C, arm)], idval[(m, C, arm)]
        print(f"{arm:15s} {rl['acc']:8.4f} | {rl['q_mean']:12.4f} {rl['q_any']:11.4f} | "
              f"{ri['q_mean']:13.4f} {ri['q_any']:12.4f}")
    accs = [line[(m, C, a)]["acc"] for a in ALL_ARMS]
    for lab, vals in (("LINE q_mean", [line[(m, C, a)]["q_mean"] for a in ALL_ARMS]),
                      ("LINE q_any", [line[(m, C, a)]["q_any"] for a in ALL_ARMS]),
                      ("IDVAL q_mean", [idval[(m, C, a)]["q_mean"] for a in ALL_ARMS]),
                      ("IDVAL q_any", [idval[(m, C, a)]["q_any"] for a in ALL_ARMS])):
        s, _ = spearman(vals, accs)
        p, _ = pearson(vals, accs)
        print(f"  rank agreement with accuracy at this cell: {lab:14s} "
              f"rho_s {s:+.4f}   r {p:+.4f}")

    print("\n" + "=" * 96)
    print("### What M3 C=512 actually shows once both units are on the table")
    print("=" * 96)
    print("Under the IDVAL unit -- the functionally sufficient span, id plus the 6-digit value --")
    print("the queried record is available SOMEWHERE across the 224 slots for essentially every")
    print("instance on three of the four methods:")
    print()
    print(f"  {'arm':15s} {'acc':>8s} {'IDVAL q_any':>12s} {'IDVAL q_mean':>13s} "
          f"{'acc - q_any':>12s}")
    for arm in ALL_ARMS:
        rl, ri = line[(m, C, arm)], idval[(m, C, arm)]
        print(f"  {arm:15s} {rl['acc']:8.4f} {ri['q_any']:12.4f} {ri['q_mean']:13.4f} "
              f"{rl['acc'] - ri['q_any']:+12.4f}")
    print()
    print("So the earlier reading -- that expected_attn answers more often than the record is")
    print("available -- was an artefact of the LINE unit being too strict, and does not survive.")
    print("There is no accuracy-above-availability anomaly here and no leak is implied.")
    print()
    print("What is left is the real finding and it cuts both ways:")
    print("  * Union availability at this cell is ~1.0 for snapkv, expected_attn and")
    print("    adakv_snapkv, while their accuracies span 0.139 to 0.396 and the floor sits at")
    print("    0.270. Availability somewhere across heads is close to uninformative here --")
    print("    which is exactly T3's claim, and it is why the union measure inverts.")
    print("  * Per-head availability is far closer to accuracy in level, and across all 44")
    print("    cells it contradicts accuracy 4.5% of the time against the union's 38.6%.")
    print("  * But at THIS cell neither measure ranks the four methods: rho_s is 0.10 for")
    print("    LINE q_mean and 0.30 for the other three. Per-head fixes the LEVEL of the")
    print("    snapkv inversion without explaining the ORDER of the four arms. Stage 1 should")
    print("    not be reported as having resolved M3 C=512; it has resolved one specific")
    print("    contradiction inside it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
