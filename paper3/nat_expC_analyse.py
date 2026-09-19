"""Experiment C analysis: G_m, I, method/floor ratio, paired instance bootstrap (no GPU).

Definitions are Paper 2's (PAPER2_DRAFT.md, Metrics):
    G_m = (A_m - A_floor) / (A_causal - A_floor)
    I   = (A_presc - A_causal) / (A_presc - A_floor)
  * refusal-to-normalise: cell with A_causal - A_floor < 0.15 reports raw only, G_m 'n/a'
  * G_m unclipped; G_m > 1 is flagged, not clipped
  * 50,000 paired resamples, percentile 95% CIs, unit = instance, numerator and denominator
    recomputed on the SAME resample (two bootstrapped means are never divided)
  * every contrast also reports the count of instances on which the two arms differ at all
Ratio R_m = A_m / A_floor is Paper 3's cost-axis quantity. Paper 3's admissibility rules are
applied mechanically: floor < 0.05 -> X (excluded); all compressed arms <= 0.02 -> V (vacuous).
Marginal cells (M3 level 1, whose full_cache anchor is 0.968 against a 0.97 ceiling, and every
C=256 cell, whose perfect-reader floor ceiling is 0.089) are marked with a dagger everywhere.
"""
import glob
import json
import sys
import zlib
from collections import defaultdict

import numpy as np

NB = 50_000
LEVELS = ("1", "3", "5")
METHODS = ("snapkv", "adakv_snapkv", "expected_attn", "keydiff")
LADDER = ("null", "random", "floor_pos", "oracle_causal", "oracle_prescient")
BUDGETS = (256, 512, 1024)


def marg(tag, lv, C):
    return "†" if (C == 256 or (tag == "M3" and lv == "1")) else ""


def load(tag):
    """-> {(arm, C, level): {iid: mean score over that level's H variants}}"""
    acc = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"runs/nvidia/natC_{tag}_C*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            for v, s in zip(r["variants"], r["score"]):
                acc[(r["arm"], r["C"], v["level"])][r["iid"]].append(s)
    out = {}
    for k, d in acc.items():
        out[k] = {i: float(np.mean(x)) for i, x in d.items() if len(x) == 4}
    return out


def boot(idx, *arrs):
    return [a[idx].mean(1) for a in arrs]


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    res = {}
    lines = []
    P = lines.append
    for tag in ("M2", "M3"):
        D = load(tag)
        if not D:
            continue
        P(f"\n## {tag}\n")
        P("`†` = marginal cell (M3 level 1 anchor; all C=256 cells). `X` floor < 0.05, `V` all compressed arms <= 0.02, `n/a` A_causal - A_floor < 0.15.\n")
        for lv in LEVELS:
            fc = D.get(("full_cache", 0, int(lv)), {})
            fcm = np.mean(list(fc.values())) if fc else float("nan")
            P(f"\n### {tag} level {lv} (full_cache {fcm:.3f}){marg(tag, lv, 0) if tag=='M3' and lv=='1' else ''}\n")
            P("| C | arm | A | R = A/floor [95% CI] | G_m [95% CI] | I [95% CI] | n differ vs floor |")
            P("|---|---|---|---|---|---|---|")
            for C in BUDGETS:
                need = [("floor_pos", C, int(lv)), ("oracle_causal", C, int(lv)),
                        ("oracle_prescient", C, int(lv))] + [(m, C, int(lv)) for m in METHODS]
                if any(k not in D for k in need):
                    continue
                ids = sorted(set.intersection(*[set(D[k]) for k in need]))
                n = len(ids)
                A = {k[0]: np.array([D[k][i] for i in ids]) for k in need}
                for extra in ("null", "random"):
                    k = (extra, C, int(lv))
                    if k in D:
                        A[extra] = np.array([D[k][i] for i in ids])
                rng = np.random.default_rng(zlib.crc32(f"{tag}|{lv}|{C}".encode()))
                idx = rng.integers(0, n, size=(NB, n))
                fl, ca, pr = A["floor_pos"], A["oracle_causal"], A["oracle_prescient"]
                bfl, bca, bpr = boot(idx, fl, ca, pr)
                flm = fl.mean()
                head = ca.mean() - flm
                cellflag = marg(tag, lv, C)
                vac = all(A[m].mean() <= 0.02 for m in METHODS)
                status = "X" if flm < 0.05 else ("V" if vac else "")
                ib = (bpr - bca) / (bpr - bfl)
                icell = f"{np.mean((pr.mean() - ca.mean()) / (pr.mean() - flm)):.3f} [{ci(ib)[0]:.3f}, {ci(ib)[1]:.3f}]" if pr.mean() - flm > 0 else "n/a"
                for arm in list(METHODS) + ["floor_pos", "oracle_causal", "oracle_prescient", "null", "random"]:
                    if arm not in A:
                        continue
                    a = A[arm]
                    am = a.mean()
                    row = dict(tag=tag, level=lv, C=C, arm=arm, n=n, A=float(am), status=status,
                               floor=float(flm), causal=float(ca.mean()), presc=float(pr.mean()))
                    if arm in METHODS:
                        (ba,) = boot(idx, a)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            rb = ba / bfl
                            gb = (ba - bfl) / (bca - bfl)
                        nz = int((bfl == 0).sum())
                        if status != "X":
                            assert nz == 0, f"{tag} L{lv} C{C}: {nz} zero-floor draws in admissible cell"
                        rtxt = f"{am / flm:.2f} [{ci(rb)[0]:.2f}, {ci(rb)[1]:.2f}]" if flm > 0 else "n/a"
                        if head < 0.15:
                            gtxt = "n/a (headroom %.3f)" % head
                            row["G"] = None
                        else:
                            g = (am - flm) / head
                            gtxt = f"{g:.2f} [{ci(gb)[0]:.2f}, {ci(gb)[1]:.2f}]"
                            row["G"] = float(g)
                            row["G_ci"] = ci(gb)
                            if g > 1:
                                gtxt += " (G>1: audit)"
                        row["R"] = float(am / flm) if flm > 0 else None
                        row["R_ci"] = ci(rb) if flm > 0 else None
                        nd = int((a != fl).sum())
                    else:
                        rtxt = gtxt = ""
                        nd = int((a != fl).sum()) if arm != "floor_pos" else 0
                    itxt = icell if arm == "oracle_prescient" else ""
                    P(f"| {C}{cellflag} {status} | {arm} | {am:.3f} | {rtxt} | {gtxt} | {itxt} | {nd}/{n} |")
                    res.setdefault(tag, []).append(row)
                res.setdefault(tag + "_I", []).append(dict(level=lv, C=C, headroom=float(head),
                                                            I_cell=icell, status=status))
    open("out/natC_report.md", "w", encoding="utf-8").write("\n".join(lines))
    json.dump(res, open("out/natC_report.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
