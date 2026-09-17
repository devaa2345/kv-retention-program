"""Stage 4 analysis — accuracy plane, crossover, k axis, ladder anchors.

Completion-form scoring (the registered primary test) lives in `stage4_score.py`. This file
reads the GENERATION packages and the k-axis captures, under the rules frozen in PREREG_P3.md
(sha256 c920c404...):

  section 9.1  a cell whose floor_pos accuracy is below 0.05 cannot express a deficit and is
               EXCLUDED from every verdict, mechanically; a cell in which every compressed arm
               is <= 0.02 is VACUOUS. Excluded cells are still tabulated and labelled.
  section 4.2  claims about the cost axis are stated in terms of p_g(c), geometry divided out.
  section 6    P-k: completion non-increasing in k at fixed c and C, and inside the
               registered bounds (k-inert upper, independent-spans lower).

Every accuracy number carries the stated deviation: generation ran at n=100, not the
registered 200 (captures ran at the full 200).
"""
from __future__ import annotations

import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

from p3 import theory as T
from p3 import theory5 as T5

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "nvidia"
MODELS = ("M2", "M3")
METHODS = ("snapkv", "adakv_snapkv", "expected_attn", "keydiff")
BUDGETS = (32, 64, 128, 256, 512)
FLOOR_MIN, VACUOUS_EPS = 0.05, 0.02
NL = chr(10)


def load(pkg, tag):
    p = RUNS / ("stage4_%s_%s.jsonl" % (pkg, tag))
    if not p.exists():
        return []
    seen, out = set(), []
    for line in p.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["key_digest"] in seen:          # dedup by digest (Stage 2: dups are identical)
            continue
        seen.add(r["key_digest"])
        out.append(r)
    return out


def fmean(v):
    return st.fmean(v) if v else float("nan")


def ci(v, n_boot=2000, seed=0):
    if not v:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    bs = sorted(st.fmean([v[rng.randrange(len(v))] for _ in range(len(v))])
                for _ in range(n_boot))
    return bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot)]


def hdr(s):
    print(NL + "=" * 100)
    print(s)
    print("=" * 100)


# ------------------------------------------------------------------------------- plane
def plane():
    hdr("ACCURACY PLANE  c x C  (generation n=100 -- stated deviation from registered 200)")
    res = {}
    for tag in MODELS:
        rows = [r for r in load("plane", tag) if r["task"] != "multispan"]
        if not rows:
            print("  %s: no data" % tag)
            continue
        agg = defaultdict(list)
        for r in rows:
            agg[(round(r["c"], 1), r["C"], r["arm"])].append(r["score"])
        cs = sorted({k[0] for k in agg})
        ns = sorted({len(v) for v in agg.values()})
        print(NL + "### %s   cell n %s" % (tag, ns if len(ns) > 1 else ns[0]))
        print("%-15s %5s" % ("arm", "C") + "".join("%10s" % ("c=%.0f" % c) for c in cs))
        for C in BUDGETS:
            for a in ("floor_pos",) + METHODS + ("oracle_causal",):
                print("%-15s %5d" % (a, C)
                      + "".join("%10.3f" % fmean(agg[(c, C, a)]) for c in cs))
            print()

        print("  RETENTION RATIO method/floor, with the section 9.1 rule applied:")
        print("  X = excluded (floor < %.2f)   V = VACUOUS (all compressed <= %.2f)"
              % (FLOOR_MIN, VACUOUS_EPS))
        print("%-15s %5s" % ("arm", "C") + "".join("%10s" % ("c=%.0f" % c) for c in cs))
        status = {}
        for C in BUDGETS:
            for c in cs:
                f = fmean(agg[(c, C, "floor_pos")])
                comp = [fmean(agg[(c, C, a)]) for a in ("floor_pos",) + METHODS]
                if max(comp) <= VACUOUS_EPS:
                    status[(c, C)] = "V"
                elif f < FLOOR_MIN:
                    status[(c, C)] = "X"
                else:
                    status[(c, C)] = "ok"
        ratios = {}
        for a in METHODS:
            for C in BUDGETS:
                cells = []
                for c in cs:
                    s = status[(c, C)]
                    f = fmean(agg[(c, C, "floor_pos")])
                    v = fmean(agg[(c, C, a)]) / f if f > 0 else float("nan")
                    ratios[(a, c, C)] = (v, s)
                    cells.append("%10s" % ("%.2f" % v if s == "ok" else s))
                print("%-15s %5d" % (a, C) + "".join(cells))
        live = sum(1 for s in status.values() if s == "ok")
        print(NL + "  coverage: %d of %d (c, C) cells admissible; %d excluded, %d vacuous"
              % (live, len(status), sum(1 for s in status.values() if s == "X"),
                 sum(1 for s in status.values() if s == "V")))

        print(NL + "  COST AXIS on admissible cells: does the ratio fall with c at fixed C?")
        falls = []
        for a in METHODS:
            for C in BUDGETS:
                seq = [(c, ratios[(a, c, C)][0]) for c in cs if ratios[(a, c, C)][1] == "ok"]
                if len(seq) < 2:
                    continue
                f_ = seq[-1][1] < seq[0][1]
                falls.append(f_)
                print("    %-14s C=%3d  %s  falls=%s"
                      % (a, C, " ".join("c=%.0f:%.2f" % s for s in seq), f_))
        res[tag] = dict(agg=agg, status=status, ratios=ratios, falls=falls, cs=cs)
        print("  ratio falls with c in %d of %d admissible (arm, C) sequences"
              % (sum(falls), len(falls)))
    return res


# ------------------------------------------------------------------------------- crossover
def crossover(res):
    hdr("CROSSOVER -- predicted winners (F5, Mode A) vs observed winners, per (model, c, C)")
    print("Predicted set = {arm : F5 completion from measured p_g > A_floor}; observed set =")
    print("{arm : accuracy > floor_pos accuracy}. Scored only on admissible cells (section 9.1).")
    pred_json = json.loads((HERE / "out" / "stage4_predictions.json").read_text(encoding="utf-8"))
    rho = pred_json["rho"]
    cap = defaultdict(list)
    for tag in MODELS:
        for r in load("capture", tag):
            if r["task"] != "multispan":
                cap[(tag, round(r["c"], 1), r["C"], r["arm"])].append(r)
    agree = tot = 0
    for tag, R in res.items():
        print(NL + "### %s" % tag)
        for c in R["cs"]:
            for C in BUDGETS:
                if R["status"][(c, C)] != "ok":
                    continue
                agg = R["agg"]
                fp = fmean(agg[(c, C, "floor_pos")])
                obs = sorted(a for a in METHODS if fmean(agg[(c, C, a)]) > fp)
                pred = []
                for a in ("snapkv", "adakv_snapkv"):   # the registered, calibrated arms
                    rs = cap.get((tag, c, C, a))
                    fl = cap.get((tag, c, C, "floor_pos"))
                    if not rs or not fl:
                        continue
                    pg = fmean([x["p_g"] for x in rs])
                    q5 = T5.q_complete(c, rho[a]["F5"], pg)
                    af = fmean([x["q_complete"] for x in fl])   # measured contiguous branch
                    if q5 > af:
                        pred.append(a)
                obs_reg = [a for a in obs if a in ("snapkv", "adakv_snapkv")]
                ok = sorted(pred) == sorted(obs_reg)
                agree += ok
                tot += 1
                print("    c=%-4.0f C=%-4d observed %-40s predicted(reg. arms) %-28s %s"
                      % (c, C, ",".join(obs) or "(none)", ",".join(pred) or "(none)",
                         "MATCH" if ok else "mismatch"))
    print(NL + "  crossover agreement on registered arms: %d of %d admissible cells" % (agree, tot))
    return agree, tot


# ------------------------------------------------------------------------------- k axis
def kaxis():
    hdr("k AXIS -- P-k containment from captures (n=200), accuracy from kaxis package (n=100)")
    pred_json = json.loads((HERE / "out" / "stage4_predictions.json").read_text(encoding="utf-8"))
    rho = pred_json["rho"]
    out = {}
    for tag in MODELS:
        cap = defaultdict(list)
        for r in load("capture", tag):
            if r["task"] == "multispan":
                cap[(r["k"], r["C"], r["arm"])].append(r)
        acc = defaultdict(list)
        for r in load("kaxis", tag):
            acc[(r["k"], r["C"], r["arm"])].append(r["score"])
        if not cap:
            print("  %s: no k captures" % tag)
            continue
        print(NL + "### %s" % tag)
        print("  %-14s %5s %3s %7s %8s %9s %8s %11s %11s %7s %9s"
              % ("arm", "C", "k", "c", "p_g", "q obs", "c_eff*", "UPPER inert",
                 "LOWER indep", "inside", "accuracy"))
        mono_raw, contain_k2, ceff_rise = [], [], []
        for a in ("snapkv", "adakv_snapkv"):
            for C in (64, 512):
                by_k = {}
                for k in (1, 2):
                    rs = cap.get((k, C, a))
                    if not rs:
                        continue
                    c = fmean([x["c"] for x in rs])
                    pg = fmean([x["p_g"] for x in rs])
                    q = fmean([x["q_complete"] for x in rs])
                    ce = math.log(q) / math.log(pg) if 0 < q < 1 and 0 < pg < 1 else float("nan")
                    up = T5.q_complete(c, rho[a]["F5"], pg)
                    lo = T5.q_complete(c / k, rho[a]["F5"], pg) ** k
                    inside = lo - 1e-9 <= q <= up + 1e-9
                    by_k[k] = dict(q=q, pg=pg, ce=ce, inside=inside)
                    if k == 2:
                        contain_k2.append(inside)
                    print("  %-14s %5d %3d %7.2f %8.4f %9.5f %8.4f %11.5f %11.5f %7s %9s"
                          % (a, C, k, c, pg, q, ce, up, lo, inside,
                             "%.3f" % fmean(acc[(k, C, a)]) if acc[(k, C, a)] else "-"))
                if 1 in by_k and 2 in by_k:
                    m = by_k[2]["q"] <= by_k[1]["q"] + 1e-9
                    r = by_k[2]["ce"] > by_k[1]["ce"]
                    mono_raw.append(m)
                    ceff_rise.append(r)
                    print("      P-k part 1, raw q non-increasing in k: %s   "
                          "(p_g k=1 %.4f -> k=2 %.4f)" % (m, by_k[1]["pg"], by_k[2]["pg"]))
                    print("      form-free: c_eff* k=1 %.4f -> k=2 %.4f   rises=%s"
                          % (by_k[1]["ce"], by_k[2]["ce"], r))

        print(NL + "  REGISTERED P-k, reported as registered:")
        print("    part 1  q non-increasing in k ........ %d of %d" % (sum(mono_raw), len(mono_raw)))
        print("    part 2  q inside bounds at k=2 ........ %d of %d" % (sum(contain_k2), len(contain_k2)))
        print("  FLAW IN THE REGISTERED TEST, stated not re-decided: at k = 1 the two bounds")
        print("  COINCIDE (k-inert and independent-spans are the same fact), so 'inside' at k=1")
        print("  only asks whether F5 is exactly right -- and F5's budget error is already")
        print("  registered (section 5.3). Containment is therefore informative at k = 2 only, and")
        print("  both bounds inherit F5's error there too. k=1 rows are printed but not counted.")
        print("  part 1 is also confounded by p_g, which is not held fixed across k: where p_g")
        print("  RISES at k=2, raw q can rise with it. Restated T2 says completion follows p_g,")
        print("  so the form-free c_eff* comparison is the cleaner reading of whether k costs")
        print("  anything beyond the retained fraction.")
        print("  form-free: c_eff* rises from k=1 to k=2 in %d of %d (arm, C) pairs"
              % (sum(ceff_rise), len(ceff_rise)))

        kcal = json.loads((HERE / "out" / "k_calibration.json").read_text(encoding="utf-8"))
        kc = kcal[tag]["chosen"]
        print(NL + "  DESIGN LIMIT -- accuracy across k is CONFOUNDED with answer length")
        print("    k=1: n_fields %d, fact cost c %.2f, answer %.1f tokens"
              % (kc["1"]["n_fields"], kc["1"]["c"], kc["1"]["answer_tokens"]))
        print("    k=2: n_fields %d, fact cost c %.2f, answer %.1f tokens"
              % (kc["2"]["n_fields"], kc["2"]["c"], kc["2"]["answer_tokens"]))
        print("  Every part must carry the record id and a part label to be identifiable, a")
        print("  FIXED per-part overhead (about 6 tokens on M2, 4 on M3). With that overhead,")
        print("  holding fact cost c constant across k and holding answer length constant")
        print("  across k are INCOMPATIBLE: adding a part adds overhead, so matching c forces")
        print("  fewer content fields and a shorter answer. The k axis can therefore hold at")
        print("  most one of the two constant. It holds c, because c is the axis the theory is")
        print("  about. This is the same arithmetic that removed k = 4 at calibration, where the")
        print("  overhead alone forced c >= 42.5 (M2) / 30.5 (M3) against a target of 19.")
        print("  Consequence, and how each result is read:")
        print("    completion across k -- matched on c, CLEAN. The registered k claim (adjacent")
        print("      parts behave as one unit) rests on this and stands.")
        print("    accuracy across k   -- span count and copy length move together. Reported")
        print("      with the confound named. NOT adjusted: a post-hoc correction on a two-point")
        print("      axis would be worse than the plain statement.")
        out[tag] = dict(mono_raw=mono_raw, contain_k2=contain_k2, ceff_rise=ceff_rise)
    return out


# ------------------------------------------------------------------------------- anchors
def anchors():
    hdr("LADDER ANCHORS -- full_cache, tripwires, oracle_prescient and I(C) at c=19")
    for tag in MODELS:
        rows = load("anchors", tag)
        pl = load("plane", tag)
        if not rows:
            print("  %s: no data" % tag)
            continue
        agg = defaultdict(list)
        for r in rows + pl:
            agg[(round(r["c"], 1), r["C"], r["arm"])].append(r["score"])
        print(NL + "### %s" % tag)
        cs = sorted({k[0] for k in agg if k[2] == "full_cache"})
        print("  full_cache (C-independent): " + "  ".join(
            "c=%.0f %.3f" % (c, fmean(agg[(c, 512, "full_cache")])) for c in cs))
        for arm in ("null", "random"):
            print("  %-7s " % arm + "  ".join(
                "c=%.0f C=%d %.3f" % (c, C, fmean(agg[(c, C, arm)]))
                for c in cs for C in (32, 512) if agg[(c, C, arm)]))
        c19 = [c for c in cs if 17 < c < 21]
        if c19:
            c = c19[0]
            print("  I(C) at c=%.1f = (prescient - causal) / (prescient - floor):" % c)
            for C in BUDGETS:
                p, q, f = (fmean(agg[(c, C, a)]) for a in
                           ("oracle_prescient", "oracle_causal", "floor_pos"))
                i = (p - q) / (p - f) if p > f else float("nan")
                print("    C=%3d  prescient %.3f  causal %.3f  floor %.3f  I=%.3f"
                      % (C, p, q, f, i))


def main() -> int:
    res = plane()
    crossover(res)
    kaxis()
    anchors()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
