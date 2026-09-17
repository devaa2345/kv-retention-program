"""Stage 1 — retrodiction against Paper 2, scored against the frozen P3_PREDICTIONS.md.

Refuses to run unless P3_PREDICTIONS.md matches the hash recorded in P3_PREDICTIONS.md.sha256,
so the ordering that makes Stage 1 a test rather than a fit is enforced by the code and not
by good intentions.

Each of 1.1-1.5 is reported separately with its own numbers. Nothing is pooled.
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from p3 import measure as ms
from p3 import theory as th
from hash_file import digest

HERE = Path(__file__).resolve().parent
PRED = HERE / "P3_PREDICTIONS.md"
FORMS = ["F1_naive", "F2_linear", "F3_power", "F4_nonparam"]
CAL_C, CAL_MODEL = 128, "M2"
ARMS = ms.METHOD_ARMS
ALL_ARMS = ms.ALL_ARMS
MODELS = ["M2", "M3"]
TOL_LOG10, TOL_ABS = 0.30, 0.05
FLOOR_OBS = 1.0 / 800.0


def check_frozen():
    rec = (HERE / "P3_PREDICTIONS.md.sha256").read_text(encoding="utf-8").split()[0]
    got = digest(PRED)
    if rec != got:
        raise SystemExit(f"P3_PREDICTIONS.md has changed since it was hashed\n"
                         f"  recorded {rec}\n  actual   {got}")
    print(f"P3_PREDICTIONS.md frozen at sha256 {got}\n")


def pearson(x, y):
    pts = [(a, b) for a, b in zip(x, y)
           if not (math.isnan(a) or math.isnan(b))]
    if len(pts) < 3:
        return float("nan"), 0
    xs, ys = zip(*pts)
    mx, my = st.fmean(xs), st.fmean(ys)
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if sx == 0 or sy == 0:
        return float("nan"), len(pts)
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy), len(pts)


def spearman(x, y):
    pts = [(a, b) for a, b in zip(x, y) if not (math.isnan(a) or math.isnan(b))]
    if len(pts) < 3:
        return float("nan"), 0

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    xs, ys = zip(*pts)
    return pearson(rank(list(xs)), rank(list(ys)))[0], len(pts)


def rho_for_form(form, p_g, q, c):
    if form in ("F1_naive", "F4_nonparam") or not (0 < p_g < 1) or not (0 < q < 1):
        return float("nan")
    ceff = math.log(q) / math.log(p_g)
    if form == "F2_linear":
        return 1.0 - (ceff - 1.0) / (c - 1.0)
    if ceff <= 0:
        return float("nan")
    return 1.0 - math.log(ceff) / math.log(c)


def hdr(s):
    print("\n" + "=" * 96)
    print(s)
    print("=" * 96)


# =============================================================================== 1.0
def test_10(tab):
    """Not a numbered test -- the T1 boundary condition, reported because Stage 0 predicted it
    and because it is the reference against which 'the framework predicts nothing' is judged."""
    hdr("1.0  T1 BOUNDARY CONDITION (context, not evidence): A_causal and A_floor vs measured")
    accs = ms.accuracy_table()
    nfree = ms.n_free_causal()
    print(f"{'model':6s} {'C':>5s} {'A_causal':>9s} {'oracle_causal':>14s} {'err':>8s} "
          f"{'A_floor':>8s} {'floor_pos':>10s} {'err':>8s} {'I(C) pred':>10s} {'I(C) meas':>10s}")
    rows = []
    for m in MODELS:
        c = st.fmean(tab[k]["c"] for k in tab if k[0] == m)
        L = st.fmean(tab[k]["L"] for k in tab if k[0] == m)
        for C in sorted({k[1] for k in tab if k[0] == m}):
            oc = accs[m].get((C, "oracle_causal"), (float("nan"), 0))[0]
            fp = accs[m].get((C, "floor_pos"), (float("nan"), 0))[0]
            op = accs[m].get((C, "oracle_prescient"), (float("nan"), 0))[0]
            ac = th.a_causal(C, c, 4, nfree[m])
            af = th.a_floor(C, c, L)
            ip = th.information_share(C, c, L, 4, nfree[m])
            im = (op - oc) / (op - fp) if not math.isnan(oc) and op != fp else float("nan")
            rows.append((ac, oc, af, fp))
            print(f"{m:6s} {C:5d} {ac:9.4f} {oc:14.4f} {ac - oc:+8.4f} "
                  f"{af:8.4f} {fp:10.4f} {af - fp:+8.4f} {ip:10.4f} {im:10.4f}")
    ce = [abs(a - b) for a, b, _, _ in rows if not math.isnan(b)]
    fe = [abs(a - b) for _, _, a, b in rows if not math.isnan(b)]
    print(f"\n  mean |error|  A_causal vs oracle_causal: {st.fmean(ce):.4f}   "
          f"A_floor vs floor_pos: {st.fmean(fe):.4f}   (both parameter-free)")
    return dict(mae_causal=st.fmean(ce), mae_floor=st.fmean(fe))


# =============================================================================== 1.1
def test_11(tab):
    hdr("1.1  MEASURE p_g AND rho PER ARM PER CELL -- are both measurable and stable within an arm?")
    print("rho is the within-fact KEEP-INDICATOR correlation (see P3_PREDICTIONS section 2:")
    print("Paper 2 stored keep sets, not scorer scores, so the score correlation is not on disk).")
    print("Three estimators per cell:")
    print("  rho_F2 / rho_F3  invert the registered form on that cell")
    print("  rho_BB           moment-match a BetaBinomial to (p_g, q_complete)")
    print("  rho_touched      moment-match to the ALL-RECORD (complete, touched) pair --")
    print("                   never touches the queried-record target")
    out = {}
    for m in MODELS:
        print(f"\n### {m}")
        print(f"{'C':>5s} {'arm':15s} {'p_g':>8s} {'p_g_pay':>8s} {'q_meas':>8s} {'c_eff*':>8s} "
              f"{'rho_F2':>8s} {'rho_F3':>8s} {'rho_BB':>8s} {'rho_tch':>8s}")
        per_arm = defaultdict(lambda: defaultdict(list))
        for C in sorted({k[1] for k in tab if k[0] == m}):
            for arm in ARMS:
                r = tab[(m, C, arm)]
                p, q, c = r["p_g"], r["q_complete"], r["c"]
                ce = math.log(q) / math.log(p) if 0 < q < 1 and 0 < p < 1 else float("nan")
                v = dict(rho_F2=rho_for_form("F2_linear", p, q, c),
                         rho_F3=rho_for_form("F3_power", p, q, c),
                         rho_BB=th.rho_from_completion(p, q, c),
                         rho_tch=th.rho_from_touched(p, r["recs_complete"] / ms.N_RECORDS,
                                                     r["recs_touched"] / ms.N_RECORDS, c),
                         c_eff=ce, p_g=p)
                for k, x in v.items():
                    per_arm[arm][k].append(x)
                print(f"{C:5d} {arm:15s} {p:8.4f} {r['p_g_payable']:8.4f} {q:8.4f} {ce:8.4f} "
                      f"{v['rho_F2']:8.4f} {v['rho_F3']:8.4f} {v['rho_BB']:8.4f} "
                      f"{v['rho_tch']:8.4f}")
        print(f"\n  stability within an arm across the {len(set(k[1] for k in tab if k[0]==m))} "
              f"budgets (mean +/- sd, and range):")
        print(f"  {'arm':15s} {'rho_F2':>22s} {'rho_F3':>22s} {'rho_BB':>22s}")
        for arm in ARMS:
            cells_ = []
            for k in ("rho_F2", "rho_F3", "rho_BB"):
                v = [x for x in per_arm[arm][k] if not math.isnan(x)]
                cells_.append(f"{st.fmean(v):.3f}+/-{st.pstdev(v):.3f} [{min(v):.3f},{max(v):.3f}]")
            print(f"  {arm:15s} " + " ".join(f"{s:>22s}" for s in cells_))
            out[(m, arm)] = {k: [x for x in per_arm[arm][k] if not math.isnan(x)]
                             for k in ("rho_F2", "rho_F3", "rho_BB", "p_g")}
    sds = [st.pstdev(v["rho_F2"]) for v in out.values()]
    print(f"\n  VERDICT 1.1: p_g and rho are both measurable at every one of the "
          f"{len(out) * 6 // 1} arm-budget cells.")
    print(f"  rho_F2 within-arm sd ranges {min(sds):.4f} to {max(sds):.4f} "
          f"(rho itself lives in [0,1]).")
    return out


# =============================================================================== 1.2
def test_12(tab, cal):
    hdr("1.2  CALIBRATE ON ONE CELL, PREDICT THE REST -- all four forms on the same held-out cells")
    print(f"Calibration cell: {CAL_MODEL} C={CAL_C}, one per arm. Held out: every other "
          f"(model, C, arm) method cell.")
    print(f"Scored as registered: hit iff |log10(pred) - log10(obs)| <= {TOL_LOG10}, both "
          f"floored at {FLOOR_OBS:.6f}.")
    print(f"Winner = lowest mean |log10 error| on the held-out cells.\n")

    rows = []
    for m in MODELS:
        for C in sorted({k[1] for k in tab if k[0] == m}):
            for arm in ARMS:
                r = tab[(m, C, arm)]
                held = not (m == CAL_MODEL and C == CAL_C)
                preds = {}
                for f in FORMS[:3]:
                    preds[f] = th.completion_pointwise(r["p_g"], r["c"],
                                                       cal[arm][f"rho_{f}"], f)
                preds["F4_nonparam"] = th.nonparam_completion(
                    r["q_complete_other"], r["c"], r["c_other"])
                rows.append(dict(m=m, C=C, arm=arm, held=held, obs=r["q_complete"], preds=preds))

    def lg(x):
        return math.log10(max(x, FLOOR_OBS))

    print(f"{'model':6s} {'C':>5s} {'arm':15s} {'obs q':>8s} "
          + "".join(f"{f.split('_')[0]:>12s}" for f in FORMS) + "   held")
    for r in rows:
        print(f"{r['m']:6s} {r['C']:5d} {r['arm']:15s} {r['obs']:8.5f} "
              + "".join(f"{r['preds'][f]:12.6f}" for f in FORMS)
              + ("   yes" if r["held"] else "   CAL"))

    held = [r for r in rows if r["held"]]
    print(f"\n  scored on {len(held)} held-out cells")
    print(f"  {'form':14s} {'mean|log10 err|':>16s} {'median':>9s} {'hit rate':>10s} "
          f"{'abs<=0.05':>10s} {'bias(log10)':>12s}")
    summary = {}
    for f in FORMS:
        errs = [lg(r["preds"][f]) - lg(r["obs"]) for r in held]
        hits = sum(1 for e in errs if abs(e) <= TOL_LOG10) / len(errs)
        absh = sum(1 for r in held if abs(r["preds"][f] - r["obs"]) <= TOL_ABS) / len(held)
        mae = st.fmean(abs(e) for e in errs)
        summary[f] = dict(mae=mae, hit=hits, abs_hit=absh, bias=st.fmean(errs))
        print(f"  {f:14s} {mae:16.4f} {st.median(abs(e) for e in errs):9.4f} "
              f"{hits:10.1%} {absh:10.1%} {st.fmean(errs):+12.4f}")

    # per-model breakdown -- M3 is a cross-model extrapolation and deserves its own line
    for m in MODELS:
        sub = [r for r in held if r["m"] == m]
        print(f"\n  {m} only (n={len(sub)}):")
        for f in FORMS:
            errs = [lg(r["preds"][f]) - lg(r["obs"]) for r in sub]
            print(f"    {f:14s} mean|log10| {st.fmean(abs(e) for e in errs):7.4f}   "
                  f"hit {sum(1 for e in errs if abs(e) <= TOL_LOG10) / len(errs):6.1%}")

    win = min(FORMS, key=lambda f: (summary[f]["mae"], -summary[f]["hit"]))
    passed = any(s["hit"] >= 0.70 for s in summary.values())
    print(f"\n  WINNER: {win}  (mean |log10 err| {summary[win]['mae']:.4f}, "
          f"hit rate {summary[win]['hit']:.1%})")
    print(f"  VERDICT 1.2: {'PASS' if passed else 'FAIL'} "
          f"-- {'at least one' if passed else 'no'} form predicts >= 70% of held-out cells "
          f"within the registered tolerance.")
    return win, summary, rows, passed


# =============================================================================== 1.3
def test_13(tab, cal, win):
    hdr("1.3  CROSSOVER (P2.3) -- does the predicted crossover set match the observed winners?")
    accs = ms.accuracy_table()
    print(f"Scoring form (declared in Stage 0 as 'whichever wins 1.2'): {win}")
    print("Predicted set = {arm : predicted completion > A_floor}, comparator (a) per-slot "
          "primary, (b) union-lifted secondary.\n")

    ok = {}
    for m in MODELS:
        C = max(k[1] for k in tab if k[0] == m)
        c = st.fmean(tab[k]["c"] for k in tab if k[0] == m)
        L = st.fmean(tab[k]["L"] for k in tab if k[0] == m)
        af = th.a_floor(C, c, L)
        fp_acc = accs[m][(C, "floor_pos")][0]
        observed = sorted(a for a in ARMS if accs[m][(C, a)][0] > fp_acc)
        print(f"### {m} C={C}    A_floor = {af:.4f}   floor_pos accuracy = {fp_acc:.4f}")
        print(f"  {'arm':15s} {'acc':>8s} {'beats floor?':>13s} {'q_pred (a)':>11s} "
              f"{'>A_floor':>9s} {'q_pred (b)':>11s} {'>A_floor':>9s} {'q_any meas':>11s}")
        pred_a, pred_b = [], []
        for arm in ARMS:
            r = tab[(m, C, arm)]
            if win == "F4_nonparam":
                q = th.nonparam_completion(r["q_complete_other"], r["c"], r["c_other"])
            else:
                q = th.completion_pointwise(r["p_g"], r["c"], cal[arm][f"rho_{win}"], win)
            qb = th.union_lift(cal[arm]["s_eff"])(q)
            if q > af:
                pred_a.append(arm)
            if qb > af:
                pred_b.append(arm)
            print(f"  {arm:15s} {accs[m][(C, arm)][0]:8.4f} "
                  f"{('YES' if accs[m][(C, arm)][0] > fp_acc else 'no'):>13s} "
                  f"{q:11.6f} {('YES' if q > af else 'no'):>9s} "
                  f"{qb:11.6f} {('YES' if qb > af else 'no'):>9s} {r['q_any']:11.4f}")
        print(f"\n  observed winner set : {observed or ['(empty)']}")
        print(f"  predicted set (a)   : {sorted(pred_a) or ['(empty)']}   "
              f"MATCH={sorted(pred_a) == observed}")
        print(f"  predicted set (b)   : {sorted(pred_b) or ['(empty)']}   "
              f"MATCH={sorted(pred_b) == observed}")
        ok[m] = (sorted(pred_a) == observed, sorted(pred_b) == observed)
        print()
    passed = ok["M3"][0]
    print(f"  VERDICT 1.3 (gate = M3 at its top budget, comparator (a)): "
          f"{'PASS' if passed else 'FAIL'}")
    print(f"  M2 corroboration cell, comparator (a): "
          f"{'match' if ok['M2'][0] else 'mismatch'}")
    return passed, ok


# =============================================================================== 1.4
def test_14(tab):
    hdr("1.4  PER-HEAD vs UNION COMPLETENESS (P3.3) -- recomputed per head on the slot dumps")
    print("q_mean = complete in a typical (layer, KV-head) slot;  q_any = complete in >= 1 slot.")
    print("For floor_pos the two coincide by construction (one global keep-set).\n")
    rows = []
    for m in MODELS:
        print(f"### {m}")
        print(f"{'C':>5s} {'arm':15s} {'acc':>8s} {'q_mean':>8s} {'q_any':>8s} {'q_maj':>8s} "
              f"{'q_any/q_mean':>13s}")
        for C in sorted({k[1] for k in tab if k[0] == m}):
            for arm in ALL_ARMS:
                r = tab[(m, C, arm)]
                ratio = r["q_any"] / r["q_mean"] if r["q_mean"] > 0 else float("inf")
                print(f"{C:5d} {arm:15s} {r['acc']:8.4f} {r['q_mean']:8.4f} {r['q_any']:8.4f} "
                      f"{r['q_maj']:8.4f} {ratio:13.2f}")
                rows.append(dict(m=m, C=C, arm=arm, acc=r["acc"], q_mean=r["q_mean"],
                                 q_any=r["q_any"], q_maj=r["q_maj"]))
        print()

    def corrs(sub, label):
        a = [r["acc"] for r in sub]
        rm, n = pearson([r["q_mean"] for r in sub], a)
        ra, _ = pearson([r["q_any"] for r in sub], a)
        rj, _ = pearson([r["q_maj"] for r in sub], a)
        sm, _ = spearman([r["q_mean"] for r in sub], a)
        sa, _ = spearman([r["q_any"] for r in sub], a)
        print(f"  {label:34s} n={n:3d}   r(acc,q_mean)={rm:+.4f}  r(acc,q_any)={ra:+.4f}  "
              f"r(acc,q_maj)={rj:+.4f}   rho_s: mean={sm:+.4f} any={sa:+.4f}")
        return rm, ra

    print("PART 1 of the registered prediction: accuracy tracks q_mean, not q_any.")
    rm_all, ra_all = corrs([r for r in rows if not math.isnan(r["acc"])], "all cells, both models")
    for m in MODELS:
        corrs([r for r in rows if r["m"] == m and not math.isnan(r["acc"])], f"{m} only")
    corrs([r for r in rows if r["arm"] != "floor_pos" and not math.isnan(r["acc"])],
          "method arms only")

    print("\nPART 2: does the per-head measure resolve the M3 C=512 inversion?")
    m, C = "M3", 512
    fp = tab[(m, C, "floor_pos")]
    print(f"  floor_pos: acc {fp['acc']:.4f}  q_mean {fp['q_mean']:.4f}  q_any {fp['q_any']:.4f}")
    inv_any, inv_mean = [], []
    for arm in ARMS:
        r = tab[(m, C, arm)]
        above_any = r["q_any"] > fp["q_any"]
        above_mean = r["q_mean"] > fp["q_mean"]
        below_acc = r["acc"] < fp["acc"]
        print(f"  {arm:15s} acc {r['acc']:.4f} ({'below' if below_acc else 'above'} floor)  "
              f"q_any {r['q_any']:.4f} ({'ABOVE' if above_any else 'below'} floor)  "
              f"q_mean {r['q_mean']:.4f} ({'ABOVE' if above_mean else 'below'} floor)")
        if below_acc and above_any:
            inv_any.append(arm)
        if below_acc and above_mean:
            inv_mean.append(arm)
    print(f"\n  arms the UNION measure gets WRONG (union says above floor, accuracy says below): "
          f"{inv_any or ['(none)']}")
    print(f"  arms the PER-HEAD measure gets WRONG (same test on q_mean): "
          f"{inv_mean or ['(none)']}")
    part2 = ("snapkv" in inv_any) and ("snapkv" not in inv_mean)
    part1 = rm_all > ra_all
    print(f"\n  part 1 (r(acc,q_mean) > r(acc,q_any)): {'PASS' if part1 else 'FAIL'} "
          f"({rm_all:+.4f} vs {ra_all:+.4f})")
    print(f"  part 2 (per-head resolves the snapkv inversion the union creates): "
          f"{'PASS' if part2 else 'FAIL'}")
    print(f"  VERDICT 1.4: {'PASS' if (part1 and part2) else 'FAIL'} "
          f"-- both parts required, as registered.")
    return (part1 and part2), part1, part2, rm_all, ra_all


# =============================================================================== 1.5
def test_15(tab):
    hdr("1.5  DELTA_head vs PER-HEAD KEEP-SET DIVERGENCE (P3.2)")
    n9 = ms.n9_cells()
    print("Divergence D = recs_touched_any / recs_touched_mean, averaged over the four method")
    print("arms at each budget. Delta_head is measured on ORACLE arms (N9), so the pairing is")
    print("per budget, not per arm.\n")
    res = {}
    for m in MODELS:
        d = n9[m]
        print(f"### {m}  ({d['n_kv_heads']} KV heads, delta_head_measured={d['delta_head_measured']}, "
              f"n9 holes {d['coverage']['holes']}/{d['coverage']['expected']})")
        print(f"{'C':>5s} {'D (divergence)':>15s} {'Delta_head':>11s} {'CI':>22s} "
              f"{'degenerate':>11s} {'usable':>7s}")
        xs, ys = [], []
        for C in sorted({k[1] for k in tab if k[0] == m}):
            cell = d["cells"].get(f"C{C}")
            D = st.fmean(tab[(m, C, a)]["recs_touched_any"] / tab[(m, C, a)]["recs_touched_mean"]
                         for a in ARMS)
            if cell is None:
                print(f"{C:5d} {D:15.4f} {'n/a':>11s} {'(cell absent from n9)':>22s} "
                      f"{'-':>11s} {'no':>7s}")
                continue
            dh = cell["delta_head"]["mean"]
            deg = cell["delta_head"]["structurally_degenerate"]
            usable = (dh != 0.0) and not deg
            print(f"{C:5d} {D:15.4f} {dh:+11.4f} "
                  f"{('[%+.4f, %+.4f]' % tuple(cell['delta_head']['ci'])):>22s} "
                  f"{str(deg):>11s} {('yes' if usable else 'no'):>7s}")
            if dh != 0.0:
                xs.append(D)
                ys.append(dh)
        if len(xs) >= 2:
            mono = all((xs[i] < xs[i + 1]) == (ys[i] > ys[i + 1]) for i in range(len(xs) - 1))
            print(f"\n  non-zero Delta_head points: n={len(xs)}   "
                  f"D {['%.3f' % v for v in xs]}   Delta_head {['%+.4f' % v for v in ys]}")
            print(f"  monotone with the predicted sign (more divergence -> more negative): {mono}")
            r, n = pearson(xs, ys)
            print(f"  Pearson r = {r if not math.isnan(r) else float('nan')} on n={n} "
                  f"(reported only to be explicit that n is too small to mean anything)")
            res[m] = dict(n=len(xs), monotone=mono, D=xs, dh=ys)
        else:
            print(f"\n  fewer than two non-zero Delta_head points -- nothing to correlate")
            res[m] = dict(n=len(xs), monotone=None)
        print()
    both = all(v.get("monotone") is True for v in res.values())
    print(f"  VERDICT 1.5: {'PASS' if both else 'FAIL/INCONCLUSIVE'} -- registered as a "
          f"sign-and-ordering check on at most two points per model, not a correlation.")
    return both, res


def main() -> int:
    check_frozen()
    tab = ms.cells("line")

    cal = {}
    for arm in ARMS:
        r = tab[(CAL_MODEL, CAL_C, arm)]
        cal[arm] = dict(
            s_eff=math.log(1.0 - r["q_any"]) / math.log(1.0 - r["q_complete"]),
            **{f"rho_{f}": rho_for_form(f, r["p_g"], r["q_complete"], r["c"])
               for f in FORMS[:3]})

    t10 = test_10(tab)
    t11 = test_11(tab)
    win, summary, rows12, p12 = test_12(tab, cal)
    p13, ok13 = test_13(tab, cal, win)
    p14, p14a, p14b, rm, ra = test_14(tab)
    p15, res15 = test_15(tab)

    hdr("STAGE 1 GATE")
    print("  1.1 measurable and stable ............ reported above, no pass/fail attached")
    print(f"  1.2 out-of-sample c_eff forms ........ {'PASS' if p12 else 'FAIL'}  "
          f"(winner {win})")
    print(f"  1.3 crossover ........................ {'PASS' if p13 else 'FAIL'}")
    print(f"  1.4 per-head vs union completeness ... {'PASS' if p14 else 'FAIL'}  "
          f"(part1 {'PASS' if p14a else 'FAIL'}, part2 {'PASS' if p14b else 'FAIL'})")
    print(f"  1.5 Delta_head vs divergence ......... "
          f"{'PASS' if p15 else 'FAIL/INCONCLUSIVE'}")
    gate = p13 or p14
    print(f"\n  GATE (1.3 or 1.4): {'PASS -- Stage 2 is earned' if gate else 'FAIL -- STOP'}")
    if not gate:
        print("  Neither test explains something Paper 2 could not. Per plan section 6 this is")
        print("  the kill criterion: the co-retention answer, as operationalised here, is wrong,")
        print("  and the programme stops at a cost of zero GPU-hours.")

    json.dump(dict(frozen_hash=digest(PRED), winner=win,
                   t10=t10,
                   t12=dict(summary={k: v for k, v in summary.items()}, passed=p12),
                   t13=dict(passed=p13, per_model={k: list(v) for k, v in ok13.items()}),
                   t14=dict(passed=p14, part1=p14a, part2=p14b,
                            r_acc_qmean=rm, r_acc_qany=ra),
                   t15=dict(passed=p15, detail={k: {kk: vv for kk, vv in v.items()}
                                                for k, v in res15.items()}),
                   gate=gate),
              open(HERE / "out" / "stage1_results.json", "w", encoding="utf-8"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
