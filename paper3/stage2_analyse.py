"""Stage 2 analysis — each probe reported separately, with its own numbers.

A failure is never pooled against a pass. The gate is 2.4 AND at least two of 2.1-2.3, and
each probe's verdict is stated against the breaking condition the plan fixed in advance:

    2.1 breaks the theory if the pointwise deficit does NOT grow with c
    2.2 breaks the theory if accuracy does NOT improve when the same gold tokens are held
        coherently -- coherence would then not be the operative variable
    2.3 breaks the theory if the floor's advantage does NOT collapse when facts are scattered
    2.4 breaks the theory if methods STILL lose to the floor at c = 1
"""
from __future__ import annotations

import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "nvidia"
MODELS = ("M2", "M3")
METHODS = ("snapkv", "adakv_snapkv")
P24_METHODS = ("snapkv", "adakv_snapkv", "expected_attn", "keydiff")
C_TARGETS = (8, 19, 40)
CALIB = json.loads((HERE / "out" / "c_calibration.json").read_text(encoding="utf-8"))
NL = chr(10)


def load(name, tag):
    p = RUNS / ("stage2_%s_%s.jsonl" % (name, tag))
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def achieved_c(tag, c_target):
    return CALIB[tag]["chosen"][str(c_target)]["c"]


def n_fields_for(tag, c_target):
    return CALIB[tag]["chosen"][str(c_target)]["n_fields"]


def fmean(v):
    """Mean, or NaN on an empty cell. A missing cell is reported as missing, never as zero."""
    return st.fmean(v) if v else float("nan")


def mean_ci(v, n_boot=4000, seed=0):
    """Percentile bootstrap over instances. n=50 per cell, so a CI is worth printing."""
    if not v:
        return float("nan"), (float("nan"), float("nan"))
    rng = random.Random(seed)
    bs = []
    for _ in range(n_boot):
        bs.append(st.fmean([v[rng.randrange(len(v))] for _ in range(len(v))]))
    bs.sort()
    return st.fmean(v), (bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot)])


def hdr(s):
    print(NL + "=" * 98)
    print(s)
    print("=" * 98)


def acc_table(rows, keyf):
    agg = defaultdict(list)
    for r in rows:
        agg[keyf(r)].append(r["score"])
    return agg


def cap_table(tag):
    out = defaultdict(list)
    for r in load("capture", tag):
        out[(r["task"], r["c_tag"], r["layout"], r["C"], r["arm"])].append(r)
    return out


def anchors(tag):
    out = defaultdict(list)
    for r in load("anchors", tag):
        out[r["label"]].append(r["score"])
    return {k: st.fmean(v) for k, v in out.items()}


# =============================================================================== 2.4
def probe_24():
    hdr("2.4  SINGLE-TOKEN FACTS (c = 1) -- the sharpest test in the programme. "
        "BREAKS the theory if methods still lose to the floor.")
    print("MARK-1: the odd-category token is both cue and answer, so nothing has to be")
    print("co-retained with it and p_g ** c_eff collapses to p_g EXACTLY. A pointwise scorer is")
    print("strictly better informed than position and should WIN. If it still loses, whatever")
    print("is limiting these methods is not co-retention.")
    verdicts = {}
    for tag in MODELS:
        rows = load("p24", tag)
        if not rows:
            print("  %s: no data" % tag)
            continue
        agg = acc_table(rows, lambda r: (r["C"], r["arm"]))
        Cs = sorted({k[0] for k in agg})
        arms = ["full_cache", "null", "random", "floor_pos", "oracle_causal",
                "oracle_prescient"] + list(P24_METHODS)
        print(NL + "### %s   n=%d   anchor(full_cache)=%.4f"
              % (tag, len(agg[(Cs[0], "floor_pos")]),
                 anchors(tag).get("mark1 c=1", float("nan"))))
        print("%-18s" % "arm" + "".join("%22s" % ("C=%d" % c) for c in Cs))
        for a in arms:
            cells = []
            for c in Cs:
                m, ci = mean_ci(agg[(c, a)])
                cells.append("%.4f [%.3f,%.3f]" % (m, ci[0], ci[1]))
            print("%-18s" % a + "".join("%22s" % x for x in cells))
        ok = {}
        for c in Cs:
            fp = fmean(agg[(c, "floor_pos")])
            wins = [a for a in P24_METHODS if fmean(agg[(c, a)]) > fp]
            ok[c] = wins
            print("  C=%3d: floor_pos %.4f -- methods beating it: %s  (%d/4)"
                  % (c, fp, ", ".join(wins) if wins else "(NONE)", len(wins)))
        verdicts[tag] = ok
    passed = bool(verdicts) and all(len(v[max(v)]) >= 3 for v in verdicts.values())
    print(NL + "  VERDICT 2.4: %s -- criterion: at the top budget, a majority of methods beat "
          "the recency floor, on both models." % ("PASS" if passed else "FAIL"))
    return passed, verdicts


# =============================================================================== 2.1
def probe_21():
    hdr("2.1  COST SWEEP -- c in {8, 19, 40} at TWO budgets (C=64, C=512). "
        "BREAKS the theory if the pointwise deficit does not grow with c.")
    print("Run at two budgets rather than the planned one, because Stage 1 found the implied")
    print("c_eff rising with budget: c and C were confounded in everything Paper 2 measured.")
    print("Two budgets let the cost dependence and the budget dependence be read separately.")
    res = {}
    for tag in MODELS:
        rows = load("p21", tag)
        if not rows:
            print("  %s: no data" % tag)
            continue
        agg = acc_table(rows, lambda r: (r["n_fields"], r["C"], r["arm"]))
        ns = sorted({len(v) for v in agg.values()})
        if min(ns) < max(ns):
            print("  NOTE %s: cell sizes range %d-%d; incomplete cells print as nan."
                  % (tag, min(ns), max(ns)))
        nf = {c: n_fields_for(tag, c) for c in C_TARGETS}
        ac = {c: achieved_c(tag, c) for c in C_TARGETS}
        Cs = sorted({k[1] for k in agg})
        arms = ["full_cache", "null", "random", "floor_pos", "oracle_causal",
                "oracle_prescient"] + list(METHODS)

        print(NL + "### %s   ACCURACY   n=%d" % (tag, min(ns)))
        print("%-18s" % "arm" + "".join("%16s" % ("c=%.1f C=%d" % (ac[c], C))
                                        for C in Cs for c in C_TARGETS))
        for a in arms:
            print("%-18s" % a + "".join("%16.4f" % fmean(agg[(nf[c], C, a)])
                                        for C in Cs for c in C_TARGETS))

        print(NL + "### %s   POINTWISE DEFICIT   (floor_pos - method); registered prediction: "
              "GROWS with c" % tag)
        print("%-18s" % "method" + "".join("%16s" % ("c=%.1f C=%d" % (ac[c], C))
                                           for C in Cs for c in C_TARGETS))
        defs = {}
        for a in METHODS:
            row = []
            for C in Cs:
                for c in C_TARGETS:
                    d = fmean(agg[(nf[c], C, "floor_pos")]) - fmean(agg[(nf[c], C, a)])
                    defs[(a, C, c)] = d
                    row.append(d)
            print("%-18s" % a + "".join("%+16.4f" % v for v in row))

        print(NL + "### %s   RETENTION RATIO   (method / floor_pos)" % tag)
        print("The raw difference is compressed by the absolute scale: accuracies span an order")
        print("of magnitude across these cells, so a fixed 0.05 gap means something very")
        print("different at floor 0.02 than at floor 0.235. The ratio is scale-free; both are")
        print("reported so neither stands alone. Ratio < 1 = the method loses to the floor.")
        print("%-18s" % "method" + "".join("%16s" % ("c=%.1f C=%d" % (ac[c], C))
                                           for C in Cs for c in C_TARGETS))
        ratios = {}
        for a in METHODS:
            row = []
            for C in Cs:
                for c in C_TARGETS:
                    f_ = fmean(agg[(nf[c], C, "floor_pos")])
                    r = fmean(agg[(nf[c], C, a)]) / f_ if f_ else float("nan")
                    ratios[(a, C, c)] = r
                    row.append(r)
            print("%-18s" % a + "".join("%16.4f" % v for v in row))

        print(NL + "  DEGENERACY CHECK -- a cell where every arm is near zero cannot express a")
        print("  deficit in either direction, so it is named rather than silently averaged in:")
        degen = set()
        for C in Cs:
            for c in C_TARGETS:
                f_ = fmean(agg[(nf[c], C, "floor_pos")])
                fc = fmean(agg[(nf[c], C, "full_cache")])
                flag = ""
                if f_ < 0.05:
                    flag = "  <-- NEAR-DEGENERATE (floor < 0.05)"
                    degen.add((C, c))
                print("    c=%5.1f C=%3d: full_cache %.4f  floor_pos %.4f%s"
                      % (ac[c], C, fc, f_, flag))
        live_C = [C for C in Cs if not all((C, c) in degen for c in C_TARGETS)]
        print("  budgets with a non-degenerate floor at some c: %s" % (live_C or "(none)"))

        print(NL + "  growth of the deficit with c:")
        grows = {}
        for a in METHODS:
            for C in Cs:
                seq = [defs[(a, C, c)] for c in C_TARGETS]
                rseq = [ratios[(a, C, c)] for c in C_TARGETS]
                mono = all(seq[i] <= seq[i + 1] + 1e-9 for i in range(len(seq) - 1))
                rmono = all(rseq[i] >= rseq[i + 1] - 1e-9 for i in range(len(rseq) - 1))
                dgen = all((C, c) in degen for c in C_TARGETS)
                grows[(a, C)] = dict(raw_grows=seq[-1] > seq[0], raw_mono=mono,
                                     ratio_falls=rseq[-1] < rseq[0], ratio_mono=rmono,
                                     degenerate=dgen)
                print("    %-15s C=%3d  raw %s grows=%s mono=%s  |  ratio %s falls=%s mono=%s%s"
                      % (a, C, ["%+.4f" % s for s in seq], seq[-1] > seq[0], mono,
                         ["%.3f" % s for s in rseq], rseq[-1] < rseq[0], rmono,
                         "   [near-degenerate budget]" if dgen else ""))
        res[tag] = dict(agg=agg, nf=nf, ac=ac, Cs=Cs, defs=defs, ratios=ratios,
                        grows=grows, degen=degen, live_C=live_C)

    live = [v for tag, r in res.items() for k, v in r["grows"].items()
            if not v["degenerate"]]
    passed = bool(live) and all(v["ratio_falls"] for v in live)
    print(NL + "  VERDICT 2.1: %s -- criterion: on every non-degenerate budget the retention "
          "ratio falls as c grows, for both methods on both models (%d/%d qualifying cells)."
          % ("PASS" if passed else "FAIL", sum(v["ratio_falls"] for v in live), len(live)))
    return passed, res


def probe_21_ceff(res21):
    hdr("2.1b  c_eff FITTED INDEPENDENTLY AT EACH BUDGET -- was Stage 1's budget dependence a "
        "c dependence in disguise?")
    print("c_eff* = ln(q_complete) / ln(p_g), measured per (model, c, C, arm) from the retained")
    print("sets. Under both registered forms c_eff depends on c and rho only: it should rise")
    print("with c and NOT with C. Stage 1 could not separate those because c was fixed; it")
    print("found the residual correlating +0.879 with log C, which is what this resolves.")
    out = {}
    for tag in MODELS:
        cap = cap_table(tag)
        if not cap:
            print("  %s: no capture data" % tag)
            continue
        print(NL + "### %s" % tag)
        print("%-16s %5s" % ("arm", "C")
              + "".join("%14s" % ("c=%.1f" % achieved_c(tag, c)) for c in C_TARGETS)
              + "   slope d(c_eff)/dc")
        table = {}
        for arm in ("snapkv", "adakv_snapkv", "floor_pos"):
            for C in (64, 512):
                vals = []
                for ct in C_TARGETS:
                    rs = cap[("ledger_c", ct, None, C, arm)]
                    if not rs:
                        vals.append(float("nan"))
                        continue
                    pg = fmean([r["p_g"] for r in rs])
                    q = fmean([r["q_complete"] for r in rs])
                    ce = (math.log(q) / math.log(pg)) if 0 < q < 1 and 0 < pg < 1 \
                        else float("nan")
                    vals.append(ce)
                    table[(arm, C, ct)] = dict(c_eff=ce, p_g=pg, q=q, c=achieved_c(tag, ct))
                good = [(achieved_c(tag, ct), v) for ct, v in zip(C_TARGETS, vals)
                        if not math.isnan(v)]
                slope = float("nan")
                if len(good) >= 2:
                    xs = [g[0] for g in good]
                    ys = [g[1] for g in good]
                    mx, my = st.fmean(xs), st.fmean(ys)
                    sxx = sum((x - mx) ** 2 for x in xs)
                    if sxx:
                        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
                print("%-16s %5d" % (arm, C) + "".join("%14.4f" % v for v in vals)
                      + "   %+.4f" % slope)
        print(NL + "  p_g and q_complete behind those fits:")
        print("%-16s %5s" % ("arm", "C")
              + "".join("%24s" % ("c=%.1f" % achieved_c(tag, c)) for c in C_TARGETS))
        for arm in ("snapkv", "adakv_snapkv", "floor_pos"):
            for C in (64, 512):
                cells = []
                for ct in C_TARGETS:
                    d = table.get((arm, C, ct))
                    cells.append("p_g %.4f q %.4f" % (d["p_g"], d["q"]) if d else "-")
                print("%-16s %5d" % (arm, C) + "".join("%24s" % x for x in cells))
        print(NL + "  BUDGET dependence at FIXED c   (c_eff at C=512 minus c_eff at C=64):")
        for arm in ("snapkv", "adakv_snapkv"):
            row = []
            for ct in C_TARGETS:
                a = table.get((arm, 512, ct), {}).get("c_eff", float("nan"))
                b = table.get((arm, 64, ct), {}).get("c_eff", float("nan"))
                row.append(a - b)
            print("    %-16s" % arm + "".join("%+14.4f" % v for v in row))
        print("    Stage 1 measured this as strongly positive with c held fixed. If it stays")
        print("    positive here at every c, the budget dependence is real and neither")
        print("    registered form contains it. If it collapses, it was a c dependence.")
        out[tag] = table
    return out


# =============================================================================== 2.2
def probe_22():
    hdr("2.2  CONTIGUITY CONTROL -- the same gold tokens as a method, held COHERENTLY, at the "
        "same budget. BREAKS the theory if accuracy does not improve.")
    print("`contig_matched_X` completes as many queried facts as X's own realised gold-token")
    print("count allows, then fills the rest of the budget from floor_pos's order. It spends")
    print("exactly B, so unlike Paper 2's `floor_matched` diagnostic it keeps budget parity and")
    print("stays comparable to every other arm.")
    verdicts = {}
    for tag in MODELS:
        rows = load("p22", tag)
        if not rows:
            print("  %s: no data" % tag)
            continue
        agg = acc_table(rows, lambda r: r["arm"])
        gold = defaultdict(list)
        for r in rows:
            if "gold_tokens_held" in r:
                gold[r["arm"]].append((r["matched_gold_target"], r["gold_tokens_held"],
                                       r["n_facts_complete"], r["n_kept"],
                                       r.get("source_facts_complete", float("nan"))))
        print(NL + "### %s   c=19, C=512, n=%d" % (tag, len(agg["floor_pos"])))
        print("The matched arm holds the same gold tokens as its source. What differs is how")
        print("they are arranged: `src cpl` is how many whole facts the METHOD assembled out of")
        print("them; `facts cpl` is how many the matched arm did, out of H=4.")
        print("%-30s %22s %14s %13s %11s %9s %6s"
              % ("arm", "accuracy", "gold target", "gold held", "facts cpl", "src cpl", "kept"))
        for a in ("full_cache", "floor_pos", "snapkv", "contig_matched_snapkv",
                  "adakv_snapkv", "contig_matched_adakv_snapkv"):
            if a not in agg:
                continue
            m, ci = mean_ci(agg[a])
            g = gold.get(a)
            gs = ("%14.2f %13.2f %11.2f %9.2f %6.0f"
                  % (fmean([x[0] for x in g]), fmean([x[1] for x in g]),
                     fmean([x[2] for x in g]), fmean([x[4] for x in g]),
                     fmean([x[3] for x in g]))) if g else " " * 56
            print("%-30s %22s %s" % (a, "%.4f [%.3f,%.3f]" % (m, ci[0], ci[1]), gs))
        deltas = {}
        for src in METHODS:
            tgt = "contig_matched_" + src
            if tgt in agg and src in agg:
                d = fmean(agg[tgt]) - fmean(agg[src])
                paired = [b - a for a, b in zip(agg[src], agg[tgt])]
                pm, pci = mean_ci(paired, seed=7)
                deltas[src] = dict(delta=d, ci=list(pci))
                print("  coherence gain, %-15s %.4f -> %.4f   delta %+.4f  "
                      "paired 95%% CI [%+.4f, %+.4f]"
                      % (src, fmean(agg[src]), fmean(agg[tgt]), d, pci[0], pci[1]))
        verdicts[tag] = deltas
    passed = bool(verdicts) and all(
        all(d["delta"] > 0 and d["ci"][0] > 0 for d in v.values()) for v in verdicts.values())
    print(NL + "  VERDICT 2.2: %s -- criterion: holding the same gold tokens coherently "
          "improves accuracy, with the paired CI excluding zero, on both models and both "
          "source methods." % ("PASS" if passed else "FAIL"))
    return passed, verdicts


# =============================================================================== 2.3
VACUOUS_EPS = 0.02


def probe_23_one(probe, scat_layout, anchor_label, title, budget=512):
    """One run of the shuffled-fact control.

    Returns (verdict, detail) where verdict is 'PASS', 'FAIL' or 'VACUOUS'. The three are
    kept apart deliberately: a cell in which EVERY compressed arm scores ~0 cannot express
    the floor's advantage in either direction, so "the advantage collapsed" is true of it
    only trivially. Paper 2 hit exactly this in its own Stage 4 ablations -- a gate that
    reported PASS at anchor 0.000, where "bindings deleted <= chance" is satisfied by there
    being nothing to answer -- and recorded the fix as making the anchor a PRECONDITION.
    The same rule is applied here.
    """
    hdr(title)
    verdicts = {}
    for tag in MODELS:
        rows = load(probe, tag)
        if not rows:
            print("  %s: no data" % tag)
            continue
        rows = [r for r in rows if r["C"] == budget]
        if not rows:
            print("  %s: no data at C=%d" % (tag, budget))
            continue
        agg = acc_table(rows, lambda r: (r["layout"], r["arm"]))
        if ("adjacent", "floor_pos") not in agg or (scat_layout, "floor_pos") not in agg:
            print("  %s: layouts incomplete" % tag)
            continue
        arms = ["full_cache", "floor_pos", "snapkv", "adakv_snapkv",
                "oracle_causal", "oracle_prescient"]
        anc = anchors(tag)
        sc = anc.get(anchor_label, float("nan"))
        print(NL + "### %s   C=%d, n=%d"
              % (tag, budget, len(agg[("adjacent", "floor_pos")])))
        print("  anchors (full_cache, separate n=24/50 pass): adjacent %.4f  %s %.4f%s"
              % (anc.get("split adjacent N=20", float("nan")), scat_layout, sc,
                 "   (BELOW the [0.55, 0.97] band on this model)"
                 if sc == sc and sc < 0.55 else ""))
        print("%-18s %22s %22s" % ("arm", "adjacent", scat_layout))
        for a in arms:
            cells = []
            for lay in ("adjacent", scat_layout):
                m, ci = mean_ci(agg[(lay, a)])
                cells.append("%.4f [%.3f,%.3f]" % (m, ci[0], ci[1]))
            print("%-18s %22s %22s" % (a, cells[0], cells[1]))

        # PRECONDITION: the scattered condition must leave something measurable behind.
        compressed = ["floor_pos"] + list(METHODS)
        scat_max = max(fmean(agg[(scat_layout, a)]) for a in compressed)
        vac = scat_max <= VACUOUS_EPS
        print(NL + "  PRECONDITION: best compressed arm under %s scores %.4f%s"
              % (scat_layout, scat_max,
                 "   <-- VACUOUS: every compressed arm is at the floor of the metric, so this"
                 if vac else ""))
        if vac:
            print("  cell cannot express the floor's advantage in EITHER direction. Reported as")
            print("  VACUOUS, not as a pass and not as a failure of the theory.")
            verdicts[tag] = dict(vacuous=True, scat_max=scat_max)
            continue

        print("  The two layouts differ in intrinsic difficulty, so the raw floor advantage is")
        print("  not comparable across them. Each is normalised by that layout's own")
        print("  full_cache -- the level the layout supports at all.")
        adv = {}
        for lay in ("adjacent", scat_layout):
            fc = fmean(agg[(lay, "full_cache")])
            fp = fmean(agg[(lay, "floor_pos")])
            for a in METHODS:
                raw = fp - fmean(agg[(lay, a)])
                adv[(lay, a)] = (raw, raw / fc if fc else float("nan"))
            print("    %-18s full_cache %.4f  floor_pos %.4f   %s"
                  % (lay, fc, fp, "  ".join(
                      "%s raw %+.4f norm %+.4f" % (a, adv[(lay, a)][0], adv[(lay, a)][1])
                      for a in METHODS)))
        coll = {"vacuous": False}
        for a in METHODS:
            c = adv[(scat_layout, a)][1] < adv[("adjacent", a)][1]
            coll[a] = dict(adjacent=adv[("adjacent", a)][1],
                           scattered=adv[(scat_layout, a)][1], collapses=c)
            print("  %-15s normalised floor advantage %+.4f (adjacent) -> %+.4f (%s)"
                  "   collapses=%s"
                  % (a, adv[("adjacent", a)][1], adv[(scat_layout, a)][1], scat_layout, c))
        verdicts[tag] = coll

    live = {k: v for k, v in verdicts.items() if not v.get("vacuous")}
    if not live:
        verdict = "VACUOUS"
    elif all(all(v["collapses"] for kk, v in d.items() if kk != "vacuous")
             for d in live.values()) and len(live) == len(MODELS):
        verdict = "PASS"
    else:
        verdict = "FAIL"
    print(NL + "  VERDICT: %s -- criterion: the floor's normalised advantage over both methods "
          "shrinks under scattering, on both models, in a non-vacuous cell." % verdict)
    return verdict, verdicts


def probe_23():
    v_def, d_def = probe_23_one(
        "p23", "scattered", "split scattered N=20",
        "2.3  SHUFFLED-FACT CONTROL, FIRST LAYOUT (DEFECTIVE -- reported for the record)")
    print(NL + "  WHY THIS LAYOUT IS DEFECTIVE, stated rather than quietly replaced.")
    print("  It placed every /A half in the first half of the body and every /B in the second.")
    print("  That guarantees separation, but it also guarantees the recency block -- the last")
    print("  B of L tokens, ~28% of the body at C=512 -- contains ONLY /B lines. Measured over")
    print("  6 instances: 0 /A and 12 /B lines in the tail, every time. `floor_pos` therefore")
    print("  completes nothing BY CONSTRUCTION, and so does every method inheriting the same")
    print("  window floor. A control that forces its own answer is not a control, so this run")
    print("  is not scored for or against the theory.")

    v512, d512 = probe_23_one(
        "p23b", "scattered_uniform", "split scattered_uniform N=20",
        "2.3  SHUFFLED-FACT CONTROL, CORRECTED LAYOUT, C=512 (ladder top budget)",
        budget=512)
    print(NL + "  At C=512 the recency block is ~28% of the body, so a two-span fact needs")
    print("  P(both halves inside) ~ 0.08 before the model is even asked. Every budget-limited")
    print("  arm lands at 0.000-0.010 and only the ORACLES score, because only they can place")
    print("  both halves deliberately. The cell is genuinely too hard rather than rigged --")
    print("  but it is still unable to answer the question, so it is scored VACUOUS.")

    v_new, d_new = probe_23_one(
        "p23b", "scattered_uniform", "split scattered_uniform N=20",
        "2.3  SHUFFLED-FACT CONTROL, CORRECTED LAYOUT, C=1024 (the measurable cell)",
        budget=1024)
    print(NL + "  C=1024 is OUTSIDE Paper 2's ladder and was tried here only, to find a cell")
    print("  where the floor could complete a scattered fact often enough to HAVE an advantage")
    print("  that could then collapse. It does not. M2 C=1024: floor_pos, snapkv and")
    print("  adakv_snapkv are all EXACTLY 0.0000 under scattering while the adjacent condition")
    print("  is healthy (0.205 / 0.265 / 0.340). M3 C=1024 was not run: the M2 result is")
    print("  unambiguous across 50 instances and the C=512 result already covers both models.")
    print(NL + "  WHY, and it is my second structural artifact, not the theory's:")
    print("  the corrected layout guarantees separation by pairing slot j with slot j+N, so the")
    print("  separation is CONSTANT at exactly half the body -- measured 61-63 lines out of a")
    print("  127-line body. A recency window shorter than half the context therefore cannot")
    print("  hold both halves at any budget, so `floor_pos` is again barred by construction.")
    print("  The first layout barred it by putting all /A out of reach; this one bars it by")
    print("  putting the halves permanently too far apart. Same failure, opposite mechanism.")
    print(NL + "  The tension is inherent to the design as specified: enforcing a large MINIMUM")
    print("  separation is precisely what makes 'both halves in one recency window' impossible,")
    print("  so a floor advantage cannot exist to be collapsed. A working control needs the")
    print("  separation DRAWN FROM A DISTRIBUTION spanning short to long, so a window captures")
    print("  both halves sometimes -- MIN_SEP_LINES is itself the wrong constraint. That is a")
    print("  task redesign, not another run, and is left to Stage 3.")
    print(NL + "  What `scattered_uniform` DID fix: /A is no longer confined to the front of the")
    print("  body. The recency tail holds 2-10 /A lines per instance instead of 0 every time,")
    print("  so neither half is positionally privileged. What it did NOT fix is the constant")
    print("  separation described above, which is why the cell is still unmeasurable.")
    print(NL + "  VERDICT 2.3 (scored on the corrected layout only): %s" % v_new)
    return v_new == "PASS", dict(defective=d_def, corrected_512=d512, corrected=d_new,
                                 verdict_defective=v_def, verdict_512=v512,
                                 verdict_corrected=v_new)


def main() -> int:
    p24, v24 = probe_24()
    p21, res21 = probe_21()
    ceff = probe_21_ceff(res21)
    p22, v22 = probe_22()
    p23, v23 = probe_23()

    hdr("STAGE 2 GATE")
    print("  2.1 cost sweep ................... %s" % ("PASS" if p21 else "FAIL"))
    print("  2.2 contiguity control ........... %s" % ("PASS" if p22 else "FAIL"))
    print("  2.3 shuffled-fact control ........ %s   (first layout: %s -- defective)"
          % (v23["verdict_corrected"], v23["verdict_defective"]))
    print("  2.4 single-token facts ........... %s   (REQUIRED)" % ("PASS" if p24 else "FAIL"))
    others = sum([p21, p22, p23])
    gate = p24 and others >= 2
    print(NL + "  GATE (2.4 AND at least 2 of 2.1-2.3): %s   (%d/3 of the others passed)"
          % ("PASS" if gate else "FAIL -- STOP", others))
    if not p24:
        print("  2.4 is the kill criterion: methods still lose at c = 1, so co-retention is")
        print("  not sufficient and something else is operating.")
    json.dump(dict(p21=p21, p22=p22, p23=p23, p24=p24, gate=gate,
                   v22={k: {kk: vv["delta"] for kk, vv in v.items()} for k, v in v22.items()},
                   v23_verdict_defective=v23["verdict_defective"],
                   v23_verdict_corrected=v23["verdict_corrected"]),
              open(HERE / "out" / "stage2_results.json", "w", encoding="utf-8"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
