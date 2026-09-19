"""Model-free checks on the natural-text dataset.

  1. Shortcut probes (query hidden). Each rule commits to ONE fact sentence without seeing the
     query. It can be right for at most one of the H queried facts, so its instance score is
     hit/H and chance is 1/(H+D) per query -- NOT H/(H+D). Pass = every rule <= chance + 0.02.
       bm25_hidden   fact with the highest BM25 against the whole context, query hidden
       regex_*       fixed lexical rules: max amount, max year, longest person name
       position_*    always the first / last / middle inserted fact
  2. Floor reachability. Analytic upper bound on floor_pos accuracy: the share of queried
     facts whose whole gold span lies inside the floor's keep-set, per level and budget. It
     assumes a perfect reader, so accuracy <= this number. This is the check Paper 2's 14B
     ladder lacked: it passed competence and still failed because the floor was degenerate.
  3. Measured c per tokenizer, and the payable cost of the H candidates against each budget.
"""
import argparse
import json
import math
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from p3 import runner
from p3.natural import facts as F
from harness import ladder  # path added by p3.runner

MODELS = {"M2": "Qwen/Qwen2.5-3B-Instruct", "M3": "meta-llama/Llama-3.2-3B-Instruct"}
BUDGETS = (128, 256, 512, 1024)
TOL = 0.02
_W = re.compile(r"[a-z0-9]+")


def toks(s):
    return _W.findall(s.lower())


def bm25(units, ctx_tokens):
    k1, b = 1.5, 0.75
    docs = [toks(u) for u in units]
    n = len(docs)
    avg = st.fmean(len(d) for d in docs)
    df = Counter(w for d in docs for w in set(d))
    out = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for w in set(ctx_tokens):
            if w in tf:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg))
        out.append(s)
    return out


def shortcut_scores(rec, H):
    fs = sorted(rec["renderings"]["M2"]["facts"], key=lambda r: r["char"][0])   # insertion order
    units = [r["text"] for r in fs]
    q = {i for i, r in enumerate(fs) if r["role"] == "queried"}
    ctx = toks(" ".join(units))
    bs = bm25(units, ctx)
    amt = [int(re.search(r"\$([\d,]+)", u).group(1).replace(",", "")) for u in units]
    yr = [int(re.search(r" in (\d{4}) ", u).group(1)) for u in units]
    pl = [len(r["elements"]["person"]["text"]) for r in fs]
    pick = dict(bm25_hidden=max(range(len(units)), key=lambda i: bs[i]),
                regex_max_amount=max(range(len(units)), key=lambda i: amt[i]),
                regex_max_year=max(range(len(units)), key=lambda i: yr[i]),
                regex_longest_name=max(range(len(units)), key=lambda i: pl[i]),
                position_first=0, position_last=len(units) - 1,
                position_middle=len(units) // 2)
    return {k: (1.0 if v in q else 0.0) / H for k, v in pick.items()}


def floor_reach(rec, tag, tk):
    """{level: {C: mean share of queried facts wholly inside floor_pos's keep-set}}."""
    r = rec["renderings"][tag]
    pre, _ = runner.templated_parts(tk, r["context"], "")
    enc = tk(pre, add_special_tokens=False, return_offsets_mapping=True)
    off = enc["offset_mapping"]
    n_ctx = len(off)
    base = pre.index(r["context"])
    out = {}
    for lv in F.LEVELS:
        facts = []
        for fr in r["facts"]:
            if fr["role"] != "queried":
                continue
            a, b = fr["gold_char"][str(lv)]
            a += base
            b += base
            idx = tuple(ti for ti, (x, y) in enumerate(off) if y > x and x < b and y > a)
            facts.append(ladder.FactSpans(fr["fact_id"], (idx,)))
        out[lv] = {}
        for C in BUDGETS:
            kept = set(ladder.floor_pos(n_ctx, C, 8, 64, facts).kept)
            out[lv][C] = st.fmean(1.0 if f.is_complete_in(kept) else 0.0 for f in facts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/natural/nat_v1.jsonl")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.data, encoding="utf-8")][:a.n]
    H, D = recs[0]["H"], recs[0]["D"]
    chance = 1.0 / (H + D)
    print("n=%d  H=%d D=%d  chance=1/(H+D)=%.4f  threshold=%.4f" % (len(recs), H, D, chance, chance + TOL))
    rows = [shortcut_scores(r, H) for r in recs]
    res = {"n": len(recs), "chance": chance, "threshold": chance + TOL, "probes": {}}
    ok = True
    print("\nSHORTCUT PROBES (query hidden)")
    for k in rows[0]:
        m = st.fmean(r[k] for r in rows)
        res["probes"][k] = m
        good = m <= chance + TOL
        ok &= good
        print("  %-20s %.4f  %s" % (k, m, "PASS" if good else "FAIL"))
    res["shortcut_pass"] = ok
    print("  VERDICT:", "PASS" if ok else "FAIL")

    toksd = {t: AutoTokenizer.from_pretrained(m) for t, m in MODELS.items()}
    res["tokens"] = {}
    res["c"] = {}
    res["floor_reach"] = {}
    res["payable"] = {}
    for tag, tk in toksd.items():
        ns = [r["renderings"][tag]["n_tokens"] for r in recs]
        res["tokens"][tag] = dict(min=min(ns), max=max(ns), mean=st.fmean(ns))
        print("\n[%s] context tokens min %d mean %.1f max %d   (target 4096 +/- 32)"
              % (tag, min(ns), st.fmean(ns), max(ns)))
        # measured c: tokens of the gold span of each queried fact, per level
        res["c"][tag] = {}
        res["payable"][tag] = {}
        for lv in F.LEVELS:
            cs = []
            for r in recs:
                for fr in r["renderings"][tag]["facts"]:
                    if fr["role"] == "queried":
                        t = fr["token_ranges"]["gold_%d" % lv]
                        cs.append(t[1] - t[0])
            res["c"][tag][lv] = dict(mean=st.fmean(cs), min=min(cs), max=max(cs))
            tot = 4 * st.fmean(cs)
            res["payable"][tag][lv] = dict(per_fact=st.fmean(cs), all_H=tot,
                                           payable_at={str(C): (tot <= C) for C in BUDGETS})
            print("  level %d: c mean %.1f [%d, %d]   H facts cost %.0f tokens; payable at %s"
                  % (lv, st.fmean(cs), min(cs), max(cs), tot,
                     [C for C in BUDGETS if tot <= C]))
        reach = [floor_reach(r, tag, tk) for r in recs]
        res["floor_reach"][tag] = {str(lv): {str(C): st.fmean(x[lv][C] for x in reach)
                                             for C in BUDGETS} for lv in F.LEVELS}
        print("  floor_pos upper bound (perfect reader), share of queried facts inside keep-set:")
        print("  %-8s" % "level" + "".join("%9s" % ("C=%d" % C) for C in BUDGETS))
        for lv in F.LEVELS:
            print("  %-8d" % lv + "".join("%9.3f" % res["floor_reach"][tag][str(lv)][str(C)]
                                          for C in BUDGETS))
    sp = [r["renderings"]["M2"]["spread"] for r in recs]
    res["spread"] = dict(first_frac_max=max(s["first_frac"] for s in sp),
                         last_frac_min=min(s["last_frac"] for s in sp),
                         max_gap_over_stride=max(s["max_gap_over_stride"] for s in sp),
                         mean_max_gap=st.fmean(s["max_gap_over_stride"] for s in sp))
    allq = [x for s in sp for x in s["queried_frac"]]
    res["spread"]["queried_frac_quartile_counts"] = [sum(1 for x in allq if lo <= x < lo + .25)
                                                     for lo in (0, .25, .5, .75)]
    print("\nSPREAD (M2 rendering):", json.dumps(res["spread"]))
    Path("out").mkdir(exist_ok=True)
    json.dump(res, open("out/nat_probes_%s.json" % Path(a.data).stem, "w"), indent=2)


if __name__ == "__main__":
    main()
