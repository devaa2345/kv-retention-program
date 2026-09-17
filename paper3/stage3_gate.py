"""Stage 3 gate — competence anchors and shortcut probes at every planned (c, k).

Two independent requirements, both reported whatever they show:

  ANCHORS   `full_cache` in [0.55, 0.97] at every planned (c, k), both models. Below the band
            the task cannot be answered even uncompressed, so no compressed arm can be read as
            a retention result; above it, the task is too easy to separate arms.
  SHORTCUTS Three query-blind rules, each of which picks ONE unit without seeing the query.
            If any lands materially above chance the task admits a surface shortcut and the
            paper's claim is not about retention. chance = 1/N = 0.025, threshold 0.045.

              bm25_hidden   rank units by BM25 against the context, query hidden
              regex         pick the unit matching a fixed surface pattern
              position      always answer with the last unit

An out-of-band anchor stops the programme here. Nothing is tuned in response.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "nvidia"
CALIB_C = json.loads((HERE / "out" / "c_calibration.json").read_text(encoding="utf-8"))
CALIB_K = json.loads((HERE / "out" / "k_calibration.json").read_text(encoding="utf-8"))
TAGS = {"Qwen/Qwen2.5-3B-Instruct": "M2", "meta-llama/Llama-3.2-3B-Instruct": "M3"}
BAND = (0.55, 0.97)
CHANCE = 1.0 / 40.0
SHORTCUT_TOL = 0.02
NL = chr(10)
_WORD = re.compile(r"[A-Za-z0-9]+")
_VALUE = re.compile(r"\b\d{6}\b")


def toks(s):
    return [w.lower() for w in _WORD.findall(s)]


# --------------------------------------------------------------------- shortcut probes

def bm25_scores(units, corpus_tokens):
    """Standard BM25 of each unit against the whole context, with the QUERY HIDDEN."""
    k1, b = 1.5, 0.75
    docs = [toks(u) for u in units]
    n = len(docs)
    avgdl = st.fmean(len(d) for d in docs) if docs else 1.0
    df = Counter()
    for d in docs:
        for w in set(d):
            df[w] += 1
    q = corpus_tokens
    out = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for w in set(q):
            if w not in tf:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avgdl))
        out.append(s)
    return out


def probe_instance(units, gold_units, H):
    """Three query-blind rules, scored on the SAME convention as the task itself.

    Each instance carries H queries and is scored as their MEAN. A query-blind rule commits to
    one unit for all H, so it can be right for at most one of them: its instance score is
    `hit / H`, and chance is 1/N per query -- not H/N. Scoring "landed on any queried unit"
    instead would put chance at H/N = 0.10 and make every probe look like a shortcut; that was
    the first version's error and it is the reason the thresholds below are 1/N based, exactly
    as Paper 2's probes were.
    """
    ctx = toks(" ".join(units))
    bs = bm25_scores(units, ctx)
    top = max(range(len(units)), key=lambda i: bs[i]) if units else -1
    hit_bm25 = (1.0 if units and units[top] in gold_units else 0.0) / H

    rx = next((u for u in units if _VALUE.search(u)), units[0] if units else "")
    hit_regex = (1.0 if rx in gold_units else 0.0) / H

    hit_pos = (1.0 if units and units[-1] in gold_units else 0.0) / H
    return hit_bm25, hit_regex, hit_pos


def units_of(inst, task):
    """The units a query-blind rule would choose among, and which are queried."""
    if task == "mark1":
        us = [ln[2:] for ln in inst.context.split(NL) if ln.startswith("- ")]
        gold = {v.answer for v in inst.variants}
        return us, gold
    pat = re.compile(r"^R\d{3}\b")
    lines = [ln for ln in inst.context.split(NL) if pat.match(ln)]
    if task == "multispan":
        by = defaultdict(list)
        for ln in lines:
            by[ln[:4]].append(ln)
        us = [" ".join(v) for v in by.values()]
        gset = {v.rec_id for v in inst.variants}
        gold = {" ".join(v) for kk, v in by.items() if kk in gset}
        return us, gold
    us = lines
    gset = {v.rec_id for v in inst.variants}
    gold = {ln for ln in lines if ln[:4] in gset}
    return us, gold


# --------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--anchors", action="store_true")
    ap.add_argument("--shortcuts", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from p3 import keys3, runner
    from p3.tasks import ledger_c, mark1, multispan

    tag = TAGS[args.model]
    tok = AutoTokenizer.from_pretrained(args.model)

    cells = []
    for ct in (8, 19, 40):
        cells.append(dict(task="ledger_c", label="c=%d k=1" % ct, k=1,
                          n_fields=CALIB_C[tag]["chosen"][str(ct)]["n_fields"],
                          c=CALIB_C[tag]["chosen"][str(ct)]["c"]))
    cells.append(dict(task="mark1", label="c=1 k=1", k=1, n_fields=None, c=1.0))
    for k in (1, 2):
        cells.append(dict(task="multispan", label="c~19 k=%d" % k, k=k,
                          n_fields=CALIB_K[tag]["chosen"][str(k)]["n_fields"],
                          c=CALIB_K[tag]["chosen"][str(k)]["c"]))

    def build(cell, sd, iid):
        if cell["task"] == "ledger_c":
            return ledger_c.build(sd, iid, n_fields=cell["n_fields"],
                                  target_tokens=2048, tokenizer=tok)
        if cell["task"] == "mark1":
            return mark1.build(sd, iid, target_tokens=2048, tokenizer=tok)
        return multispan.build(sd, iid, k=cell["k"], n_fields=cell["n_fields"],
                               target_tokens=2048, tokenizer=tok)

    SCORE = {"ledger_c": ledger_c.score_instance, "mark1": mark1.score_instance,
             "multispan": multispan.score_instance}

    # ---------------- shortcut probes (model-free) ----------------
    if args.shortcuts:
        print("=" * 92)
        print("SHORTCUT PROBES  %s   chance %.4f, threshold %.4f, n=%d"
              % (tag, CHANCE, CHANCE + SHORTCUT_TOL, args.n))
        print("=" * 92)
        print("%-14s %8s %11s %11s %11s %8s" % ("cell", "c", "bm25_hid", "regex",
                                                "position", "verdict"))
        rows = []
        for cell in cells:
            acc = [[], [], []]
            for i in range(args.n):
                iid = "gate_%05d" % i
                sd = keys3.instance_seed(task=cell["task"], instance_id=iid, model=args.model,
                                         model_revision="gate", n_fields=cell["n_fields"],
                                         layout=None, n_records=None, protocol="agnostic",
                                         device="nvidia", backend="cuda-12.8",
                                         transformers_version="5.2.0",
                                         kvpress_version="0.5.4", dtype="bfloat16",
                                         batch_size=1)
                inst = build(cell, sd, iid)
                us, gold = units_of(inst, cell["task"])
                for j, v in enumerate(probe_instance(us, gold, len(inst.variants))):
                    acc[j].append(v)
            m = [st.fmean(a) for a in acc]
            ok = all(v <= CHANCE + SHORTCUT_TOL for v in m)
            rows.append(dict(cell=cell["label"], c=cell["c"], bm25=m[0], regex=m[1],
                             position=m[2], pass_=ok))
            print("%-14s %8.2f %11.4f %11.4f %11.4f %8s"
                  % (cell["label"], cell["c"], m[0], m[1], m[2], "PASS" if ok else "FAIL"))
        allok = all(r["pass_"] for r in rows)
        print(NL + "  SHORTCUT VERDICT %s: %s" % (tag, "PASS" if allok else "FAIL"))
        json.dump(rows, open(HERE / "out" / ("gate_shortcuts_%s.json" % tag), "w",
                             encoding="utf-8"), indent=2)

    # ---------------- anchors (GPU) ----------------
    if args.anchors:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        out = RUNS / ("stage3_anchors_%s.jsonl" % tag)
        out.parent.mkdir(parents=True, exist_ok=True)
        done = set()
        if out.exists():
            for line in out.open(encoding="utf-8"):
                try:
                    done.add(json.loads(line)["key_digest"])
                except Exception:
                    pass
        t0 = time.time()
        for cell in cells:
            if cell["task"] != "multispan":
                continue          # ledger_c and mark1 anchors already measured at Stage 2
            for i in range(args.n):
                iid = "gate_%05d" % i
                kd = dict(task=cell["task"], instance_id=iid, model=args.model,
                          model_revision="gate", arm="full_cache", B=0, C=0,
                          n_fields=cell["n_fields"], n_records=cell["k"], layout=None,
                          matched_to=None, max_new=0, seed=0, protocol="agnostic",
                          device="nvidia", backend="cuda-12.8",
                          transformers_version="5.2.0", kvpress_version="0.5.4",
                          dtype="bfloat16", batch_size=1)
                dg = keys3.digest(kd)
                if dg in done:
                    continue
                sd = keys3.instance_seed(**{**kd, "arm": None})
                inst = build(cell, sd, iid)
                pre, _ = runner.templated_parts(tok, inst.context, "")
                posts = [runner.templated_parts(tok, inst.context, v.query)[1]
                         for v in inst.variants]
                mns = [len(tok(v.answer, add_special_tokens=False)["input_ids"]) + 16
                       for v in inst.variants]
                t1 = time.time()
                outs, flags = runner.generate_with(model, tok, pre, posts, None, mns)
                row = dict(key_digest=dg, key=kd, model_tag=tag, label=cell["label"],
                           task=cell["task"], k=cell["k"], c=cell["c"],
                           score=SCORE[cell["task"]](outs, inst), gen=outs,
                           answers=[v.answer for v in inst.variants], stop_flags=flags,
                           wall_s=time.time() - t1, meta=inst.meta)
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + NL)
                done.add(dg)
            print("  anchor %s %s done  (%.1f min)" % (tag, cell["label"],
                                                       (time.time() - t0) / 60), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
