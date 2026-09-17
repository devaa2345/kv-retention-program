"""Completion counts for Stage 2 — prefill only, no generation.

Probe 2.1 asks for completion counts as well as accuracy, and the `c_eff` fit needs `p_g` and
the measured per-fact completion at each (model, c, C, arm). Those come from the retained sets,
not from generations, so this is a separate cheap pass over the same instances.

Emits, per (instance, cell, arm), the same quantities Paper 2's `frag_perinstance_*.jsonl`
carried, so `p3.theory` and the Stage 1 estimators apply unchanged:

    p_g          gold tokens retained / gold tokens present, mean over (layer, KV-head) slots
    q_complete   P(the queried fact is retained WHOLE), mean over slots      <- T2's target
    q_any        the same, union over slots
    recs_*       all-record touched/complete accounting

`floor_pos` has one global keep-set, so its per-slot and union values coincide by construction.
Budget parity is asserted on every captured cell.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import ladder
from p3 import keys3, runner
from stage2_run import (CALIB, ENV, TAGS, build_instance, nf_for)

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "nvidia"
N_SINK, N_WINDOW = runner.N_SINK, runner.N_WINDOW
METHODS = ("snapkv", "adakv_snapkv")
REC = re.compile(r"^R(\d{3})\b")


def all_fact_tokens(inst, tok, pre):
    """Token sets for EVERY record (not just the queried ones), for the all-record accounting."""
    enc = tok(pre, add_special_tokens=False, return_offsets_mapping=True)
    off = enc["offset_mapping"]
    base = pre.index(inst.context)
    out: dict[str, set[int]] = {}
    for line in inst.context.split("\n"):
        m = REC.match(line)
        if not m:
            continue
        a = inst.context.index(line) + base
        b = a + len(line)
        out.setdefault(m.group(1), set()).update(
            ti for ti, (x, y) in enumerate(off) if y > x and x < b and y > a)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    rev = getattr(model.config, "_commit_hash", None) or "unresolved"
    tag = TAGS[args.model]
    out = RUNS / f"stage2_capture_{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)

    cells = [("ledger_c", nf_for(tag, c), None, None, c, C)
             for c in (8, 19, 40) for C in (64, 512)]
    cells += [("mark1", None, None, None, 1, C) for C in (64, 512)]
    cells += [("split_ledger", 4, lay, 20, 19, 512) for lay in ("adjacent", "scattered")]

    t0 = time.time()
    for i in range(args.n):
        iid = f"s2_{i:05d}"
        built = {}
        for task, nf, lay, nr, c_tag, C in cells:
            tkey = (task, nf, lay, nr)
            if tkey not in built:
                sd = keys3.instance_seed(task=task, instance_id=iid, model=args.model,
                                         model_revision=rev, n_fields=nf, layout=lay,
                                         n_records=nr, **ENV)
                inst = build_instance(task, sd, iid, tok, n_fields=nf, layout=lay,
                                      n_records=nr)
                pre, _ = runner.templated_parts(tok, inst.context, "")
                facts, n_ctx = runner.facts_and_ctx(inst, tok, pre)
                allrec = all_fact_tokens(inst, tok, pre) if task != "mark1" else {}
                built[tkey] = (sd, inst, pre, facts, n_ctx, allrec)
            sd, inst, pre, facts, n_ctx, allrec = built[tkey]

            for arm in ("floor_pos",) + METHODS:
                if arm == "floor_pos":
                    keepsets = [runner.ladder_kept("floor_pos", n_ctx, C, facts, sd)]
                    parity = C + N_SINK + N_WINDOW
                else:
                    cap = runner.capture_keepsets(model, tok, pre, arm, C, n_ctx)
                    parity = runner.assert_budget_parity(cap, C, n_ctx, arm)
                    keepsets = [cap.per_head[k] for k in cap.heads()]

                gold_tok = {t for f in facts for s in f.spans for t in s}
                pg, qc, ranks = [], [], []
                for kk in keepsets:
                    pg.append(len(gold_tok & kk) / max(1, len(gold_tok)))
                    qc.append(sum(1 for f in facts if f.is_complete_in(kk)) / len(facts))
                q_any = st.fmean(
                    [max(1.0 if f.is_complete_in(kk) else 0.0 for kk in keepsets)
                     for f in facts])
                rc = rt = 0.0
                if allrec:
                    rc = st.fmean([sum(1 for s in allrec.values() if s and s <= kk)
                                   for kk in keepsets])
                    rt = st.fmean([sum(1 for s in allrec.values() if s & kk)
                                   for kk in keepsets])
                row = dict(model=args.model, model_tag=tag, instance_id=iid, task=task,
                           n_fields=nf, layout=lay, n_records=nr, c_tag=c_tag, C=C, arm=arm,
                           n_slots=len(keepsets), n_ctx=n_ctx, B_asserted=parity,
                           gold_tokens=len(gold_tok), n_facts=len(facts),
                           p_g=st.fmean(pg), q_complete=st.fmean(qc), q_any=q_any,
                           recs_complete=rc, recs_touched=rt,
                           n_all_records=len(allrec))
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
        if (i + 1) % 10 == 0 or i == args.n - 1:
            print(f"  capture {tag}: instance {i + 1}/{args.n}  "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
