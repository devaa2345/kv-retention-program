"""Validity check on the retention predicate itself. NOT a hypothesis about the leak.

Every method arm's retention is measured over (layer, KV-head) SLOTS -- 72 on M2 (36 layers x 2
heads), 224 on M3 (28 x 8) -- while `floor_pos` has exactly ONE global keep-set. So every
method-vs-floor contrast in this study compares a slot-aggregated statistic against an exact one.
If the aggregation understates retention for head-wise arms, the matched-gold-token comparison
inherits that bias.

Two aggregations are in use, and they are not the same:

  * the leak test used a MAJORITY vote  (retained iff complete in > 50% of slots)
  * the fragmentation tables use a MEAN over slots (`qcpl` = mean per-slot completeness)

Both are exact for `floor_pos` (one slot) and aggregated for methods. This script measures how
much either one hides.

--mode leak
    KeyDiff, M2, aware, C=512. Replaces the binary predicate with the FRACTION of slots in which
    the value tokens (and separately the whole gold record) survive, on the
    correct-without-gold population, and correlates that fraction with correctness.

--mode completeness
    For the fragmentation captures: per arm per budget, of the queried records scored INCOMPLETE
    by the majority-vote predicate, what fraction are complete in AT LEAST ONE slot. Near zero
    for `floor_pos` by construction (one slot: majority and any coincide). If it is large for the
    method arms, the completeness contrast needs restating.
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import ladder, methods, press
from harness.keys import seed_key_without_seed
from harness.tasks import ledger
from stage5_ladder_validation import facts_and_ctx, generate_with, templated_parts
from run_grid_aware import aware_parts, spans_over

N_SINK, N_WINDOW = 8, 64
ARMS = ["snapkv", "expected_attn", "keydiff", "adakv_snapkv"]


def pearson(x, y):
    if len(x) < 3:
        return float("nan")
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def build(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    return tok, model, (getattr(model.config, "_commit_hash", None) or "x")


def seed_of(tok_model, iid, rev):
    return seed_key_without_seed(
        task="ledger", instance_id=iid, model=tok_model, model_revision=rev, arm="grid", B=1,
        protocol="agnostic", device="nvidia", backend="cuda-12.8",
        torch_version=torch.__version__, transformers_version="5.2.0",
        kvpress_version="0.5.4", dtype="bfloat16")


# ------------------------------------------------------------------ mode: leak
def mode_leak(args):
    M = "Qwen/Qwen2.5-3B-Instruct"
    C = 512
    tok, model, rev = build(M)
    rows = []
    for i in range(args.n):
        iid = "grid_%05d" % i
        sd = seed_of(M, iid, rev)
        inst = ledger.build(sd, iid, target_tokens=2048, tokenizer=tok)
        for v in inst.variants:
            pre, post = aware_parts(tok, inst.context, v.query)
            facts, n_ctx = spans_over(inst, tok, pre)
            gold = {t for f in facts if f.fact_id == v.rec_id for t in f.tokens}
            enc = tok(pre, add_special_tokens=False, return_offsets_mapping=True)
            off = enc["offset_mapping"]
            base = pre.index(inst.context)
            line = next((ln for ln in inst.context.split("\n")
                         if ln.startswith(v.rec_id + " | ")), "")
            a = inst.context.index(line) + base
            b = a + len(line)
            vtok = {ti for ti, (x, y) in enumerate(off) if y > x and x < b and y > b - 6}

            ratio = press._ratio_for(C + N_SINK + N_WINDOW, n_ctx)
            cap = methods.Capture()
            p = methods.make_capturing(
                methods.make_floor_constrained(
                    methods.build_method("keydiff", ratio), n_ctx, N_SINK, N_WINDOW), cap)
            p.compression_ratio = ratio
            out = generate_with(model, tok, pre, [post], p)[0]
            ok = 1.0 if v.answer in out else 0.0

            hs = cap.heads()
            if not hs:
                continue
            gfrac = statistics.fmean([1.0 if gold <= cap.per_head[k] else 0.0 for k in hs])
            vfrac = statistics.fmean([1.0 if (vtok and vtok <= cap.per_head[k]) else 0.0
                                      for k in hs])
            # partial presence: any single value token present at all
            vany = statistics.fmean([1.0 if (vtok & cap.per_head[k]) else 0.0 for k in hs])
            rows.append(dict(ok=ok, gfrac=gfrac, vfrac=vfrac, vany=vany, n_slots=len(hs)))
        if (i + 1) % 25 == 0:
            print("  [%d/%d]" % (i + 1, args.n), flush=True)

    pop = [r for r in rows if r["gfrac"] <= 0.5]          # the correct-without-gold population
    print("\nKeyDiff  M2 aware  C=512   n=%d prefixes, %d slots per capture"
          % (len(rows), rows[0]["n_slots"] if rows else 0))
    print("  majority-vote population (gold slot-fraction <= 0.5): n=%d, acc=%.4f"
          % (len(pop), statistics.fmean(r["ok"] for r in pop) if pop else float("nan")))
    if not pop:
        return 0
    for field, label in (("vfrac", "VALUE complete"), ("gfrac", "GOLD RECORD complete"),
                         ("vany", "ANY value token present")):
        x = [r[field] for r in pop]
        y = [r["ok"] for r in pop]
        c = [r[field] for r in pop if r["ok"] > 0.5]
        w = [r[field] for r in pop if r["ok"] <= 0.5]
        print("\n  slot fraction, %s" % label)
        print("    mean over population      %.4f   (min %.4f  max %.4f)"
              % (statistics.fmean(x), min(x), max(x)))
        print("    mean | answered CORRECTLY %.4f   (n=%d)"
              % (statistics.fmean(c) if c else float("nan"), len(c)))
        print("    mean | answered WRONG     %.4f   (n=%d)"
              % (statistics.fmean(w) if w else float("nan"), len(w)))
        print("    r(fraction, correctness)  %+.4f" % pearson(x, y))
        print("    %% of population with fraction exactly 0: %.1f%%"
              % (100.0 * sum(1 for a in x if a == 0.0) / len(x)))
    return 0


# ---------------------------------------------------------- mode: completeness
def mode_completeness(args):
    tok, model, rev = build(args.model)
    budgets = [int(b) for b in args.budgets.split(",")]
    agg = defaultdict(lambda: [0, 0])          # (C, arm) -> [n_incomplete, n_any_complete]

    for i in range(args.n):
        iid = "grid_%05d" % i
        sd = seed_of(args.model, iid, rev)
        inst = ledger.build(sd, iid, target_tokens=2048, tokenizer=tok)
        pre, _ = templated_parts(tok, inst.context, "")
        facts, n_ctx = facts_and_ctx(inst, tok, pre)
        enc = tok(pre, add_special_tokens=False, return_offsets_mapping=True)
        off = enc["offset_mapping"]
        base = pre.index(inst.context)
        qrec = {}
        for v in inst.variants:
            line = next((ln for ln in inst.context.split("\n")
                         if ln.startswith(v.rec_id + " | ")), "")
            a = inst.context.index(line) + base
            b = a + len(line)
            qrec[v.rec_id] = {ti for ti, (x, y) in enumerate(off) if y > x and x < b and y > a}

        for C in budgets:
            ratio = press._ratio_for(C + N_SINK + N_WINDOW, n_ctx)
            keepsets = {}
            keepsets["floor_pos"] = [set(
                ladder.floor_pos(n_ctx, C, N_SINK, N_WINDOW, facts).kept)]
            for arm in ARMS:
                cap = methods.Capture()
                p = methods.make_capturing(
                    methods.make_floor_constrained(
                        methods.build_method(arm, ratio), n_ctx, N_SINK, N_WINDOW), cap)
                p.compression_ratio = ratio
                ids = tok(pre, add_special_tokens=False, return_tensors="pt").to(model.device)
                with torch.inference_mode(), p(model):
                    model(**ids, use_cache=True)
                keepsets[arm] = [cap.per_head[k] for k in cap.heads()]

            for arm, ks in keepsets.items():
                if not ks:
                    continue
                for toks in qrec.values():
                    if not toks:
                        continue
                    n_c = sum(1 for kk in ks if toks <= kk)
                    majority = (n_c / len(ks)) > 0.5
                    if not majority:
                        agg[(C, arm)][0] += 1
                        agg[(C, arm)][1] += int(n_c >= 1)
        if (i + 1) % 20 == 0:
            print("  [%d/%d]" % (i + 1, args.n), flush=True)

    print("\n%s   n=%d instances" % (args.model, args.n))
    print("Of queried records scored INCOMPLETE by majority vote, what fraction are complete "
          "in >= 1 slot?")
    print("%5s %-16s %12s %14s %10s" % ("C", "arm", "n_incomplete", "n_any_complete", "fraction"))
    for C in budgets:
        for arm in ["floor_pos"] + ARMS:
            n_i, n_a = agg.get((C, arm), [0, 0])
            if not n_i:
                continue
            print("%5d %-16s %12d %14d %10.4f" % (C, arm, n_i, n_a, n_a / n_i))
        print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["leak", "completeness"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--budgets", default="16,32,64,128,256,512")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    return mode_leak(args) if args.mode == "leak" else mode_completeness(args)


if __name__ == "__main__":
    raise SystemExit(main())
