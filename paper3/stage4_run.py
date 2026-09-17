"""Stage 4 — the designed experiment. PINNED ENV ONLY, single process.

Grid is PREREG_P3.md section 6, frozen at sha256 c920c404..., which this script verifies before
running anything.

**Stated deviation from the registered n.** The prereg registers 200 instances. Costed against
Stage 2's measured throughput the full grid is 27-49 GPU-hours against the plan's 15-18, so:

    CAPTURES   n = 200, the full registered n. Prefill only, no generation. These carry the
               scoring of the registered completion predictions, which is what Stage 4 is for.
    GENERATION n = 100. Halves the accuracy CIs (widening them by sqrt(2)) and changes nothing
               else. No cell is dropped and no arm is dropped from the load-bearing plane.

The deviation is in precision, not design, and is reported with every accuracy number.

Packages, run in this order so an interruption leaves the load-bearing plane finished first:

    capture   prefill-only completion accounting, n=200, the whole grid
    plane     c x C, 6 arms, generation                       <- the load-bearing plane
    anchors   full_cache (C-independent), null/random tripwires, oracle_prescient at c=19
    kaxis     k in {1,2} at C in {64,512}
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import ladder, methods, press
from p3 import keys3, runner
from p3.tasks import ledger_c, mark1, multispan

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "nvidia"
PREREG_SHA = "c920c404fd43777e522916ef0188362805a958aa7253512ec2be40641a846eee"
CALIB_C = json.loads((HERE / "out" / "c_calibration.json").read_text(encoding="utf-8"))
CALIB_K = json.loads((HERE / "out" / "k_calibration.json").read_text(encoding="utf-8"))
TAGS = {"Qwen/Qwen2.5-3B-Instruct": "M2", "meta-llama/Llama-3.2-3B-Instruct": "M3"}
ENV = dict(backend="cuda-12.8", transformers_version="5.2.0", kvpress_version="0.5.4",
           dtype="bfloat16", device="nvidia", batch_size=1, protocol="agnostic")
N_SINK, N_WINDOW = runner.N_SINK, runner.N_WINDOW
BUDGETS = (32, 64, 128, 256, 512)
C_TAGS = (1, 8, 19, 40)
PLANE_ARMS = ("floor_pos", "snapkv", "adakv_snapkv", "expected_attn", "keydiff",
              "oracle_causal")
CAP_ARMS = ("floor_pos", "snapkv", "adakv_snapkv", "expected_attn", "keydiff")
LADDER = ("full_cache", "null", "random", "floor_pos", "oracle_causal", "oracle_prescient")
NL = chr(10)
REC = re.compile(r"^R(\d{3})\b")

SCORERS = {"ledger_c": ledger_c.score_instance, "mark1": mark1.score_instance,
           "multispan": multispan.score_instance}
SCORE_ONE = {"ledger_c": ledger_c.score_one, "mark1": mark1.score_one,
             "multispan": multispan.score_one}


def check_prereg():
    from hash_file import digest
    got = digest(HERE / "PREREG_P3.md")
    if got != PREREG_SHA:
        raise SystemExit("PREREG_P3.md has changed since freezing%s  recorded %s%s  actual %s"
                         % (NL, PREREG_SHA, NL, got))
    print("  PREREG_P3.md verified at %s" % got)


# --------------------------------------------------------------------------- cells

def spec(tag, c_tag=None, k=None):
    """One task configuration: (task, n_fields, k, achieved c, label)."""
    if c_tag == 1:
        return dict(task="mark1", n_fields=None, k=1, c=1.0, label="c=1")
    if k is not None:
        d = CALIB_K[tag]["chosen"][str(k)]
        return dict(task="multispan", n_fields=d["n_fields"], k=k, c=d["c"],
                    label="k=%d" % k)
    d = CALIB_C[tag]["chosen"][str(c_tag)]
    return dict(task="ledger_c", n_fields=d["n_fields"], k=1, c=d["c"], label="c=%d" % c_tag)


def build_instance(sp, sd, iid, tok):
    if sp["task"] == "mark1":
        return mark1.build(sd, iid, target_tokens=2048, tokenizer=tok)
    if sp["task"] == "multispan":
        return multispan.build(sd, iid, k=sp["k"], n_fields=sp["n_fields"],
                               target_tokens=2048, tokenizer=tok)
    return ledger_c.build(sd, iid, n_fields=sp["n_fields"], target_tokens=2048, tokenizer=tok)


def max_new_for(tok, inst, task):
    if task == "mark1":
        return [8] * len(inst.variants)
    return [len(tok(v.answer, add_special_tokens=False)["input_ids"]) + runner.MAX_NEW_SLACK
            for v in inst.variants]


def cells_for(package, tag):
    out = []
    if package == "plane":
        for ct in C_TAGS:
            sp = spec(tag, c_tag=ct)
            for C in BUDGETS:
                out.append(dict(**sp, C=C, arms=PLANE_ARMS))
    elif package == "anchors":
        for ct in C_TAGS:
            sp = spec(tag, c_tag=ct)
            out.append(dict(**sp, C=512, arms=("full_cache",)))     # C-independent
            for C in (32, 512):
                out.append(dict(**sp, C=C, arms=("null", "random")))
        sp = spec(tag, c_tag=19)
        for C in BUDGETS:
            out.append(dict(**sp, C=C, arms=("oracle_prescient",)))
    elif package == "kaxis":
        for k in (1, 2):
            sp = spec(tag, k=k)
            for C in (64, 512):
                out.append(dict(**sp, C=C, arms=PLANE_ARMS))
    elif package == "capture":
        for ct in C_TAGS:
            sp = spec(tag, c_tag=ct)
            for C in BUDGETS:
                out.append(dict(**sp, C=C, arms=CAP_ARMS))
        for k in (1, 2):
            sp = spec(tag, k=k)
            for C in (64, 512):
                out.append(dict(**sp, C=C, arms=CAP_ARMS))
    else:
        raise ValueError(package)
    return out


def all_unit_tokens(inst, tok, pre, task):
    """Token sets for every record/entry, for the all-unit accounting."""
    enc = tok(pre, add_special_tokens=False, return_offsets_mapping=True)
    off = enc["offset_mapping"]
    base = pre.index(inst.context)
    out = defaultdict(set)
    for line in inst.context.split(NL):
        key = None
        if task == "mark1":
            if line.startswith("- "):
                key = line[2:]
        else:
            m = REC.match(line)
            if m:
                key = m.group(1)
        if key is None:
            continue
        a = inst.context.index(line) + base
        b = a + len(line)
        out[key].update(ti for ti, (x, y) in enumerate(off) if y > x and x < b and y > a)
    return out


# --------------------------------------------------------------------------- loop

def load_done(path):
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["key_digest"])
                except Exception:
                    continue
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("package", choices=("capture", "plane", "anchors", "kaxis"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, required=True)
    args = ap.parse_args()
    check_prereg()

    tag = TAGS[args.model]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    rev = getattr(model.config, "_commit_hash", None) or "unresolved"

    cells = cells_for(args.package, tag)
    out = RUNS / ("stage4_%s_%s.jsonl" % (args.package, tag))
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    total = sum(len(c["arms"]) for c in cells) * args.n
    print("  %s %s: %d cells, %d arm-cells, n=%d -> %d rows (%d already present)"
          % (args.package, tag, len(cells), sum(len(c["arms"]) for c in cells), args.n,
             total, len(done)), flush=True)

    t0 = time.time()
    written = skipped = failed = 0
    for i in range(args.n):
        iid = "s4_%05d" % i
        built = {}
        for cell in cells:
            tkey = (cell["task"], cell["n_fields"], cell["k"])
            if tkey not in built:
                sd = keys3.instance_seed(task=cell["task"], instance_id=iid, model=args.model,
                                         model_revision=rev, n_fields=cell["n_fields"],
                                         layout=None, n_records=cell["k"], **ENV)
                inst = build_instance(cell, sd, iid, tok)
                pre, _ = runner.templated_parts(tok, inst.context, "")
                facts, n_ctx = runner.facts_and_ctx(inst, tok, pre)
                posts = [runner.templated_parts(tok, inst.context, v.query)[1]
                         for v in inst.variants]
                units = (all_unit_tokens(inst, tok, pre, cell["task"])
                         if args.package == "capture" else None)
                built[tkey] = (sd, inst, pre, facts, n_ctx, posts,
                               max_new_for(tok, inst, cell["task"]), units)
            sd, inst, pre, facts, n_ctx, posts, mns, units = built[tkey]
            C = cell["C"]
            B = C + N_SINK + N_WINDOW

            for arm in cell["arms"]:
                kd = dict(task=cell["task"], instance_id=iid, model=args.model,
                          model_revision=rev, arm=arm, B=B, C=C, seed=sd,
                          n_fields=cell["n_fields"], n_records=cell["k"],
                          layout=args.package, matched_to=None, max_new=max(mns), **ENV)
                dg = keys3.digest(kd)
                if dg in done:
                    skipped += 1
                    continue
                t1 = time.time()
                try:
                    if args.package == "capture":
                        if arm == "floor_pos":
                            keeps = [runner.ladder_kept("floor_pos", n_ctx, C, facts, sd)]
                            parity = B
                        else:
                            cap = runner.capture_keepsets(model, tok, pre, arm, C, n_ctx)
                            parity = runner.assert_budget_parity(cap, C, n_ctx, arm)
                            keeps = [cap.per_head[h] for h in cap.heads()]
                        gold = {t for f in facts for s in f.spans for t in s}
                        pg = [len(gold & kk) / max(1, len(gold)) for kk in keeps]
                        qc = [sum(1 for f in facts if f.is_complete_in(kk)) / len(facts)
                              for kk in keeps]
                        q_any = st.fmean([max(1.0 if f.is_complete_in(kk) else 0.0
                                              for kk in keeps) for f in facts])
                        rc = st.fmean([sum(1 for s in units.values() if s and s <= kk)
                                       for kk in keeps])
                        rt = st.fmean([sum(1 for s in units.values() if s & kk)
                                       for kk in keeps])
                        row = dict(key_digest=dg, key=kd, model_tag=tag, package="capture",
                                   task=cell["task"], label=cell["label"], c=cell["c"],
                                   k=cell["k"], C=C, B=B, arm=arm, n_ctx=n_ctx,
                                   B_asserted=parity, n_slots=len(keeps),
                                   gold_tokens=len(gold), n_facts=len(facts),
                                   p_g=st.fmean(pg), q_complete=st.fmean(qc), q_any=q_any,
                                   units_complete=rc, units_touched=rt,
                                   n_units=len(units), wall_s=time.time() - t1)
                    else:
                        if arm == "oracle_prescient":
                            outs, flags = [], []
                            for v, post, mn in zip(inst.variants, posts, mns):
                                gf = next(f for f in facts if f.fact_id == v.rec_id)
                                p, _ = press.build_arm(arm, n_ctx=n_ctx, C=C, n_sink=N_SINK,
                                                       n_window=N_WINDOW, facts=facts,
                                                       gold=gf, seed=sd)
                                o, fl = runner.generate_with(model, tok, pre, [post], p, [mn])
                                outs.append(o[0])
                                flags.append(fl[0])
                        elif arm in LADDER:
                            p, _ = press.build_arm(arm, n_ctx=n_ctx, C=C, n_sink=N_SINK,
                                                   n_window=N_WINDOW, facts=facts, seed=sd)
                            outs, flags = runner.generate_with(model, tok, pre, posts, p, mns)
                        else:
                            ratio = press._ratio_for(B, n_ctx)
                            p = methods.make_floor_constrained(
                                methods.build_method(arm, ratio), n_ctx, N_SINK, N_WINDOW)
                            p.compression_ratio = ratio
                            outs, flags = runner.generate_with(model, tok, pre, posts, p, mns)
                        row = dict(key_digest=dg, key=kd, model_tag=tag,
                                   package=args.package, task=cell["task"],
                                   label=cell["label"], c=cell["c"], k=cell["k"], C=C, B=B,
                                   arm=arm, n_ctx=n_ctx,
                                   score=SCORERS[cell["task"]](outs, inst),
                                   per_variant=[SCORE_ONE[cell["task"]](o, v.answer)
                                                for v, o in zip(inst.variants, outs)],
                                   gen=outs, answers=[v.answer for v in inst.variants],
                                   stop_flags=flags, wall_s=time.time() - t1,
                                   meta=inst.meta)
                    with out.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + NL)
                    done.add(dg)
                    written += 1
                except Exception as e:
                    failed += 1
                    with out.with_suffix(".failures.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(dict(key_digest=dg, key=kd,
                                                error="%s: %s" % (type(e).__name__, e))) + NL)
        if (i + 1) % 5 == 0 or i == args.n - 1:
            el = time.time() - t0
            rate = max(1e-9, (written + skipped) / el)
            print("  [%s/%s] inst %d/%d  written %d skipped %d failed %d  %.1f min  "
                  "%.2f s/row  eta %.1f min"
                  % (args.package, tag, i + 1, args.n, written, skipped, failed, el / 60,
                     el / max(1, written), (total - written - skipped) / rate / 60),
                  flush=True)
    print("  DONE %s/%s: written %d skipped %d failed %d in %.1f min"
          % (args.package, tag, written, skipped, failed, (time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
