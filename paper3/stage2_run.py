"""Stage 2 runner — the adversarial probes. PINNED ENV ONLY, single process.

One JSONL row per (instance, cell, arm), append-only and resumable on the dedup key, so an
interrupted probe restarts where it stopped. Writes only under `runs/nvidia/`.

Usage:
    stage2_run.py anchors  --model ... --n 24
    stage2_run.py p21      --model ... --n 50
    stage2_run.py p22      --model ... --n 50
    stage2_run.py p23      --model ... --n 50
    stage2_run.py p24      --model ... --n 50
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import methods, press
from p3 import keys3, runner
from p3.tasks import ledger_c, mark1, scatter

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "nvidia"
CALIB = json.loads((HERE / "out" / "c_calibration.json").read_text(encoding="utf-8"))

TAGS = {"Qwen/Qwen2.5-3B-Instruct": "M2", "meta-llama/Llama-3.2-3B-Instruct": "M3"}
ENV = dict(backend="cuda-12.8", transformers_version="5.2.0", kvpress_version="0.5.4",
           dtype="bfloat16", device="nvidia", batch_size=1, protocol="agnostic")
N_SINK, N_WINDOW = runner.N_SINK, runner.N_WINDOW

LADDER = ("full_cache", "null", "random", "floor_pos", "oracle_causal", "oracle_prescient")
P21_METHODS = ("snapkv", "adakv_snapkv")
P24_METHODS = ("snapkv", "adakv_snapkv", "expected_attn", "keydiff")


# --------------------------------------------------------------------------- instances

def build_instance(task: str, seed: int, iid: str, tok, *, n_fields=None, layout=None,
                   n_records=None):
    if task == "ledger_c":
        return ledger_c.build(seed, iid, n_fields=n_fields,
                              n_records=n_records or ledger_c.N_RECORDS,
                              target_tokens=2048, tokenizer=tok)
    if task == "mark1":
        return mark1.build(seed, iid, target_tokens=2048, tokenizer=tok)
    if task == "split_ledger":
        return scatter.build(seed, iid, layout=layout, n_fields=n_fields,
                             n_records=n_records or scatter.N_RECORDS,
                             target_tokens=2048, tokenizer=tok)
    raise ValueError(task)


SCORERS = {"ledger_c": ledger_c.score_instance,
           "mark1": mark1.score_instance,
           "split_ledger": scatter.score_instance}

SCORE_ONE = {"ledger_c": ledger_c.score_one,
             "mark1": mark1.score_one,
             "split_ledger": scatter.score_one}


def max_new_for(tok, inst, task):
    if task == "mark1":
        return [8] * len(inst.variants)
    return [len(tok(v.answer, add_special_tokens=False)["input_ids"]) + runner.MAX_NEW_SLACK
            for v in inst.variants]


# --------------------------------------------------------------------------- the loop

def load_done(path: Path) -> set[str]:
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["key_digest"])
                except Exception:
                    continue
    return done


def run_cells(model, tok, M, rev, out: Path, cells, n: int, note: str = ""):
    """`cells` is a list of dicts: task, n_fields, layout, C, arms, label."""
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    print(f"  resuming: {len(done)} rows already in {out.name}")
    t0 = time.time()
    written = skipped = failed = 0
    total = sum(len(c["arms"]) for c in cells) * n

    for i in range(n):
        iid = f"s2_{i:05d}"
        by_task: dict[tuple, object] = {}
        for cell in cells:
            task, nf, lay, C = cell["task"], cell["n_fields"], cell["layout"], cell["C"]
            nr = cell.get("n_records")
            tkey = (task, nf, lay, nr)
            if tkey not in by_task:
                sd = keys3.instance_seed(task=task, instance_id=iid, model=M,
                                         model_revision=rev, n_fields=nf, layout=lay,
                                         n_records=nr, **ENV)
                inst = build_instance(task, sd, iid, tok, n_fields=nf, layout=lay,
                                      n_records=nr)
                pre, _ = runner.templated_parts(tok, inst.context, "")
                facts, n_ctx = runner.facts_and_ctx(inst, tok, pre)
                posts = [runner.templated_parts(tok, inst.context, v.query)[1]
                         for v in inst.variants]
                by_task[tkey] = (sd, inst, pre, facts, n_ctx, posts,
                                 max_new_for(tok, inst, task))
            sd, inst, pre, facts, n_ctx, posts, mns = by_task[tkey]
            B = C + N_SINK + N_WINDOW

            for arm in cell["arms"]:
                kd = dict(task=task, instance_id=iid, model=M, model_revision=rev, arm=arm,
                          B=B, C=C, seed=sd, n_fields=nf, n_records=nr, layout=lay,
                          matched_to=cell.get("matched_to"), max_new=max(mns), **ENV)
                dg = keys3.digest(kd)
                if dg in done:
                    skipped += 1
                    continue
                try:
                    extra = {}
                    if arm == "oracle_prescient":
                        sc, flags, outs = [], [], []
                        for v, post, mn in zip(inst.variants, posts, mns):
                            gf = next(f for f in facts if f.fact_id == v.rec_id)
                            p, _ = press.build_arm(arm, n_ctx=n_ctx, C=C, n_sink=N_SINK,
                                                   n_window=N_WINDOW, facts=facts,
                                                   gold=gf, seed=sd)
                            o, fl = runner.generate_with(model, tok, pre, [post], p, [mn])
                            outs.append(o[0])
                            flags.append(fl[0])
                        score = SCORERS[task](outs, inst)
                    elif arm in LADDER:
                        p, _ = press.build_arm(arm, n_ctx=n_ctx, C=C, n_sink=N_SINK,
                                               n_window=N_WINDOW, facts=facts, seed=sd)
                        outs, flags = runner.generate_with(model, tok, pre, posts, p, mns)
                        score = SCORERS[task](outs, inst)
                    elif arm.startswith(runner.SYNTHETIC_PREFIX):
                        src = arm[len(runner.SYNTHETIC_PREFIX):]
                        cap = runner.capture_keepsets(model, tok, pre, src, C, n_ctx)
                        runner.assert_budget_parity(cap, C, n_ctx, src)
                        gold_tok = {t for f in facts for s in f.spans for t in s}
                        ks = [cap.per_head[k] for k in cap.heads()]
                        g = st.fmean(len(gold_tok & kk) for kk in ks)
                        # the SOURCE method's own coherence, so the contrast is visible in the
                        # report rather than inferred: same gold tokens, how many whole facts?
                        src_complete = st.fmean(
                            sum(1 for f in facts if f.is_complete_in(kk)) for kk in ks)
                        p, info = runner.build_contig_matched(n_ctx, C, facts,
                                                             int(round(g)))
                        outs, flags = runner.generate_with(model, tok, pre, posts, p, mns)
                        score = SCORERS[task](outs, inst)
                        extra = dict(matched_gold_target=g,
                                     source_facts_complete=src_complete, **info)
                    else:
                        ratio = press._ratio_for(B, n_ctx)
                        p = methods.make_floor_constrained(
                            methods.build_method(arm, ratio), n_ctx, N_SINK, N_WINDOW)
                        p.compression_ratio = ratio
                        outs, flags = runner.generate_with(model, tok, pre, posts, p, mns)
                        score = SCORERS[task](outs, inst)

                    row = dict(key_digest=dg, key=kd, score=score, n_ctx=n_ctx, C=C, B=B,
                               task=task, n_fields=nf, n_records=nr, layout=lay, arm=arm,
                               model_tag=TAGS[M],
                               label=cell["label"], stop_flags=flags,
                               per_variant=[SCORE_ONE[task](o, v.answer)
                                            for v, o in zip(inst.variants, outs)],
                               gen=outs,
                               answers=[v.answer for v in inst.variants],
                               meta=inst.meta, **extra)
                    with out.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")
                    done.add(dg)
                    written += 1
                except Exception as e:
                    failed += 1
                    with out.with_suffix(".failures.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(dict(key_digest=dg, key=kd,
                                                error=f"{type(e).__name__}: {e}")) + "\n")
        if (i + 1) % 5 == 0 or i == n - 1:
            el = time.time() - t0
            rate = (written + skipped) / max(1e-9, el)
            print(f"  [{note}] instance {i + 1}/{n}  written {written} skipped {skipped} "
                  f"failed {failed}  {el / 60:.1f} min  eta "
                  f"{(total - written - skipped) / max(1e-9, rate) / 60:.1f} min", flush=True)
    print(f"  DONE {note}: written {written} skipped {skipped} failed {failed} "
          f"in {(time.time() - t0) / 60:.1f} min")
    return written, failed


# --------------------------------------------------------------------------- probes

def nf_for(tag: str, c_target: int) -> int:
    return CALIB[tag]["chosen"][str(c_target)]["n_fields"]


def cells_anchors(tag):
    out = []
    for c in (8, 19, 40):
        out.append(dict(task="ledger_c", n_fields=nf_for(tag, c), layout=None, C=512,
                        arms=("full_cache",), label=f"ledger_c c={c}"))
    out.append(dict(task="mark1", n_fields=None, layout=None, C=512,
                    arms=("full_cache",), label="mark1 c=1"))
    # SPLIT-LEDGER at two record densities. N=40 records means 80 near-identical half-lines,
    # and the observed failure mode is picking the WRONG /A line, i.e. aliasing rather than
    # retention. N=20 halves the aliasing at the same line count LEDGER-C has.
    for nr in (40, 20):
        for lay in ("adjacent", "scattered"):
            out.append(dict(task="split_ledger", n_fields=4, layout=lay, C=512, n_records=nr,
                            arms=("full_cache",), label=f"split {lay} N={nr}"))
    out.append(dict(task="split_ledger", n_fields=4, layout="scattered_uniform", C=512,
                    n_records=20, arms=("full_cache",),
                    label="split scattered_uniform N=20"))
    return out


def cells_p21(tag):
    arms = LADDER + P21_METHODS
    return [dict(task="ledger_c", n_fields=nf_for(tag, c), layout=None, C=C,
                 arms=arms, label=f"p21 c={c} C={C}")
            for c in (8, 19, 40) for C in (64, 512)]


def cells_p22(tag):
    arms = ("full_cache", "floor_pos", "snapkv", "adakv_snapkv",
            "contig_matched_snapkv", "contig_matched_adakv_snapkv")
    return [dict(task="ledger_c", n_fields=nf_for(tag, 19), layout=None, C=512,
                 arms=arms, label="p22 c=19 C=512")]


def cells_p23(tag):
    arms = ("full_cache", "floor_pos", "snapkv", "adakv_snapkv",
            "oracle_causal", "oracle_prescient")
    # N=20 records, not 40. At N=40 the context holds 80 near-identical half-lines and the
    # observed failure was picking the WRONG /A line -- aliasing, not retention. The anchor
    # measured 0.177 scattered / 0.729 adjacent at N=40 against 0.542 / 0.906 at N=20, so
    # N=20 is the density at which the scattered condition is answerable at all.
    return [dict(task="split_ledger", n_fields=4, layout=lay, C=512, n_records=20, arms=arms,
                 label=f"p23 {lay} C=512 N=20") for lay in ("adjacent", "scattered")]


def cells_p24(tag):
    arms = LADDER + P24_METHODS
    return [dict(task="mark1", n_fields=None, layout=None, C=C, arms=arms,
                 label=f"p24 c=1 C={C}") for C in (64, 512)]


def cells_p23b(tag):
    """Probe 2.3, corrected layout.

    The first `scattered` layout put every /A in the first half of the body and every /B in
    the second. That guarantees separation, but it also guarantees the recency block can hold
    only /B lines, so `floor_pos` completes nothing BY CONSTRUCTION -- measured at 0.0000 for
    every compressed arm on both models. A control that forces the answer is not a control.
    `scattered_uniform` draws both halves from the same uniform distribution over the whole
    body and enforces separation as a minimum rather than a partition.
    """
    arms = ("full_cache", "floor_pos", "snapkv", "adakv_snapkv",
            "oracle_causal", "oracle_prescient")
    # C=1024 as well as 512. At C=512 the corrected layout is still VACUOUS: a two-span fact
    # needs BOTH halves inside the recency block, which is ~28% of the body there, so
    # P(both) ~ 0.08 and every budget-limited arm measured 0.000-0.010 on both models. That
    # cell cannot express the floor's advantage in either direction. At C=1024 the block is
    # ~53% of the body, P(both) ~ 0.28, which is measurable at n=50. C=1024 is outside Paper
    # 2's ladder and is used HERE ONLY, because the probe needs a non-degenerate cell and the
    # ladder's top budget cannot provide one for a two-span fact.
    return [dict(task="split_ledger", n_fields=4, layout=lay, C=C, n_records=20, arms=arms,
                 label="p23b %s C=%d N=20" % (lay, C))
            for lay in ("adjacent", "scattered_uniform") for C in (512, 1024)]


BUILDERS = {"anchors": cells_anchors, "p21": cells_p21, "p22": cells_p22,
            "p23": cells_p23, "p23b": cells_p23b, "p24": cells_p24}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", choices=sorted(BUILDERS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    rev = getattr(model.config, "_commit_hash", None) or "unresolved"
    tag = TAGS[args.model]

    cells = BUILDERS[args.probe](tag)
    out = RUNS / f"stage2_{args.probe}_{tag}.jsonl"
    print(f"{args.probe}  {tag}  {args.model}  n={args.n}  "
          f"{len(cells)} cells, {sum(len(c['arms']) for c in cells)} arm-cells")
    run_cells(model, tok, args.model, rev, out, cells, args.n, note=f"{args.probe}/{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
