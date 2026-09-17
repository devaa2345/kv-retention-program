"""A40 stratum -- Stage 0: env verify, LEDGER-C n_fields calibration, rate benchmark, anchor gate.

Single process. Device is its own stratum ("a40"), never differenced against Machine N
(RTX 5070, device="nvidia") or Machine A (RX 7900 XTX). Model: Qwen/Qwen2.5-14B-Instruct, bf16.

Reuses Paper 3's task/harness code unmodified (p3.tasks.ledger_c, p3.runner, harness.press)
so every invariant (position_ids from the uncompressed length, generation-only press checks,
mandatory sink/window floors, CRC32 seeds, dedup key fields) is inherited rather than
reimplemented.
"""
from __future__ import annotations

import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper3"))  # exposes p3 package; p3.runner inserts paper2 itself

from p3 import keys3, runner          # noqa: E402
from p3.tasks import ledger_c, mark1  # noqa: E402
from harness import press             # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
LOG = HERE / "logs"
OUT.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "/workspace/models/Qwen2.5-14B-Instruct"
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
MODEL_REV = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
ENV = dict(backend="cuda-12.8", torch_version=torch.__version__,
           transformers_version="5.2.0", kvpress_version="0.5.4",
           dtype="bfloat16", device="a40", batch_size=1, protocol="agnostic")
TARGETS = (1, 8, 19, 40)
N_SINK, N_WINDOW = runner.N_SINK, runner.N_WINDOW
BAND = (0.55, 0.97)


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa")
    model.eval()
    return model, tok


# --------------------------------------------------------------------- calibration

def calibrate(tok, targets=TARGETS, target_tokens=2048):
    seeds = [(1000 + i, f"cal_{i:05d}") for i in range(8)]
    table = {}
    for nf in range(1, 40):
        cs, Ls, ansl = [], [], []
        for sd, iid in seeds:
            inst = ledger_c.build(sd, iid, n_fields=nf, target_tokens=target_tokens, tokenizer=tok)
            pre, _ = runner.templated_parts(tok, inst.context, "")
            enc = tok(pre, add_special_tokens=False, return_offsets_mapping=True)
            off = enc["offset_mapping"]
            Ls.append(len(off))
            base = pre.index(inst.context)
            import re
            REC = re.compile(r"^R(\d{3}) \| ")
            for line in inst.context.split("\n"):
                m = REC.match(line)
                if not m or not (1 <= int(m.group(1)) <= ledger_c.N_RECORDS):
                    continue
                a = inst.context.index(line) + base
                b = a + len(line)
                cs.append(sum(1 for (x, y) in off if y > x and x < b and y > a))
            for v in inst.variants:
                ansl.append(len(tok(v.answer, add_special_tokens=False)["input_ids"]))
        table[nf] = dict(c=st.fmean(cs), L=st.fmean(Ls), answer_tokens=st.fmean(ansl))
        if table[nf]["c"] > max(targets) + 8:
            break
    chosen = {}
    for t in targets:
        nf = min(table, key=lambda k: abs(table[k]["c"] - t))
        chosen[str(t)] = dict(n_fields=nf, **table[nf])
    return dict(table=table, chosen=chosen)


# --------------------------------------------------------------------- instance build

def build_instance(c_tag, n_fields, i, tok, prefix):
    iid = f"{prefix}_c{c_tag}_{i:05d}"
    sd = keys3.instance_seed(task="ledger_c", instance_id=iid, model=MODEL_NAME,
                             model_revision=MODEL_REV, n_fields=n_fields,
                             n_records=ledger_c.N_RECORDS, layout=None, **ENV)
    inst = ledger_c.build(sd, iid, n_fields=n_fields, target_tokens=2048, tokenizer=tok)
    pre, _ = runner.templated_parts(tok, inst.context, "")
    facts, n_ctx = runner.facts_and_ctx(inst, tok, pre)
    posts = [runner.templated_parts(tok, inst.context, v.query)[1] for v in inst.variants]
    max_new = [len(tok(v.answer, add_special_tokens=False)["input_ids"]) + runner.MAX_NEW_SLACK
               for v in inst.variants]
    return dict(iid=iid, seed=sd, inst=inst, pre=pre, facts=facts, n_ctx=n_ctx,
                posts=posts, max_new=max_new)


def build_instance_mark1(i, tok, prefix):
    """c = 1 anchor, via MARK-1 (single-token facts) -- LEDGER-C cannot reach c = 1
    (n_fields=1 already measures c ~ 8 on this tokenizer: record id + separator + one
    multi-token surname). This mirrors Paper 3's own PREREG_P3 convention exactly."""
    iid = f"{prefix}_c1_{i:05d}"
    sd = keys3.instance_seed(task="mark1", instance_id=iid, model=MODEL_NAME,
                             model_revision=MODEL_REV, n_fields=None,
                             n_records=mark1.N_ENTRIES, layout=None, **ENV)
    inst = mark1.build(sd, iid, target_tokens=2048, tokenizer=tok)
    pre, _ = runner.templated_parts(tok, inst.context, "")
    facts, n_ctx = runner.facts_and_ctx(inst, tok, pre)
    posts = [runner.templated_parts(tok, inst.context, v.query)[1] for v in inst.variants]
    max_new = [len(tok(v.answer, add_special_tokens=False)["input_ids"]) + runner.MAX_NEW_SLACK
               for v in inst.variants]
    return dict(iid=iid, seed=sd, inst=inst, pre=pre, facts=facts, n_ctx=n_ctx,
                posts=posts, max_new=max_new)


# --------------------------------------------------------------------- step 5: rate benchmark

def rate_benchmark(model, tok, n_fields_19, n=200, C=512):
    times = []
    keepsets = {}
    n_ctx_seen = []
    for i in range(n):
        rec = build_instance(19, n_fields_19, i, tok, "rate")
        inst, pre, facts, n_ctx, posts, max_new = (
            rec["inst"], rec["pre"], rec["facts"], rec["n_ctx"], rec["posts"], rec["max_new"])
        n_ctx_seen.append(n_ctx)
        kept = runner.ladder_kept("floor_pos", n_ctx, C, facts, rec["seed"])
        keepsets[rec["iid"]] = np.array(sorted(kept), dtype=np.int32)
        p, _ = press.build_arm("floor_pos", n_ctx=n_ctx, C=C, n_sink=N_SINK, n_window=N_WINDOW)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        texts, flags = runner.generate_with(model, tok, pre, posts, p, max_new)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if i == 0:
            score0 = ledger_c.score_instance(texts, inst)
    np.savez(OUT / "keepsets_floor_pos_c19_C512_rate200.npz",
              **{k: v for k, v in keepsets.items()})
    return dict(n=n, C=C, n_fields=n_fields_19, mean_n_ctx=st.fmean(n_ctx_seen),
                mean_sec_per_record=st.fmean(times), median_sec_per_record=st.median(times),
                stdev_sec_per_record=st.pstdev(times), min_sec=min(times), max_sec=max(times),
                total_sec=sum(times), records_per_hour=3600.0 / st.fmean(times),
                sample_first_score=score0, per_record_sec=times)


# --------------------------------------------------------------------- step 6: anchor gate

def anchor_cell(model, tok, c_tag, n_fields, n=100):
    scores = []
    flags_all = []
    scorer = mark1.score_instance if c_tag == 1 else ledger_c.score_instance
    for i in range(n):
        if c_tag == 1:
            rec = build_instance_mark1(i, tok, "anchor")
        else:
            rec = build_instance(c_tag, n_fields, i, tok, "anchor")
        inst, pre, posts, max_new = rec["inst"], rec["pre"], rec["posts"], rec["max_new"]
        texts, flags = runner.generate_with(model, tok, pre, posts, None, max_new)
        scores.append(scorer(texts, inst))
        flags_all.extend(flags)
    acc = st.fmean(scores)
    return dict(c_tag=c_tag, n_fields=n_fields, n=n, accuracy=acc,
                task=("mark1" if c_tag == 1 else "ledger_c"),
                in_band=BAND[0] <= acc <= BAND[1],
                eos_rate=sum(1 for f in flags_all if f == "eos") / len(flags_all),
                per_instance_scores=scores)


def main():
    print("loading model...", flush=True)
    model, tok = load_model()
    torch.cuda.synchronize()
    vram_gb = torch.cuda.memory_allocated() / 1e9
    print(f"loaded. VRAM allocated: {vram_gb:.2f} GB", flush=True)

    print("calibrating n_fields for c in", TARGETS, flush=True)
    calib = calibrate(tok)
    (OUT / "calibration.json").write_text(json.dumps(calib, indent=2), encoding="utf-8")
    for t in TARGETS:
        print(f"  target c={t:3d} -> n_fields={calib['chosen'][str(t)]['n_fields']:2d} "
              f"achieved c={calib['chosen'][str(t)]['c']:.3f}", flush=True)

    n_fields_19 = calib["chosen"]["19"]["n_fields"]
    print(f"\nstep 5: rate benchmark, 200 records, LEDGER-C c~19 (n_fields={n_fields_19}), "
          f"floor_pos C=512", flush=True)
    rb = rate_benchmark(model, tok, n_fields_19, n=200, C=512)
    rb_summary = {k: v for k, v in rb.items() if k != "per_record_sec"}
    (OUT / "rate_benchmark.json").write_text(json.dumps(rb_summary, indent=2), encoding="utf-8")
    np.save(OUT / "rate_benchmark_per_record_sec.npy", np.array(rb["per_record_sec"]))
    print(f"  mean {rb['mean_sec_per_record']:.4f} s/record "
          f"({rb['records_per_hour']:.1f} records/hour), "
          f"median {rb['median_sec_per_record']:.4f} s, VRAM used {vram_gb:.2f} GB", flush=True)

    print("\nstep 6: full_cache competence anchor gate, LEDGER-C, n=100 per cell", flush=True)
    gate = {}
    for t in TARGETS:
        nf = None if t == 1 else calib["chosen"][str(t)]["n_fields"]
        cell = anchor_cell(model, tok, t, nf, n=100)
        gate[str(t)] = cell
        band_txt = "IN BAND" if cell["in_band"] else "OUT OF BAND"
        nf_txt = "None" if nf is None else f"{nf:2d}"
        print(f"  c={t:3d} (n_fields={nf_txt}): accuracy={cell['accuracy']:.4f}  {band_txt}",
              flush=True)
        (OUT / f"anchor_cell_c{t}.json").write_text(json.dumps(cell, indent=2), encoding="utf-8")
    gate_summary = {k: {kk: vv for kk, vv in v.items() if kk != "per_instance_scores"}
                    for k, v in gate.items()}
    (OUT / "anchor_gate.json").write_text(json.dumps(gate_summary, indent=2), encoding="utf-8")
    (OUT / "anchor_gate_full.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    print("\n=== GATE SUMMARY ===")
    any_out = False
    for t in TARGETS:
        c = gate[str(t)]
        print(f"c={t:3d}: acc={c['accuracy']:.4f} n={c['n']} "
              f"{'IN' if c['in_band'] else 'OUT OF'} band {BAND}")
        if not c["in_band"]:
            any_out = True
    if any_out:
        print("AT LEAST ONE ANCHOR IS OUT OF BAND -- stop, do not start grid work.")
    else:
        print("All four anchors in band. STOP HERE per instructions -- do not start grid work.")


if __name__ == "__main__":
    main()
