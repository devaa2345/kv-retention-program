"""ChunkKV on the existing LEDGER grid. Same instances, seeds, scorers and generation path as
stage4_run.py's plane package; only the arm differs. PINNED ENV, single process, resumable.

Cells: c in {1, 8, 19, 40} (MARK-1 at c = 1), C = 512 by default, n = 100, both models.
Rows land in runs/nvidia/chunkkv_plane_<tag>.jsonl (never in stage4_*.jsonl).
Each row carries `realised_kept` (measured from the compressed cache length after prefill, on
the first 3 instances per cell) and `deficit = B - realised_kept`.
"""
import argparse
import json
import statistics as st
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import stage4_run as S4
from harness import press
from p3 import chunkkv, keys3, runner

NL = chr(10)


def realised_kept(model, tok, pre, p):
    ids = tok(pre, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.inference_mode(), p(model):
        out = model(**ids, use_cache=True)
    pkv = out.past_key_values
    n = pkv.get_seq_length() if hasattr(pkv, "get_seq_length") else None
    del out
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--C", type=int, default=512)
    a = ap.parse_args()
    tag = S4.TAGS[a.model]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    rev = getattr(model.config, "_commit_hash", None) or "unresolved"
    out = S4.RUNS / f"chunkkv_plane_{tag}.jsonl"
    done = S4.load_done(out)
    C = a.C
    B = C + S4.N_SINK + S4.N_WINDOW
    t0, w = time.time(), 0
    for i in range(a.n):
        iid = "s4_%05d" % i
        for ct in S4.C_TAGS:
            sp = S4.spec(tag, c_tag=ct)
            sd = keys3.instance_seed(task=sp["task"], instance_id=iid, model=a.model,
                                     model_revision=rev, n_fields=sp["n_fields"], layout=None,
                                     n_records=sp["k"], **S4.ENV)
            inst = S4.build_instance(sp, sd, iid, tok)
            pre, _ = runner.templated_parts(tok, inst.context, "")
            facts, n_ctx = runner.facts_and_ctx(inst, tok, pre)
            posts = [runner.templated_parts(tok, inst.context, v.query)[1] for v in inst.variants]
            mns = S4.max_new_for(tok, inst, sp["task"])
            kd = dict(task=sp["task"], instance_id=iid, model=a.model, model_revision=rev,
                      arm="chunkkv", B=B, C=C, seed=sd, n_fields=sp["n_fields"],
                      n_records=sp["k"], layout="plane", matched_to=None, max_new=max(mns),
                      **S4.ENV)
            dg = keys3.digest(kd)
            if dg in done:
                continue
            ratio = press._ratio_for(B, n_ctx)
            p = chunkkv.build_chunkkv(ratio, n_ctx, S4.N_SINK, S4.N_WINDOW)
            rk = None
            if i < 3:
                rk = realised_kept(model, tok, pre, chunkkv.build_chunkkv(ratio, n_ctx))
                exp = chunkkv.expected_kept(n_ctx, B)
                assert rk == exp, f"realised {rk} != expected {exp} (B={B}, n_ctx={n_ctx})"
                assert 0 <= B - rk < chunkkv.CHUNK, (B, rk)
            t1 = time.time()
            outs, flags = runner.generate_with(model, tok, pre, posts, p, mns)
            row = dict(key_digest=dg, key=kd, model_tag=tag, package="chunkkv_plane",
                       task=sp["task"], label=sp["label"], c=sp["c"], k=sp["k"], C=C, B=B,
                       arm="chunkkv", n_ctx=n_ctx, realised_kept=rk,
                       expected_kept=chunkkv.expected_kept(n_ctx, B),
                       deficit=B - chunkkv.expected_kept(n_ctx, B),
                       score=S4.SCORERS[sp["task"]](outs, inst),
                       per_variant=[S4.SCORE_ONE[sp["task"]](o, v.answer)
                                    for v, o in zip(inst.variants, outs)],
                       gen=outs, answers=[v.answer for v in inst.variants], stop_flags=flags,
                       wall_s=time.time() - t1, meta=inst.meta)
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + NL)
            done.add(dg)
            w += 1
        if (i + 1) % 5 == 0:
            el = (time.time() - t0) / 60
            print(f"  [chunkkv {tag} C={C}] {i + 1}/{a.n}  {el:.1f} min  eta "
                  f"{el / (i + 1) * (a.n - i - 1):.1f} min", flush=True)
    print("DONE", tag, C, flush=True)


if __name__ == "__main__":
    main()
