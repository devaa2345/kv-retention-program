"""Stage 4 competence tuning (v1 §4.2 knobs, §4.5 band). PINNED ENV ONLY.

v1 §4.5: a model enters a cell only if its uncompressed `full_cache` anchor sits in
[0.55, 0.97]. Above 0.97 the ceiling cannot be distinguished from the floor; below 0.55 the
floor effect dominates. v1 §4.2 names the knobs: N (records), H (bindings), hop count.

At the shipped N=96 / H=4 the anchor is 0.067 on Qwen2.5-1.5B -- far below band. This sweeps N
to find a setting that lands every admitted model inside it. **Tuning happens here, against
`full_cache` only, and is frozen before Stage 6 -- never after method results are visible.**
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness.keys import seed_key_without_seed
from harness.tasks import ledger

MAX_NEW = 32  # B9 RESOLVED: raised from 12, which zeroed Llama on format alone


@torch.inference_mode()
def anchor(model, tok, n_records: int, n_bindings: int, n: int, repo: str, rev: str,
           max_new: int = MAX_NEW) -> dict:
    hits = wf = 0
    samples = []
    for i in range(n):
        iid = f"sw{n_records}_{n_bindings}_{i:04d}"
        seed = seed_key_without_seed(
            task="ledger", instance_id=iid, model=repo, model_revision=rev, arm="full_cache",
            B=1, protocol="agnostic", device="nvidia", backend="cuda-12.8",
            torch_version=torch.__version__, transformers_version="5.2.0",
            kvpress_version="0.5.4", dtype="bfloat16",
        )
        inst = ledger.build(seed, iid, n_records=n_records, n_bindings=n_bindings,
                            target_tokens=2048, tokenizer=tok)
        # Decision 7: the instance score is the MEAN of the per-variant scores, so every one
        # of the H queries is asked. `score_instance` gives partial credit (0, 1/H, ... 1).
        outs = []
        for v in inst.variants:
            msgs = [{"role": "user", "content": f"{inst.context}\n\n{v.query}"}]
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(p, return_tensors="pt").to(model.device)
            g = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            outs.append(tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
        if len(samples) < 3:
            samples.append(outs[0])
        hits += ledger.score_instance(outs, inst)
        wf += sum(bool(re.search(r"\b\d{6}\b", o)) for o in outs) / len(outs)
    return {"n_records": n_records, "n_bindings": n_bindings, "n": n, "max_new": max_new,
            "anchor": round(hits / n, 4), "well_formed": round(wf / n, 4),
            "chance": round(1 / n_records, 4), "in_band": bool(0.55 <= hits / n <= 0.97),
            "sample": samples}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--records", default="96,48,24,12")
    ap.add_argument("--bindings", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=MAX_NEW,
                    help="DIAGNOSTIC ONLY. v1 §5.4 pins 12 for Task B.")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda").eval()
    rev = getattr(model.config, "_commit_hash", None) or "unresolved"

    rows = []
    print(f"{args.model}   (H={args.bindings}, n={args.n} per point)")
    print(f"  {'N':>4} {'chance':>7} {'anchor':>7} {'well-fm':>8}  in band [0.55,0.97]")
    for nr in [int(x) for x in args.records.split(",")]:
        r = anchor(model, tok, nr, args.bindings, args.n, args.model, rev, args.max_new)
        rows.append(r)
        print(f"  {nr:4d} {r['chance']:7.4f} {r['anchor']:7.4f} {r['well_formed']:8.3f}  "
              f"{'YES' if r['in_band'] else 'no'}")

    tag = args.model.split("/")[-1].replace(".", "_")
    p = Path(__file__).resolve().parent / "gates" / "nvidia" / f"stage4_competence_{tag}_mn{args.max_new}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"model": args.model, "model_revision": rev,
                             "n_bindings": args.bindings, "band": [0.55, 0.97],
                             "sweep": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
