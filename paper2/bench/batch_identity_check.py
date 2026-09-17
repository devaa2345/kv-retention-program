"""Batch-vs-batch-1 bit-identity check (v1 §6.1). PINNED ENV ONLY.

Decision 7 makes every instance H generations that share one 2048-token context and differ
only in a short query suffix, and decode is ~91% of an instance-arm's cost. Batching the H
variants together is therefore the obvious throughput lever.

v1 §6.1 permits it only under a hard condition:

    "batch 8 is the target for throughput and must be validated to produce **bit-identical
     outputs to batch 1** before use (batching changes reduction order -- if outputs differ,
     batch size joins the dedup key or batching is abandoned)."

This compares generated token IDs exactly, not decoded text, because text comparison can hide
a differing token that detokenises to the same string. Left padding is used, since decoder-only
generation with right padding is wrong regardless of this check.

Exit 0 = identical (adopt batching). Exit 1 = differs (batch_size joins the dedup key).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# This module lives in bench/, so running it as `python bench/...` puts bench/ on sys.path
# rather than the repo root. Put the repo root first so `harness` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness.keys import seed_key_without_seed
from harness.tasks import ledger

MAX_NEW = 32


@torch.inference_mode()
def gen_single(model, tok, prompts: list[str]) -> list[list[int]]:
    out = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").to(model.device)
        g = model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=False,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
        out.append(g[0][ids["input_ids"].shape[1]:].tolist())
    return out


@torch.inference_mode()
def gen_batched(model, tok, prompts: list[str]) -> list[list[int]]:
    old_side = tok.padding_side
    tok.padding_side = "left"          # decoder-only generation requires left padding
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    g = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                       pad_token_id=tok.pad_token_id)
    tok.padding_side = old_side
    plen = enc["input_ids"].shape[1]
    return [row[plen:].tolist() for row in g]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=25, help="instances (each contributes H prompts)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda").eval()
    rev = getattr(model.config, "_commit_hash", None) or "unresolved"

    n_prompt = n_ident = 0
    n_inst_ident = 0
    first_diff = None
    for i in range(args.n):
        iid = f"bid{i:05d}"
        seed = seed_key_without_seed(
            task="ledger", instance_id=iid, model=args.model, model_revision=rev,
            arm="full_cache", B=1, protocol="agnostic", device="nvidia",
            backend="cuda-12.8", torch_version=torch.__version__,
            transformers_version="5.2.0", kvpress_version="0.5.4", dtype="bfloat16",
        )
        inst = ledger.build(seed, iid, target_tokens=2048, tokenizer=tok)
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": f"{inst.context}\n\n{v.query}"}],
                tokenize=False, add_generation_prompt=True,
            )
            for v in inst.variants
        ]
        a = gen_single(model, tok, prompts)
        b = gen_batched(model, tok, prompts)
        same = [x == y for x, y in zip(a, b)]
        n_prompt += len(same)
        n_ident += sum(same)
        n_inst_ident += all(same)
        if first_diff is None and not all(same):
            k = same.index(False)
            first_diff = {
                "instance": iid, "variant": k,
                "batch1_ids": a[k][:12], "batched_ids": b[k][:12],
                "batch1_text": tok.decode(a[k], skip_special_tokens=True),
                "batched_text": tok.decode(b[k], skip_special_tokens=True),
            }

    ident = n_ident == n_prompt
    res = {
        "model": args.model, "model_revision": rev, "n_instances": args.n,
        "H": ledger.N_BINDINGS, "batch_size": ledger.N_BINDINGS, "max_new": MAX_NEW,
        "prompts_compared": n_prompt, "prompts_identical": n_ident,
        "instances_identical": n_inst_ident,
        "token_identical": ident,
        "first_divergence": first_diff,
        "verdict": ("ADOPT batching -- outputs are bit-identical to batch 1" if ident else
                    "DO NOT batch silently -- outputs differ; batch_size must join the dedup key"),
    }
    print(f"{args.model}")
    print(f"  prompts compared      {n_prompt}")
    print(f"  bit-identical         {n_ident}  ({n_ident / max(1, n_prompt):.4f})")
    print(f"  instances all-identical {n_inst_ident}/{args.n}")
    print(f"  VERDICT: {res['verdict']}")
    if first_diff:
        print(f"  first divergence: {first_diff['instance']} variant {first_diff['variant']}")
        print(f"    batch1  : {first_diff['batch1_ids']}")
        print(f"    batched : {first_diff['batched_ids']}")

    tag = args.model.split("/")[-1].replace(".", "_")
    p = Path(__file__).resolve().parent.parent / "gates" / "nvidia" / f"batch_identity_{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {p}")
    return 0 if ident else 1


if __name__ == "__main__":
    raise SystemExit(main())
