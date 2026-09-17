"""Engine self-check: does run_compressed with keep=ALL reproduce the plain full-cache anchor?

If it does not, the engine is wrong and any arm run through it is meaningless. This is the
"assert the invariant on live data" practice from Paper 1: a reference arm that behaves
impossibly is the tripwire.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from harness.engine import run_compressed
from harness.keys import seed_key_without_seed
from harness.tasks import ledger

M = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-3B-Instruct"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 6

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
rev = getattr(model.config, "_commit_hash", None) or "x"


def templated_parts(tok, context: str, query: str):
    """Split the chat-templated prompt into (prefix+context) and (query+generation prompt).

    The query must be appended AFTER compression (agnostic protocol), so the template has to be
    cut at the context/query boundary rather than applied to the concatenation.
    """
    marker = "QUERY"
    full = tok.apply_chat_template(
        [{"role": "user", "content": context + "\n\n" + marker}],
        tokenize=False, add_generation_prompt=True)
    pre, post = full.split(marker)
    return pre, query + post


plain = comp = 0.0
for i in range(N):
    iid = f"sc{i:04d}"
    seed = seed_key_without_seed(
        task="ledger", instance_id=iid, model=M, model_revision=rev, arm="selfcheck",
        B=1, protocol="agnostic", device="nvidia", backend="cuda-12.8",
        torch_version=torch.__version__, transformers_version="5.2.0",
        kvpress_version="0.5.4", dtype="bfloat16")
    inst = ledger.build(seed, iid, target_tokens=2048, tokenizer=tok)

    # (a) plain: whole prompt through generate(), the way the anchors were measured
    outs = []
    for v in inst.variants:
        p = tok.apply_chat_template(
            [{"role": "user", "content": f"{inst.context}\n\n{v.query}"}],
            tokenize=False, add_generation_prompt=True)
        ids = tok(p, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            g = model.generate(**ids, max_new_tokens=32, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        outs.append(tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
    plain += ledger.score_instance(outs, inst)

    # (b) engine with keep = EVERYTHING: must match (a)
    pre, _ = templated_parts(tok, inst.context, "")
    n_pre = len(tok(pre, add_special_tokens=False)["input_ids"])
    qs = [templated_parts(tok, inst.context, v.query)[1] for v in inst.variants]
    outs2 = run_compressed(model, tok, pre, qs, list(range(n_pre)), 32)
    comp += ledger.score_instance(outs2, inst)
    if i == 0:
        print("  plain[0] :", outs[0][:60])
        print("  engine[0]:", outs2[0][:60])

print(f"\n  plain full-cache      {plain / N:.4f}")
print(f"  engine keep=ALL       {comp / N:.4f}")
print(f"  MATCH: {abs(plain - comp) < 1e-9}")
