"""Item 4: bound the selection bias in Delta_head, M2 C=32.

The equal-candidate-count guard refused 47 of 200 instances. Those refusals are NOT
outcome-neutral: on the 47, `floor_pos` scores 0.2606; on the 153 kept, 0.0000. The guard drops
instances whose records overlap the sink/window floor -- exactly the ones the floor can answer.
Delta_head is now a headline result, so the bias needs bounding rather than assuming.

This runs `oracle_causal_perhead` with `press.PERHEAD_STRICT = False` so the refused instances
also produce a cell, and reports Delta_head three ways:

    matched 153   -- guard respected, equal candidate count per head (the admissible measurement)
    excluded 47   -- guard-refused instances only
    all 200       -- both pooled

The 47 are NOT an admissible Delta_head measurement: their heads hold different candidate
counts, so the contrast there mixes head allocation with a budget difference. They are computed
only to bound how far the matched estimate could be from the population value.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import press, stats
from harness.keys import seed_key_without_seed
from harness.tasks import ledger
from stage5_ladder_validation import facts_and_ctx, generate_with, templated_parts

M = "Qwen/Qwen2.5-3B-Instruct"
C = 32
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
N_SINK, N_WINDOW = 8, 64
OUT = Path("runs/nvidia/diag_dhead_guard_M2_C32.jsonl")

press.PERHEAD_STRICT = False          # deliberately relaxed; see module docstring

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
rev = getattr(model.config, "_commit_hash", None) or "x"
n_kv = int(getattr(model.config, "num_key_value_heads", model.config.num_attention_heads))
OUT.unlink(missing_ok=True)

# which instances did the guard refuse in the real N9 run?
have = set()
for line in Path("runs/nvidia/n9_M2_ledger.jsonl").open(encoding="utf-8"):
    r = json.loads(line)
    if r["C"] == C and r["key"]["arm"] == "oracle_causal_perhead":
        have.add(r["key"]["instance_id"])
causal = {}
for line in Path("runs/nvidia/n9_M2_ledger.jsonl").open(encoding="utf-8"):
    r = json.loads(line)
    if r["C"] == C and r["key"]["arm"] == "oracle_causal":
        causal[r["key"]["instance_id"]] = r["score"]
print("  guard kept %d instances at C=%d; recomputing all %d with the guard relaxed"
      % (len(have), C, N))

ph = {}
for i in range(N):
    iid = "grid_%05d" % i
    sd = seed_key_without_seed(
        task="ledger", instance_id=iid, model=M, model_revision=rev, arm="grid", B=1,
        protocol="agnostic", device="nvidia", backend="cuda-12.8",
        torch_version=torch.__version__, transformers_version="5.2.0",
        kvpress_version="0.5.4", dtype="bfloat16")
    inst = ledger.build(sd, iid, target_tokens=2048, tokenizer=tok)
    pre, _ = templated_parts(tok, inst.context, "")
    facts, n_ctx = facts_and_ctx(inst, tok, pre)
    posts = [templated_parts(tok, inst.context, v.query)[1] for v in inst.variants]
    try:
        p, _ = press.build_arm("oracle_causal_perhead", n_ctx=n_ctx, C=C, n_sink=N_SINK,
                               n_window=N_WINDOW, facts=facts, seed=sd, n_heads=n_kv)
        sc = ledger.score_instance(generate_with(model, tok, pre, posts, p), inst)
        ph[iid] = sc
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"instance_id": iid, "C": C, "score": sc,
                                "guard_kept": iid in have}) + "\n")
    except Exception as e:
        print("    %s failed: %s" % (iid, type(e).__name__))
    if (i + 1) % 25 == 0:
        print("  [%d/%d]" % (i + 1, N), flush=True)

print("\nDelta_head = A(perhead) - A(causal),  M2  C=%d,  %d KV heads" % (C, n_kv))
print("  %-16s %6s %10s %10s %10s %24s"
      % ("subset", "n", "perhead", "causal", "delta", "95% CI"))
groups = [("matched 153", [i for i in ph if i in have]),
          ("excluded 47", [i for i in ph if i not in have]),
          ("all 200", list(ph))]
for name, ids in groups:
    ids = [i for i in ids if i in causal]
    if not ids:
        continue
    a = [ph[i] for i in ids]
    b = [causal[i] for i in ids]
    d = stats.paired_contrast(a, b, seed=32, n_boot=20000)
    print("  %-16s %6d %10.4f %10.4f %+10.4f  [%+.4f, %+.4f]"
          % (name, len(ids), statistics.fmean(a), statistics.fmean(b),
             d.mean_diff, d.ci_low, d.ci_high))
print("\n  NOTE: the 'excluded 47' and 'all 200' rows are NOT admissible Delta_head")
print("  measurements -- those heads hold unequal candidate counts, so the contrast mixes")
print("  head allocation with a budget difference. They bound the bias; they do not replace")
print("  the matched estimate.")
print("  wrote %s" % OUT)
