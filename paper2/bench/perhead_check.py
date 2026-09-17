import sys; sys.path.insert(0, ".")
from transformers import AutoTokenizer
from harness import press
from harness.keys import seed_key_without_seed
from harness.tasks import ledger
from stage5_ladder_validation import facts_and_ctx, templated_parts

M = "meta-llama/Llama-3.2-3B-Instruct"
tok = AutoTokenizer.from_pretrained(M)
sd = seed_key_without_seed(task="ledger", instance_id="grid_00000", model=M,
    model_revision="x", arm="grid", B=1, protocol="agnostic", device="nvidia",
    backend="cuda-12.8", torch_version="t", transformers_version="5.2.0",
    kvpress_version="0.5.4", dtype="bfloat16")
inst = ledger.build(sd, "grid_00000", target_tokens=2048, tokenizer=tok)
pre, _ = templated_parts(tok, inst.context, "")
facts, n_ctx = facts_and_ctx(inst, tok, pre)
print("n_ctx", n_ctx, "candidates", len(facts))

for C in (16, 32, 64, 128, 256, 512):
    ph = press._build_perhead(n_ctx, C, facts, 8, 64, 8)
    sets = ph.keep_by_head
    sizes = sorted({len(v) for v in sets.values()})
    distinct = len({v for v in sets.values()})
    union = set().union(*sets.values())
    g, _ = press.build_arm("oracle_causal", n_ctx=n_ctx, C=C, n_sink=8, n_window=64,
                           facts=facts, seed=sd)
    print(f"C={C:4d} kept_sizes={sizes} distinct_head_sets={distinct}/8 "
          f"union={len(union)}")

# how much does the candidate set actually cost? Delta_head/Delta_temporal can only be
# non-zero where the budget FORCES a choice among candidates.
from harness import ladder
floor, region = ladder._floors(n_ctx, 8, 64)
rs = set(region)
costs = [(f.fact_id, len({t for s in f.spans for t in s if t in rs})) for f in facts]
print("\npayable cost per candidate:", costs)
print("total payable:", sum(c for _, c in costs))
for C in (16, 32, 64, 128, 256, 512):
    k = press._optimal_count([(len({t for s in f.spans for t in s if t in rs}), f,
                               {t for s in f.spans for t in s if t in rs}) for f in facts], C)
    print(f"  C={C:4d}  optimal complete candidates k={k} of {len(facts)}"
          f"   {'BINDING' if k < len(facts) else 'not binding -- all candidates fit'}")
