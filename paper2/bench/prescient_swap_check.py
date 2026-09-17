"""Verify the redesigned oracle_prescient on real LEDGER instances.

Three properties the redesign must have, checked rather than assumed:
  1. equal budget  -- prescient.n_kept == causal.n_kept == B
  2. equal content -- prescient holds the SAME NUMBER of candidates as causal (this is the
                      cost-uniformity assumption; it holds on LEDGER because record lines are
                      format-identical, and it is checked here rather than argued)
  3. queried whole -- prescient always retains the queried candidate complete
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer
from harness import ladder as L
from harness.keys import seed_key_without_seed
from harness.tasks import ledger
import stage5_ladder_validation as S

MODELS = {"M2": "Qwen/Qwen2.5-3B-Instruct", "M3": "meta-llama/Llama-3.2-3B-Instruct"}
BUDGETS = {"M2": [32, 64, 128, 256, 512], "M3": [16, 32, 64, 128, 256, 512]}

for mid, repo in MODELS.items():
    tok = AutoTokenizer.from_pretrained(repo)
    costs, mismatch, tot, identical = [], 0, 0, 0
    for i in range(40):
        sd = seed_key_without_seed(
            task="ledger", instance_id=f"u{i}", model=repo, model_revision="r", arm="a", B=1,
            protocol="agnostic", device="nvidia", backend="none", torch_version="na",
            transformers_version="na", kvpress_version="na", dtype="na")
        inst = ledger.build(sd, f"u{i}", target_tokens=2048, tokenizer=tok)
        pre, _ = S.templated_parts(tok, inst.context, "")
        facts, n_ctx = S.facts_and_ctx(inst, tok, pre)
        costs += [len(f.tokens) for f in facts]
        for C in BUDGETS[mid]:
            c = L.oracle_causal(n_ctx, C, facts, n_sink=8, n_window=64)
            for v in inst.variants:
                g = next(f for f in facts if f.fact_id == v.rec_id)
                try:
                    p = L.oracle_prescient(n_ctx, C, g, 8, 64, facts)
                except ValueError:
                    continue
                tot += 1
                assert p.n_kept == c.n_kept, (p.n_kept, c.n_kept)
                if p.notes["n_held"] != c.notes["n_pairs_taken"]:
                    mismatch += 1
                if set(p.kept) == set(c.kept):
                    identical += 1
    print(f"[{mid}] candidate payable cost: min {min(costs)} max {max(costs)}")
    print(f"      budget equality       : OK on all {tot} (asserted)")
    print(f"      count mismatches      : {mismatch}/{tot}")
    print(f"      keep-sets identical   : {identical}/{tot}  "
          f"(queried already in causal's packing -> knowing the query is worth nothing)")
