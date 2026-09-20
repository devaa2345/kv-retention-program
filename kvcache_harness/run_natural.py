"""Natural-text validation of the retention/promotion decomposition (Paper 1, Option B item 2).

Arms (as in Phase 1, permanent arms + iso-token tiered arm at 8-bit):
  1 no protection | 2 structural protection | 4 protection + recoverable tier | 5 retention oracle
  6 full-cache reference.
Dataset: data/natural_l2048/nat_l2048_v1.jsonl (built by build_natural_dataset.py from Paper 3's
generator; NOT nat_v1). Model Qwen2.5-1.5B-Instruct bf16 eager, same engine and defaults as Phase 1.

Run: python -m kvcache_harness.run_natural --n 10 --budgets 256,512,1024 --out results/natural/pilot.jsonl
"""
import argparse
import json
import os
import time
import traceback

import torch

from .models import load_model_and_tokenizer
from .engine import CacheEngine
from .cache.base_policy import BudgetSpec
from .cache.permanent_evict import PermanentEvictPolicy
from .cache.structural_protection import StructuralProtectionWrapper
from .cache.recoverable_tier import RecoverableTierPolicy
from .cache.oracle_static import OracleStaticPolicy
from .tasks import natural_text as NT

DATA = "data/natural_l2048/nat_l2048_v1.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--budgets", default="256,512")
    ap.add_argument("--out", default="results/natural/raw_results.jsonl")
    ap.add_argument("--data", default=DATA)
    a = ap.parse_args()
    budgets = [int(x) for x in a.budgets.split(",")]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                r = json.loads(l)
                done.add((r["budget"], r["arm"], r["iid"]))
            except Exception:
                pass
    model, tok = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct", device="cuda")
    prompts = NT.load(a.data, a.n)
    f = open(a.out, "a")
    t0 = time.time()
    for k, p in enumerate(prompts):
        turn_texts, body_offset = NT.build_turn_texts(tok, p)
        oracle, protected, groups, n_pat = NT.compute_positions(tok, turn_texts[0], body_offset, p)
        n_ctx = len(tok(turn_texts[0])["input_ids"])
        for budget in budgets:
            spec = BudgetSpec(total_budget=budget, full_fraction=0.5)
            arms = {
                "1_no_protect_permanent": lambda: PermanentEvictPolicy(spec),
                "2_protect_permanent": lambda: StructuralProtectionWrapper(PermanentEvictPolicy(spec), groups=groups),
                "4_recoverable_protected": lambda: StructuralProtectionWrapper(RecoverableTierPolicy(spec), groups=groups),
                "5_oracle_static": lambda: OracleStaticPolicy(spec),
                "6_full_cache_ref": lambda: PermanentEvictPolicy(BudgetSpec(100_000, 1.0)),
            }
            for arm, make in arms.items():
                b_key = 0 if arm == "6_full_cache_ref" else budget      # budget-free reference: run once
                if (b_key, arm, p.iid) in done:
                    continue
                try:
                    eng = CacheEngine(model, tok, make(), device="cuda")
                    trace, answers, _ = eng.generate_multi_turn(
                        turn_texts, protected, oracle, max_answer_tokens=24, rebalance_every=1,
                        eos_token_id=tok.eos_token_id, recency_window=64, keep_sink=True)
                    correct = [NT.score_turn(answers[j], p, j) for j in range(len(p.turn_order))]
                    f.write(json.dumps({"budget": b_key, "arm": arm, "iid": p.iid,
                                        "frac_retrieved": sum(correct) / len(correct),
                                        "per_turn_correct": correct, "n_ctx": n_ctx,
                                        "n_protected_tokens": len(protected), "n_pattern_sentences": n_pat,
                                        "n_oracle_tokens": len(oracle), "answers": answers}) + "\n")
                    f.flush()
                    done.add((b_key, arm, p.iid))
                except Exception as e:
                    print(f"ERROR {p.iid} b={budget} {arm}: {e}", flush=True)
                    traceback.print_exc()
                    torch.cuda.empty_cache()
        print(f"[{k+1}/{len(prompts)}] {(time.time()-t0)/60:.1f}min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
