"""Sensitivity of the iso-memory dissociation to `quant_byte_cost`, which
was chosen (0.25) post-hash and is load-bearing for the Phase 1 iso-memory
result. One cell: budget=257, arm=4_recoverable_protected, n=150 seeds,
swept over quant_byte_cost. Everything else identical to Phase 1.

    T = nominal_budget / (full_fraction*1.0 + (1-full_fraction)*quant_byte_cost)

Run: python -m kvcache_harness.run_quant_sensitivity
"""
from __future__ import annotations

import json
import os
import time
import traceback

import torch

from .models import load_model_and_tokenizer
from .engine import CacheEngine
from .cache.base_policy import BudgetSpec
from .cache.recoverable_tier import RecoverableTierPolicy
from .cache.structural_protection import StructuralProtectionWrapper
from .tasks.multi_credential import make_multi_credential_prompt, score_turn
from .run_phase0r_calibration import build_turn_texts, compute_positions

NOMINAL_BUDGET = 257
FULL_FRACTION = 0.5
QUANT_COSTS = [0.125, 0.25, 0.30, 0.50]
N_PROMPTS = 150
SEED_START = 3000
OUT = "results/quant_sensitivity/raw_results.jsonl"

CFG = dict(n_credentials=6, value_len=14, n_distractors=20, words_per_paragraph=50,
           protect_after_chars=2, max_answer_tokens=22, rebalance_every=1,
           recency_window=64, keep_sink=True)


def tokens_for(quant_cost):
    return round(NOMINAL_BUDGET / (FULL_FRACTION * 1.0 + (1 - FULL_FRACTION) * quant_cost))


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
                done.add((r["quant_byte_cost"], r["seed"]))
            except Exception:
                pass

    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct", device="cuda")
    f = open(OUT, "a")
    t0 = time.time()

    for qc in QUANT_COSTS:
        T = tokens_for(qc)
        print(f"=== quant_byte_cost={qc} -> true token budget {T} (nominal {NOMINAL_BUDGET}) ===", flush=True)
        for i in range(N_PROMPTS):
            seed = SEED_START + i
            if (qc, seed) in done:
                continue
            try:
                prompt = make_multi_credential_prompt(
                    seed=seed, n_credentials=CFG["n_credentials"], n_distractors=CFG["n_distractors"],
                    words_per_paragraph=CFG["words_per_paragraph"], value_len=CFG["value_len"])
                turn_texts, body_offset = build_turn_texts(tokenizer, prompt)
                oracle_important, protected, protected_groups, _ = compute_positions(
                    tokenizer, turn_texts[0], body_offset, prompt, CFG["protect_after_chars"])

                spec = BudgetSpec(total_budget=T, full_fraction=FULL_FRACTION)
                policy = StructuralProtectionWrapper(RecoverableTierPolicy(spec), groups=protected_groups)
                engine = CacheEngine(model, tokenizer, policy, device="cuda")
                trace, turn_answers, _ = engine.generate_multi_turn(
                    turn_texts, protected, oracle_important,
                    max_answer_tokens=CFG["max_answer_tokens"], rebalance_every=CFG["rebalance_every"],
                    eos_token_id=tokenizer.eos_token_id, recency_window=CFG["recency_window"],
                    keep_sink=CFG["keep_sink"])
                n = len(prompt.credentials)
                correct = [score_turn(turn_answers[j], prompt.credentials[prompt.turn_order[j]]) for j in range(n)]
                f.write(json.dumps({"quant_byte_cost": qc, "true_tokens": T, "seed": seed,
                                     "frac_retrieved": sum(correct) / n}) + "\n")
                f.flush()
            except Exception as e:
                print(f"ERROR qc={qc} seed={seed}: {e}", flush=True)
                traceback.print_exc()
                torch.cuda.empty_cache()
            if (i + 1) % 25 == 0:
                print(f"  qc={qc} {i+1}/{N_PROMPTS} elapsed={(time.time()-t0)/60:.1f}min", flush=True)
    f.close()
    print("quant sensitivity complete", flush=True)


if __name__ == "__main__":
    main()
