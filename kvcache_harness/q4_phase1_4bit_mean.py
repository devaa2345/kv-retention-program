"""QUEUE 4 — Phase 1 at 4-bit under MEAN accumulation, budget 514 only.

The −0.683 was measured under sum. The interaction should not move (a bias
shared by both arms cancels in the difference), but 4-bit is the regime where
retention quality matters most: better ranking means more credential tokens
survive to be quantized. Confirm the largest single effect in the paper
before it ships.

Arm 2 (permanent eviction, no cold tier) is bit-invariant but NOT score-mode
invariant — it ranks by attention — so it is re-run here under mean rather
than reused from the sum results.

Run: python -m kvcache_harness.q4_phase1_4bit_mean
"""
from __future__ import annotations

import json
import os
import random
import time
from statistics import mean

from . import engine as engine_mod
from .models import load_model_and_tokenizer
from .engine import CacheEngine
from .cache.base_policy import BudgetSpec
from .cache.permanent_evict import PermanentEvictPolicy
from .cache.recoverable_tier import RecoverableTierPolicy
from .cache.structural_protection import StructuralProtectionWrapper
from .tasks.multi_credential import make_multi_credential_prompt, score_turn
from .run_phase0r_calibration import build_turn_texts, compute_positions

BUDGET = 514
SEEDS = list(range(3000, 3150))
OUT = "results/q4_phase1_4bit_mean/raw.jsonl"
CFG = dict(n_credentials=6, value_len=14, n_distractors=20, words_per_paragraph=50,
           protect_after_chars=2, max_answer_tokens=22, rebalance_every=1,
           recency_window=64, keep_sink=True)


def main():
    engine_mod.SCORE_MODE = "mean"
    engine_mod.QUANT_BITS = 4
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    model, tok = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct", device="cuda")
    f = open(OUT, "w")
    t0 = time.time()

    arms = {
        "2_protect_permanent": lambda spec, g: StructuralProtectionWrapper(
            PermanentEvictPolicy(spec), groups=g),
        "4_recoverable_protected": lambda spec, g: StructuralProtectionWrapper(
            RecoverableTierPolicy(spec), groups=g),
    }
    got = {a: {} for a in arms}

    for i, seed in enumerate(SEEDS):
        p = make_multi_credential_prompt(
            seed=seed, n_credentials=CFG["n_credentials"], n_distractors=CFG["n_distractors"],
            words_per_paragraph=CFG["words_per_paragraph"], value_len=CFG["value_len"])
        tt, bo = build_turn_texts(tok, p)
        oi, prot, grp, _ = compute_positions(tok, tt[0], bo, p, CFG["protect_after_chars"])
        spec = BudgetSpec(total_budget=BUDGET, full_fraction=0.5)
        for arm, factory in arms.items():
            eng = CacheEngine(model, tok, factory(spec, grp), device="cuda")
            _, answers, _ = eng.generate_multi_turn(
                tt, prot, oi, max_answer_tokens=CFG["max_answer_tokens"],
                rebalance_every=CFG["rebalance_every"], eos_token_id=tok.eos_token_id,
                recency_window=CFG["recency_window"], keep_sink=CFG["keep_sink"])
            n = len(p.credentials)
            fr = sum(score_turn(answers[j], p.credentials[p.turn_order[j]])
                     for j in range(n)) / n
            got[arm][seed] = fr
            f.write(json.dumps({"score_mode": "mean", "quant_bits": 4, "budget": BUDGET,
                                 "arm": arm, "seed": seed, "frac_retrieved": fr}) + "\n")
        f.flush()
        if (i + 1) % 25 == 0:
            print(f"[{i+1}/{len(SEEDS)}] elapsed={(time.time()-t0)/60:.1f}min", flush=True)

    f.close()

    a2, a4 = got["2_protect_permanent"], got["4_recoverable_protected"]
    common = sorted(set(a2) & set(a4))
    d = [a4[s] - a2[s] for s in common]
    random.seed(0)
    boot = []
    for _ in range(10000):
        smp = [d[random.randrange(len(d))] for _ in range(len(d))]
        boot.append(mean(smp))
    boot.sort()

    print(f"\nBudget {BUDGET}, 4-bit, SCORE_MODE=mean, n={len(common)}")
    print(f"  protection alone    : {mean(a2[s] for s in common):.3f}")
    print(f"  protection + tiered : {mean(a4[s] for s in common):.3f}")
    print(f"  interaction         : {mean(d):+.4f}  95% CI [{boot[250]:+.4f}, {boot[9750]:+.4f}]")
    print(f"  prompts differing   : {sum(1 for x in d if x != 0)}/{len(d)}")
    print(f"  sum-mode reference  : -0.6833  [-0.7200, -0.6456]")

    engine_mod.SCORE_MODE = "sum"
    engine_mod.QUANT_BITS = 8


if __name__ == "__main__":
    main()
