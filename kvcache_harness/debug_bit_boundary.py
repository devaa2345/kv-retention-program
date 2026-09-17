"""Locate the free -> damaging boundary of the quantized tier with measured
edges rather than inferred from the 8-to-4 jump.

The reconstruction-error curve is smooth (roughly doubling per bit, no
cliff), so the boundary has to be located by ACCURACY, not error. Sweeps
all-FULL vs all-QUANT with retention held identical at every bit-width from
8 down to 3, budget 257.
"""
from __future__ import annotations

from statistics import mean

from . import engine as engine_mod
from .models import load_model_and_tokenizer
from .engine import CacheEngine
from .cache.base_policy import BudgetSpec
from .cache.protected_split_tier import ProtectedSplitSignalTierPolicy
from .cache.promotion_signals import AttentionPromotion
from .tasks.multi_credential import make_multi_credential_prompt, score_turn
from .run_phase0r_calibration import build_turn_texts, compute_positions

BUDGET = 257
SEEDS = list(range(3000, 3025))
BITS = [8, 7, 6, 5, 4, 3]
CFG = dict(n_credentials=6, value_len=14, n_distractors=20, words_per_paragraph=50,
           protect_after_chars=2, max_answer_tokens=22, rebalance_every=1,
           recency_window=64, keep_sink=True)


def run(model, tokenizer, seed, full_fraction):
    prompt = make_multi_credential_prompt(
        seed=seed, n_credentials=CFG["n_credentials"], n_distractors=CFG["n_distractors"],
        words_per_paragraph=CFG["words_per_paragraph"], value_len=CFG["value_len"])
    turn_texts, body_offset = build_turn_texts(tokenizer, prompt)
    oracle_important, protected, groups, _ = compute_positions(
        tokenizer, turn_texts[0], body_offset, prompt, CFG["protect_after_chars"])
    spec = BudgetSpec(total_budget=BUDGET, full_fraction=full_fraction)
    policy = ProtectedSplitSignalTierPolicy(spec, AttentionPromotion(), groups=groups)
    eng = CacheEngine(model, tokenizer, policy, device="cuda")
    _, answers, _ = eng.generate_multi_turn(
        turn_texts, protected, oracle_important,
        max_answer_tokens=CFG["max_answer_tokens"], rebalance_every=CFG["rebalance_every"],
        eos_token_id=tokenizer.eos_token_id, recency_window=CFG["recency_window"],
        keep_sink=CFG["keep_sink"])
    n = len(prompt.credentials)
    return sum(score_turn(answers[j], prompt.credentials[prompt.turn_order[j]])
               for j in range(n)) / n


def main():
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct", device="cuda")
    baseline = mean(run(model, tokenizer, s, 1.0) for s in SEEDS)
    print(f"all-FULL reference (bit-width irrelevant): {baseline:.3f}  n={len(SEEDS)}", flush=True)
    print(f"{'bits':>5s} {'all-QUANT':>10s} {'gap':>8s}")
    for b in BITS:
        engine_mod.QUANT_BITS = b
        q = mean(run(model, tokenizer, s, 0.01) for s in SEEDS)
        print(f"{b:5d} {q:10.3f} {baseline-q:+8.3f}", flush=True)
    engine_mod.QUANT_BITS = 8


if __name__ == "__main__":
    main()
