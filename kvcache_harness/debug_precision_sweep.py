"""At what precision does the quantized recoverable tier stop being free?

Phase 2 diagnostics showed that at 8-bit, holding retention fixed and
putting 100% of retained tokens in QUANT costs nothing versus 100% FULL --
so the FULL/QUANT decision is a no-op and no promotion signal can matter.
That makes H-ORTH untestable at 8-bit rather than false.

This sweeps the tier's bit-width and, at each, compares all-FULL against
all-QUANT with retention held identical. The bit-width where a gap opens is
where the promotion decision starts to carry information -- and therefore
the only regime where Phase 2 could measure anything.

Run: python -m kvcache_harness.debug_precision_sweep
"""
from __future__ import annotations

from statistics import mean

import torch

from . import engine as engine_mod
from .models import load_model_and_tokenizer
from .engine import CacheEngine, _quant_dequant
from .cache.base_policy import BudgetSpec
from .cache.protected_split_tier import ProtectedSplitSignalTierPolicy
from .cache.promotion_signals import AttentionPromotion
from .tasks.multi_credential import make_multi_credential_prompt, score_turn
from .run_phase0r_calibration import build_turn_texts, compute_positions

BUDGET = 257
SEEDS = list(range(3000, 3020))
BITS = [8, 4, 3, 2]
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
    _, turn_answers, _ = eng.generate_multi_turn(
        turn_texts, protected, oracle_important,
        max_answer_tokens=CFG["max_answer_tokens"], rebalance_every=CFG["rebalance_every"],
        eos_token_id=tokenizer.eos_token_id, recency_window=CFG["recency_window"],
        keep_sink=CFG["keep_sink"])
    n = len(prompt.credentials)
    correct = [score_turn(turn_answers[j], prompt.credentials[prompt.turn_order[j]]) for j in range(n)]
    return sum(correct) / n


def main():
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct", device="cuda")

    print("numerical error introduced per bit-width (realistic KV vector):")
    torch.manual_seed(0)
    v = torch.randn(2, 128, dtype=torch.bfloat16)
    for b in BITS:
        q = _quant_dequant(v, bits=b)
        err = (q.float() - v.float()).norm() / v.float().norm()
        print(f"  {b}-bit: relative L2 error = {err*100:.2f}%")

    print()
    print(f"{'bits':>5s} {'all-FULL':>9s} {'all-QUANT':>10s} {'gap':>8s}")
    baseline = mean(run(model, tokenizer, s, 1.0) for s in SEEDS)
    for b in BITS:
        engine_mod.QUANT_BITS = b
        q_acc = mean(run(model, tokenizer, s, 0.01) for s in SEEDS)
        print(f"{b:5d} {baseline:9.3f} {q_acc:10.3f} {baseline-q_acc:+8.3f}", flush=True)
    engine_mod.QUANT_BITS = 8


if __name__ == "__main__":
    main()
