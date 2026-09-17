"""Two checks for the 4-bit Phase 2 run.

BAND: at 4-bit, what is the usable range the promotion decision can move
outcomes within, per budget? Upper edge = all retained at FULL, lower edge =
all retained at QUANT, retention held identical. If a budget's band is
near-zero the arms must tie for a *second* reason (tier too damaging to
recover from) and that budget cannot test H-ORTH either.

ERROR: quantization error measured on REAL cached K/V tensors from an actual
prefill rather than a synthetic vector, with the denominator stated
explicitly, reported separately for keys and values, at every bit-width.
"""
from __future__ import annotations

from statistics import mean

import torch
from transformers import DynamicCache

from . import engine as engine_mod
from .models import load_model_and_tokenizer
from .engine import CacheEngine, _quant_dequant
from .cache.base_policy import BudgetSpec
from .cache.protected_split_tier import ProtectedSplitSignalTierPolicy
from .cache.promotion_signals import AttentionPromotion
from .tasks.multi_credential import make_multi_credential_prompt, score_turn
from .run_phase0r_calibration import build_turn_texts, compute_positions

SEEDS = list(range(3000, 3015))
BITS = [8, 7, 6, 5, 4, 3, 2]
CFG = dict(n_credentials=6, value_len=14, n_distractors=20, words_per_paragraph=50,
           protect_after_chars=2, max_answer_tokens=22, rebalance_every=1,
           recency_window=64, keep_sink=True)


def run(model, tokenizer, seed, budget, full_fraction):
    prompt = make_multi_credential_prompt(
        seed=seed, n_credentials=CFG["n_credentials"], n_distractors=CFG["n_distractors"],
        words_per_paragraph=CFG["words_per_paragraph"], value_len=CFG["value_len"])
    turn_texts, body_offset = build_turn_texts(tokenizer, prompt)
    oracle_important, protected, groups, _ = compute_positions(
        tokenizer, turn_texts[0], body_offset, prompt, CFG["protect_after_chars"])
    spec = BudgetSpec(total_budget=budget, full_fraction=full_fraction)
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


@torch.no_grad()
def real_kv_error(model, tokenizer):
    """Quantization error on real cached K/V from a real prefill.

    Denominator, stated explicitly: for each layer, error is
        ||dequant(quant(X)) - X||_F  /  ||X||_F
    where X is that layer's cache tensor of shape (batch, kv_heads, seq, head_dim),
    Frobenius norm over ALL elements of the layer tensor, computed in float32.
    Reported as the mean over layers, keys and values separately. Note the
    quantizer itself is applied per (kv_heads, head_dim) slice for one position
    -- matching engine._apply_changes -- so this aggregates per-position
    quantization error up to a per-layer relative norm.
    """
    prompt = make_multi_credential_prompt(seed=3000, n_credentials=6, n_distractors=20,
                                           words_per_paragraph=50, value_len=14)
    turn_texts, _ = build_turn_texts(tokenizer, prompt)
    ids = tokenizer(turn_texts[0], return_tensors="pt")["input_ids"].to("cuda")
    cache = DynamicCache(config=model.config)
    model(input_ids=ids, past_key_values=cache, use_cache=True)

    def quant_per_position(X, bits):
        """Vectorized equivalent of applying engine._quant_dequant to each
        position's (kv_heads, head_dim) slice independently: min/max are
        taken per position over dims (heads, head_dim)."""
        levels = float(2 ** bits - 1)
        Xf = X.float()
        mn = Xf.amin(dim=(0, 2), keepdim=True)
        mx = Xf.amax(dim=(0, 2), keepdim=True)
        scale = (mx - mn) / levels
        safe = scale.abs() >= 1e-8
        q = ((Xf - mn) / torch.where(safe, scale, torch.ones_like(scale))).round().clamp(0, levels)
        deq = q * scale + mn
        return torch.where(safe, deq, Xf)

    out = {}
    for bits in BITS:
        kerrs, verrs = [], []
        for layer in cache.layers:
            for tensor, acc in ((layer.keys, kerrs), (layer.values, verrs)):
                X = tensor[0]                      # (kv_heads, seq, head_dim)
                Q = quant_per_position(X, bits)
                acc.append(((Q - X.float()).norm() / X.float().norm()).item())
        out[bits] = (mean(kerrs), mean(verrs))
    return out


def main():
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct", device="cuda")

    print("=== ERROR on REAL cached K/V (relative Frobenius norm per layer tensor, "
          "float32, mean over 28 layers) ===")
    errs = real_kv_error(model, tokenizer)
    print(f"{'bits':>5s} {'keys':>9s} {'values':>9s}")
    for b in BITS:
        k, v = errs[b]
        print(f"{b:5d} {k*100:8.2f}% {v*100:8.2f}%")

    print()
    print("=== BAND at 4-bit: what range can the promotion decision move outcomes within? ===")
    engine_mod.QUANT_BITS = 4
    print(f"{'budget':>7s} {'all-FULL':>9s} {'all-QUANT':>10s} {'band':>7s}")
    for budget in (257, 154):
        hi = mean(run(model, tokenizer, s, budget, 1.0) for s in SEEDS)
        lo = mean(run(model, tokenizer, s, budget, 0.01) for s in SEEDS)
        print(f"{budget:7d} {hi:9.3f} {lo:10.3f} {hi-lo:7.3f}", flush=True)
    engine_mod.QUANT_BITS = 8


if __name__ == "__main__":
    main()
