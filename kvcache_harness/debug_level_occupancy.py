"""Level-occupancy histogram per bit-width — the last untested hypothesis
for the non-monotonic accuracy curve.

Level COUNTS were already verified correct (<= 2^bits at every width, on
10,640 calls intercepted during real generation). But a correct count says
nothing about the DISTRIBUTION across those levels. Under per-position
affine min-max quantization, a vector with outliers pushes most of its mass
into a few central levels while the extremes sit nearly empty — so the
*effective* resolution can be far below the nominal one.

If 5- and 6-bit concentrate mass into few levels while 4-bit spreads across
its own, that is the mechanism: effective resolution, not nominal
bit-width, would be what the model sees, and it need not be monotone.

Reports, on real cached K/V:
  * effective level count = 2^H where H is the Shannon entropy (bits) of
    the level-occupancy distribution — the number of levels actually
    carrying the mass
  * occupancy ratio = effective / nominal
  * share of mass in the single most-used level, and in the top 4

Run: python -m kvcache_harness.debug_level_occupancy
"""
from __future__ import annotations

import math
from statistics import mean

import torch
from transformers import DynamicCache

from .models import load_model_and_tokenizer
from .tasks.multi_credential import make_multi_credential_prompt
from .run_phase0r_calibration import build_turn_texts

BITS = [8, 7, 6, 5, 4, 3]
SEEDS = [3000, 3001, 3002]


@torch.no_grad()
def level_stats(X: torch.Tensor, bits: int):
    """X: (kv_heads, seq, head_dim) for one layer. Quantizes each position's
    (kv_heads, head_dim) slice exactly as engine._quant_dequant does, and
    returns the level indices actually produced."""
    levels = float(2 ** bits - 1)
    Xf = X.float()
    mn = Xf.amin(dim=(0, 2), keepdim=True)
    mx = Xf.amax(dim=(0, 2), keepdim=True)
    scale = (mx - mn) / levels
    safe = scale.abs() >= 1e-8
    q = ((Xf - mn) / torch.where(safe, scale, torch.ones_like(scale))).round().clamp(0, levels)
    return q[safe.expand_as(q)].to(torch.int64)


def entropy_bits(counts: torch.Tensor) -> float:
    total = counts.sum().item()
    if total <= 0:
        return 0.0
    p = counts.float() / total
    p = p[p > 0]
    return float(-(p * p.log2()).sum().item())


def main():
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct", device="cuda")

    caches = []
    for seed in SEEDS:
        p = make_multi_credential_prompt(seed=seed, n_credentials=6, n_distractors=20,
                                          words_per_paragraph=50, value_len=14)
        tt, _ = build_turn_texts(tokenizer, p)
        ids = tokenizer(tt[0], return_tensors="pt")["input_ids"].to("cuda")
        c = DynamicCache(config=model.config)
        with torch.no_grad():
            model(input_ids=ids, past_key_values=c, use_cache=True)
        caches.append(c)

    print("Level occupancy on real cached K/V (per-position affine min-max, "
          f"{len(SEEDS)} prompts x 28 layers)")
    print(f"{'bits':>5s} {'nominal':>8s} {'eff_K':>8s} {'eff_V':>8s} {'ratio_K':>8s} "
          f"{'ratio_V':>8s} {'top1_K':>8s} {'top4_K':>8s}")

    for bits in BITS:
        nominal = 2 ** bits
        effK, effV, t1K, t4K = [], [], [], []
        for c in caches:
            for layer in c.layers:
                for T, eff_acc, is_key in ((layer.keys, effK, True), (layer.values, effV, False)):
                    q = level_stats(T[0], bits)
                    if q.numel() == 0:
                        continue
                    counts = torch.bincount(q, minlength=nominal).cpu()
                    eff_acc.append(2 ** entropy_bits(counts))
                    if is_key:
                        srt = counts.sort(descending=True).values.float()
                        tot = srt.sum().item()
                        if tot > 0:
                            t1K.append(srt[0].item() / tot)
                            t4K.append(srt[:4].sum().item() / tot)
        ek, ev = mean(effK), mean(effV)
        print(f"{bits:5d} {nominal:8d} {ek:8.1f} {ev:8.1f} {ek/nominal:8.3f} "
              f"{ev/nominal:8.3f} {mean(t1K)*100:7.1f}% {mean(t4K)*100:7.1f}%")


if __name__ == "__main__":
    main()
