"""Metrics for Phase 0. Retrieval accuracy is exact; Global LIR is exact
(computed from recorded promotion/demotion events); starvation-gap and FMM
are approximations of the Paper-1 / QEvict concepts, computed from a single
extra full-KV (no eviction) reference pass over prompt+generated tokens —
documented as approximations, not claimed to be bit-identical to either
paper's formal definition.
"""

from __future__ import annotations

import torch


def global_lir(events) -> float:
    """Fraction of QUANT demotions that are later reversed by a promotion
    back to FULL (recoverable-tier arms only; 0.0 for non-tiered arms)."""
    demotions = sum(len(e.demotions) for e in events)
    promotions = sum(len(e.promotions) for e in events)
    if demotions == 0:
        return 0.0
    return promotions / demotions


@torch.no_grad()
def reference_full_attention(model, tokenizer, full_token_ids: list, device: str):
    """One no-eviction, teacher-forced forward pass over the full
    prompt+generated sequence, used as ground truth for FMM /
    starvation-gap. Returns attn_by_query[i] = 1D tensor of attention mass
    (mean over layers & heads) that query position i assigns to every key
    position <= i.
    """
    ids = torch.tensor([full_token_ids], device=device)
    out = model(input_ids=ids, use_cache=False, output_attentions=True)
    # Accumulate the layer-mean incrementally instead of torch.stack-ing every
    # layer's (1, heads, L, L) tensor at once — for L in the low thousands
    # that stack is a multi-GB transient allocation we don't need.
    acc = None
    n_layers = len(out.attentions)
    for layer_attn in out.attentions:
        layer_mean = layer_attn.mean(dim=1)[0]     # (Q, K), mean over heads
        acc = layer_mean if acc is None else acc + layer_mean
    avg = acc / n_layers
    return avg.detach().to("cpu")


def future_missed_mass(eviction_step_to_positions: dict, ref_attn, prompt_len: int) -> float:
    """Approximate FMM: for every position evicted at absolute step t_evict,
    sum the reference attention mass that *later* queries (position >
    t_evict, i.e. after the eviction happened) assign back to it; divide by
    the total attention mass those same later queries assign to anything.
    Higher = more of the future's "true" attention need was thrown away.
    """
    Q, K = ref_attn.shape
    missed = 0.0
    total = 0.0
    for t_evict, positions in eviction_step_to_positions.items():
        if t_evict + 1 >= Q:
            continue
        future_block = ref_attn[t_evict + 1:, :]      # (Q', K)
        total += future_block.sum().item()
        valid_positions = [p for p in positions if p < K]
        if valid_positions:
            missed += future_block[:, valid_positions].sum().item()
    if total <= 0:
        return 0.0
    return missed / total


def cost_summary(events, baseline_step_time_s: float | None = None) -> dict:
    step_times = [e.wall_time_s for e in events if e.wall_time_s > 0]
    mean_step = sum(step_times) / len(step_times) if step_times else 0.0
    out = {
        "mean_step_time_s": mean_step,
        "n_steps": len(step_times),
        "n_evictions": sum(len(e.evictions) for e in events),
        "n_promotions": sum(len(e.promotions) for e in events),
        "n_demotions": sum(len(e.demotions) for e in events),
        "materializes_attention_matrix": True,  # every arm here uses eager attn
    }
    if baseline_step_time_s:
        out["pct_of_baseline_step"] = 100.0 * mean_step / baseline_step_time_s
    return out
