"""Minimal KV-compression engine — enough to run the Stage 3 random-span control.

Stage 3 (N1) precedes Stage 5 (the kvpress ladder build), so it cannot use the kvpress
integration: that does not exist yet. This module implements eviction directly, by the
mechanism eviction actually is:

    1. prefill the context, keeping the full KV cache
    2. slice every layer's key/value tensors down to the retained token indices
    3. append the query AFTER compression (agnostic protocol, v1 §5.2)
    4. greedy-decode

**Conventions, stated because they are choices:**

* **No key re-rotation.** Retained keys keep the RoPE phase they were given at their original
  positions; new tokens continue from the *compressed* cache length. This is the default
  convention for most kvpress presses (kvpress ships `KeyRerotationPress` precisely because
  re-rotation is the *other* option), and it is what StreamingLLM does. Stage 5 must use the
  same convention or the two are not comparable.
* **Sliced, not masked.** Evicted tokens are removed from the cache rather than masked out, so
  the retained count is exactly `B` and the memory saving is real rather than notional.
* This engine is deliberately small and is **not** the Stage 5 harness. It exists so that a
  hard gate which logically precedes the ladder can be run without pre-supposing the ladder.
"""

from __future__ import annotations

import torch


def _slice_cache(past, keep: torch.Tensor):
    """Slice every layer's K/V down to `keep` (a LongTensor of positions to retain)."""
    layers = getattr(past, "layers", None)
    if layers is not None:                       # transformers >= 5 Cache with .layers
        for lyr in layers:
            lyr.keys = lyr.keys.index_select(2, keep).contiguous()
            lyr.values = lyr.values.index_select(2, keep).contiguous()
        return past
    if hasattr(past, "key_cache"):               # older Cache API
        for i in range(len(past.key_cache)):
            past.key_cache[i] = past.key_cache[i].index_select(2, keep).contiguous()
            past.value_cache[i] = past.value_cache[i].index_select(2, keep).contiguous()
        return past
    raise TypeError(f"unsupported cache type: {type(past)}")


def _cache_len(past) -> int:
    layers = getattr(past, "layers", None)
    if layers is not None:
        return layers[0].keys.shape[2]
    return past.key_cache[0].shape[2]


@torch.inference_mode()
def run_compressed(model, tok, context: str, queries: list[str], keep_idx,
                   max_new: int = 32) -> list[str]:
    """Prefill `context`, compress the cache to `keep_idx`, then answer each query.

    The compressed cache is built ONCE and reused across all queries — that is the whole point
    of the query-agnostic protocol (compress once, query many times) and it is what makes the
    H-variant scoring affordable.
    """
    dev = model.device
    ctx = tok(context, add_special_tokens=False, return_tensors="pt").to(dev)
    n_ctx = ctx["input_ids"].shape[1]

    keep = torch.as_tensor(sorted(keep_idx), dtype=torch.long, device=dev)
    if keep.numel() and int(keep[-1]) >= n_ctx:
        raise ValueError(f"keep index {int(keep[-1])} out of range for context of {n_ctx}")

    out = model(**ctx, use_cache=True)
    base = out.past_key_values
    _slice_cache(base, keep)
    kept_len = _cache_len(base)

    answers = []
    for q in queries:
        # Deep-copy the compressed cache per query so queries do not contaminate each other.
        past = _clone_cache(base)
        q_ids = tok(q, add_special_tokens=False, return_tensors="pt").to(dev)["input_ids"]
        cur = q_ids
        toks = []
        for _ in range(max_new):
            o = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = o.past_key_values
            nxt = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            t = int(nxt)
            if t == tok.eos_token_id:
                break
            toks.append(t)
            cur = nxt
        answers.append(tok.decode(toks, skip_special_tokens=True))
    return answers


def _clone_cache(past):
    import copy
    layers = getattr(past, "layers", None)
    new = copy.copy(past)
    if layers is not None:
        new.layers = [copy.copy(l) for l in layers]
        for l in new.layers:
            l.keys = l.keys.clone()
            l.values = l.values.clone()
        return new
    new.key_cache = [k.clone() for k in past.key_cache]
    new.value_cache = [v.clone() for v in past.value_cache]
    return new


__all__ = ["run_compressed"]
