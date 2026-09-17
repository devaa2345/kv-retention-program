"""Tier-A method presses, with per-head retained-set capture and score permutation.

Two traps this module exists to avoid, both of which produce FALSE NEGATIVES in the admission
gates (frozen PREREG §3.9(c): "every press check must generate tokens"):

1. **`get_seq_length()` does not shrink for head-wise presses.** `AdaKVPress` does not gather
   the cache at all — it leaves keys/values full length and records `module.masked_key_indices`,
   which kvpress's globally-patched attention function then masks. So a cache-length check would
   read AdaKV as "no compression applied" when it is compressing correctly, and a retained-set
   read from cache shape would be empty. Its retained set must come from the MASK.
2. **Prefill logits are identical at every ratio.** Compression happens in a forward hook AFTER
   the attention output is computed, so the prefill logits never move. Any check that compares
   prefill logits will see "no effect" for every method at every ratio. Only generation shows
   the difference — hence every gate here generates tokens.

Retained sets are accounted **per (layer, head)**. A global count would mislead for head-wise
presses: AdaKV and LU-KV spend different budgets on different heads by design, and a
head-agnostic union would hide exactly the behaviour they are being tested for.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import torch
from kvpress import (AdaKVPress, ExpectedAttentionPress, KeyDiffPress, SnapKVPress, TOVAPress)
from kvpress.presses.scorer_press import ScorerPress

try:
    from kvpress import LUKVPress
except Exception:                                   # pragma: no cover
    LUKVPress = None


# --------------------------------------------------------------------- capture / permute

@dataclass
class Capture:
    """Per-(layer, head) retained index sets, filled during a prefill."""
    per_head: dict = field(default_factory=dict)      # (layer_idx, head) -> set[int]
    n_kept: dict = field(default_factory=dict)        # (layer_idx, head) -> int
    seq_len: int = 0

    def reset(self):
        self.per_head.clear()
        self.n_kept.clear()
        self.seq_len = 0

    def heads(self):
        return sorted(self.per_head)


def _layer_idx(module) -> int:
    return int(getattr(module, "layer_idx", -1))


class _CaptureMixin:
    """Records what the press retained, without altering what it retains."""

    def _record_from_scores(self, module, scores: torch.Tensor):
        n = scores.shape[-1]
        n_kept = int(n * (1 - self.compression_ratio))
        idx = scores.topk(n_kept, dim=-1).indices          # (b, h, n_kept)
        li = _layer_idx(module)
        cap = self._capture
        cap.seq_len = n
        for h in range(idx.shape[1]):
            cap.per_head[(li, h)] = set(idx[0, h].tolist())
            cap.n_kept[(li, h)] = n_kept

    def _record_from_mask(self, module, k_len: int, n_heads: int):
        """AdaKV path: retained = complement of module.masked_key_indices, per head."""
        mk = getattr(module, "masked_key_indices", None)
        li = _layer_idx(module)
        cap = self._capture
        cap.seq_len = k_len
        if mk is None:
            for h in range(n_heads):
                cap.per_head[(li, h)] = set(range(k_len))
                cap.n_kept[(li, h)] = k_len
            return
        _, head_idx, seq_idx = mk
        masked: dict[int, set[int]] = {h: set() for h in range(n_heads)}
        for h, s in zip(head_idx.tolist(), seq_idx.tolist()):
            masked[h].add(s)
        for h in range(n_heads):
            keep = set(range(k_len)) - masked[h]
            cap.per_head[(li, h)] = keep
            cap.n_kept[(li, h)] = len(keep)


def make_capturing(press, capture: Capture):
    """Return a press that behaves identically but records its per-head retained sets."""
    cls = type(press)

    if isinstance(press, AdaKVPress):
        class _CapAda(cls, _CaptureMixin):
            def compress(self, module, hidden_states, keys, values, attentions, kwargs):
                out = super().compress(module, hidden_states, keys, values, attentions, kwargs)
                self._record_from_mask(module, keys.shape[2], keys.shape[1])
                return out
        new = _CapAda(press=press.press, alpha_safeguard=press.alpha_safeguard)
        new._capture = capture
        return new

    class _Cap(cls, _CaptureMixin):
        def compress(self, module, hidden_states, keys, values, attentions, kwargs):
            if self.compression_ratio > 0:
                s = self.score(module, hidden_states, keys, values, attentions, kwargs)
                self._record_from_scores(module, s)
            return super().compress(module, hidden_states, keys, values, attentions, kwargs)

    new = _clone_press(press, _Cap)
    new._capture = capture
    return new


def make_floor_constrained(press, n_ctx: int, n_sink: int = 8, n_window: int = 64):
    """Force a method press to honour the MANDATORY FLOORS, as the frozen spec requires.

    PREREG §4.1 budget accounting, uniform across every arm:

        B = C + n_sink + n_window       tokens retained, identical for all arms
        mandatory floors (all arms):    first n_sink, last n_window
        C:                              tokens each arm chooses from the compressible region

    kvpress presses rank all `n_ctx` positions freely, so left unconstrained they evict sink and
    window tokens and spend the whole of `B` on their own ranking. Measured on Qwen2.5-3B at
    C=512: SnapKV and AdaKV retained only 0.54 of the sink; KeyDiff retained 0.33 of the recency
    window and only 0.044 of it at C=64. That is not the accounting the ladder arms run under,
    and N1 measured the sink alone as worth ~0.27 accuracy — so an unconstrained method arm is
    handicapped for a reason that has nothing to do with the quality of its ranking.

    This pins the floors at +inf before the press's own topk, leaving it exactly `C` tokens to
    allocate by its own score inside the compressible region. That is what the frozen spec says
    every arm gets; enforcing it is implementing the prereg, not amending it.
    """
    if isinstance(press, AdaKVPress):
        inner = make_floor_constrained(press.press, n_ctx, n_sink, n_window)
        return AdaKVPress(press=inner, alpha_safeguard=press.alpha_safeguard)

    cls = type(press)
    KEEP = 1.0e6

    class _Floored(cls):
        def score(self, module, hidden_states, keys, values, attentions, kwargs):
            s = super().score(module, hidden_states, keys, values, attentions, kwargs)
            n = s.shape[-1]
            s = s.clone()
            s[..., :min(n_sink, n)] = KEEP
            if n_window > 0:
                s[..., max(0, n - n_window):] = KEEP
            return s

    return _clone_press(press, _Floored)


def make_permuted(press, seed: int):
    """Return a press whose score vector is randomly permuted per (batch, head).

    G2's question: is the method distinguishable from its own permutation? If permuting the
    scores leaves accuracy unchanged, the ordering carries no information and the method is
    measuring nothing.
    """
    if isinstance(press, AdaKVPress):
        inner = make_permuted(press.press, seed)
        return AdaKVPress(press=inner, alpha_safeguard=press.alpha_safeguard)

    cls = type(press)

    class _Perm(cls):
        def score(self, module, hidden_states, keys, values, attentions, kwargs):
            s = super().score(module, hidden_states, keys, values, attentions, kwargs)
            b, h, n = s.shape
            g = torch.Generator(device="cpu")
            g.manual_seed(zlib.crc32(f"perm|{seed}|{_layer_idx(module)}|{n}".encode())
                          & 0xFFFFFFFF)
            out = s.clone()
            for bi in range(b):
                for hi in range(h):
                    perm = torch.randperm(n, generator=g).to(s.device)
                    out[bi, hi] = s[bi, hi][perm]
            return out

    return _clone_press(press, _Perm)


def _clone_press(press, new_cls):
    """Rebuild a press under a new subclass, carrying its dataclass fields across.

    Only `init=True` fields are passed: several presses carry internal state as non-init
    dataclass fields (LU-KV's `_budget_curves`, for one), and forwarding those to __init__
    raises. Non-init state is copied across afterwards so the clone is behaviourally identical.
    """
    import dataclasses
    fields = dataclasses.fields(press)
    kw = {f.name: getattr(press, f.name) for f in fields if f.init}
    new = new_cls(**kw)
    for f in fields:
        if not f.init and hasattr(press, f.name):
            object.__setattr__(new, f.name, getattr(press, f.name))
    return new


# --------------------------------------------------------------------- the roster

def build_method(name: str, compression_ratio: float):
    """Tier-A roster, frozen PREREG §4.2. H2O/ObservedAttention is struck (decision 9(d))."""
    if name == "snapkv":
        return SnapKVPress(compression_ratio=compression_ratio)
    if name == "tova":
        return TOVAPress(compression_ratio=compression_ratio)
    if name == "expected_attn":
        return ExpectedAttentionPress(compression_ratio=compression_ratio)
    if name == "keydiff":
        return KeyDiffPress(compression_ratio=compression_ratio)
    if name == "adakv_snapkv":
        return AdaKVPress(press=SnapKVPress(compression_ratio=compression_ratio))
    if name == "lukv":
        if LUKVPress is None:
            raise RuntimeError("LUKVPress not available in this kvpress build")
        return LUKVPress(compression_ratio=compression_ratio)
    raise ValueError(f"unknown method: {name}")


TIER_A = ("snapkv", "tova", "expected_attn", "keydiff", "adakv_snapkv", "lukv")

__all__ = ["build_method", "TIER_A", "Capture", "make_capturing", "make_permuted",
           "make_floor_constrained"]
