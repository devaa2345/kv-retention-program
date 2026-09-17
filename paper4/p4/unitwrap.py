"""Unit-aware allocation around an existing scorer press. Paper 4, Stage 1.

The claim under test is that the methods lose to `floor_pos` at large fact cost `c` because they
FRAGMENT: they spend budget on many partial facts rather than a few whole ones. To isolate
fragmentation as the cause, U-X must differ from X in allocation ONLY:

  * the SCORE FUNCTION is untouched. The wrapper calls the wrapped press's own `score()` exactly
    once -- including the mandatory-floor pinning from `harness.methods.make_floor_constrained` --
    and never modifies the returned tensor;
  * the BUDGET is identical: exactly B = C + n_sink + n_window per (layer, KV head) for scorer
    presses, and exactly B * n_heads per layer for AdaKV, which is the invariant AdaKV itself
    satisfies;
  * the FLOORS (first n_sink, last n_window) are free and always kept, as for every arm.

What changes: X keeps the top-C region tokens by score. U-X aggregates the per-token scores over
units (mean over a unit's payable tokens, i.e. score-per-token), then fills C greedily with WHOLE
units in descending score-per-token, all-or-nothing -- a unit that does not fit in the remaining
budget is skipped, never truncated. Region tokens that belong to no unit are singleton units, so
they compete on the same score-per-token scale.

Consequence used as an exact check: if every unit is a singleton, U-X is X (same keep-set up to
exact score ties, which topk breaks arbitrarily).

Exact budget: singleton filler normally absorbs any remainder smaller than the next unit. If the
greedy ends with budget left and nothing whole fits, the remainder is filled with the
highest-scoring leftover tokens and COUNTED (`fallback`), so any departure from all-or-nothing is
reported rather than hidden.

AdaKV: its allocation has two parts, a per-head safeguard (top `alpha * n_kept` per head) and a
global layer-wide top-k. Both are allocation, so both become unit-wise: phase 1 gives each head
its safeguard budget by the same greedy, phase 2 runs the greedy over (head, unit) items across
the whole layer.
"""
from __future__ import annotations

import numpy as np
import torch
from kvpress import AdaKVPress

from harness import methods


class UnitIndex:
    """Per-instance unit structure over the compressible region.

    `units` are token-index collections in the uncompressed prefill. Tokens inside the floors are
    removed from a unit's payable set (they are kept by every arm and cost nothing), exactly as
    `harness.ladder.oracle_causal` charges payable cost. Units may overlap at a shared boundary
    token; overlap is handled by charging only not-yet-chosen tokens.
    """

    def __init__(self, n_ctx: int, units, n_sink: int, n_window: int):
        if n_ctx <= n_sink + n_window:
            raise ValueError("no compressible region")
        floor = np.zeros(n_ctx, dtype=bool)
        floor[:n_sink] = True
        floor[n_ctx - n_window:] = True
        covered = np.zeros(n_ctx, dtype=bool)
        payable = []
        for u in units:
            t = np.unique(np.asarray(list(u), dtype=np.int64))
            t = t[(t >= 0) & (t < n_ctx)]
            covered[t] = True
            t = t[~floor[t]]
            if t.size:
                payable.append(t)
        self.n_ctx = n_ctx
        self.floor = floor
        self.floor_idx = np.flatnonzero(floor)
        self.units = payable
        self.unit_first = np.array([u[0] for u in payable], dtype=np.int64)
        self.singletons = np.flatnonzero(~floor & ~covered)
        self.n_region = int((~floor).sum())


def _unit_means(s: np.ndarray, ui: UnitIndex) -> np.ndarray:
    return np.array([s[u].mean() for u in ui.units], dtype=np.float64)


def allocate(s: np.ndarray, ui: UnitIndex, budget: int, chosen: np.ndarray | None = None):
    """Greedy whole-unit allocation of `budget` region tokens for ONE score row.

    `chosen` (bool, n_ctx) is updated in place and returned; floors are never set in it.
    """
    if chosen is None:
        chosen = np.zeros(ui.n_ctx, dtype=bool)
    region = ~ui.floor
    if not np.all(np.isfinite(s[region])):
        raise ValueError("non-finite scores in the compressible region")
    nu = len(ui.units)
    means = _unit_means(s, ui)
    sing = ui.singletons
    score = np.concatenate([means, s[sing]])
    first = np.concatenate([ui.unit_first, sing])
    order = np.lexsort((first, -score))              # score desc, then position asc
    is_unit = order < nu
    unit_pos = np.flatnonzero(is_unit)

    remaining = int(budget) - int(chosen.sum())
    taken = 0
    i, N = 0, len(order)
    while i < N and remaining > 0:
        if not is_unit[i]:
            k = unit_pos[np.searchsorted(unit_pos, i)] if unit_pos.size and unit_pos[-1] > i else N
            run = sing[order[i:k] - nu]
            run = run[~chosen[run]][:remaining]
            chosen[run] = True
            remaining -= run.size
            i = k
            continue
        u = ui.units[order[i]]
        new = u[~chosen[u]]
        if 0 < new.size <= remaining:
            chosen[new] = True
            remaining -= new.size
            taken += 1
        i += 1

    fallback = 0
    if remaining > 0:
        cand = np.flatnonzero(region & ~chosen)
        top = cand[np.lexsort((cand, -s[cand]))][:remaining]
        chosen[top] = True
        fallback = int(top.size)
    return chosen, dict(units_taken=taken, fallback=fallback)


def unit_keep_per_head(S: np.ndarray, ui: UnitIndex, C: int):
    """Scorer presses: independent whole-unit allocation of C per KV head."""
    keeps, taken, fb = [], 0, 0
    for h in range(S.shape[0]):
        ch, st = allocate(S[h], ui, C)
        if int(ch.sum()) != C:
            raise AssertionError(f"head {h}: allocated {int(ch.sum())} != C={C}")
        keeps.append(np.flatnonzero(ch | ui.floor))
        taken += st["units_taken"]
        fb += st["fallback"]
    return keeps, dict(units_taken=taken, fallback=fb)


def unit_keep_adakv(S: np.ndarray, ui: UnitIndex, C: int, n_kept: int, alpha: float):
    """AdaKV: unit-wise safeguard per head, then unit-wise global allocation across the layer."""
    h, n = S.shape
    nf = len(ui.floor_idx)
    n_safe = int(n_kept * alpha)                    # AdaKVPress's own definition
    safe_region = min(C, max(0, n_safe - nf))       # floors are pinned highest, so they fill it first
    chosen = np.zeros((h, n), dtype=bool)
    taken = fb = 0
    for hi in range(h):
        _, st = allocate(S[hi], ui, safe_region, chosen[hi])
        taken += st["units_taken"]
        fb += st["fallback"]

    remaining = h * C - int(chosen.sum())
    nu, ns = len(ui.units), len(ui.singletons)
    means = np.stack([_unit_means(S[hi], ui) for hi in range(h)]) if nu else np.zeros((h, 0))
    item_score = np.concatenate([means.ravel(), S[:, ui.singletons].ravel()])
    item_head = np.concatenate([np.repeat(np.arange(h), nu), np.repeat(np.arange(h), ns)])
    item_idx = np.concatenate([np.tile(np.arange(nu), h), np.tile(np.arange(ns), h)])
    item_first = np.concatenate([np.tile(ui.unit_first, h), np.tile(ui.singletons, h)])
    item_unit = np.concatenate([np.ones(h * nu, dtype=bool), np.zeros(h * ns, dtype=bool)])
    order = np.lexsort((item_first, item_head, -item_score))
    is_unit = item_unit[order]
    unit_pos = np.flatnonzero(is_unit)

    i, N = 0, len(order)
    while i < N and remaining > 0:
        if not is_unit[i]:
            k = unit_pos[np.searchsorted(unit_pos, i)] if unit_pos.size and unit_pos[-1] > i else N
            seg = order[i:k]
            hh = item_head[seg]
            tt = ui.singletons[item_idx[seg]]
            free = ~chosen[hh, tt]
            hh, tt = hh[free][:remaining], tt[free][:remaining]
            chosen[hh, tt] = True
            remaining -= hh.size
            i = k
            continue
        j = order[i]
        hi = item_head[j]
        u = ui.units[item_idx[j]]
        new = u[~chosen[hi, u]]
        if 0 < new.size <= remaining:
            chosen[hi, new] = True
            remaining -= new.size
            taken += 1
        i += 1

    if remaining > 0:
        hh, tt = np.nonzero(~chosen & ~ui.floor[None, :])
        o = np.lexsort((tt, hh, -S[hh, tt]))[:remaining]
        chosen[hh[o], tt[o]] = True
        fb += int(o.size)
    if int(chosen.sum()) != h * C:
        raise AssertionError(f"AdaKV layer total {int(chosen.sum())} != h*C={h * C}")
    keeps = [np.flatnonzero(chosen[hi] | ui.floor) for hi in range(h)]
    return keeps, dict(units_taken=taken, fallback=fb)


def _record(capture, stats, li, keeps, st, S=None):
    if capture is not None:
        for h, k in enumerate(keeps):
            capture.per_head[(li, h)] = set(k.tolist())
            capture.n_kept[(li, h)] = int(k.size)
    if stats is not None:
        stats["units_taken"] = stats.get("units_taken", 0) + st["units_taken"]
        stats["fallback"] = stats.get("fallback", 0) + st["fallback"]
        stats["score_calls"] = stats.get("score_calls", 0) + 1
        if S is not None and stats.get("store_scores"):
            stats.setdefault("scores", {})[li] = S


def make_unit_aware(press, ui: UnitIndex, C: int, capture=None, stats: dict | None = None):
    """Return U-X: `press`'s own scores, whole-unit allocation. `press` must already be floored."""
    if isinstance(press, AdaKVPress):
        alpha = press.alpha_safeguard

        class _UAda(AdaKVPress):
            def compress(self, module, hidden_states, keys, values, attentions, kwargs):
                if self.compression_ratio == 0:
                    return keys, values
                assert module.config._attn_implementation != "eager", "eager mode not supported"
                scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)
                b, h, k_len = scores.shape
                if b != 1 or k_len != ui.n_ctx:
                    raise AssertionError(f"shape {tuple(scores.shape)} vs n_ctx {ui.n_ctx}")
                n_kept = int(k_len * (1 - self.compression_ratio))
                if n_kept - len(ui.floor_idx) != C:
                    raise AssertionError(f"n_kept {n_kept} does not match C={C} + floors")
                S = scores[0].detach().to(torch.float64).cpu().numpy()
                keeps, st = unit_keep_adakv(S, ui, C, n_kept, alpha)
                mask = np.ones((h, k_len), dtype=bool)
                for hi, k in enumerate(keeps):
                    mask[hi, k] = False
                hh, ss = np.nonzero(mask)
                dev = keys.device
                hi_t = torch.as_tensor(hh, dtype=torch.long, device=dev)
                module.masked_key_indices = (torch.zeros_like(hi_t), hi_t,
                                             torch.as_tensor(ss, dtype=torch.long, device=dev))
                _record(capture, stats, int(module.layer_idx), keeps, st, S)
                return keys, values

        return _UAda(press=press.press, alpha_safeguard=alpha)

    cls = type(press)

    class _U(cls):
        def compress(self, module, hidden_states, keys, values, attentions, kwargs):
            if self.compression_ratio == 0:
                return keys, values
            scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
            b, h, k_len = scores.shape
            if b != 1 or k_len != ui.n_ctx:
                raise AssertionError(f"shape {tuple(scores.shape)} vs n_ctx {ui.n_ctx}")
            n_kept = int(k_len * (1 - self.compression_ratio))
            if n_kept - len(ui.floor_idx) != C:
                raise AssertionError(f"n_kept {n_kept} does not match C={C} + floors")
            S = scores[0].detach().to(torch.float64).cpu().numpy()
            keeps, st = unit_keep_per_head(S, ui, C)
            idx = torch.as_tensor(np.stack(keeps), dtype=torch.long, device=keys.device)
            idx = idx.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
            keys = keys.gather(2, idx).contiguous()
            values = values.gather(2, idx).contiguous()
            _record(capture, stats, int(module.layer_idx), keeps, st, S)
            return keys, values

    return methods._clone_press(press, _U)


__all__ = ["UnitIndex", "allocate", "unit_keep_per_head", "unit_keep_adakv", "make_unit_aware"]
