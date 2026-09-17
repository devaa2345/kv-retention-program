"""Stage 5 — the reference ladder as real kvpress presses (frozen PREREG_P2.md §4.1).

*** SESSION A MUST NOT READ THIS FILE (A6 firewall, repo CLAUDE.md). ***

Every ladder arm is a `kvpress.ScorerPress` subclass. kvpress keeps the top
`n_kept = int(k_len * (1 - compression_ratio))` positions by score, so an arm that needs an
EXACT keep-set expresses it as a score vector: `KEEP` for retained positions, `DROP` otherwise,
with `compression_ratio` solved so that `n_kept` equals `B` exactly. `set_budget()` does that
solve and asserts the realised count, because an off-by-one here would silently change what
every ratio in the paper is normalised against.

Arms (PREREG_P2.md §4.1):

    full_cache          no compression                          anchor
    null                keep the last B                         tripwire
    random              uniform random B from the region        tripwire
    floor_pos           sink + most recent                      THE DENOMINATOR
    oracle_causal       optimal complete-candidate packing      numerator ceiling
    oracle_prescient    the queried candidate + floor filler    absolute bound

`oracle_causal` and `oracle_prescient` are instance-specific: their keep-sets come from
`harness.ladder`, which owns decision 7's packing and its per-instance optimality assertion.
This module only *transports* that decision into kvpress; it does not re-decide it. In
particular `oracle_prescient` still RAISES on a VOID cell rather than truncating gold, because
that raise lives in `harness.ladder`.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import torch
from kvpress.presses.scorer_press import ScorerPress

from harness import ladder

KEEP, DROP = 1.0e6, -1.0e6


@dataclass
class _ExactSetPress(ScorerPress):
    """Base for arms defined by an exact set of retained positions."""

    keep: frozenset[int] = field(default_factory=frozenset)
    _n_kept: int = 0

    def set_keep(self, keep_idx, seq_len: int) -> "_ExactSetPress":
        keep = frozenset(int(i) for i in keep_idx)
        if not keep:
            raise ValueError("empty keep-set")
        if max(keep) >= seq_len:
            raise ValueError(f"keep index {max(keep)} out of range for seq_len {seq_len}")
        self.keep = keep
        self._solve_ratio(len(keep), seq_len)
        return self

    def _solve_ratio(self, n_keep: int, seq_len: int) -> None:
        """Choose compression_ratio so kvpress's floor() lands on exactly n_keep."""
        if n_keep >= seq_len:
            self.compression_ratio = 0.0
            self._n_kept = seq_len
            return
        # n_kept = int(seq_len * (1 - r)); aim at the middle of the half-open bin.
        self.compression_ratio = 1.0 - (n_keep + 0.5) / seq_len
        got = int(seq_len * (1 - self.compression_ratio))
        if got != n_keep:                      # fall back to a direct search on the boundary
            for eps in (1e-9, 1e-7, 1e-5):
                for cand in (1.0 - (n_keep + 0.5) / seq_len + eps,
                             1.0 - (n_keep + 0.5) / seq_len - eps):
                    if 0 <= cand < 1 and int(seq_len * (1 - cand)) == n_keep:
                        self.compression_ratio = cand
                        got = n_keep
                        break
                if got == n_keep:
                    break
        if got != n_keep:
            raise RuntimeError(
                f"cannot express keep={n_keep} of {seq_len} as a kvpress compression_ratio "
                f"(got {got}); the arm's realised budget would not equal B"
            )
        self._n_kept = n_keep

    def score(self, module, hidden_states, keys, values, attentions, kwargs) -> torch.Tensor:
        b, h, n, _ = keys.shape
        s = torch.full((n,), DROP, dtype=torch.float32, device=keys.device)
        idx = torch.tensor(sorted(self.keep), dtype=torch.long, device=keys.device)
        idx = idx[idx < n]
        s[idx] = KEEP
        # Head-agnostic by construction: the same set for every KV head. Head-wise allocation
        # is a separate question (decision 9(a)) and is not expressed by the ladder.
        return s.view(1, 1, n).expand(b, h, n).contiguous()


@dataclass
class NullPress(ScorerPress):
    """Pure truncation: keep the last B. Tripwire, not a competitor."""
    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        b, h, n, _ = keys.shape
        s = torch.arange(n, dtype=torch.float32, device=keys.device)
        return s.view(1, 1, n).expand(b, h, n).contiguous()


@dataclass
class RandomPress(ScorerPress):
    """Uniform random retention, CRC32-seeded (repo rule 5). Tripwire."""
    seed: int = 0

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        b, h, n, _ = keys.shape
        g = torch.Generator(device="cpu")
        g.manual_seed(zlib.crc32(f"randpress|{self.seed}|{n}".encode()) & 0xFFFFFFFF)
        s = torch.rand(n, generator=g).to(keys.device)
        return s.view(1, 1, n).expand(b, h, n).contiguous()


@dataclass
class FloorPosPress(ScorerPress):
    """THE DENOMINATOR: attention sinks plus the most recent content (StreamingLLM-shaped)."""
    n_sink: int = 8

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        b, h, n, _ = keys.shape
        s = torch.arange(n, dtype=torch.float32, device=keys.device)
        s[: min(self.n_sink, n)] = KEEP        # sinks always survive
        return s.view(1, 1, n).expand(b, h, n).contiguous()


class OracleCausalPress(_ExactSetPress):
    """Numerator ceiling. Keep-set comes from harness.ladder (decision 7's packing)."""


class OraclePrescientPress(_ExactSetPress):
    """Absolute bound. Keep-set comes from harness.ladder; raises on a VOID cell."""


@dataclass
class OracleCausalPerHeadPress(ScorerPress):
    """N9 diagnostic: the causal oracle allowed to hold DIFFERENT candidates in different heads.

    `oracle_causal` is head-agnostic -- one keep-set replicated across every KV head. This arm
    relaxes exactly that and nothing else: each head still gets the same budget C, still holds
    complete candidates only, but head h is offered the candidate list ROTATED by h, so across
    heads the union covers more of the candidate set than any single head can.

    Delta_head = A(perhead) - A(global) is therefore the value of head-wise allocation with the
    total budget and the completeness rule held fixed. Per frozen PREREG §3.9(a) it is measured
    on M3/M4 only: M1 and M2 have 2 KV heads, so there is almost no allocation to express and the
    contrast is structurally near-degenerate there.
    """
    keep_by_head: dict = field(default_factory=dict)      # head -> frozenset[int]
    floor: frozenset = field(default_factory=frozenset)

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        b, h, n, _ = keys.shape
        s = torch.full((b, h, n), DROP, dtype=torch.float32, device=keys.device)
        for hi in range(h):
            idx = sorted(self.keep_by_head.get(hi % max(1, len(self.keep_by_head)), self.floor))
            t = torch.tensor([i for i in idx if i < n], dtype=torch.long, device=keys.device)
            if t.numel():
                s[:, hi, :].index_fill_(1, t, KEEP)
        return s.contiguous()


def build_arm(arm: str, *, n_ctx: int, C: int, n_sink: int = 8, n_window: int = 64,
              facts=None, gold=None, seed: int = 0, n_heads: int = 1):
    """Instantiate one ladder arm at budget B = C + n_sink + n_window.

    Returns (press_or_None, selection_or_None). `None` press means full_cache (no compression).
    """
    B = C + n_sink + n_window
    if arm == "full_cache":
        return None, None

    if arm == "null":
        p = NullPress()
        p.compression_ratio = _ratio_for(B, n_ctx)
        return p, None
    if arm == "random":
        p = RandomPress(seed=seed)
        p.compression_ratio = _ratio_for(B, n_ctx)
        return p, None
    if arm == "floor_pos":
        p = FloorPosPress(n_sink=n_sink)
        p.compression_ratio = _ratio_for(B, n_ctx)
        return p, None

    if arm == "oracle_causal":
        sel = ladder.oracle_causal(n_ctx, C, facts, n_sink=n_sink, n_window=n_window)
        return OracleCausalPress().set_keep(sel.kept, n_ctx), sel
    if arm == "oracle_causal_perhead":
        return _build_perhead(n_ctx, C, facts, n_sink, n_window, n_heads), None

    if arm == "oracle_prescient":
        # Raises on a VOID cell (C < k_gold) rather than truncating gold -- kept deliberately.
        sel = ladder.oracle_prescient(n_ctx, C, gold, n_sink, n_window, facts)
        return OraclePrescientPress().set_keep(sel.kept, n_ctx), sel

    raise ValueError(f"unknown ladder arm: {arm}")


# When True (default) the per-head arm REFUSES to emit a cell whose heads hold different
# candidate counts, because Delta_head would then confound head allocation with a budget
# difference. Set False ONLY to measure how much that refusal biases the result -- the
# resulting cells are not admissible as Delta_head measurements.
PERHEAD_STRICT = True


def _perhead_keepsets(payable, C, n_heads):
    """Per-head complete-candidate sets: same budget C, same COUNT, different identities.

    A rotation of the *input list* does nothing here -- `oracle_causal` sorts by
    `(payable_cost, fact_id)`, so its packing is order-independent and every head would receive
    the identical set, making `Delta_head` identically zero by construction rather than by
    measurement. The rotation must therefore act on the greedy's *scan offset* within the
    cost-ascending order.

    Head h scans the cost-ascending candidate list starting at position h and wrapping, taking
    any candidate that still fits in C, until it holds `k` of them -- where `k` is the count the
    global optimal packing achieves. If a head's rotation lands it on expensive candidates and it
    cannot reach `k`, it back-fills from the cheapest unused candidates that still fit. So every
    head spends at most C, every head completes the same number of candidates, and the heads hold
    *different* candidates. That is exactly the one faculty `Delta_head` is meant to price.
    """
    order = sorted(range(len(payable)), key=lambda i: (payable[i][0], payable[i][1].fact_id))
    k = _optimal_count(payable, C)
    H = len(order)
    by_head = {}
    for hi in range(max(1, n_heads)):
        chosen: set[int] = set()
        taken: list[int] = []
        off = hi % H if H else 0
        for step in range(H):                      # rotated scan
            i = order[(off + step) % H]
            if len(taken) >= k:
                break
            toks = payable[i][2]
            if len(chosen | toks) <= C:
                chosen |= toks
                taken.append(i)
        if len(taken) < k:                         # back-fill from the cheapest unused
            for i in order:
                if len(taken) >= k:
                    break
                if i in taken:
                    continue
                toks = payable[i][2]
                if len(chosen | toks) <= C:
                    chosen |= toks
                    taken.append(i)
        by_head[hi] = (frozenset(chosen), len(taken))
    return by_head, k


def _optimal_count(payable, C):
    from harness.ladder import _optimal_pair_count
    return _optimal_pair_count([c for c, _, _ in payable], C)


def _build_perhead(n_ctx, C, facts, n_sink, n_window, n_heads):
    """One packing per head over a rotated scan offset, at equal budget and equal count."""
    floor, region = ladder._floors(n_ctx, n_sink, n_window)
    region_set = set(region)
    payable = []
    for f in facts:
        toks = {t for s in f.spans for t in s if t in region_set}
        payable.append((len(toks), f, toks))

    by_head, k = _perhead_keepsets(payable, C, n_heads)

    counts = {h: c for h, (_, c) in by_head.items()}
    if len(set(counts.values())) > 1 and PERHEAD_STRICT:
        raise AssertionError(
            f"per-head arm does not hold a constant candidate count across heads: {counts}. "
            "Delta_head would then confound head allocation with a budget difference."
        )

    keep_by_head = {}
    for hi, (chosen, _) in by_head.items():
        kept = ladder._pad_from_floor(floor | set(chosen), region, C - len(chosen))
        keep_by_head[hi] = frozenset(kept)

    if (len(set(keep_by_head.values())) == 1 and max(1, n_heads) > 1
            and len(payable) > k and PERHEAD_STRICT):
        raise AssertionError(
            "per-head arm produced an identical keep-set for every head while more candidates "
            f"than the packed {k} were available -- Delta_head would be zero by construction."
        )

    p = OracleCausalPerHeadPress()
    p.keep_by_head = keep_by_head
    p.floor = frozenset(floor)
    p.compression_ratio = _ratio_for(C + n_sink + n_window, n_ctx)
    return p


def _ratio_for(n_keep: int, seq_len: int) -> float:
    if n_keep >= seq_len:
        return 0.0
    r = 1.0 - (n_keep + 0.5) / seq_len
    if int(seq_len * (1 - r)) != n_keep:
        raise RuntimeError(f"cannot express keep={n_keep} of {seq_len}")
    return r


LADDER_ARMS = ("full_cache", "null", "random", "floor_pos",
               "oracle_causal", "oracle_prescient")
N9_ARMS = ("floor_pos", "oracle_causal", "oracle_causal_perhead", "oracle_prescient")

__all__ = ["build_arm", "LADDER_ARMS", "N9_ARMS", "OracleCausalPerHeadPress", "NullPress", "RandomPress", "FloorPosPress",
           "OracleCausalPress", "OraclePrescientPress"]
