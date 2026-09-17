"""CPU tests for the unit-aware wrapper, against kvpress's own compress on mock scores."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent / "paper3"), str(HERE.parent / "paper2")]

from kvpress import AdaKVPress                          # noqa: E402
from kvpress.presses.scorer_press import ScorerPress    # noqa: E402

from harness import methods                             # noqa: E402
from p4 import unitwrap as UW                           # noqa: E402

N_SINK, N_WINDOW = 8, 64


@dataclass
class FixedScore(ScorerPress):
    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        module.calls = getattr(module, "calls", 0) + 1
        return module.scores.clone()


def mock(h, n, seed, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(1, h, n, generator=g).to(dtype)
    keys = torch.arange(n, dtype=torch.float32).view(1, 1, n, 1).expand(1, h, n, 4).contiguous()
    m = SimpleNamespace(head_dim=4, layer_idx=0, scores=s,
                        config=SimpleNamespace(_attn_implementation="sdpa"))
    return m, keys


def ratio_for(B, n):
    return 1.0 - (B + 0.5) / n


def floored(n, B):
    p = methods.make_floor_constrained(FixedScore(compression_ratio=ratio_for(B, n)), n, N_SINK, N_WINDOW)
    p.compression_ratio = ratio_for(B, n)
    return p


def kept_from_keys(k):
    return [set(k[0, h, :, 0].long().tolist()) for h in range(k.shape[1])]


def random_units(n, rng, size_lo=5, size_hi=40, gap=6):
    units, t = [], N_SINK + 3
    while t + size_hi < n - N_WINDOW + 20:
        L = int(rng.integers(size_lo, size_hi))
        units.append(list(range(t, t + L)))
        t += L + int(rng.integers(1, gap))
    return units


def test_identity_scorer():
    n, C = 700, 128
    B = C + N_SINK + N_WINDOW
    for seed in range(5):
        m, keys = mock(4, n, seed)
        ref = floored(n, B)
        kx, _ = ref.compress(m, None, keys, keys, None, {})
        ui = UW.UnitIndex(n, [], N_SINK, N_WINDOW)
        up = UW.make_unit_aware(floored(n, B), ui, C)
        up.compression_ratio = ratio_for(B, n)
        ku, _ = up.compress(m, None, keys, keys, None, {})
        assert kept_from_keys(kx) == kept_from_keys(ku)


def test_identity_adakv():
    n, C = 700, 128
    B = C + N_SINK + N_WINDOW
    for seed in range(5):
        m, keys = mock(4, n, seed)
        ref = AdaKVPress(press=floored(n, B))
        ref.compress(m, None, keys, keys, None, {})
        _, hx, sx = m.masked_key_indices
        ui = UW.UnitIndex(n, [], N_SINK, N_WINDOW)
        up = UW.make_unit_aware(AdaKVPress(press=floored(n, B)), ui, C)
        m2, _ = mock(4, n, seed)
        up.compress(m2, None, keys, keys, None, {})
        _, hu, su = m2.masked_key_indices
        a = sorted(zip(hx.tolist(), sx.tolist()))
        b = sorted(zip(hu.tolist(), su.tolist()))
        assert a == b


def test_all_or_nothing_budget_floors_scorer():
    rng = np.random.default_rng(0)
    n, C = 900, 256
    B = C + N_SINK + N_WINDOW
    for seed in range(5):
        m, keys = mock(3, n, seed)
        units = random_units(n, rng)
        ui = UW.UnitIndex(n, units, N_SINK, N_WINDOW)
        stats = {}
        up = UW.make_unit_aware(floored(n, B), ui, C, stats=stats)
        up.compression_ratio = ratio_for(B, n)
        ku, _ = up.compress(m, None, keys, keys, None, {})
        assert m.calls == 1, "score must be called exactly once"
        assert stats["fallback"] == 0
        floor = set(range(N_SINK)) | set(range(n - N_WINDOW, n))
        for kk in kept_from_keys(ku):
            assert len(kk) == B
            assert floor <= kk
            for u in ui.units:
                inside = sum(1 for t in u.tolist() if t in kk)
                assert inside in (0, len(u)), "partial unit retained"


def test_all_or_nothing_budget_floors_adakv():
    rng = np.random.default_rng(1)
    n, C, h = 900, 256, 4
    B = C + N_SINK + N_WINDOW
    for seed in range(5):
        m, keys = mock(h, n, seed)
        units = random_units(n, rng)
        ui = UW.UnitIndex(n, units, N_SINK, N_WINDOW)
        stats = {}
        up = UW.make_unit_aware(AdaKVPress(press=floored(n, B)), ui, C, stats=stats)
        up.compress(m, None, keys, keys, None, {})
        assert m.calls == 1
        assert stats["fallback"] == 0
        _, hh, ss = m.masked_key_indices
        masked = {hi: set() for hi in range(h)}
        for a, b in zip(hh.tolist(), ss.tolist()):
            masked[a].add(b)
        keeps = [set(range(n)) - masked[hi] for hi in range(h)]
        assert sum(len(k) for k in keeps) == h * B
        floor = set(range(N_SINK)) | set(range(n - N_WINDOW, n))
        n_safe_region = int(B * 0.2) - len(floor)
        for kk in keeps:
            assert floor <= kk
            assert len(kk) - len(floor) >= min(n_safe_region, 1)
            for u in ui.units:
                inside = sum(1 for t in u.tolist() if t in kk)
                assert inside in (0, len(u))


def test_fewer_touched_more_complete():
    """On fragmenting scores (iid), U completes more units and touches fewer than X."""
    rng = np.random.default_rng(2)
    n, C = 2000, 512
    units = random_units(n, rng, 35, 45, 10)
    ui = UW.UnitIndex(n, units, N_SINK, N_WINDOW)
    S = rng.standard_normal((8, n))
    keeps, _ = UW.unit_keep_per_head(S, ui, C)
    region = np.flatnonzero(~ui.floor)
    tx = cx = tu = cu = 0
    for h in range(8):
        top = set(region[np.argsort(-S[h, region], kind="stable")[:C]].tolist())
        ku = set(keeps[h].tolist())
        for u in ui.units:
            us = set(u.tolist())
            tx += bool(us & top); cx += us <= top
            tu += bool(us & ku); cu += us <= ku
    assert tu < tx and cu > cx


def test_overlap_and_fallback():
    n = 300
    # region is [8, 236); three overlapping units cover it exactly, so there are no singletons.
    ui = UW.UnitIndex(n, [list(range(8, 60)), list(range(59, 120)), list(range(119, 236))],
                      N_SINK, N_WINDOW)
    assert ui.singletons.size == 0
    s = np.linspace(1, 0, n)
    ch, st = UW.allocate(s, ui, 100)
    # unit 1 (52 tokens) fits; unit 2 needs 60 new (one shared) > 48 left; unit 3 needs 116.
    assert int(ch.sum()) == 100
    assert st["units_taken"] == 1 and st["fallback"] == 48


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print("ALL %d TESTS PASS" % len(fns))
