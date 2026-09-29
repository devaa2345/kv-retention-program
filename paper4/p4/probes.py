"""Tier-1 probe arms (CANDIDATES.md A1-A6). Separate from `unitwrap.py`, which Stage 3 uses and
which is deliberately not modified here.

Every probe keeps the invariants the frozen prereg asserts: exactly B retained per slot (per-layer
B*heads for AdaKV), mandatory floors retained, the wrapped press's own `score()` called once and
unmodified.
"""
from __future__ import annotations

import re

import numpy as np
import torch

import bisect

from harness import methods
from kvpress import AdaKVPress
from p4 import units as UN
from p4.unitwrap import UnitIndex, _record, _unit_means, allocate

EARLY_LAMBDA = 0.25        # A2: prior strength, in units of the SD of unit mean scores
HYBRID_FRAC = 0.5          # A3: fraction of C reserved for the contiguous floor tail


# ----------------------------------------------------------------- unit index variants (A4)

def _line_spans(inst, task):
    at, out = 0, []
    for line in inst.context.split("\n"):
        out.append((line, at, at + len(line)))
        at += len(line) + 1
    return out


def supra_units(inst, task, tok, pre):
    """A record line merged with the line that follows it (record + its adjacent filler)."""
    spans, prev_rec = [], None
    for line, a, b in _line_spans(inst, task):
        is_rec = bool(UN.REC.match(line)) if task == "ledger_c" else line.startswith("- ")
        if is_rec:
            spans.append([a, b])
            prev_rec = len(spans) - 1
        elif prev_rec is not None and line.strip():
            spans[prev_rec][1] = b
            prev_rec = None
    return _to_units(inst, tok, pre, [(a, b) for a, b in spans])


def sub_units(inst, task, tok, pre):
    """One unit per field: the record line split on '|' (LEDGER-C), else the entry word."""
    out = []
    for line, a, b in _line_spans(inst, task):
        if task == "ledger_c" and UN.REC.match(line):
            off = a
            for part in line.split("|"):
                s = off + (len(part) - len(part.lstrip()))
                e = off + len(part.rstrip())
                if e > s:
                    out.append((s, e))
                off += len(part) + 1
        elif task == "mark1" and line.startswith("- ") and len(line) > 2:
            out.append((a + 2, b))
    return _to_units(inst, tok, pre, out)


def _to_units(inst, tok, pre, char_spans):
    enc = tok(pre, add_special_tokens=False, return_offsets_mapping=True)
    off = enc["offset_mapping"]
    base = pre.index(inst.context)
    units = []
    for a, b in char_spans:
        a, b = a + base, b + base
        units.append([ti for ti, (x, y) in enumerate(off) if y > x and x < b and y > a])
    return units, len(off)


def unit_index(inst, task, tok, pre, n_sink, n_window, granularity="line"):
    if granularity == "line":
        return UN.oracle_unit_index(inst, task, tok, pre, n_sink, n_window)
    units, n_ctx = (supra_units if granularity == "supra" else sub_units)(inst, task, tok, pre)
    return UnitIndex(n_ctx, units, n_sink, n_window)


# ----------------------------------------------------------------- allocation variants

def _unit_means(s, ui):
    return np.array([s[u].mean() for u in ui.units], dtype=np.float64)


def allocate_with_prior(s, ui, budget, lam, chosen=None):
    """A2: same greedy, but unit rank gets a prior favouring EARLY units.

    Implemented by scoring units on `mean + lam * sd(means) * (1 - first_token / n_ctx)` while
    leaving singleton scores untouched. The press's own score() is not modified.
    """
    if not ui.units:
        return allocate(s, ui, budget, chosen)
    means = _unit_means(s, ui)
    sd = means.std() or 1.0
    bonus = lam * sd * (1.0 - ui.unit_first / ui.n_ctx)
    s2 = np.array(s, dtype=np.float64, copy=True)
    # push the prior into the token scores of each unit so `allocate` ranks with it
    for j, u in enumerate(ui.units):
        s2[u] = s2[u] + bonus[j]
    return allocate(s2, ui, budget, chosen)


def keep_hybrid(S, ui, C, frac=HYBRID_FRAC):
    """A3: reserve `frac*C` for the contiguous most-recent region tokens, unit-pack the rest."""
    n_tail = int(round(frac * C))
    region = np.flatnonzero(~ui.floor)
    tail = region[-n_tail:] if n_tail else np.array([], dtype=np.int64)
    keeps, taken, fb = [], 0, 0
    for h in range(S.shape[0]):
        chosen = np.zeros(ui.n_ctx, bool)
        chosen[tail] = True
        ch, st = allocate(S[h], ui, C, chosen)
        if int(ch.sum()) != C:
            raise AssertionError(f"hybrid head {h}: {int(ch.sum())} != C={C}")
        keeps.append(np.flatnonzero(ch | ui.floor))
        taken += st["units_taken"]
        fb += st["fallback"]
    return keeps, dict(units_taken=taken, fallback=fb)


MMR_LAMBDA = 0.5   # Lever 2: one fixed value, a neutral default (equal relevance/diversity
                    # weight, same convention as HYBRID_FRAC=0.5) -- not tuned, per instruction
                    # only one lambda is tested before deciding whether to size a sweep.


def keep_mmr(S, ui, C, lam=MMR_LAMBDA):
    """Lever 2: MMR-style redundancy-penalized greedy selection, SCORE-SPACE similarity (not
    field-content similarity -- LEVER2_DESIGN.md's evidence-backed choice). similarity(u,v) =
    1 - |norm_score(u) - norm_score(v)|, unit mean scores min-max normalized per instance.
    Greedy: repeatedly pick argmax(norm_score(u) - lam * max_{v selected} sim(u,v)) among
    not-yet-selected units, filling remaining budget with singletons by raw score exactly as
    `allocate()` does elsewhere (unchanged fallback discipline)."""
    keeps, taken, fb = [], 0, 0
    n_units = len(ui.units)
    for h in range(S.shape[0]):
        chosen = np.zeros(ui.n_ctx, dtype=bool)
        remaining = C
        if n_units:
            means = _unit_means(S[h], ui)
            lo, hi = means.min(), means.max()
            norm = (means - lo) / (hi - lo + 1e-12)
            cand = list(range(n_units))
            selected_norm = []
            while cand and remaining > 0:
                if selected_norm:
                    sn = np.array(selected_norm)
                    penalty = np.array([np.max(1 - np.abs(norm[u] - sn)) for u in cand])
                else:
                    penalty = np.zeros(len(cand))
                mmr_score = norm[cand] - lam * penalty
                best_i = int(np.argmax(mmr_score))
                u_idx = cand[best_i]
                u = ui.units[u_idx]
                new = u[~chosen[u]]
                if 0 < new.size <= remaining:
                    chosen[new] = True
                    remaining -= new.size
                    selected_norm.append(norm[u_idx])
                    taken += 1
                cand.pop(best_i)
        if remaining > 0:
            region = ~ui.floor
            avail = np.flatnonzero(region & ~chosen)
            top = avail[np.argsort(-S[h][avail], kind="stable")][:remaining]
            chosen[top] = True
            fb += int(top.size)
        keeps.append(np.flatnonzero(chosen | ui.floor))
    return keeps, dict(units_taken=taken, fallback=fb)


def ufloor_keep(ui, C):
    """Item 3 (U_FLOOR_PREREG.md): whole records only, recency order, no scorer, no query.
    Deterministic given the document layout -- identical across every layer and head. Any
    budget left after whole-record packing (a record didn't fit the remaining gap) is filled
    with the most-recent remaining non-floor tokens, same recency-only spirit, no scorer --
    needed only to hit the exact B budget every arm in this project asserts."""
    order = np.argsort(-ui.unit_first)
    chosen = np.zeros(ui.n_ctx, dtype=bool)
    remaining = C
    for idx in order:
        u = ui.units[idx]
        if 0 < u.size <= remaining:
            chosen[u] = True
            remaining -= u.size
    if remaining > 0:
        region = ~np.isin(np.arange(ui.n_ctx), ui.floor_idx)
        avail = np.flatnonzero(region & ~chosen)
        top = avail[np.argsort(-avail)][:remaining]   # most recent remaining positions
        chosen[top] = True
    chosen[ui.floor_idx] = True
    return chosen


def make_ufloor_press(press, ui, C, capture=None, stats=None):
    """press for `U-floor`. `press` supplies compress()'s signature/gather mechanics only --
    its own score() is never called; the keep-set is pure structure, computed once and reused
    across every layer/head."""
    cls = type(press)
    kept_once = {}

    class _UF(cls):
        def compress(self, module, hidden_states, keys, values, attentions, kwargs):
            if self.compression_ratio == 0:
                return keys, values
            if "keep" not in kept_once:
                kept_once["keep"] = np.flatnonzero(ufloor_keep(ui, C))
            keep = kept_once["keep"]
            n_kept = int(keys.shape[2] * (1 - self.compression_ratio))
            if n_kept - len(ui.floor_idx) != C:
                raise AssertionError(f"n_kept {n_kept} != C={C} + floors")
            h = keys.shape[1]
            idx = torch.as_tensor(keep, dtype=torch.long, device=keys.device)
            idx = idx.view(1, 1, -1, 1).expand(1, h, -1, module.head_dim)
            keys = keys.gather(2, idx).contiguous()
            values = values.gather(2, idx).contiguous()
            if capture is not None:
                for hh in range(h):
                    capture.per_head[(int(module.layer_idx), hh)] = set(keep.tolist())
                    capture.n_kept[(int(module.layer_idx), hh)] = len(keep)
            return keys, values

    return methods._clone_press(press, _UF)


def keep_consensus(S, ui, C):
    """A1: one unit set from head-averaged scores, replicated to every head."""
    mean_scores = S.mean(axis=0)
    ch, st = allocate(mean_scores, ui, C)
    if int(ch.sum()) != C:
        raise AssertionError(f"consensus: {int(ch.sum())} != C={C}")
    keep = np.flatnonzero(ch | ui.floor)
    return [keep for _ in range(S.shape[0])], dict(units_taken=st["units_taken"] * S.shape[0],
                                                   fallback=st["fallback"] * S.shape[0])


def keep_early(S, ui, C, lam=EARLY_LAMBDA):
    keeps, taken, fb = [], 0, 0
    for h in range(S.shape[0]):
        ch, st = allocate_with_prior(S[h], ui, C, lam)
        if int(ch.sum()) != C:
            raise AssertionError(f"early head {h}: {int(ch.sum())} != C={C}")
        keeps.append(np.flatnonzero(ch | ui.floor))
        taken += st["units_taken"]
        fb += st["fallback"]
    return keeps, dict(units_taken=taken, fallback=fb)


# ----------------------------------------------------------------- press factories

def make_probe_press(press, ui, C, mode, capture=None, stats=None, frac=None):
    """mode in {'consensus', 'hybrid', 'early'}; `press` must already be floor-constrained.
    `frac` overrides HYBRID_FRAC for mode='hybrid' (E2: ρ in {0.10, 0.25, 0.50}); ignored otherwise.
    """
    cls = type(press)
    if mode == "hybrid" and frac is not None:
        import functools
        fn = functools.partial(keep_hybrid, frac=frac)
    else:
        fn = {"consensus": keep_consensus, "hybrid": keep_hybrid, "early": keep_early,
              "mmr": keep_mmr}[mode]

    class _P(cls):
        def compress(self, module, hidden_states, keys, values, attentions, kwargs):
            if self.compression_ratio == 0:
                return keys, values
            scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
            b, h, k_len = scores.shape
            if b != 1 or k_len != ui.n_ctx:
                raise AssertionError(f"shape {tuple(scores.shape)} vs n_ctx {ui.n_ctx}")
            n_kept = int(k_len * (1 - self.compression_ratio))
            if n_kept - len(ui.floor_idx) != C:
                raise AssertionError(f"n_kept {n_kept} != C={C} + floors")
            S = scores[0].detach().to(torch.float64).cpu().numpy()
            keeps, st = fn(S, ui, C)
            keeps = [k[np.argsort(-S[h][k], kind="stable")] for h, k in enumerate(keeps)]  # A4
            idx = torch.as_tensor(np.stack(keeps), dtype=torch.long, device=keys.device)
            idx = idx.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
            keys = keys.gather(2, idx).contiguous()
            values = values.gather(2, idx).contiguous()
            _record(capture, stats, int(module.layer_idx), keeps, st, S)
            return keys, values

    return methods._clone_press(press, _P)


def allocate_spaced(s, ui, budget, K, chosen=None):
    """PREREG_SPACING.md (31f318d9...), K=24. A candidate unit is skipped if it would sit within
    K non-unit tokens -- in ASCENDING POSITION order among the retained set -- of the nearest
    already-chosen unit. Ambiguity resolution, stated in DIAGNOSIS/report: "final kept-sequence
    order (not source order)" means position order restricted to the RETAINED subset (what
    Track 2 actually measured for adjacency), not the score-order cache WRITE sequence introduced
    by A4 -- write order is mechanistically irrelevant to attention (RoPE encodes true position).

    Two passes, per the prereg's relaxation clause:
      1. STRICT: standard greedy (as `allocate`), but a unit is accepted only if both neighbouring
         already-chosen UNITS (not singletons) are >= K tokens away in position order, or no
         neighbour exists on that side yet. Skipped units are logged with their violation
         (K - actual gap).
      2. RELAX: if this head's completion is needed to bring the INSTANCE's units_complete into
         the matched band, skipped units are re-admitted in ascending violation order until the
         instance-level target is met (or every skip is re-admitted, reproducing unconstrained
         allocation exactly -- worst case, zero net spacing effect, logged as such).

    This function performs pass 1 only (returns the skip log with violations); pass 2 (relaxation
    across heads to hit the instance-level completion band) is orchestrated in
    `spaced_keep_with_relaxation`, because it needs every head's result before deciding whether
    and how much to relax.
    """
    if chosen is None:
        chosen = np.zeros(ui.n_ctx, dtype=bool)
    region = ~ui.floor
    nu = len(ui.units)
    means = _unit_means(s, ui)
    sing = ui.singletons
    score = np.concatenate([means, s[sing]])
    first = np.concatenate([ui.unit_first, sing])
    order = np.lexsort((first, -score))
    is_unit = order < nu
    unit_pos = np.flatnonzero(is_unit)

    chosen_unit_spans = []      # sorted list of (start, end) of ACCEPTED units, position order
    skipped = []                 # (unit_idx, violation) for units rejected by spacing

    def gap_ok(u_start, u_end):
        import bisect
        starts = [s0 for s0, _ in chosen_unit_spans]
        idx = bisect.bisect_left(starts, u_start)
        viol = 0
        if idx > 0:
            _, prev_end = chosen_unit_spans[idx - 1]
            gap = u_start - prev_end - 1
            if gap < K:
                viol = max(viol, K - gap)
        if idx < len(chosen_unit_spans):
            next_start, _ = chosen_unit_spans[idx]
            gap = next_start - u_end - 1
            if gap < K:
                viol = max(viol, K - gap)
        return viol

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
        ui_idx = order[i]
        u = ui.units[ui_idx]
        new = u[~chosen[u]]
        if 0 < new.size <= remaining:
            u_start, u_end = int(u.min()), int(u.max())
            viol = gap_ok(u_start, u_end)
            if viol == 0:
                chosen[new] = True
                remaining -= new.size
                taken += 1
                import bisect
                bisect.insort(chosen_unit_spans, (u_start, u_end))
            else:
                skipped.append((ui_idx, viol, u, new.size))
        i += 1

    unit_token_mask = np.zeros(ui.n_ctx, dtype=bool)     # tokens belonging to ACCEPTED units --
    for start, end in chosen_unit_spans:                  # protected from relaxation eviction
        unit_token_mask[start:end + 1] = True
    fallback = 0
    if remaining > 0:
        cand = np.flatnonzero(region & ~chosen)
        top = cand[np.lexsort((cand, -s[cand]))][:remaining]
        chosen[top] = True
        fallback = int(top.size)
    return chosen, dict(units_taken=taken, fallback=fallback), skipped, remaining, unit_token_mask


def spaced_keep_with_relaxation(S, ui, C, K, line_units, target_units_complete):
    """Instance-level orchestration: strict spacing per head, then relax the least-violating
    skips (pooled across heads) until this instance's mean units_complete (the SAME quantity
    `p4.common.keep_metrics` reports -- count of `line_units` fully retained, mean over slots,
    NOT restricted to queried facts) re-enters [target-1, target+1]. Returns (keeps, stats)
    where stats logs relaxations performed. `line_units`: dict unit_id -> token-index set, as
    returned by `stage4_run.all_unit_tokens` / carried in `p4.common.build`'s `line_units`.
    """
    n_heads = S.shape[0]
    keeps, skip_logs, protected, taken_totals, fb_totals = [], [], [], 0, 0
    for h in range(n_heads):
        ch, st, skipped, remaining, prot = allocate_spaced(S[h], ui, C, K)
        keeps.append(ch)
        skip_logs.append(skipped)
        protected.append(prot)
        taken_totals += st["units_taken"]
        fb_totals += st["fallback"]

    def units_complete_now():
        return np.mean([sum(1 for s in line_units.values() if s and s <= set(np.flatnonzero(keeps[h] | ui.floor)))
                        for h in range(n_heads)])

    n_relax = 0
    violations_relaxed = []
    if target_units_complete is not None:
        pool = sorted(((h, ui_idx, viol, u, size) for h, skipped in enumerate(skip_logs)
                      for ui_idx, viol, u, size in skipped), key=lambda t: t[2])
        pi = 0
        while pi < len(pool) and units_complete_now() < target_units_complete - 1.0:
            h, ui_idx, viol, u, size = pool[pi]
            new_toks = u[~keeps[h][u]]
            need = int(new_toks.size)
            budget_left = int(C - keeps[h].sum())
            if need > budget_left:
                # evict lowest-scoring EVICTABLE (non-floor, non-accepted-unit) kept tokens --
                # never evict floor or an already-accepted whole unit, per the prereg's intent
                evictable = np.flatnonzero(keeps[h] & ~ui.floor & ~protected[h])
                to_free = need - budget_left
                if evictable.size >= to_free:
                    victims = evictable[np.argsort(S[h][evictable])][:to_free]
                    keeps[h][victims] = False
                    budget_left += to_free
                else:
                    pi += 1
                    continue          # cannot free enough room; try the next candidate instead
            if need <= budget_left:
                keeps[h][new_toks] = True
                protected[h][u] = True         # this unit is now accepted; protect it too
                n_relax += 1
                violations_relaxed.append(int(viol))
            pi += 1

    kept_final = [np.flatnonzero(keeps[h] | ui.floor) for h in range(n_heads)]
    return kept_final, dict(units_taken=taken_totals, fallback=fb_totals,
                            n_relaxations=n_relax, violations_relaxed=violations_relaxed,
                            fully_relaxed=(target_units_complete is not None and
                                          n_relax == sum(len(s) for s in skip_logs) and n_relax > 0))


def make_spaced_press(press, ui, C, K, line_units, target_box, capture=None, stats=None):
    """press for `spaced-U-snapkv`. `line_units` is the same dict `p4.common.build` returns;
    `target_box` is a 1-element dict {"target": <units_complete count from the stored
    unconstrained U-snapkv row for this instance, or None>}, read at compress-time so the same
    press object can be reused across an instance's forward pass without rebuilding it."""
    cls = type(press)

    class _S(cls):
        def compress(self, module, hidden_states, keys, values, attentions, kwargs):
            if self.compression_ratio == 0:
                return keys, values
            scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
            b, h, k_len = scores.shape
            if b != 1 or k_len != ui.n_ctx:
                raise AssertionError(f"shape {tuple(scores.shape)} vs n_ctx {ui.n_ctx}")
            n_kept = int(k_len * (1 - self.compression_ratio))
            if n_kept - len(ui.floor_idx) != C:
                raise AssertionError(f"n_kept {n_kept} != C={C} + floors")
            S = scores[0].detach().to(torch.float64).cpu().numpy()
            keeps, st = spaced_keep_with_relaxation(S, ui, C, K, line_units,
                                                    target_box.get("target"))
            keeps = [k[np.argsort(-S[h][k], kind="stable")] for h, k in enumerate(keeps)]  # A4
            idx = torch.as_tensor(np.stack(keeps), dtype=torch.long, device=keys.device)
            idx = idx.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
            keys = keys.gather(2, idx).contiguous()
            values = values.gather(2, idx).contiguous()
            _record(capture, stats, int(module.layer_idx), keeps, st)
            if stats is not None:
                stats["n_relaxations"] = stats.get("n_relaxations", 0) + st["n_relaxations"]
                stats.setdefault("violations_relaxed", []).extend(st["violations_relaxed"])
                stats["layers_fully_relaxed"] = stats.get("layers_fully_relaxed", 0) + int(st["fully_relaxed"])
            return keys, values

    return methods._clone_press(press, _S)


def make_ascending_gather(press, capture=None, stats=None):
    """A6: identical keep-set to the base press, gathered in ascending position order."""
    cls = type(press)

    class _A(cls):
        def compress(self, module, hidden_states, keys, values, attentions, kwargs):
            if self.compression_ratio == 0:
                return keys, values
            scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
            k_len = keys.shape[2]
            n_kept = int(k_len * (1 - self.compression_ratio))
            idx = scores.topk(n_kept, dim=-1).indices.sort(dim=-1).values      # ASCENDING
            if capture is not None or stats is not None:
                keeps = [np.sort(idx[0, hh].detach().cpu().numpy()) for hh in range(idx.shape[1])]
                _record(capture, stats, int(module.layer_idx), keeps,
                        dict(units_taken=0, fallback=0))
            g = idx.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
            return keys.gather(2, g).contiguous(), values.gather(2, g).contiguous()

    return methods._clone_press(press, _A)


# ----------------------------------------------------------------- AdaKV spacing (Amendment S2)

def _gap_violation(chosen_unit_spans, u_start, u_end, K):
    """K - actual gap to the nearest already-chosen unit span in this head, or 0 if OK."""
    starts = [s0 for s0, _ in chosen_unit_spans]
    idx = bisect.bisect_left(starts, u_start)
    viol = 0
    if idx > 0:
        _, prev_end = chosen_unit_spans[idx - 1]
        gap = u_start - prev_end - 1
        if gap < K:
            viol = max(viol, K - gap)
    if idx < len(chosen_unit_spans):
        next_start, _ = chosen_unit_spans[idx]
        gap = next_start - u_end - 1
        if gap < K:
            viol = max(viol, K - gap)
    return viol


def allocate_spaced_adakv(S, ui, C, K, alpha, n_kept, line_units, target_units_complete):
    """Cross-head-aware spacing allocator for AdaKV -- NEW SCOPE per Amendment S2, not a swap
    onto the per-head `allocate_spaced`. Mirrors `p4.unitwrap.unit_keep_adakv`'s own two-phase
    mechanism exactly (per-head safeguard, then a single global cross-head top-k over the whole
    layer budget h*C), with a spacing constraint added to BOTH phases and a single relaxation
    pass afterward that pools skips from both phases and re-admits the least-violating ones,
    evicting the lowest-scoring non-floor/non-unit token in that SAME head to make room where
    the layer is already at its budget -- extended from `spaced_keep_with_relaxation`'s per-head
    eviction discipline to AdaKV's shared h*C budget.
    """
    h, n = S.shape
    nf = len(ui.floor_idx)
    n_safe = int(n_kept * alpha)
    safe_region = min(C, max(0, n_safe - nf))
    chosen = np.zeros((h, n), dtype=bool)
    unit_token_mask = np.zeros((h, n), dtype=bool)          # protected: belongs to an accepted unit
    chosen_unit_spans = [[] for _ in range(h)]              # per head, sorted (start, end)
    skip_pool = []                                          # (head, ui_idx, viol, tokens, size)
    taken = 0

    def _accept(hi, ui_idx, u_start, u_end, toks):
        nonlocal taken
        new = toks[~chosen[hi, toks]]
        chosen[hi, new] = True
        unit_token_mask[hi, new] = True
        bisect.insort(chosen_unit_spans[hi], (u_start, u_end))
        taken += 1
        return int(new.size)

    # ---- phase 1: per-head safeguard, spacing-strict
    for hi in range(h):
        means = _unit_means(S[hi], ui)
        sing = ui.singletons
        score = np.concatenate([means, S[hi][sing]])
        first = np.concatenate([ui.unit_first, sing])
        nu = len(ui.units)
        order = np.lexsort((first, -score))
        is_unit = order < nu
        unit_pos = np.flatnonzero(is_unit)
        remaining = safe_region
        i, N = 0, len(order)
        while i < N and remaining > 0:
            if not is_unit[i]:
                k = unit_pos[np.searchsorted(unit_pos, i)] if unit_pos.size and unit_pos[-1] > i else N
                run = sing[order[i:k] - nu]
                run = run[~chosen[hi, run]][:remaining]
                chosen[hi, run] = True
                remaining -= run.size
                i = k
                continue
            ui_idx = order[i]
            u = ui.units[ui_idx]
            new = u[~chosen[hi, u]]
            if 0 < new.size <= remaining:
                u_start, u_end = int(u.min()), int(u.max())
                viol = _gap_violation(chosen_unit_spans[hi], u_start, u_end, K)
                if viol == 0:
                    remaining -= _accept(hi, ui_idx, u_start, u_end, u)
                else:
                    skip_pool.append((hi, ui_idx, viol, u, new.size))
            i += 1

    # ---- phase 2: global cross-head top-k, spacing-strict (mirrors unit_keep_adakv exactly,
    # with a spacing check added when a unit is offered to a specific head)
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
        hi = int(item_head[j])
        ui_idx = int(item_idx[j])
        u = ui.units[ui_idx]
        new = u[~chosen[hi, u]]
        if 0 < new.size <= remaining:
            u_start, u_end = int(u.min()), int(u.max())
            viol = _gap_violation(chosen_unit_spans[hi], u_start, u_end, K)
            if viol == 0:
                remaining -= _accept(hi, ui_idx, u_start, u_end, u)
            else:
                skip_pool.append((hi, ui_idx, viol, u, new.size))
        i += 1

    fb = 0
    if remaining > 0:
        hh, tt = np.nonzero(~chosen & ~ui.floor[None, :])
        o = np.lexsort((tt, hh, -S[hh, tt]))[:remaining]
        chosen[hh[o], tt[o]] = True
        fb += int(o.size)
    if int(chosen.sum()) != h * C:
        raise AssertionError(f"AdaKV spaced layer total {int(chosen.sum())} != h*C={h * C}")

    # ---- relaxation: pool skips from BOTH phases, re-admit least-violating first, evicting
    # non-floor/non-unit fill in that SAME head if the layer has no free budget left
    n_relax, violations_relaxed = 0, []
    if target_units_complete is not None:
        skip_pool.sort(key=lambda t: t[2])

        def units_complete_now():
            kept_sets = [set(np.flatnonzero(chosen[hi] | ui.floor)) for hi in range(h)]
            return np.mean([sum(1 for v in line_units.values() if v and v <= ks) for ks in kept_sets])

        pi = 0
        while pi < len(skip_pool) and units_complete_now() < target_units_complete - 1.0:
            hi, ui_idx, viol, u, size = skip_pool[pi]
            new_toks = u[~chosen[hi, u]]
            need = int(new_toks.size)
            if int(chosen.sum()) + need > h * C:
                evictable = np.flatnonzero(chosen[hi] & ~ui.floor & ~unit_token_mask[hi])
                to_free = int(chosen.sum()) + need - h * C
                if evictable.size >= to_free:
                    victims = evictable[np.argsort(S[hi][evictable])][:to_free]
                    chosen[hi, victims] = False
                else:
                    pi += 1
                    continue
            u_start, u_end = int(u.min()), int(u.max())
            chosen[hi, new_toks] = True
            unit_token_mask[hi, new_toks] = True
            bisect.insort(chosen_unit_spans[hi], (u_start, u_end))
            n_relax += 1
            violations_relaxed.append(int(viol))
            pi += 1
        if int(chosen.sum()) != h * C:
            raise AssertionError(f"AdaKV spaced relaxation broke layer total: "
                                 f"{int(chosen.sum())} != h*C={h * C}")

    keeps = [np.flatnonzero(chosen[hi] | ui.floor) for hi in range(h)]
    return keeps, dict(units_taken=taken, fallback=fb, n_relaxations=n_relax,
                       violations_relaxed=violations_relaxed,
                       fully_relaxed=(target_units_complete is not None
                                     and n_relax == len(skip_pool) and n_relax > 0))


def make_spaced_adakv_press(press, ui, C, K, line_units, target_box, capture=None, stats=None):
    """press for `spaced-U-adakv_snapkv`. `press` is the AdaKVPress whose inner press is already
    floor-constrained by the caller, as usual."""
    assert isinstance(press, AdaKVPress)
    alpha = press.alpha_safeguard

    class _SAda(AdaKVPress):
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
                raise AssertionError(f"n_kept {n_kept} != C={C} + floors")
            S = scores[0].detach().to(torch.float64).cpu().numpy()
            keeps, st = allocate_spaced_adakv(S, ui, C, K, alpha, n_kept, line_units,
                                             target_box.get("target"))
            mask = np.ones((h, k_len), dtype=bool)
            for hi, k in enumerate(keeps):
                mask[hi, k] = False
            hh, ss = np.nonzero(mask)
            dev = keys.device
            hi_t = torch.as_tensor(hh, dtype=torch.long, device=dev)
            module.masked_key_indices = (torch.zeros_like(hi_t), hi_t,
                                         torch.as_tensor(ss, dtype=torch.long, device=dev))
            _record(capture, stats, int(module.layer_idx), keeps, st)
            if stats is not None:
                stats["n_relaxations"] = stats.get("n_relaxations", 0) + st["n_relaxations"]
                stats.setdefault("violations_relaxed", []).extend(st["violations_relaxed"])
                stats["layers_fully_relaxed"] = stats.get("layers_fully_relaxed", 0) + int(st["fully_relaxed"])
            return keys, values

    return _SAda(press=press.press, alpha_safeguard=alpha)
