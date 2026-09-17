"""Phase 2 policy: recoverable tiering with the eviction signal and the
promotion signal decoupled.

Eviction (who leaves the retained set permanently) stays ranked by
cumulative attention in every arm — that is QEvict as published, and holding
it fixed is what makes the promotion signal the only varied factor.
FULL-vs-QUANT among the retained set is ordered by the pluggable
`PromotionSignal`.

Also records, per selection call, the Spearman rank correlation between the
eviction signal and the promotion signal over the retained candidates, so
H-ORTH's continuous form ("benefit tracks |rho| -> 0") can be tested rather
than just the P1-vs-P2 contrast.
"""

from __future__ import annotations

from .base_policy import EvictionPolicy, SelectionContext, SelectionResult, TokenStatus


def _spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None

    def ranks(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


class SplitSignalTierPolicy(EvictionPolicy):
    tiered = True

    def __init__(self, budget, promotion_signal):
        super().__init__(budget)
        self.promotion_signal = promotion_signal
        self.name = f"tier_{promotion_signal.name}"
        self.rho_samples = []   # Spearman(eviction signal, promotion signal) per call

    def select(self, ctx: SelectionContext) -> SelectionResult:
        result = SelectionResult()
        full_slots = self.budget.full_slots
        quant_slots = self.budget.quant_slots

        active = [p for p in ctx.active_positions
                  if ctx.statuses.get(p, TokenStatus.FULL) != TokenStatus.EVICTED]

        # --- eviction decision: cumulative attention, as published ---
        by_attention = sorted(active, key=lambda p: ctx.scores.get(p, 0.0), reverse=True)
        retained = by_attention[:full_slots + quant_slots]
        dropped = by_attention[full_slots + quant_slots:]

        # --- FULL vs QUANT among retained: the varied promotion signal ---
        promo = self.promotion_signal.values(retained, ctx)
        by_promo = sorted(retained, key=lambda p: promo.get(p, 0.0), reverse=True)
        new_full = set(by_promo[:full_slots])

        if len(retained) >= 3:
            rho = _spearman([ctx.scores.get(p, 0.0) for p in retained],
                            [promo.get(p, 0.0) for p in retained])
            if rho is not None:
                self.rho_samples.append(rho)

        for p in retained:
            old = ctx.statuses.get(p, TokenStatus.FULL)
            new = TokenStatus.FULL if p in new_full else TokenStatus.QUANT
            if new == old:
                continue
            result.new_statuses[p] = new
            if new > old:
                result.promotions.append(p)
            else:
                result.demotions.append(p)

        for p in dropped:
            if ctx.statuses.get(p, TokenStatus.FULL) != TokenStatus.EVICTED:
                result.new_statuses[p] = TokenStatus.EVICTED
                result.evictions.append(p)

        return result
