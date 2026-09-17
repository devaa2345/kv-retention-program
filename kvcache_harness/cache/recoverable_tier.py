"""Arms 3 & 4 base: QEvict-style three-tier Full <-> Quantized -> Evicted.
Positions can be promoted from QUANT back to FULL if their cumulative score
rises again (that's the reversibility); only QUANT -> EVICTED is permanent.
"""

from __future__ import annotations

from .base_policy import EvictionPolicy, SelectionContext, SelectionResult, TokenStatus


class RecoverableTierPolicy(EvictionPolicy):
    name = "recoverable_tier"
    tiered = True

    def select(self, ctx: SelectionContext) -> SelectionResult:
        result = SelectionResult()
        full_slots = self.budget.full_slots
        quant_slots = self.budget.quant_slots

        active = [p for p in ctx.active_positions if ctx.statuses.get(p, TokenStatus.FULL) != TokenStatus.EVICTED]
        active.sort(key=lambda p: ctx.scores.get(p, 0.0), reverse=True)

        new_full = set(active[:full_slots])
        new_quant = set(active[full_slots:full_slots + quant_slots])
        new_evicted = set(active[full_slots + quant_slots:])

        for p in active:
            old = ctx.statuses.get(p, TokenStatus.FULL)
            if p in new_full:
                new = TokenStatus.FULL
            elif p in new_quant:
                new = TokenStatus.QUANT
            else:
                new = TokenStatus.EVICTED

            if new == old:
                continue
            result.new_statuses[p] = new
            if new > old:
                result.promotions.append(p)
            elif new == TokenStatus.EVICTED:
                result.evictions.append(p)
            else:
                result.demotions.append(p)
        return result
