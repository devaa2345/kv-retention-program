"""Arm 1: H2O-style cumulative-attention eviction, permanent (no protection,
no recoverable tier). Once a position drops below the budget cutoff by
score, it is evicted and gone for good.
"""

from __future__ import annotations

from .base_policy import EvictionPolicy, SelectionContext, SelectionResult, TokenStatus


class PermanentEvictPolicy(EvictionPolicy):
    name = "permanent_evict"
    tiered = False

    def select(self, ctx: SelectionContext) -> SelectionResult:
        result = SelectionResult()
        budget = self.budget.total_budget

        # Rank currently-active (non-evicted) positions by score, descending.
        active = [p for p in ctx.active_positions if ctx.statuses.get(p, TokenStatus.FULL) != TokenStatus.EVICTED]
        active.sort(key=lambda p: ctx.scores.get(p, 0.0), reverse=True)

        keep = set(active[:budget])
        for p in active:
            if p not in keep and ctx.statuses.get(p) != TokenStatus.EVICTED:
                result.new_statuses[p] = TokenStatus.EVICTED
                result.evictions.append(p)
            elif p in keep and ctx.statuses.get(p) != TokenStatus.FULL:
                result.new_statuses[p] = TokenStatus.FULL
        return result
