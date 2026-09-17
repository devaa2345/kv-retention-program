"""Arm 5: oracle_static. Perfect knowledge of which positions matter
(ctx.oracle_important, from the task's ground truth), encoded once at the
first selection call and never revisited afterward — this is the "perfect
information, applied once" control the scoping doc calls for.
"""

from __future__ import annotations

from .base_policy import EvictionPolicy, SelectionContext, SelectionResult, TokenStatus


class OracleStaticPolicy(EvictionPolicy):
    name = "oracle_static"
    tiered = False

    def __init__(self, budget):
        super().__init__(budget)
        self._decided = False
        self._keep = set()

    def select(self, ctx: SelectionContext) -> SelectionResult:
        result = SelectionResult()

        if not self._decided:
            budget = self.budget.total_budget
            # Sort, not `list(a_set)` — set iteration order is hash-based,
            # not logical/document order. Slicing an unsorted set when
            # `important` exceeds budget silently and arbitrarily drops
            # whichever positions the hash order happened to put last
            # (this was a real bug: it was cutting off a near-arbitrary
            # subset of credentials rather than making a principled choice,
            # and — worse — fragmenting individual credentials' label+value
            # spans rather than dropping any of them cleanly).
            important = sorted(ctx.oracle_important)
            if len(important) > budget:
                self.truncated_important = True  # oracle isn't actually oracular under this budget — a calibration signal, not a policy bug
            filler = [p for p in ctx.active_positions if p not in ctx.oracle_important]
            self._keep = set(important[:budget])
            if len(self._keep) < budget:
                self._keep |= set(filler[: budget - len(self._keep)])
            self._decided = True

        for p in ctx.active_positions:
            old = ctx.statuses.get(p, TokenStatus.FULL)
            if p in self._keep:
                if old != TokenStatus.FULL:
                    result.new_statuses[p] = TokenStatus.FULL
            else:
                if old != TokenStatus.EVICTED:
                    result.new_statuses[p] = TokenStatus.EVICTED
                    result.evictions.append(p)
        return result
