"""Phase 2 policy: structural protection + recoverable tiering, with
retention and precision separated.

Why this exists (found by the Phase 2 smoke test): composing
`StructuralProtectionWrapper` with a tiering policy makes the promotion
signal inert. The wrapper pins every seated protected group straight to
FULL and hands the inner policy whatever budget is left -- which, at the
budgets where this task has headroom, is approximately zero. The
credentials therefore never enter the QUANT tier, nothing is ever
demoted, and all five promotion signals produce byte-identical results.
That is the same "the recoverability factor is inert in every cell" trap
that the original 2x2 fell into.

Separating the two decisions fixes it and is the more faithful mechanism
anyway:

  * RETENTION (eviction immunity): structural protection decides this.
    A protected group is kept whole, or dropped whole, exactly as in
    cache/structural_protection.py.
  * PRECISION (FULL vs QUANT among everything retained): the pluggable
    promotion signal decides this.

So protection says "do not delete this", and the promotion signal says
"of the things we kept, which deserve full precision" -- which is exactly
the decision H-ORTH is a claim about.
"""

from __future__ import annotations

from .base_policy import EvictionPolicy, SelectionContext, SelectionResult, TokenStatus
from .split_signal_tier import _spearman


class ProtectedSplitSignalTierPolicy(EvictionPolicy):
    tiered = True

    def __init__(self, budget, promotion_signal, groups: dict | None = None):
        super().__init__(budget)
        self.promotion_signal = promotion_signal
        self.groups = groups or {}
        self.name = f"protected_tier_{promotion_signal.name}"
        self.rho_samples = []

    def select(self, ctx: SelectionContext) -> SelectionResult:
        result = SelectionResult()
        full_slots = self.budget.full_slots
        retain_slots = self.budget.full_slots + self.budget.quant_slots

        active = [p for p in ctx.active_positions
                  if ctx.statuses.get(p, TokenStatus.FULL) != TokenStatus.EVICTED]

        # --- RETENTION: protected groups seated whole, by best-member attention ---
        protected_active = [p for p in active if p in ctx.protected]
        group_map: dict = {}
        for p in protected_active:
            group_map.setdefault(self.groups.get(p, ("_singleton", p)), []).append(p)

        ranked_groups = sorted(group_map.values(),
                               key=lambda ms: max(ctx.scores.get(p, 0.0) for p in ms),
                               reverse=True)
        retained: list = []
        budget_left = retain_slots
        for members in ranked_groups:
            if len(members) <= budget_left:
                retained.extend(members)
                budget_left -= len(members)

        # remaining retention capacity goes to unprotected content by attention
        retained_set = set(retained)
        others = [p for p in active if p not in retained_set]
        others.sort(key=lambda p: ctx.scores.get(p, 0.0), reverse=True)
        retained.extend(others[:max(0, budget_left)])
        retained_set = set(retained)

        # --- PRECISION: FULL vs QUANT over everything retained, by promotion signal ---
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

        for p in active:
            if p not in retained_set and ctx.statuses.get(p, TokenStatus.FULL) != TokenStatus.EVICTED:
                result.new_statuses[p] = TokenStatus.EVICTED
                result.evictions.append(p)

        return result
