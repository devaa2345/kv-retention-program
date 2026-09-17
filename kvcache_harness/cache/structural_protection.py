"""Arms 2 & 4: structural protection wrapper (Transactional Attention /
Protection-paper style sponsorship).

Design history (both fixed during Phase 0-R calibration, see engine.py /
run_phase0r_calibration.py for how these surfaced):

v1 reserved every structurally-matched position unconditionally, exempt
from budget entirely — unrealistic, and it mechanically saturates
protection at ~100% whenever protected content is a large share of
context, leaving no headroom for G3 and no opening for recoverability to
ever matter.

v2 made protection a per-TOKEN scoring bonus, competing for the shared
budget individually. That fixed the saturation problem but introduced a
worse one: when protected content exceeds budget, individual tokens within
the *same* credential value can win or lose the ranking independently,
fragmenting the value (some characters survive, others don't) — corrupting
it in a way that produced visibly garbled generations in calibration
(`sk-123456`-looking hallucinations, not near-misses).

v3 (current): protection operates on whole GROUPS (one structurally-matched
line = one group, via `groups`, a {token_pos: group_id} map — see
tasks/credential_retrieval.py:char_spans_to_token_groups). Groups are
ranked by their best member's real attention score and seated into budget
greedily, atomically — a group is either kept whole or dropped whole, never
split. Whatever budget remains after protected groups are seated goes to
the wrapped base policy, competing over the unprotected content as before.
"""

from __future__ import annotations

from .base_policy import EvictionPolicy, SelectionContext, SelectionResult, TokenStatus, BudgetSpec


class StructuralProtectionWrapper(EvictionPolicy):
    def __init__(self, base_policy: EvictionPolicy, groups: dict | None = None):
        super().__init__(base_policy.budget)
        self.base_policy = base_policy
        self.name = f"protected_{base_policy.name}"
        self.tiered = base_policy.tiered
        self.groups = groups or {}   # pos -> group_id; positions absent default to a singleton group

    def select(self, ctx: SelectionContext) -> SelectionResult:
        result = SelectionResult()

        protected_active = [p for p in ctx.active_positions if p in ctx.protected]
        group_map: dict = {}
        for p in protected_active:
            gid = self.groups.get(p, ("_singleton", p))
            group_map.setdefault(gid, []).append(p)

        def group_score(members):
            return max(ctx.scores.get(p, 0.0) for p in members)

        ranked_groups = sorted(group_map.values(), key=group_score, reverse=True)

        kept_protected: list = []
        budget_left = self.budget.total_budget
        for members in ranked_groups:
            if len(members) <= budget_left:
                kept_protected.extend(members)
                budget_left -= len(members)
            # else: whole group dropped, contests the remaining budget below like any other content

        for p in kept_protected:
            if ctx.statuses.get(p) != TokenStatus.FULL:
                result.new_statuses[p] = TokenStatus.FULL
                result.promotions.append(p)

        remaining_budget = BudgetSpec(total_budget=max(0, budget_left), full_fraction=self.budget.full_fraction)
        kept_set = set(kept_protected)
        remaining_active = [p for p in ctx.active_positions if p not in kept_set]

        sub_ctx = SelectionContext(
            scores=ctx.scores,
            statuses=ctx.statuses,
            protected=set(),
            oracle_important=ctx.oracle_important,
            active_positions=remaining_active,
            budget=remaining_budget,
            is_prefill=ctx.is_prefill,
            step=ctx.step,
        )
        self.base_policy.budget = remaining_budget
        sub_result = self.base_policy.select(sub_ctx)

        result.new_statuses.update(sub_result.new_statuses)
        result.promotions.extend(sub_result.promotions)
        result.demotions.extend(sub_result.demotions)
        result.evictions.extend(sub_result.evictions)
        return result
