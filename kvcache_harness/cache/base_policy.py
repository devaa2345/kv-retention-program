"""Shared types and the policy interface every cache arm implements.

Design note (documented simplification for the Phase 0 pilot): importance
scores are computed *globally* (mean attention received, averaged over all
layers and heads) rather than per-layer independently. Real H2O/SnapKV-style
systems often score per-layer. We use one shared kept/tier schedule across
layers because (a) it keeps the harness tractable, (b) it makes structural
protection and the oracle trivial to wire (both operate on token identity /
position, not on a per-layer score), and (c) the research question here is
about mechanism composition (protection x recoverability), not about
reproducing H2O's exact per-layer scoring behavior. This is called out again
in README.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class TokenStatus(IntEnum):
    EVICTED = 0
    QUANT = 1
    FULL = 2


@dataclass
class BudgetSpec:
    """Matched-budget definition shared across arms.

    total_budget: max number of *retained* positions (FULL + QUANT combined).
    full_fraction: for tiered policies, fraction of total_budget held at
        FULL precision; the remainder is QUANT. Non-tiered policies ignore
        full_fraction and hold everything they retain at FULL.
    """

    total_budget: int
    full_fraction: float = 0.5

    @property
    def full_slots(self) -> int:
        return max(1, round(self.total_budget * self.full_fraction))

    @property
    def quant_slots(self) -> int:
        return max(0, self.total_budget - self.full_slots)


@dataclass
class SelectionContext:
    scores: dict            # pos -> float, cumulative attention received
    statuses: dict          # pos -> TokenStatus, current status of every known pos
    protected: set          # pos set, forced-FULL positions (structural protection)
    oracle_important: set   # pos set, ground-truth important positions (oracle only)
    active_positions: list  # positions currently materialized in the cache, in order
    budget: BudgetSpec
    is_prefill: bool
    step: int
    # Phase 2: additional per-position signals a policy may use *instead of*
    # attention for the promotion decision, e.g. {"epiphany": {pos: float},
    # "future_need": {pos: float}}. Empty for Phase 0/1 policies.
    aux_signals: dict = field(default_factory=dict)


@dataclass
class SelectionResult:
    new_statuses: dict = field(default_factory=dict)   # pos -> TokenStatus changes only
    promotions: list = field(default_factory=list)      # pos list, QUANT/EVICTED -> FULL this step
    demotions: list = field(default_factory=list)       # pos list, FULL -> QUANT this step
    evictions: list = field(default_factory=list)       # pos list, * -> EVICTED this step


class EvictionPolicy:
    """Base interface. Subclasses implement `select`."""

    name = "base"
    tiered = False  # whether this policy ever uses TokenStatus.QUANT

    def __init__(self, budget: BudgetSpec):
        self.budget = budget

    def select(self, ctx: SelectionContext) -> SelectionResult:
        raise NotImplementedError
