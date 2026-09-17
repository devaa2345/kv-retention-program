"""Arm definitions (spec section 5) and phase-2 promotion signals (section 7)."""
from __future__ import annotations
from .policy import PolicyConfig

# [GAP-R] arm 5's eviction column is "--" in the spec table. Read as permanent (FULL/EVICT):
# oracle_static is a *retention* ceiling, so tiering would confound it with a promotion effect.
ARMS = {
    1: dict(protection="none",          eviction="permanent", label="none/permanent"),
    2: dict(protection="structural",    eviction="permanent", label="structural/permanent"),
    3: dict(protection="none",          eviction="tiered",    label="none/tiered"),
    4: dict(protection="structural",    eviction="tiered",    label="structural/tiered"),
    5: dict(protection="oracle_static", eviction="permanent", label="oracle_static"),
    6: dict(protection="none",          eviction="none",      label="full_cache_ref"),
}

PROMOTION_SIGNALS = ["attention", "epiphany", "random", "roundrobin", "oracle"]
PROMO_IDS = {"attention": "P1", "epiphany": "P2", "random": "P3",
             "roundrobin": "P4", "oracle": "P5"}


def make_cfg(arm: int, budget: int, iso_condition="iso_token", quant_bits=4,
             quant_byte_cost=0.25, promotion="attention", recency_window=64,
             full_fraction=0.5) -> PolicyConfig:
    a = ARMS[arm]
    from .cache_engine import iso_memory_tokens
    eff_budget = budget
    # startswith, not ==, so sensitivity conditions like "iso_memory_c0.28" also receive the
    # byte-cost-adjusted token grant. An exact match silently gave them the iso-token budget,
    # which made the [SPEC-GAP 3] sensitivity a no-op that reproduced the iso-token numbers.
    if iso_condition.startswith("iso_memory") and a["eviction"] == "tiered":
        eff_budget = iso_memory_tokens(budget, full_fraction, quant_byte_cost)
    return PolicyConfig(
        total_budget=eff_budget, protection=a["protection"], eviction=a["eviction"],
        quant_bits=quant_bits, quant_byte_cost=quant_byte_cost, promotion=promotion,
        iso_condition=iso_condition, recency_window=recency_window,
        full_fraction=full_fraction)
