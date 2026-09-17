"""Retention, protection, and promotion policy (spec sections 4, 5, 7).

Every selection here is over explicitly-ordered lists. No Python `set` is ever iterated or
sliced in a way that reaches the output (spec section 1). Ties are broken by ascending position
index so that two processes agree exactly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

FULL, QUANT, EVICT = 0, 1, 2


@dataclass
class PolicyConfig:
    total_budget: int
    recency_window: int = 64
    sink: int = 1
    protection: str = "none"          # none | structural | oracle_static
    eviction: str = "permanent"       # permanent | tiered | none
    full_fraction: float = 0.5
    quant_bits: int = 4
    quant_byte_cost: float = 0.25
    promotion: str = "attention"      # attention | epiphany | random | roundrobin | oracle
    iso_condition: str = "iso_token"  # iso_token | iso_memory
    rebalance_every: int = 1
    score_norm: str = "mean"   # [SPEC-GAP 4] "sum" = raw accumulated mass; "mean" = per-query

    def competitive_room(self) -> int:
        """Spec section 4: total_budget - recency_window - sink. Must be > 0."""
        return self.total_budget - self.recency_window - self.sink

    def assert_budget_valid(self) -> None:
        room = self.competitive_room()
        if room <= 0:
            raise ValueError(
                f"VOID CELL: total_budget({self.total_budget}) - recency_window"
                f"({self.recency_window}) - sink({self.sink}) = {room} <= 0. "
                "Spec section 4 requires this be positive; refusing to run."
            )


def select_retained(
    n: int,
    budget: int,
    scores: list[float],
    line_id: list[int],
    is_protected_line: list[bool],
    is_oracle_must: list[bool],
    cfg: PolicyConfig,
) -> tuple[list[int], dict]:
    """Return the ascending list of cache slots to retain, plus diagnostics.

    Order of operations (spec section 4):
      1. floors first, identically in every arm: sink, then recency window;
      2. protection seats whole lines atomically, ranked by best member score;
      3. remaining budget filled by attention-score rank.
    """
    if budget >= n:
        return list(range(n)), {"cut_score": float("-inf"), "n_protected_lines": 0,
                                "oracle_oversubscribed": False, "surplus": budget - n}

    kept_flag = [False] * n

    # --- 1. floors ---
    for i in range(min(cfg.sink, n)):
        kept_flag[i] = True
    for i in range(max(0, n - cfg.recency_window), n):
        kept_flag[i] = True
    n_floor = sum(kept_flag)
    remaining = budget - n_floor
    if remaining < 0:
        raise ValueError(f"floors ({n_floor}) exceed budget ({budget}) at n={n}")

    diag = {"n_floor": n_floor, "oracle_oversubscribed": False, "n_protected_lines": 0}

    # --- 2a. oracle_static must-keep set (not greedy: mandatory) ---
    if cfg.protection == "oracle_static":
        must = [i for i in range(n) if is_oracle_must[i] and not kept_flag[i]]
        if len(must) > remaining:
            diag["oracle_oversubscribed"] = True
            diag["oracle_deficit"] = len(must) - remaining
            # Seat as many as fit, highest score first, then ascending index for ties.
            must = sorted(must, key=lambda i: (-scores[i], i))[:remaining]
        for i in must:
            kept_flag[i] = True
        remaining = budget - sum(kept_flag)

    # --- 2b. structural protection: whole matched lines, atomic, greedy by best member ---
    elif cfg.protection == "structural":
        lines: dict[int, list[int]] = {}
        for i in range(n):
            lid = line_id[i]
            if lid >= 0 and is_protected_line[i]:
                lines.setdefault(lid, []).append(i)
        # explicit sort: best member score desc, then smallest line id for a stable tie-break
        ordered = sorted(lines.items(), key=lambda kv: (-max(scores[j] for j in kv[1]), kv[0]))
        seated = 0
        for lid, members in ordered:
            need = [j for j in members if not kept_flag[j]]
            if not need:
                seated += 1
                continue
            if len(need) <= remaining:          # atomic: all or nothing
                for j in need:
                    kept_flag[j] = True
                remaining -= len(need)
                seated += 1
        diag["n_protected_lines"] = seated

    # --- 3. fill the rest by attention rank ---
    cut_score = float("-inf")
    if remaining > 0:
        cands = [i for i in range(n) if not kept_flag[i]]
        cands.sort(key=lambda i: (-scores[i], i))     # explicit, deterministic
        chosen = cands[:remaining]
        if chosen:
            cut_score = scores[chosen[-1]]
        for i in chosen:
            kept_flag[i] = True

    keep = [i for i in range(n) if kept_flag[i]]
    diag["cut_score"] = cut_score
    diag["surplus"] = budget - len(keep)
    return keep, diag


def assign_tiers(
    keep: list[int],
    n: int,
    promo_scores: list[float],
    cfg: PolicyConfig,
    rng: random.Random | None = None,
    rr_counter: int = 0,
) -> list[bool]:
    """Return quant_mask over `keep`: True means that retained slot is QUANT tier.

    [GAP-C]: sink and recency positions are FULL by priority; the remainder is split by the
    promotion signal, top `full_fraction` FULL.
    """
    m = len(keep)
    if cfg.eviction != "tiered":
        return [False] * m           # permanent arms hold everything retained at FULL

    n_full_target = int(round(cfg.full_fraction * m))

    priority = []                    # slots that are FULL by floor priority
    rest = []
    for pos_in_keep, slot in enumerate(keep):
        if slot < cfg.sink or slot >= n - cfg.recency_window:
            priority.append(pos_in_keep)
        else:
            rest.append(pos_in_keep)

    full_flags = [False] * m
    for p in priority[:n_full_target]:
        full_flags[p] = True
    left = max(0, n_full_target - len(priority))

    if left > 0 and rest:
        # Every signal is now expressed as an explicit per-position promotion score, ranked the
        # same way. Previously `random` and `roundrobin` bypassed promo_scores entirely, which
        # meant the probe measuring rho(eviction, promotion) had to reconstruct them and could
        # not measure what the policy actually did. One code path, one measurable quantity.
        order = sorted(rest, key=lambda p: (-promo_scores[keep[p]], keep[p]))
        for p in order[:left]:
            full_flags[p] = True

    return [not f for f in full_flags]
