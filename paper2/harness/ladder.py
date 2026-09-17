"""The reference ladder (v1 §4.1) — floor_pos, oracle_causal, oracle_prescient.

*** SESSION A MUST NOT READ THIS FILE. ***
A6 is the blind reimplementation of this ladder and is only worth running if session A has
genuinely not seen the primary implementation (repo CLAUDE.md, "Blind-reimplementation
firewall"). Paper 2's entire result is a ratio of reference arms: an error in `floor_pos` or
`oracle_causal` does not produce a slightly-wrong detail, it produces a wrong headline with
correct-looking error bars.

Budget accounting, uniform across every arm:

    B = C + n_sink + n_window          tokens retained, identical for all arms
    mandatory floors (all arms):       first n_sink, last n_window
    compressible region:               [n_sink, n_ctx - n_window)
    C:                                 tokens each arm chooses from that region

`null` is the one exception and deliberately so: it is pure truncation (last B), it does not
honour the sink, and it exists as a tripwire rather than as a competitor.

=============================================================================
DECISION 7 — RESOLVED. `oracle_causal` packs complete span-pairs optimally.
=============================================================================
The causal oracle must be the OPTIMAL POLICY GIVEN ITS INFORMATION. It knows content but not
the query; the query is uniform over the H bound facts; a fact is answerable only if BOTH its
spans are retained. So expected accuracy is (complete pairs retained) / H, and the argmax is
forced: retain as many COMPLETE pairs as fit in C, never a partial one.

Implementation: greedy knapsack over pairs in ascending token cost. For a pure
maximise-the-count objective under one budget this is provably optimal, and it is additionally
verified per instance by exhaustive search over pair subsets (H is small), so the assertion is
measured rather than argued.

STRUCK, with reasons (all complete fewer pairs than achievable, so all are policies wearing a
ceiling's label rather than ceilings):
  document-order / earliest-position -- spends budget on whichever spans come first, splitting pairs
  shortest-first (spans)             -- packs short spans, systematically orphaning long partners
  round-robin by fact                -- one span per fact before completing any; completes none
  seeded-random                      -- scatters across candidates, completes ~0
Each was measured completing strictly fewer pairs than the optimal packing.

Remainder of the budget is filled from `floor_pos`'s own preference order.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Iterable, Sequence


STRUCK_RULES = (
    "earliest_position", "shortest_first", "seeded_random", "round_robin", "document_order",
)


@dataclass(frozen=True)
class FactSpans:
    """One answerable fact, as the token indices of its constituent spans.

    On LEDGER a fact is two disjoint spans: the binding sentence (s1) and the record line
    (s2). Retaining s2 without s1 is useless (you do not know which surname); retaining s1
    without s2 is useless (you do not know the value). `complete` is what makes that
    checkable rather than assumed.
    """
    fact_id: str
    spans: tuple[tuple[int, ...], ...]

    @property
    def tokens(self) -> tuple[int, ...]:
        return tuple(sorted({t for s in self.spans for t in s}))

    def is_complete_in(self, kept: set[int]) -> bool:
        return all(all(t in kept for t in s) for s in self.spans)


@dataclass
class Selection:
    """What an arm retained, plus the diagnostics the prereg requires to be logged."""
    arm: str
    kept: tuple[int, ...]
    n_kept: int
    C: int
    B: int
    n_candidate_tokens_kept: int = 0
    n_candidate_spans_complete: int = 0
    n_facts_complete: int = 0
    degradation_rule: str | None = None
    degraded: bool = False
    notes: dict = field(default_factory=dict)


def _floors(n_ctx: int, n_sink: int, n_window: int) -> tuple[set[int], list[int]]:
    """Returns (mandatory floor tokens, compressible region indices)."""
    if n_ctx <= n_sink + n_window:
        raise ValueError(
            f"context ({n_ctx}) is not longer than the floors ({n_sink}+{n_window}); "
            "no compressible region exists"
        )
    floor = set(range(min(n_sink, n_ctx))) | set(range(max(0, n_ctx - n_window), n_ctx))
    region = [i for i in range(n_ctx) if i not in floor]
    return floor, region


def _finish(arm: str, kept: set[int], C: int, n_sink: int, n_window: int,
            facts: Sequence[FactSpans] | None = None, **kw) -> Selection:
    B = C + n_sink + n_window
    sel = Selection(arm=arm, kept=tuple(sorted(kept)), n_kept=len(kept), C=C, B=B, **kw)
    if facts:
        cand_tokens = {t for f in facts for t in f.tokens}
        sel.n_candidate_tokens_kept = len(cand_tokens & kept)
        sel.n_candidate_spans_complete = sum(
            1 for f in facts for s in f.spans if all(t in kept for t in s)
        )
        sel.n_facts_complete = sum(1 for f in facts if f.is_complete_in(kept))
    return sel


# --------------------------------------------------------------------------- tripwires

def null_arm(n_ctx: int, C: int, n_sink: int = 8, n_window: int = 64) -> Selection:
    """Pure truncation: the last B tokens. Tripwire, not a competitor."""
    B = C + n_sink + n_window
    kept = set(range(max(0, n_ctx - B), n_ctx))
    return _finish("null", kept, C, n_sink, n_window)


def random_arm(n_ctx: int, C: int, seed: int, n_sink: int = 8, n_window: int = 64,
               facts: Sequence[FactSpans] | None = None) -> Selection:
    """Uniform random C from the compressible region. Tripwire.

    Seeded by CRC32 (repo rule 5) so it is reproducible across processes and machines.
    """
    floor, region = _floors(n_ctx, n_sink, n_window)
    rs = zlib.crc32(f"random|{seed}|{n_ctx}|{C}".encode()) & 0xFFFFFFFF
    import random as _r
    rng = _r.Random(rs)
    pick = region if C >= len(region) else rng.sample(region, C)
    return _finish("random", floor | set(pick), C, n_sink, n_window, facts)


# --------------------------------------------------------------------------- floor

def floor_pos(n_ctx: int, C: int, n_sink: int = 8, n_window: int = 64,
              facts: Sequence[FactSpans] | None = None) -> Selection:
    """THE DENOMINATOR (v1 §1.1, decision 2).

    StreamingLLM-shaped: attention sinks plus the most recent content. Non-absorbing (it
    scores > 0 whenever gold happens to sit early or late), genuinely zero-cost, and the
    empirically strongest trivial baseline in the published matched-budget audit. Beating
    Paper 1's absorbing `random` floor meant nothing; beating this one means something.
    """
    floor, region = _floors(n_ctx, n_sink, n_window)
    kept = floor | set(region[-C:] if C < len(region) else region)
    return _finish("floor_pos", kept, C, n_sink, n_window, facts)


def _pad_from_floor(kept: set[int], region: Sequence[int], budget_left: int) -> set[int]:
    """Fill remaining budget with floor_pos's own preference order (most recent first)."""
    if budget_left <= 0:
        return kept
    for i in reversed(region):
        if budget_left == 0:
            break
        if i not in kept:
            kept.add(i)
            budget_left -= 1
    return kept


# --------------------------------------------------------------------------- ceilings

def oracle_prescient(n_ctx: int, C: int, gold: FactSpans,
                     n_sink: int = 8, n_window: int = 64,
                     facts: Sequence[FactSpans] | None = None) -> Selection:
    """Knows WHICH candidate is queried. Differs from `oracle_causal` in exactly that.

    ===================================================================================
    REDESIGNED 2026-09-06 after check B failed twice. Read this before changing it.
    ===================================================================================
    The previous definition was "the queried gold span + filler from `floor_pos`". That made
    the two ceiling arms differ in TWO faculties at once:

      * information -- prescient knows which record is asked, causal does not; and
      * content     -- causal retained all H packed records while prescient retained ONE
                       record and spent the rest of its budget on filler.

    Both arms always contain the queried gold, so the information advantage bought prescient
    nothing, while causal's extra clean records displaced mildly harmful filler. The result was
    a REPLICATED inversion, `oracle_causal` > `oracle_prescient` (0.9988 vs 0.9950 at n=200,
    3 generations in 800), which made `I` negative -- and `I` is defined as if only information
    differed.

    **The fix: prescient holds the SAME candidate records as causal, with the queried one
    guaranteed to be among them.** Concretely it starts from causal's packing and, if the
    queried candidate was not chosen, swaps out the most expensive chosen candidate for it.
    Everything else -- budget, filler order, floors -- is identical.

    **Why not a strict superset (causal's set PLUS the queried record)?** It is unreachable
    inside a matched budget. `oracle_causal` packs the *maximum* number of candidates that fit
    in `C`; if the queried one was not chosen then by maximality no packing of that many plus
    one fits, so adding it would break the equal-`B` invariant that every arm in the ladder
    depends on. The swap preserves both the record count and the budget.

    The arms now differ in exactly one faculty: **when the budget cannot hold all H
    candidates, causal must choose without knowing the query while prescient always includes
    the queried one.** That is the faculty `I` measures.

    **The swap is cost-neutral only when candidate costs are near-uniform.** If the queried
    candidate is much more expensive than the one it displaces, prescient may end up holding
    FEWER candidates than causal -- which would reintroduce a content difference, in the
    opposite direction. On LEDGER this does not arise by construction: every record line is
    format-identical (`R0dd | Surname | Dept | 123456`) so payable costs differ by at most a
    token or two. The selection records `n_held` and causal's count so the assumption is
    checkable per instance rather than assumed, and `stage5_ladder_validation` reports any
    instance where the two differ.
    """
    floor, region = _floors(n_ctx, n_sink, n_window)
    region_set = set(region)

    want = {t for t in gold.tokens if t in region_set}
    if len(want) > C:
        raise ValueError(
            f"oracle_prescient: gold needs {len(want)} tokens but C={C}. This cell is VOID "
            "and must have been excluded at Stage 1, not run."
        )

    cands = list(facts) if facts else [gold]
    payable = {f.fact_id: {t for s in f.spans for t in s if t in region_set} for f in cands}
    costs = {fid: len(toks) for fid, toks in payable.items()}

    # Reproduce causal's packing exactly: greedy over candidates, ascending payable cost.
    order = sorted(payable, key=lambda fid: (costs[fid], fid))
    chosen: list[str] = []
    used: set[int] = set()
    for fid in order:
        if len(used | payable[fid]) <= C:
            used |= payable[fid]
            chosen.append(fid)

    swapped_out = None
    if gold.fact_id not in chosen:
        # Swap the most expensive chosen candidate for the queried one. Same count, same
        # budget; the only difference from causal is WHICH candidates are held.
        victim = max(chosen, key=lambda fid: (costs[fid], fid))
        trial = (set(chosen) - {victim}) | {gold.fact_id}
        toks = set().union(*(payable[fid] for fid in trial))
        if len(toks) > C:
            # Fall back to dropping victims until the queried candidate fits. Deterministic.
            keep_ids = [gold.fact_id]
            toks = set(payable[gold.fact_id])
            for fid in order:
                if fid == gold.fact_id:
                    continue
                if len(toks | payable[fid]) <= C:
                    toks |= payable[fid]
                    keep_ids.append(fid)
            trial = set(keep_ids)
        swapped_out = victim
        chosen = sorted(trial)
        used = set().union(*(payable[fid] for fid in chosen))

    if not payable[gold.fact_id] <= used:
        raise AssertionError("oracle_prescient must always retain the queried candidate whole")

    kept = _pad_from_floor(floor | used, region, C - len(used))
    sel = _finish("oracle_prescient", kept, C, n_sink, n_window, cands,
                  degradation_rule="prescient_swap", degraded=len(chosen) < len(cands),
                  notes={"K_all_payable": len({t for v in payable.values() for t in v}),
                         "candidates_held": sorted(chosen), "n_held": len(chosen),
                         "queried": gold.fact_id, "swapped_out": swapped_out,
                         "H": len(cands)})
    if gold.fact_id not in {f.fact_id for f in cands if f.is_complete_in(set(sel.kept))}:
        raise AssertionError("queried candidate is not complete in the prescient keep-set")
    return sel


def _optimal_pair_count(costs: list[int], C: int) -> int:
    """Exhaustive check: the most complete pairs any admissible packing can fit in C.

    H is small (4 by construction), so this is 2**H subsets -- cheap enough to run per
    instance rather than trusting the greedy argument. This is the assertion decision 7
    requires: "no admissible packing completes more pairs".
    """
    best = 0
    n = len(costs)
    for mask in range(1 << n):
        tot = cnt = 0
        for i in range(n):
            if mask >> i & 1:
                tot += costs[i]
                cnt += 1
        if tot <= C and cnt > best:
            best = cnt
    return best


def oracle_causal(n_ctx: int, C: int, candidates: Sequence[FactSpans],
                  *, n_sink: int = 8, n_window: int = 64,
                  verify_optimal: bool = True) -> Selection:
    """THE NUMERATOR CEILING (v1 §1.2, decision 3; packing fixed by decision 7).

    Knows the ground-truth token identity of every answerable fact, but not which is queried.
    Because the query is uniform over the H facts and a fact scores only if BOTH its spans
    survive, expected accuracy is (complete pairs) / H -- so the optimal policy is to fit as
    many COMPLETE pairs as possible and never retain a partial one.

    Greedy over pairs in ascending token cost, which is optimal for maximising a count under a
    single budget, and verified exhaustively per instance.
    """
    floor, region = _floors(n_ctx, n_sink, n_window)
    region_set = set(region)

    # Payable cost of each fact: its candidate tokens outside the free floors. Tokens inside
    # the sink or recency window are retained by every arm anyway and must not be charged.
    payable: list[tuple[int, FactSpans, set[int]]] = []
    for f in candidates:
        toks = {t for s in f.spans for t in s if t in region_set}
        payable.append((len(toks), f, toks))

    all_tokens = {t for _, _, toks in payable for t in toks}
    fits_all = len(all_tokens) <= C

    # Greedy knapsack: ascending payable cost, whole pairs only.
    order = sorted(range(len(payable)), key=lambda i: (payable[i][0], payable[i][1].fact_id))
    chosen: set[int] = set()
    taken: list[str] = []
    for i in order:
        cost, f, toks = payable[i]
        if len(chosen | toks) <= C:
            chosen |= toks
            taken.append(f.fact_id)

    if verify_optimal:
        best = _optimal_pair_count([c for c, _, _ in payable], C)
        if len(taken) < best:
            raise AssertionError(
                f"oracle_causal packing is not optimal: completed {len(taken)} pairs but an "
                f"admissible packing completes {best} within C={C}. Decision 7 requires the "
                "causal oracle to be the optimal policy given its information."
            )

    kept = _pad_from_floor(floor | chosen, region, C - len(chosen))
    sel = _finish("oracle_causal", kept, C, n_sink, n_window, candidates,
                  degradation_rule="pair_knapsack_optimal", degraded=not fits_all,
                  notes={"K_all_payable": len(all_tokens), "fits_all": fits_all,
                         "pairs_taken": taken, "n_pairs_taken": len(taken),
                         "H": len(candidates),
                         "expected_accuracy": len(taken) / max(1, len(candidates))})
    # No partial pairs, ever.
    assert sel.n_facts_complete == len(taken), (
        f"partial pair retained: {sel.n_facts_complete} complete vs {len(taken)} packed"
    )
    return sel


__all__ = [
    "STRUCK_RULES", "FactSpans", "Selection",
    "null_arm", "random_arm", "floor_pos", "oracle_prescient", "oracle_causal",
]
