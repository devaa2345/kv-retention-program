"""Phase 2 (H-ORTH): the promotion signal, varied while everything else is
held fixed.

H-ORTH: "recoverability has value if and only if the promotion signal is
statistically independent of the eviction signal." Phase 1 found
recoverable tiering inert under matched-token comparison; the mechanism
hypothesis is that this is *because* promotion re-scores on the same
cumulative-attention signal that caused the token to be demoted in the
first place, so a dormant token can never earn its way back.

Each signal below supplies the FULL-vs-QUANT ordering among retained
tokens. Eviction (who leaves the retained set entirely) stays
attention-ranked in every arm, as published.
"""

from __future__ import annotations

import random


class PromotionSignal:
    name = "base"

    def values(self, candidates, ctx) -> dict:
        """Return {pos: float} — higher means 'promote to FULL sooner'."""
        raise NotImplementedError


class AttentionPromotion(PromotionSignal):
    """P1 — QEvict as published: promote on the same cumulative-attention
    score that drives eviction. The circularity case."""
    name = "P1_attention"

    def values(self, candidates, ctx):
        return {p: ctx.scores.get(p, 0.0) for p in candidates}


class EpiphanyPromotion(PromotionSignal):
    """P2 — EpiKV-style: promote on representation change (how much a token's
    hidden state moves across layers during its own forward pass), read from
    hidden states rather than the attention matrix. Computed once per token
    at arrival, so it is structurally independent of accumulated attention."""
    name = "P2_epiphany"

    def values(self, candidates, ctx):
        epi = ctx.aux_signals.get("epiphany", {})
        return {p: epi.get(p, 0.0) for p in candidates}


class RandomPromotion(PromotionSignal):
    """P3 — signal-free control. Stable per position within a run (a random
    but consistent preference), so it is a genuinely uninformative signal
    rather than per-step churn (that is P4's job)."""
    name = "P3_random"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._cache = {}

    def values(self, candidates, ctx):
        out = {}
        for p in candidates:
            if p not in self._cache:
                self._cache[p] = random.Random((self.seed, p).__hash__()).random()
            out[p] = self._cache[p]
        return out


class RoundRobinPromotion(PromotionSignal):
    """P4 — signal-free *coverage* control: rotation. Every retained token
    cycles through the FULL tier regardless of any measured importance.
    This is the arm to watch — in the adjacent block-diffusion work, blind
    rotation beat measured ranking, and ranking's penalty grew with budget."""
    name = "P4_roundrobin"

    def values(self, candidates, ctx):
        n = max(1, len(candidates))
        return {p: float((p + ctx.step) % n) for p in candidates}


class FutureNeedPromotion(PromotionSignal):
    """P5 — oracle promotion: knows which credential the *next* query will
    ask for. Ceiling on what any promotion signal could buy."""
    name = "P5_oracle_future"

    def values(self, candidates, ctx):
        fut = ctx.aux_signals.get("future_need", {})
        return {p: fut.get(p, -1e9) for p in candidates}


ALL_SIGNALS = {
    "P1_attention": AttentionPromotion,
    "P2_epiphany": EpiphanyPromotion,
    "P3_random": RandomPromotion,
    "P4_roundrobin": RoundRobinPromotion,
    "P5_oracle_future": FutureNeedPromotion,
}
