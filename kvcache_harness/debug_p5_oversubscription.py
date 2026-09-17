"""Is P5 an actual upper bound, or "oracle on the subset that fits"?

P5 assigns a high promotion value to credential tokens and -1e9 to
everything else, then the policy takes the top `full_slots` by that value.
If the number of credential tokens surviving RETENTION exceeds `full_slots`,
P5 cannot promote all of them, something has to choose the remainder, and
the arm is not a ceiling -- it is oracle-on-a-subset and must not be
reported as an upper bound.

This instruments every selection call at budget 257, 4-bit, and records:
  * full_slots available
  * how many tokens P5 actively wants at FULL (finite promo value) that
    survived retention
  * whether that demand exceeded the slots (oversubscription)
  * the final tier disposition of credential tokens

It also reports what the tie-break beyond the must-promote set actually is,
since `sorted` is stable and `retained` is built attention-first.
"""
from __future__ import annotations

from statistics import mean

from . import engine as engine_mod
from .models import load_model_and_tokenizer
from .engine import CacheEngine
from .cache.base_policy import BudgetSpec, TokenStatus
from .cache.protected_split_tier import ProtectedSplitSignalTierPolicy
from .cache.promotion_signals import FutureNeedPromotion
from .tasks.multi_credential import make_multi_credential_prompt, score_turn
from .run_phase0r_calibration import build_turn_texts, compute_positions
from .run_phase2 import build_future_need

BUDGET = 257
SEEDS = list(range(3000, 3012))
CFG = dict(n_credentials=6, value_len=14, n_distractors=20, words_per_paragraph=50,
           protect_after_chars=2, max_answer_tokens=22, rebalance_every=1,
           recency_window=64, keep_sink=True)


class InstrumentedPolicy(ProtectedSplitSignalTierPolicy):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.calls = []

    def select(self, ctx):
        full_slots = self.budget.full_slots
        result = super().select(ctx)
        # reconstruct what the promotion stage saw
        retained = [p for p, s in result.new_statuses.items()
                    if s in (TokenStatus.FULL, TokenStatus.QUANT)]
        retained += [p for p in ctx.active_positions
                     if p not in result.new_statuses
                     and ctx.statuses.get(p, TokenStatus.FULL) != TokenStatus.EVICTED]
        promo = self.promotion_signal.values(retained, ctx)
        wanted = [p for p in retained if promo.get(p, -1e9) > -1e8]
        self.calls.append((full_slots, len(wanted), len(retained)))
        return result


def main():
    engine_mod.QUANT_BITS = 4
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct", device="cuda")

    rows = []
    for seed in SEEDS:
        prompt = make_multi_credential_prompt(
            seed=seed, n_credentials=CFG["n_credentials"], n_distractors=CFG["n_distractors"],
            words_per_paragraph=CFG["words_per_paragraph"], value_len=CFG["value_len"])
        turn_texts, body_offset = build_turn_texts(tokenizer, prompt)
        oracle_important, protected, groups, cred_pos = compute_positions(
            tokenizer, turn_texts[0], body_offset, prompt, CFG["protect_after_chars"])
        future_need = build_future_need(tokenizer, turn_texts[0], body_offset, prompt)

        spec = BudgetSpec(total_budget=BUDGET, full_fraction=0.5)
        policy = InstrumentedPolicy(spec, FutureNeedPromotion(), groups=groups)
        eng = CacheEngine(model, tokenizer, policy, device="cuda")
        trace, answers, _ = eng.generate_multi_turn(
            turn_texts, protected, oracle_important,
            max_answer_tokens=CFG["max_answer_tokens"], rebalance_every=CFG["rebalance_every"],
            eos_token_id=tokenizer.eos_token_id, recency_window=CFG["recency_window"],
            keep_sink=CFG["keep_sink"], future_need=future_need)

        n = len(prompt.credentials)
        acc = sum(score_turn(answers[j], prompt.credentials[prompt.turn_order[j]])
                  for j in range(n)) / n

        all_cred = set()
        for ps in cred_pos.values():
            all_cred |= ps
        st = trace.final_statuses
        tot = max(1, len(all_cred))
        oversub = [c for c in policy.calls if c[1] > c[0]]
        rows.append(dict(
            acc=acc,
            future_need_size=len(future_need),
            mean_full_slots=mean(c[0] for c in policy.calls),
            mean_wanted=mean(c[1] for c in policy.calls),
            max_wanted=max(c[1] for c in policy.calls),
            frac_calls_oversubscribed=len(oversub) / max(1, len(policy.calls)),
            cred_full=sum(1 for p in all_cred if st.get(p) == TokenStatus.FULL) / tot,
            cred_quant=sum(1 for p in all_cred if st.get(p) == TokenStatus.QUANT) / tot,
            cred_evicted=sum(1 for p in all_cred if p not in st) / tot,
        ))

    print(f"P5 at budget {BUDGET}, 4-bit, n={len(rows)} prompts")
    print(f"  future_need set size (label+value tokens, all 6 creds): {mean(r['future_need_size'] for r in rows):.1f}")
    print(f"  FULL slots available per call:                          {mean(r['mean_full_slots'] for r in rows):.1f}")
    print(f"  tokens P5 wants at FULL that survived retention (mean): {mean(r['mean_wanted'] for r in rows):.1f}")
    print(f"  ...(max over calls):                                    {mean(r['max_wanted'] for r in rows):.1f}")
    print(f"  fraction of selection calls OVERSUBSCRIBED:             {mean(r['frac_calls_oversubscribed'] for r in rows):.3f}")
    print()
    print(f"  credential tokens ending FULL:    {mean(r['cred_full'] for r in rows):.3f}")
    print(f"  credential tokens ending QUANT:   {mean(r['cred_quant'] for r in rows):.3f}")
    print(f"  credential tokens EVICTED:        {mean(r['cred_evicted'] for r in rows):.3f}")
    print(f"  accuracy:                          {mean(r['acc'] for r in rows):.3f}")
    engine_mod.QUANT_BITS = 8


if __name__ == "__main__":
    main()
