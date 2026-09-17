"""Two diagnostics for the Phase 2 interim result (oracle promotion signal
performing no better than random).

A. WIRING: does P5's future_need information actually reach the promotion
   decision? Measured behaviourally, not by inspection: what fraction of
   credential value tokens end the run in the FULL tier under P5 vs P3
   (random)? If P5 is wired, it must be strictly higher.

B. MECHANISM: does the QUANT tier degrade retrieval *at all*? Force
   full_fraction to ~0 (everything retained is quantized) vs ~1 (everything
   retained is full precision), holding retention identical. If accuracy is
   the same, 8-bit KV quantization is near-lossless for this task, the
   FULL/QUANT split is a no-op, and no promotion signal could ever matter --
   which would explain Phase 1's null and Phase 2's flat arms as one thing.

Run: python -m kvcache_harness.debug_p5_and_quant
"""
from __future__ import annotations

from statistics import mean

from .models import load_model_and_tokenizer
from .engine import CacheEngine
from .cache.base_policy import BudgetSpec, TokenStatus
from .cache.protected_split_tier import ProtectedSplitSignalTierPolicy
from .cache.promotion_signals import RandomPromotion, FutureNeedPromotion, AttentionPromotion
from .tasks.multi_credential import make_multi_credential_prompt, score_turn
from .run_phase0r_calibration import build_turn_texts, compute_positions
from .run_phase2 import build_future_need

BUDGET = 257
SEEDS = list(range(3000, 3010))
CFG = dict(n_credentials=6, value_len=14, n_distractors=20, words_per_paragraph=50,
           protect_after_chars=2, max_answer_tokens=22, rebalance_every=1,
           recency_window=64, keep_sink=True)


def run(model, tokenizer, seed, signal, full_fraction=0.5):
    prompt = make_multi_credential_prompt(
        seed=seed, n_credentials=CFG["n_credentials"], n_distractors=CFG["n_distractors"],
        words_per_paragraph=CFG["words_per_paragraph"], value_len=CFG["value_len"])
    turn_texts, body_offset = build_turn_texts(tokenizer, prompt)
    oracle_important, protected, groups, cred_pos = compute_positions(
        tokenizer, turn_texts[0], body_offset, prompt, CFG["protect_after_chars"])
    future_need = build_future_need(tokenizer, turn_texts[0], body_offset, prompt)

    spec = BudgetSpec(total_budget=BUDGET, full_fraction=full_fraction)
    policy = ProtectedSplitSignalTierPolicy(spec, signal, groups=groups)
    engine = CacheEngine(model, tokenizer, policy, device="cuda")
    trace, turn_answers, _ = engine.generate_multi_turn(
        turn_texts, protected, oracle_important,
        max_answer_tokens=CFG["max_answer_tokens"], rebalance_every=CFG["rebalance_every"],
        eos_token_id=tokenizer.eos_token_id, recency_window=CFG["recency_window"],
        keep_sink=CFG["keep_sink"],
        capture_epiphany=False,
        future_need=future_need if isinstance(signal, FutureNeedPromotion) else None)

    n = len(prompt.credentials)
    correct = [score_turn(turn_answers[j], prompt.credentials[prompt.turn_order[j]]) for j in range(n)]

    all_cred_pos = set()
    for ps in cred_pos.values():
        all_cred_pos |= ps
    statuses = trace.final_statuses
    full_c = sum(1 for p in all_cred_pos if statuses.get(p) == TokenStatus.FULL)
    quant_c = sum(1 for p in all_cred_pos if statuses.get(p) == TokenStatus.QUANT)
    gone_c = sum(1 for p in all_cred_pos if p not in statuses)
    tot = max(1, len(all_cred_pos))
    return dict(frac=sum(correct) / n, cred_full=full_c / tot, cred_quant=quant_c / tot,
                cred_evicted=gone_c / tot,
                n_future_need=len(future_need),
                n_future_need_overlap=len(set(future_need) & all_cred_pos))


def main():
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct", device="cuda")

    print("=== A. WIRING: credential tokens ending in FULL tier, P5 vs P3 vs P1 ===")
    for label, sig_factory in [("P5_oracle", lambda s: FutureNeedPromotion()),
                                ("P3_random", lambda s: RandomPromotion(seed=s)),
                                ("P1_attention", lambda s: AttentionPromotion())]:
        res = [run(model, tokenizer, s, sig_factory(s)) for s in SEEDS]
        print(f"{label:14s} acc={mean(r['frac'] for r in res):.3f} "
              f"cred_FULL={mean(r['cred_full'] for r in res):.3f} "
              f"cred_QUANT={mean(r['cred_quant'] for r in res):.3f} "
              f"cred_EVICTED={mean(r['cred_evicted'] for r in res):.3f} "
              f"(future_need entries={res[0]['n_future_need']}, overlap with cred tokens={res[0]['n_future_need_overlap']})",
              flush=True)

    print()
    print("=== B. MECHANISM: does QUANT degrade at all? (retention identical, precision varied) ===")
    for label, ff in [("all FULL   (full_fraction=1.0)", 1.0),
                       ("half/half  (full_fraction=0.5)", 0.5),
                       ("all QUANT  (full_fraction=0.01)", 0.01)]:
        res = [run(model, tokenizer, s, AttentionPromotion(), full_fraction=ff) for s in SEEDS]
        print(f"{label:32s} acc={mean(r['frac'] for r in res):.3f} "
              f"cred_FULL={mean(r['cred_full'] for r in res):.3f} "
              f"cred_QUANT={mean(r['cred_quant'] for r in res):.3f}", flush=True)


if __name__ == "__main__":
    main()
