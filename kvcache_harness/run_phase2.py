"""Phase 2 — H-ORTH: does recoverability need a promotion signal that is
statistically independent of the eviction signal?

Protection is held ON and recoverable tiering ON in every arm (per the
test-program design: that is the realistic regime, and Phase 1 showed
protection is what carries the task). Eviction stays attention-ranked
throughout. The ONLY varied factor is the promotion signal:

  P1 attention (same as eviction; QEvict as published, the circularity case)
  P2 epiphany  (EpiKV-style representation delta; the orthogonal candidate)
  P3 random    (signal-free control)
  P4 roundrobin(signal-free COVERAGE control -- the arm to watch)
  P5 oracle    (knows which tokens the queries need; ceiling)

Budgets: 257 (the one budget clearing every calibration gate with real
headroom -- protection alone 0.423 vs oracle 0.943) and 154 (qualified,
secondary). 514 is skipped: protection already saturates there (0.964), so
no promotion signal could show anything.

Comparison baseline is Phase 1's protection-alone arm at the same budget,
iso-token. Also logs mean Spearman rho(eviction signal, promotion signal)
per run, for H-ORTH's continuous form.

Run: python -m kvcache_harness.run_phase2
"""
from __future__ import annotations

import json
import os
import time
import traceback
from statistics import mean

import torch

from .models import load_model_and_tokenizer
from .engine import CacheEngine
from .cache.base_policy import BudgetSpec
from .cache.protected_split_tier import ProtectedSplitSignalTierPolicy
from .cache.promotion_signals import ALL_SIGNALS
from .tasks.multi_credential import make_multi_credential_prompt, score_turn
from .run_phase0r_calibration import build_turn_texts, compute_positions
from .tasks.credential_retrieval import char_spans_to_token_positions

BUDGETS = [257, 154]
N_PROMPTS = 150
SEED_START = 3000
FULL_FRACTION = 0.5

# Bit-width of the quantized recoverable tier. At 8 the FULL/QUANT split has
# no effect on quality (addendum §11), which makes every promotion signal
# equivalent by construction and H-ORTH untestable. 4-bit is the only swept
# precision where QUANT is degraded but not destroyed, i.e. the only regime
# where the promotion decision carries information.
QUANT_BITS_FOR_RUN = 4
OUT = f"results/phase2_{QUANT_BITS_FOR_RUN}bit/raw_results.jsonl"

CFG = dict(n_credentials=6, value_len=14, n_distractors=20, words_per_paragraph=50,
           protect_after_chars=2, max_answer_tokens=22, rebalance_every=1,
           recency_window=64, keep_sink=True)


def build_future_need(tokenizer, turn0_text, body_offset, prompt):
    """Oracle promotion target (P5): credential tokens the queries will need,
    with earlier-queried credentials ranked more urgent."""
    enc = tokenizer(turn0_text, return_offsets_mapping=True, add_special_tokens=True)
    om = enc["offset_mapping"]
    n = len(prompt.credentials)
    need = {}
    for idx, cred in enumerate(prompt.credentials):
        rank = prompt.turn_order.index(idx)
        spans = [(cred.label_char_span[0] + body_offset, cred.label_char_span[1] + body_offset),
                 (cred.value_char_span[0] + body_offset, cred.value_char_span[1] + body_offset)]
        for p in char_spans_to_token_positions(om, spans, protect_after_chars=0):
            need[p] = 1.0 + (n - rank) / n
    return need


def load_done():
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
                done.add((r["budget"], r["signal"], r["seed"]))
            except Exception:
                pass
    return done


def main():
    from . import engine as engine_mod
    engine_mod.QUANT_BITS = QUANT_BITS_FOR_RUN

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = load_done()
    print(f"quant tier bit-width = {QUANT_BITS_FOR_RUN}; resuming: {len(done)} cells already done", flush=True)

    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct", device="cuda")
    f = open(OUT, "a")
    t0 = time.time()

    for i in range(N_PROMPTS):
        seed = SEED_START + i
        prompt = make_multi_credential_prompt(
            seed=seed, n_credentials=CFG["n_credentials"], n_distractors=CFG["n_distractors"],
            words_per_paragraph=CFG["words_per_paragraph"], value_len=CFG["value_len"])
        turn_texts, body_offset = build_turn_texts(tokenizer, prompt)
        oracle_important, protected, protected_groups, _ = compute_positions(
            tokenizer, turn_texts[0], body_offset, prompt, CFG["protect_after_chars"])
        future_need = build_future_need(tokenizer, turn_texts[0], body_offset, prompt)

        for budget in BUDGETS:
            spec = BudgetSpec(total_budget=budget, full_fraction=FULL_FRACTION)
            for sig_name, sig_cls in ALL_SIGNALS.items():
                if (budget, sig_name, seed) in done:
                    continue
                try:
                    signal = sig_cls(seed=seed) if sig_name == "P3_random" else sig_cls()
                    policy = ProtectedSplitSignalTierPolicy(spec, signal, groups=protected_groups)
                    inner = policy
                    engine = CacheEngine(model, tokenizer, policy, device="cuda")
                    trace, turn_answers, _ = engine.generate_multi_turn(
                        turn_texts, protected, oracle_important,
                        max_answer_tokens=CFG["max_answer_tokens"], rebalance_every=CFG["rebalance_every"],
                        eos_token_id=tokenizer.eos_token_id, recency_window=CFG["recency_window"],
                        keep_sink=CFG["keep_sink"],
                        capture_epiphany=(sig_name == "P2_epiphany"),
                        future_need=future_need if sig_name == "P5_oracle_future" else None)
                    n = len(prompt.credentials)
                    correct = [score_turn(turn_answers[j], prompt.credentials[prompt.turn_order[j]])
                               for j in range(n)]
                    rho = mean(inner.rho_samples) if inner.rho_samples else None
                    f.write(json.dumps({
                        "quant_bits": QUANT_BITS_FOR_RUN,
                        "budget": budget, "signal": sig_name, "seed": seed,
                        "frac_retrieved": sum(correct) / n,
                        "mean_spearman_rho_evict_vs_promote": rho,
                        "n_promotions": sum(len(e.promotions) for e in trace.events),
                        "n_demotions": sum(len(e.demotions) for e in trace.events),
                    }) + "\n")
                    f.flush()
                except Exception as e:
                    print(f"ERROR budget={budget} sig={sig_name} seed={seed}: {e}", flush=True)
                    traceback.print_exc()
                    torch.cuda.empty_cache()

        if (i + 1) % 5 == 0:
            print(f"[{i+1}/{N_PROMPTS} prompts] elapsed={(time.time()-t0)/60:.1f}min", flush=True)

    f.close()
    print("Phase 2 complete", flush=True)


if __name__ == "__main__":
    main()
