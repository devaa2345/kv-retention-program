"""H-ORTH factorial, our half (Paper 1, Option B item 3).

Cell A (already exists): OUR tier-seating rule x OUR distractor format  -> results/phase2_4bit/
Cell B (this script):    OUR tier-seating rule x AMD's distractor format -> results/phase2_4bit_amdfmt/

Our tier-seating rule = cache/protected_split_tier.py (protection decides retention, the
promotion signal decides FULL vs QUANT among retained). Distractor format switches to the
independent implementation's (all 26 lines `LABEL_i_ID: sk-<14 hex>`), via
make_multi_credential_prompt(distractor_format="amd"). Nothing else changes: same model, 4-bit
tier, budget 257, seeds 3000-3149, n=150, signals P1-P5.

Cell B needs its own context, because the format change alters what structural protection
competes against (26 pattern-matching lines that all look like credentials). So besides P1-P5
this script also runs, at the same budget and seeds, the reference arms that make P1-P5
interpretable under the new format:
  R1 no-protection permanent (arm 1)      R2 structural-protection permanent (arm 2)
  R5 oracle_static (retention oracle)     R6 full_cache_ref
  BAND_full / BAND_quant: retention held identical (protected retention, attention signal) with
      full_fraction=1.0 (all retained FULL) and 0.0 (all retained QUANT, 4-bit). Their gap is
      the room the promotion decision has; if it is ~0 under this format, P1-P5 cannot separate.

Run: python -m kvcache_harness.run_phase2_amdfmt
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
from . import engine as engine_mod
from .cache.base_policy import BudgetSpec
from .cache.permanent_evict import PermanentEvictPolicy
from .cache.structural_protection import StructuralProtectionWrapper
from .cache.oracle_static import OracleStaticPolicy
from .cache.protected_split_tier import ProtectedSplitSignalTierPolicy
from .cache.promotion_signals import ALL_SIGNALS
from .tasks.multi_credential import make_multi_credential_prompt, score_turn
from .run_phase0r_calibration import build_turn_texts, compute_positions
from .run_phase2 import build_future_need, CFG, SEED_START, N_PROMPTS

QUANT_BITS_FOR_RUN = 4
BUDGET = 257
OUT = "results/phase2_4bit_amdfmt/raw_results.jsonl"
FMT = "amd"


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
    engine_mod.QUANT_BITS = QUANT_BITS_FOR_RUN
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = load_done()
    print(f"distractor_format={FMT} bits={QUANT_BITS_FOR_RUN} budget={BUDGET}; {len(done)} cells done", flush=True)
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct", device="cuda")
    f = open(OUT, "a")
    t0 = time.time()
    spec = BudgetSpec(total_budget=BUDGET, full_fraction=0.5)

    for i in range(N_PROMPTS):
        seed = SEED_START + i
        prompt = make_multi_credential_prompt(
            seed=seed, n_credentials=CFG["n_credentials"], n_distractors=CFG["n_distractors"],
            words_per_paragraph=CFG["words_per_paragraph"], value_len=CFG["value_len"],
            distractor_format=FMT)
        turn_texts, body_offset = build_turn_texts(tokenizer, prompt)
        oracle_important, protected, groups, _ = compute_positions(
            tokenizer, turn_texts[0], body_offset, prompt, CFG["protect_after_chars"])
        future_need = build_future_need(tokenizer, turn_texts[0], body_offset, prompt)
        ctx_len = len(tokenizer(turn_texts[0])["input_ids"])

        def arm_specs():
            for name, cls in ALL_SIGNALS.items():
                yield name, (lambda cls=cls, name=name: ProtectedSplitSignalTierPolicy(
                    spec, cls(seed=seed) if name == "P3_random" else cls(), groups=groups))
            yield "BAND_full", lambda: ProtectedSplitSignalTierPolicy(
                BudgetSpec(BUDGET, 1.0), ALL_SIGNALS["P1_attention"](), groups=groups)
            yield "BAND_quant", lambda: ProtectedSplitSignalTierPolicy(
                BudgetSpec(BUDGET, 0.0), ALL_SIGNALS["P1_attention"](), groups=groups)
            yield "R1_no_protect_permanent", lambda: PermanentEvictPolicy(spec)
            yield "R2_protect_permanent", lambda: StructuralProtectionWrapper(PermanentEvictPolicy(spec), groups=groups)
            yield "R5_oracle_static", lambda: OracleStaticPolicy(spec)
            yield "R6_full_cache_ref", lambda: PermanentEvictPolicy(BudgetSpec(100_000, 1.0))

        for name, make in arm_specs():
            if (BUDGET, name, seed) in done:
                continue
            try:
                policy = make()
                engine = CacheEngine(model, tokenizer, policy, device="cuda")
                trace, answers, _ = engine.generate_multi_turn(
                    turn_texts, protected, oracle_important,
                    max_answer_tokens=CFG["max_answer_tokens"], rebalance_every=CFG["rebalance_every"],
                    eos_token_id=tokenizer.eos_token_id, recency_window=CFG["recency_window"],
                    keep_sink=CFG["keep_sink"], capture_epiphany=(name == "P2_epiphany"),
                    future_need=future_need if name == "P5_oracle_future" else None)
                n = len(prompt.credentials)
                correct = [score_turn(answers[j], prompt.credentials[prompt.turn_order[j]]) for j in range(n)]
                rho = mean(policy.rho_samples) if getattr(policy, "rho_samples", None) else None
                f.write(json.dumps({
                    "distractor_format": FMT, "quant_bits": QUANT_BITS_FOR_RUN, "budget": BUDGET,
                    "signal": name, "seed": seed, "frac_retrieved": sum(correct) / n,
                    "mean_spearman_rho_evict_vs_promote": rho, "context_tokens": ctx_len,
                    "n_protected_tokens": len(protected),
                }) + "\n")
                f.flush()
            except Exception as e:
                print(f"ERROR sig={name} seed={seed}: {e}", flush=True)
                traceback.print_exc()
                torch.cuda.empty_cache()
        if (i + 1) % 5 == 0:
            print(f"[{i+1}/{N_PROMPTS}] elapsed={(time.time()-t0)/60:.1f}min", flush=True)
    f.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
