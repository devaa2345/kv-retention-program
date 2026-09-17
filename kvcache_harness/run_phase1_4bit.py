"""Phase 1 at 4-bit, iso-token (item 3 of the cross-implementation audit).

Our Phase 1 ran at 8-bit only and the spec never stated it — our defect. At
8-bit the quantized tier is a no-op (ADDENDUM §11), so the tiered arms were
being compared in a regime where the cold tier cost nothing. At 4-bit it
demonstrably costs something, and the independent RX 7900 XTX
reimplementation reports a *negative* interaction there that grows with
budget (−0.046 at 257, −0.116 at 514).

Arms 1 and 2 are permanent eviction with no quantized tier, so they are
bit-width invariant and are REUSED from results/phase1/ rather than re-run.
Only arms 3 and 4 (and the tiered-relevant references) are executed here.

Run: python -m kvcache_harness.run_phase1_4bit
"""
from __future__ import annotations

import json
import os
import time
import traceback

import torch

from . import engine as engine_mod
from .models import load_model_and_tokenizer
from .engine import CacheEngine
from .cache.base_policy import BudgetSpec
from .cache.recoverable_tier import RecoverableTierPolicy
from .cache.structural_protection import StructuralProtectionWrapper
from .cache.oracle_static import OracleStaticPolicy
from .cache.permanent_evict import PermanentEvictPolicy
from .tasks.multi_credential import make_multi_credential_prompt, score_turn
from .run_phase0r_calibration import build_turn_texts, compute_positions

QUANT_BITS_FOR_RUN = 4
BUDGETS = [154, 257, 514]
N_PROMPTS = 150
SEED_START = 3000
FULL_FRACTION = 0.5
OUT = "results/phase1_4bit/raw_results.jsonl"

CFG = dict(n_credentials=6, value_len=14, n_distractors=20, words_per_paragraph=50,
           protect_after_chars=2, max_answer_tokens=22, rebalance_every=1,
           recency_window=64, keep_sink=True)


def load_done():
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
                done.add((r["budget"], r["arm"], r["seed"]))
            except Exception:
                pass
    return done


def main():
    engine_mod.QUANT_BITS = QUANT_BITS_FOR_RUN
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = load_done()
    print(f"quant_bits={QUANT_BITS_FOR_RUN}, iso-token; resuming: {len(done)} cells done", flush=True)

    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct", device="cuda")
    f = open(OUT, "a")
    t0 = time.time()

    for i in range(N_PROMPTS):
        seed = SEED_START + i
        prompt = make_multi_credential_prompt(
            seed=seed, n_credentials=CFG["n_credentials"], n_distractors=CFG["n_distractors"],
            words_per_paragraph=CFG["words_per_paragraph"], value_len=CFG["value_len"])
        turn_texts, body_offset = build_turn_texts(tokenizer, prompt)
        oracle_important, protected, groups, _ = compute_positions(
            tokenizer, turn_texts[0], body_offset, prompt, CFG["protect_after_chars"])

        for budget in BUDGETS:
            spec = BudgetSpec(total_budget=budget, full_fraction=FULL_FRACTION)
            ref_spec = BudgetSpec(total_budget=100_000, full_fraction=1.0)
            arms = {
                # arms 1 and 2 are bit-invariant and reused from results/phase1/ — not run here
                "3_recoverable_no_protection": lambda: RecoverableTierPolicy(spec),
                "4_recoverable_protected": lambda: StructuralProtectionWrapper(
                    RecoverableTierPolicy(spec), groups=groups),
                "5_oracle_static": lambda: OracleStaticPolicy(spec),
                "6_full_cache_ref": lambda: PermanentEvictPolicy(ref_spec),
            }
            for arm_name, factory in arms.items():
                if (budget, arm_name, seed) in done:
                    continue
                try:
                    engine = CacheEngine(model, tokenizer, factory(), device="cuda")
                    trace, answers, _ = engine.generate_multi_turn(
                        turn_texts, protected, oracle_important,
                        max_answer_tokens=CFG["max_answer_tokens"],
                        rebalance_every=CFG["rebalance_every"],
                        eos_token_id=tokenizer.eos_token_id,
                        recency_window=CFG["recency_window"], keep_sink=CFG["keep_sink"])
                    n = len(prompt.credentials)
                    correct = [score_turn(answers[j], prompt.credentials[prompt.turn_order[j]])
                               for j in range(n)]
                    f.write(json.dumps({
                        "quant_bits": QUANT_BITS_FOR_RUN, "iso_condition": "iso_token",
                        "budget": budget, "arm": arm_name, "seed": seed,
                        "frac_retrieved": sum(correct) / n,
                        "per_credential_correct": correct,
                        "n_evictions": sum(len(e.evictions) for e in trace.events),
                        "n_promotions": sum(len(e.promotions) for e in trace.events),
                        "n_demotions": sum(len(e.demotions) for e in trace.events),
                    }) + "\n")
                    f.flush()
                except Exception as e:
                    print(f"ERROR b={budget} arm={arm_name} seed={seed}: {e}", flush=True)
                    traceback.print_exc()
                    torch.cuda.empty_cache()

        if (i + 1) % 5 == 0:
            print(f"[{i+1}/{N_PROMPTS}] elapsed={(time.time()-t0)/60:.1f}min", flush=True)

    f.close()
    engine_mod.QUANT_BITS = 8
    print("Phase 1 @ 4-bit complete", flush=True)


if __name__ == "__main__":
    main()
