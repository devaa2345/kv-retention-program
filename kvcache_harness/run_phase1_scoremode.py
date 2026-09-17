"""Does switching attention accumulation from sum to mean move our headline
numbers? (Blocking open item from the cross-implementation audit, §2.2.)

Our retention ranking accumulates raw attention mass summed over queries,
which carries a -0.717 correlation with position: early tokens outrank later
ones for having existed longer. Mean-normalising by the number of attending
queries removes that. The two readings are near-uncorrelated (rho = +0.014)
and share only 13.5% of the retained set, so this could change everything or
nothing and reasoning cannot settle which.

Runs arms 1-4 at 8-bit, iso-token, budgets 154/257/514, n=150, under
SCORE_MODE="mean". Arms 5 (oracle_static, ground-truth keep-set) and 6
(full_cache_ref, never evicts) are attention-independent and therefore
invariant under score mode, so they are not re-run.

Begins with a REGRESSION CHECK: under SCORE_MODE="sum" the patched engine
must still reproduce the shipped 8-bit numbers, otherwise the mean results
are uninterpretable.

Run: python -m kvcache_harness.run_phase1_scoremode
"""
from __future__ import annotations

import json
import os
import time
import traceback
from statistics import mean

import torch

from . import engine as engine_mod
from .models import load_model_and_tokenizer
from .engine import CacheEngine
from .cache.base_policy import BudgetSpec
from .cache.permanent_evict import PermanentEvictPolicy
from .cache.recoverable_tier import RecoverableTierPolicy
from .cache.structural_protection import StructuralProtectionWrapper
from .tasks.multi_credential import make_multi_credential_prompt, score_turn
from .run_phase0r_calibration import build_turn_texts, compute_positions

BUDGETS = [154, 257, 514]
N_PROMPTS = 150
SEED_START = 3000
FULL_FRACTION = 0.5
OUT = "results/phase1_meanscore/raw_results.jsonl"

CFG = dict(n_credentials=6, value_len=14, n_distractors=20, words_per_paragraph=50,
           protect_after_chars=2, max_answer_tokens=22, rebalance_every=1,
           recency_window=64, keep_sink=True)

# shipped 8-bit sum-mode values the regression check must reproduce
SHIPPED = {(257, "1_no_protect_permanent"): 0.010, (257, "2_protect_permanent"): 0.423}


def build_arms(spec, groups):
    return {
        "1_no_protect_permanent": lambda: PermanentEvictPolicy(spec),
        "2_protect_permanent": lambda: StructuralProtectionWrapper(
            PermanentEvictPolicy(spec), groups=groups),
        "3_recoverable_no_protection": lambda: RecoverableTierPolicy(spec),
        "4_recoverable_protected": lambda: StructuralProtectionWrapper(
            RecoverableTierPolicy(spec), groups=groups),
    }


def one_run(model, tok, seed, budget, arm_name):
    prompt = make_multi_credential_prompt(
        seed=seed, n_credentials=CFG["n_credentials"], n_distractors=CFG["n_distractors"],
        words_per_paragraph=CFG["words_per_paragraph"], value_len=CFG["value_len"])
    turn_texts, body_offset = build_turn_texts(tok, prompt)
    oracle_important, protected, groups, _ = compute_positions(
        tok, turn_texts[0], body_offset, prompt, CFG["protect_after_chars"])
    spec = BudgetSpec(total_budget=budget, full_fraction=FULL_FRACTION)
    policy = build_arms(spec, groups)[arm_name]()
    eng = CacheEngine(model, tok, policy, device="cuda")
    _, answers, _ = eng.generate_multi_turn(
        turn_texts, protected, oracle_important,
        max_answer_tokens=CFG["max_answer_tokens"], rebalance_every=CFG["rebalance_every"],
        eos_token_id=tok.eos_token_id, recency_window=CFG["recency_window"],
        keep_sink=CFG["keep_sink"])
    n = len(prompt.credentials)
    correct = [score_turn(answers[j], prompt.credentials[prompt.turn_order[j]]) for j in range(n)]
    return sum(correct) / n, correct


def main():
    model, tok = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct", device="cuda")

    # ---- regression check: sum mode must still reproduce shipped values ----
    engine_mod.SCORE_MODE = "sum"
    print("REGRESSION CHECK (SCORE_MODE='sum', 25 seeds, budget 257):", flush=True)
    ok = True
    for arm, expected in [("1_no_protect_permanent", SHIPPED[(257, "1_no_protect_permanent")]),
                           ("2_protect_permanent", SHIPPED[(257, "2_protect_permanent")])]:
        got = mean(one_run(model, tok, s, 257, arm)[0] for s in range(3000, 3025))
        delta = got - expected
        flag = "OK" if abs(delta) < 0.08 else "*** DRIFT ***"
        if abs(delta) >= 0.08:
            ok = False
        print(f"  {arm:28s} shipped={expected:.3f} now={got:.3f} delta={delta:+.3f}  {flag}", flush=True)
    if not ok:
        print("Regression check FAILED — patch changed sum behaviour. Aborting.", flush=True)
        return
    print("  regression clean; proceeding to mean run\n", flush=True)

    # ---- the mean run ----
    engine_mod.SCORE_MODE = "mean"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
                done.add((r["budget"], r["arm"], r["seed"]))
            except Exception:
                pass
    print(f"SCORE_MODE=mean; resuming: {len(done)} cells done", flush=True)

    f = open(OUT, "a")
    t0 = time.time()
    for i in range(N_PROMPTS):
        seed = SEED_START + i
        for budget in BUDGETS:
            for arm_name in ["1_no_protect_permanent", "2_protect_permanent",
                              "3_recoverable_no_protection", "4_recoverable_protected"]:
                if (budget, arm_name, seed) in done:
                    continue
                try:
                    frac, correct = one_run(model, tok, seed, budget, arm_name)
                    f.write(json.dumps({
                        "score_mode": "mean", "quant_bits": 8, "iso_condition": "iso_token",
                        "budget": budget, "arm": arm_name, "seed": seed,
                        "frac_retrieved": frac, "per_credential_correct": correct,
                    }) + "\n")
                    f.flush()
                except Exception as e:
                    print(f"ERROR b={budget} arm={arm_name} seed={seed}: {e}", flush=True)
                    traceback.print_exc()
                    torch.cuda.empty_cache()
        if (i + 1) % 5 == 0:
            print(f"[{i+1}/{N_PROMPTS}] elapsed={(time.time()-t0)/60:.1f}min", flush=True)

    f.close()
    engine_mod.SCORE_MODE = "sum"
    print("mean-score Phase 1 complete", flush=True)


if __name__ == "__main__":
    main()
