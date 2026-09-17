"""One-off diagnostic (not part of the harness proper): is oracle_static's
degradation at low Phase-1 budgets a real "must-keep set exceeds budget"
effect, or a reoccurrence of the old nondeterministic-set-slicing bug?
Run twice with different PYTHONHASHSEED and diff the retained sets — if
they match exactly, the old bug is not back and the degradation is a
budget/recency-window sizing fact about the sweep, not a code defect.
"""
import json
import sys

from kvcache_harness.models import load_model_and_tokenizer
from kvcache_harness.engine import CacheEngine
from kvcache_harness.cache.base_policy import BudgetSpec
from kvcache_harness.cache.oracle_static import OracleStaticPolicy
from kvcache_harness.tasks.multi_credential import make_multi_credential_prompt, score_turn
from kvcache_harness.run_phase0r_calibration import build_turn_texts, compute_positions

BUDGETS = [51, 103, 154, 257, 514]
SEEDS = list(range(2000, 2010))  # calibration seeds
RECENCY_WINDOW = 64

model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct", device="cuda")

out = {}
for budget in BUDGETS:
    budget_spec = BudgetSpec(total_budget=budget, full_fraction=0.5)
    accs = []
    for seed in SEEDS:
        prompt = make_multi_credential_prompt(seed=seed, n_credentials=6, n_distractors=20,
                                               words_per_paragraph=50, value_len=14)
        turn_texts, body_offset = build_turn_texts(tokenizer, prompt)
        oracle_important, protected, protected_groups, cred_value_positions = compute_positions(
            tokenizer, turn_texts[0], body_offset, prompt, protect_after_chars=2
        )
        competitive = budget - RECENCY_WINDOW - 1  # minus sink
        policy = OracleStaticPolicy(budget_spec)
        engine = CacheEngine(model, tokenizer, policy, device="cuda")
        trace, turn_answers, _ = engine.generate_multi_turn(
            turn_texts, protected, oracle_important, max_answer_tokens=22, rebalance_every=1,
            eos_token_id=tokenizer.eos_token_id, recency_window=RECENCY_WINDOW, keep_sink=True,
        )
        n = len(prompt.credentials)
        correct = [score_turn(turn_answers[i], prompt.credentials[prompt.turn_order[i]]) for i in range(n)]
        frac = sum(correct) / n
        accs.append(frac)
        kept_sorted = sorted(policy._keep)
        out[f"{budget}_{seed}"] = {
            "frac": frac,
            "oracle_important_count": len(oracle_important),
            "competitive_budget": competitive,
            "truncated": len(oracle_important) > competitive,
            "kept_set_sig": kept_sorted,
        }
    print(f"budget={budget:4d} competitive={competitive:5d} oracle_important~{len(oracle_important):3d} "
          f"mean_acc={sum(accs)/len(accs):.3f}", flush=True)

with open(sys.argv[1], "w") as f:
    json.dump(out, f)
