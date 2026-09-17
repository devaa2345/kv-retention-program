"""Phase 1: mechanism decomposition, done properly. 2 (protection) x 2
(recoverability) x 5 budgets x 2 (iso-token / iso-memory), plus
oracle_static and full_cache_ref references at each budget, n>=150 prompts,
per PREREG.md. This is a long-running background job — writes incrementally
to a JSONL file and is resumable (re-running skips any (budget,
iso_condition, arm, seed) already present in the output file).

Run with:  python -m kvcache_harness.run_phase1 [config_path]
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

import torch
import yaml

from .models import load_model_and_tokenizer
from .engine import CacheEngine
from .cache.base_policy import BudgetSpec
from .cache.permanent_evict import PermanentEvictPolicy
from .cache.structural_protection import StructuralProtectionWrapper
from .cache.recoverable_tier import RecoverableTierPolicy
from .cache.oracle_static import OracleStaticPolicy
from .tasks.multi_credential import make_multi_credential_prompt, question_for, score_turn
from .run_phase0r_calibration import build_turn_texts, compute_positions

QUANT_BYTE_COST = 0.25   # accounted relative cost of a QUANT-tier slot vs a FULL slot (1.0);
                          # see PREREG.md / iso-memory derivation: this harness doesn't
                          # actually store quant tokens in a smaller dtype (they're
                          # dequantized back into the same tensor — see engine.py's
                          # _quant_dequant), so iso-memory parity is enforced at the
                          # accounting level, not measured from real GPU bytes.


def iso_memory_budget_tokens(permanent_budget_tokens: int, full_fraction: float = 0.5) -> int:
    """Token count T for a tiered arm such that its accounted byte cost
    (full_fraction*T*1.0 + (1-full_fraction)*T*QUANT_BYTE_COST) equals the
    permanent arm's byte cost (permanent_budget_tokens * 1.0)."""
    cost_per_token = full_fraction * 1.0 + (1 - full_fraction) * QUANT_BYTE_COST
    return round(permanent_budget_tokens / cost_per_token)


def load_done_keys(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    done.add((row["budget"], row["iso_condition"], row["arm"], row["seed"]))
                except Exception:
                    continue
    return done


def run_one(engine, turn_texts, protected, oracle_important, prompt, cfg):
    t0 = time.time()
    trace, turn_answers, turn_start_steps = engine.generate_multi_turn(
        turn_texts, protected, oracle_important,
        max_answer_tokens=cfg["max_answer_tokens"], rebalance_every=cfg["rebalance_every"],
        eos_token_id=engine.tokenizer.eos_token_id, recency_window=cfg["recency_window"],
        keep_sink=cfg["keep_sink"],
    )
    n = len(prompt.credentials)
    correct = [score_turn(turn_answers[i], prompt.credentials[prompt.turn_order[i]]) for i in range(n)]
    return {
        "frac_retrieved": sum(correct) / n,
        "per_credential_correct": correct,
        "wall_time_s": time.time() - t0,
        "n_evictions": sum(len(e.evictions) for e in trace.events),
        "n_promotions": sum(len(e.promotions) for e in trace.events),
        "n_demotions": sum(len(e.demotions) for e in trace.events),
    }


def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model, tokenizer = load_model_and_tokenizer(cfg["model_name"], device=cfg["device"])
    device = cfg["device"]

    os.makedirs(cfg["output_dir"], exist_ok=True)
    out_path = os.path.join(cfg["output_dir"], "raw_results.jsonl")
    done = load_done_keys(out_path)
    print(f"resuming: {len(done)} (budget,iso,arm,seed) cells already done", flush=True)

    out_f = open(out_path, "a")

    def write_row(budget, iso_condition, arm, seed, metrics):
        row = {"budget": budget, "iso_condition": iso_condition, "arm": arm, "seed": seed, **metrics}
        out_f.write(json.dumps(row) + "\n")
        out_f.flush()
        done.add((budget, iso_condition, arm, seed))

    budgets = cfg["budgets"]
    seeds = list(range(cfg["seed_start"], cfg["seed_start"] + cfg["n_prompts"]))
    full_fraction = cfg.get("full_fraction", 0.5)

    total_cells = len(budgets) * len(seeds) * (2 + 2 * 2 + 2)  # permanent(2) + recoverable(2)*iso(2) + refs(2)
    done_count = 0
    t_start = time.time()

    for seed in seeds:
        prompt = make_multi_credential_prompt(
            seed=seed, n_credentials=cfg["n_credentials"], n_distractors=cfg["n_distractors"],
            words_per_paragraph=cfg["words_per_paragraph"], value_len=cfg["value_len"],
        )
        turn_texts, body_offset = build_turn_texts(tokenizer, prompt)
        oracle_important, protected, protected_groups, cred_value_positions = compute_positions(
            tokenizer, turn_texts[0], body_offset, prompt, cfg["protect_after_chars"]
        )

        for budget in budgets:
            budget_spec = BudgetSpec(total_budget=budget, full_fraction=full_fraction)

            # --- permanent arms: identical under iso_token and iso_memory (only tiered arms
            # change based on iso condition) — compute ONCE, write both rows, rather than
            # re-running the same policy/budget twice for a label that doesn't affect it.
            for arm_name, policy_factory in [
                ("1_no_protect_permanent", lambda: PermanentEvictPolicy(budget_spec)),
                ("2_protect_permanent", lambda: StructuralProtectionWrapper(PermanentEvictPolicy(budget_spec), groups=protected_groups)),
            ]:
                needed_isos = [iso for iso in ("iso_token", "iso_memory") if (budget, iso, arm_name, seed) not in done]
                if needed_isos:
                    try:
                        engine = CacheEngine(model, tokenizer, policy_factory(), device=device)
                        metrics = run_one(engine, turn_texts, protected, oracle_important, prompt, cfg)
                        for iso in needed_isos:
                            write_row(budget, iso, arm_name, seed, metrics)
                    except Exception as e:
                        print(f"ERROR budget={budget} arm={arm_name} seed={seed}: {e}", flush=True)
                        traceback.print_exc()
                        torch.cuda.empty_cache()
                done_count += 2

            # --- references: iso-token only (not tiered, iso-memory is meaningless for them) ---
            ref_budget = BudgetSpec(total_budget=100_000, full_fraction=1.0)
            for arm_name, policy in [
                ("5_oracle_static", OracleStaticPolicy(budget_spec)),
                ("6_full_cache_ref", PermanentEvictPolicy(ref_budget)),
            ]:
                key = (budget, "iso_token", arm_name, seed)
                if key in done:
                    done_count += 1
                    continue
                try:
                    engine = CacheEngine(model, tokenizer, policy, device=device)
                    metrics = run_one(engine, turn_texts, protected, oracle_important, prompt, cfg)
                    write_row(budget, "iso_token", arm_name, seed, metrics)
                except Exception as e:
                    print(f"ERROR budget={budget} arm={arm_name} seed={seed}: {e}", flush=True)
                    traceback.print_exc()
                    torch.cuda.empty_cache()
                done_count += 1

            # --- recoverable (tiered) arms: budget depends on iso condition ---
            for iso in ("iso_token", "iso_memory"):
                tiered_budget_tokens = budget if iso == "iso_token" else iso_memory_budget_tokens(budget, full_fraction)
                tiered_spec = BudgetSpec(total_budget=tiered_budget_tokens, full_fraction=full_fraction)
                for arm_name, policy in [
                    ("3_recoverable_no_protection", RecoverableTierPolicy(tiered_spec)),
                    ("4_recoverable_protected", StructuralProtectionWrapper(RecoverableTierPolicy(tiered_spec), groups=protected_groups)),
                ]:
                    key = (budget, iso, arm_name, seed)
                    if key in done:
                        done_count += 1
                        continue
                    try:
                        engine = CacheEngine(model, tokenizer, policy, device=device)
                        metrics = run_one(engine, turn_texts, protected, oracle_important, prompt, cfg)
                        write_row(budget, iso, arm_name, seed, metrics)
                    except Exception as e:
                        print(f"ERROR budget={budget} iso={iso} arm={arm_name} seed={seed}: {e}", flush=True)
                        traceback.print_exc()
                        torch.cuda.empty_cache()
                    done_count += 1

        elapsed = time.time() - t_start
        print(f"[seed {seed} done] cells_done~{done_count}/{total_cells} elapsed={elapsed/60:.1f}min "
              f"({len(done)} rows on disk)", flush=True)

    out_f.close()
    print("Phase 1 run complete.", flush=True)


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "configs", "phase1.yaml"
    )
    main(cfg_path)
