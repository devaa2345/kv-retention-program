"""Phase 0 pilot driver: 5 arms x N synthetic credential-retrieval prompts,
matched budget, on one small model. Run with:

    python -m kvcache_harness.run_phase0_pilot [config_path]

Validates the two known results first (arm 1 ~0%, arm 2 ~100%) before the
new cells (arms 3-5) are trusted — see plan's verification section.
"""

from __future__ import annotations

import json
import os
import sys
from statistics import mean

import torch
import yaml

from .models import load_model_and_tokenizer
from .engine import CacheEngine
from .metrics import global_lir, reference_full_attention, future_missed_mass, cost_summary
from .cache.base_policy import BudgetSpec
from .cache.permanent_evict import PermanentEvictPolicy
from .cache.structural_protection import StructuralProtectionWrapper
from .cache.recoverable_tier import RecoverableTierPolicy
from .cache.oracle_static import OracleStaticPolicy
from .tasks.credential_retrieval import (
    make_credential_prompt, score_response, find_structural_label_spans,
    char_spans_to_token_positions, build_chat_prompt,
)


def build_arms(budget: BudgetSpec):
    return {
        "1_permanent_no_protection": PermanentEvictPolicy(budget),
        "2_permanent_protected": StructuralProtectionWrapper(PermanentEvictPolicy(budget)),
        "3_recoverable_no_protection": RecoverableTierPolicy(budget),
        "4_recoverable_protected": StructuralProtectionWrapper(RecoverableTierPolicy(budget)),
        "5_oracle_static": OracleStaticPolicy(budget),
    }


def eviction_step_to_abs_positions(events, prompt_len: int) -> dict:
    out = {}
    for e in events:
        if not e.evictions:
            continue
        t_evict_abs = (prompt_len - 1) if e.step == 0 else (prompt_len + e.step - 1)
        out.setdefault(t_evict_abs, []).extend(e.evictions)
    return out


def run(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model, tokenizer = load_model_and_tokenizer(cfg["model_name"], device=cfg["device"])
    device = cfg["device"]
    budget = BudgetSpec(total_budget=cfg["total_budget"], full_fraction=cfg["full_fraction"])

    per_arm_results = {name: [] for name in build_arms(budget).keys()}

    for i in range(cfg["n_prompts"]):
        seed = cfg["seed_start"] + i
        prompt = make_credential_prompt(
            seed=seed,
            n_distractors=cfg["n_distractors"],
            words_per_paragraph=cfg["words_per_paragraph"],
        )
        prompt_ids, offset_mapping, body_offset = build_chat_prompt(tokenizer, prompt)

        shifted_target_span = (prompt.target_char_span[0] + body_offset, prompt.target_char_span[1] + body_offset)
        oracle_important = char_spans_to_token_positions(offset_mapping, [shifted_target_span], protect_after_chars=0)
        label_spans = [(s + body_offset, e + body_offset) for (s, e) in find_structural_label_spans(prompt.text)]
        protected = char_spans_to_token_positions(offset_mapping, label_spans, protect_after_chars=cfg["protect_after_chars"])

        print(f"[prompt {i}] seed={seed} prompt_len={prompt_ids.shape[1]} "
              f"oracle_tokens={len(oracle_important)} protected_tokens={len(protected)}", flush=True)

        arms = build_arms(budget)
        for arm_name, policy in arms.items():
            engine = CacheEngine(model, tokenizer, policy, device=device)
            trace = engine.generate(
                prompt_ids=prompt_ids,
                max_new_tokens=cfg["max_new_tokens"],
                protected=protected,
                oracle_important=oracle_important,
                rebalance_every=cfg["rebalance_every"],
                eos_token_id=tokenizer.eos_token_id,
                recency_window=cfg.get("recency_window", 8),
                keep_sink=cfg.get("keep_sink", True),
            )
            correct = score_response(trace.generated_text, prompt.target_value)

            lir = global_lir(trace.events)
            full_ids = prompt_ids[0].tolist() + trace.generated_token_ids
            try:
                ref_attn = reference_full_attention(model, tokenizer, full_ids, device)
                ev_map = eviction_step_to_abs_positions(trace.events, trace.prompt_len)
                fmm = future_missed_mass(ev_map, ref_attn, trace.prompt_len)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                fmm = None

            cost = cost_summary(trace.events)

            per_arm_results[arm_name].append({
                "seed": seed,
                "correct": correct,
                "global_lir": lir,
                "fmm": fmm,
                "generated_text": trace.generated_text,
                **cost,
            })
            print(f"  {arm_name:30s} correct={correct} lir={lir:.3f} fmm={fmm} "
                  f"n_evict={cost['n_evictions']} n_promote={cost['n_promotions']}", flush=True)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    out_path = os.path.join(cfg["output_dir"], "raw_results.json")
    with open(out_path, "w") as f:
        json.dump(per_arm_results, f, indent=2)

    summary_lines = ["| arm | accuracy | mean_global_lir | mean_fmm | mean_step_time_s |",
                      "|---|---|---|---|---|"]
    for arm_name, rows in per_arm_results.items():
        acc = mean(r["correct"] for r in rows)
        lir = mean(r["global_lir"] for r in rows)
        fmm_vals = [r["fmm"] for r in rows if r["fmm"] is not None]
        fmm_mean = mean(fmm_vals) if fmm_vals else float("nan")
        step_time = mean(r["mean_step_time_s"] for r in rows)
        summary_lines.append(f"| {arm_name} | {acc:.1%} | {lir:.3f} | {fmm_mean:.3f} | {step_time*1000:.1f} ms |")

    summary = "\n".join(summary_lines)
    print("\n" + summary)
    with open(os.path.join(cfg["output_dir"], "summary.md"), "w") as f:
        f.write(summary + "\n")


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "configs", "phase0_pilot.yaml"
    )
    run(cfg_path)
