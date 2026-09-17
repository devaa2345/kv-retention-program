"""Phase 0-R calibration sweep: tune (N, budget, context length, distractor
count) until gates G1-G5 pass, at n=10 per candidate config. Nothing here
is a result — this is instrument validation, per the test-program design.

Run with:  python -m kvcache_harness.run_phase0r_calibration [config_path]
"""

from __future__ import annotations

import json
import os
import sys
from statistics import mean

import yaml

from .models import load_model_and_tokenizer
from .engine import CacheEngine
from .cache.base_policy import BudgetSpec
from .cache.permanent_evict import PermanentEvictPolicy
from .cache.structural_protection import StructuralProtectionWrapper
from .cache.oracle_static import OracleStaticPolicy
from .tasks.multi_credential import (
    make_multi_credential_prompt, question_for, score_turn, find_structural_label_spans,
)
from .tasks.credential_retrieval import char_spans_to_token_positions, char_spans_to_token_groups


def build_turn_texts(tokenizer, prompt):
    n = len(prompt.credentials)
    first_q = question_for(prompt, prompt.turn_order[0])
    first_content = prompt.dump_text + "\n\n" + first_q
    messages = [{"role": "user", "content": first_content}]
    turn0_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    body_offset = turn0_text.find(prompt.dump_text)
    assert body_offset >= 0

    turn_texts = [turn0_text]
    for i in range(1, n):
        q = question_for(prompt, prompt.turn_order[i])
        turn_texts.append(f"<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n")
    return turn_texts, body_offset


def compute_positions(tokenizer, turn0_text, body_offset, prompt, protect_after_chars):
    enc = tokenizer(turn0_text, return_offsets_mapping=True, add_special_tokens=True)
    offset_mapping = enc["offset_mapping"]

    oracle_important = set()
    cred_value_positions = {}   # credential_idx -> token positions (for G5)
    for i, cred in enumerate(prompt.credentials):
        # oracle needs the LABEL as well as the bare value — without the
        # "CRED_i_KEY:" text anchoring it, a floating hex span has no way
        # to tell the model which credential it belongs to. This was
        # silently wrong in the first calibration pass (oracle_important
        # was value-only) and tanked oracle_static's score well below its
        # theoretical ceiling.
        label_span = (cred.label_char_span[0] + body_offset, cred.label_char_span[1] + body_offset)
        value_span = (cred.value_char_span[0] + body_offset, cred.value_char_span[1] + body_offset)
        pos = char_spans_to_token_positions(offset_mapping, [label_span, value_span], protect_after_chars=0)
        oracle_important |= pos
        cred_value_positions[i] = char_spans_to_token_positions(offset_mapping, [value_span], protect_after_chars=0)

    label_spans = [(s + body_offset, e + body_offset) for (s, e) in find_structural_label_spans(prompt.dump_text)]
    protected = char_spans_to_token_positions(offset_mapping, label_spans, protect_after_chars=protect_after_chars)
    protected_groups = char_spans_to_token_groups(offset_mapping, label_spans, protect_after_chars=protect_after_chars)

    return oracle_important, protected, protected_groups, cred_value_positions


def dormancy_eviction_hit(events, turn_start_steps, prompt, cred_value_positions):
    """G5 proxy: did >=1 credential have any of its value tokens evicted or
    demoted before its own turn started?"""
    evicted_or_demoted_before = {}
    for e in events:
        for p in e.evictions + e.demotions:
            evicted_or_demoted_before.setdefault(p, []).append(e.step)

    hit = False
    for cred_idx, positions in cred_value_positions.items():
        rank = prompt.turn_order.index(cred_idx)
        own_start = turn_start_steps[rank]
        for p in positions:
            if any(s < own_start for s in evicted_or_demoted_before.get(p, [])):
                hit = True
                break
        if hit:
            break
    return hit


def run_config(model, tokenizer, cfg, device):
    budget = BudgetSpec(total_budget=cfg["total_budget"], full_fraction=cfg.get("full_fraction", 0.5))
    ref_budget = BudgetSpec(total_budget=100_000, full_fraction=1.0)

    arm_scores = {"1_no_protect_permanent": [], "2_protect_permanent": [],
                  "5_oracle_static": [], "6_full_cache_ref": []}
    dormancy_hits = []

    for i in range(cfg["n_prompts"]):
        seed = cfg["seed_start"] + i
        prompt = make_multi_credential_prompt(
            seed=seed, n_credentials=cfg["n_credentials"], n_distractors=cfg["n_distractors"],
            words_per_paragraph=cfg["words_per_paragraph"], value_len=cfg["value_len"],
        )
        turn_texts, body_offset = build_turn_texts(tokenizer, prompt)
        oracle_important, protected, protected_groups, cred_value_positions = compute_positions(
            tokenizer, turn_texts[0], body_offset, prompt, cfg["protect_after_chars"]
        )

        arms = {
            "1_no_protect_permanent": PermanentEvictPolicy(budget),
            "2_protect_permanent": StructuralProtectionWrapper(PermanentEvictPolicy(budget), groups=protected_groups),
            "5_oracle_static": OracleStaticPolicy(budget),
            "6_full_cache_ref": PermanentEvictPolicy(ref_budget),
        }

        for arm_name, policy in arms.items():
            engine = CacheEngine(model, tokenizer, policy, device=device)
            trace, turn_answers, turn_start_steps = engine.generate_multi_turn(
                turn_texts, protected, oracle_important,
                max_answer_tokens=cfg["max_answer_tokens"], rebalance_every=cfg["rebalance_every"],
                eos_token_id=tokenizer.eos_token_id, recency_window=cfg["recency_window"],
                keep_sink=cfg["keep_sink"],
            )
            n = len(prompt.credentials)
            correct = [score_turn(turn_answers[i], prompt.credentials[prompt.turn_order[i]]) for i in range(n)]
            frac = sum(correct) / n
            arm_scores[arm_name].append(frac)

            if arm_name == "1_no_protect_permanent":
                dormancy_hits.append(dormancy_eviction_hit(trace.events, turn_start_steps, prompt, cred_value_positions))

        print(f"[seed {seed}] " + " ".join(f"{k}={v[-1]:.2f}" for k, v in arm_scores.items()), flush=True)

    summary = {k: mean(v) for k, v in arm_scores.items()}
    summary["G5_dormancy_eviction_rate"] = sum(dormancy_hits) / len(dormancy_hits)
    return summary


def check_gates(summary):
    g1 = summary["5_oracle_static"] >= 0.95
    g2 = summary["1_no_protect_permanent"] <= 0.10
    g3 = 0.45 <= summary["2_protect_permanent"] <= 0.75
    g4 = summary["6_full_cache_ref"] >= 0.95
    g5 = summary["G5_dormancy_eviction_rate"] >= 0.80
    return {"G1_ceiling": g1, "G2_floor": g2, "G3_headroom": g3, "G4_no_eviction_ref": g4, "G5_dormancy": g5}


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "configs", "phase0r_calibration.yaml"
    )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model, tokenizer = load_model_and_tokenizer(cfg["model_name"], device=cfg["device"])
    summary = run_config(model, tokenizer, cfg, cfg["device"])
    gates = check_gates(summary)

    print("\n--- summary ---")
    for k, v in summary.items():
        print(f"{k}: {v:.3f}")
    print("\n--- gates ---")
    for k, v in gates.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(os.path.join(cfg["output_dir"], "calibration_summary.json"), "w") as f:
        json.dump({"config": cfg, "summary": summary, "gates": gates}, f, indent=2)
