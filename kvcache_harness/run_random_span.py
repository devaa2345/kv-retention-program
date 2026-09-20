"""Random-span control (Paper 1, Option B item 1).

Question: does structural protection identify genuinely useful positions, or does it just
reserve capacity? Arm 7 keeps the mechanism of arm 2 (StructuralProtectionWrapper over the same
PermanentEvictPolicy, same budget, atomic groups seated by best-member attention) and changes
ONLY which positions are protected: the protected set is a set of random spans placed uniformly
in the dump text, one random contiguous token span per structural group, each with exactly that group's token
count. So the protected token count and the group-size distribution match arm 2
by construction, but the choice of WHAT to protect carries no information about credentials.

Everything else (task, seeds 3000-3149, engine, sum score mode, recency window, sink, budgets)
is identical to Phase 1. Arms 1, 2 and 5 are NOT re-run for the main contrast: their rows in
results/phase1/raw_results.jsonl are the reference (permanent arms are identical under both
iso conditions). A small determinism check re-runs arm 2 and compares.

Only VALID/PARTIAL budgets per ADDENDUM_2026-09-03.md: 154, 257, 514.

Run: python -m kvcache_harness.run_random_span [--check-determinism]
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback

import torch

from .models import load_model_and_tokenizer
from .engine import CacheEngine
from .cache.base_policy import BudgetSpec
from .cache.permanent_evict import PermanentEvictPolicy
from .cache.structural_protection import StructuralProtectionWrapper
from .tasks.multi_credential import make_multi_credential_prompt, score_turn, find_structural_label_spans
from .tasks.credential_retrieval import char_spans_to_token_groups
from .run_phase0r_calibration import build_turn_texts, compute_positions

BUDGETS = [154, 257, 514]
N_PROMPTS = 150
SEED_START = 3000
OUT = "results/random_span/raw_results.jsonl"
CFG = dict(n_credentials=6, value_len=14, n_distractors=20, words_per_paragraph=50,
           protect_after_chars=2, max_answer_tokens=22, rebalance_every=1,
           recency_window=64, keep_sink=True)


def random_token_spans_like(n_body_tokens: int, lengths: list, rng: random.Random) -> list:
    """One random contiguous token span per entry of `lengths` (token counts), non-overlapping,
    uniformly placed over the body tokens [0, n_body_tokens). Returns [(start, length), ...]."""
    spans = []
    for L in sorted(lengths, reverse=True):          # place long spans first so rejection stays cheap
        for _ in range(10_000):
            s = rng.randint(0, n_body_tokens - L)
            if all(s + L <= a or s >= a + b for a, b in spans):
                spans.append((s, L))
                break
        else:
            raise RuntimeError("could not place a non-overlapping random span")
    return sorted(spans)


def build_random_protection(tokenizer, turn0_text, body_offset, prompt, struct_groups, seed):
    """Random-span protection matched to structural protection in TOKEN space: one random
    contiguous span per structural group, with that group's exact token count. (Matching in
    character space does not work: random prose spans tokenize to far fewer tokens per
    character than hex-value lines, giving 132 vs 310 protected tokens.)"""
    om = tokenizer(turn0_text, return_offsets_mapping=True, add_special_tokens=True)["offset_mapping"]
    lo, hi = body_offset, body_offset + len(prompt.dump_text)
    body_tokens = [i for i, (a, b) in enumerate(om) if b > a and a >= lo and b <= hi]
    first = body_tokens[0]
    assert body_tokens == list(range(first, first + len(body_tokens)))      # contiguous body block
    sizes = {}
    for gid in struct_groups.values():
        sizes[gid] = sizes.get(gid, 0) + 1
    rng = random.Random(10_000_000 + seed)           # independent of the task's own stream
    spans = random_token_spans_like(len(body_tokens), sorted(sizes.values()), rng)
    groups = {}
    for gid, (st, L) in enumerate(spans):
        for k in range(L):
            groups[first + st + k] = gid
    return set(groups), groups


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


def run_arm(engine, turn_texts, protected, oracle_important, prompt):
    trace, answers, _ = engine.generate_multi_turn(
        turn_texts, protected, oracle_important,
        max_answer_tokens=CFG["max_answer_tokens"], rebalance_every=CFG["rebalance_every"],
        eos_token_id=engine.tokenizer.eos_token_id, recency_window=CFG["recency_window"],
        keep_sink=CFG["keep_sink"])
    n = len(prompt.credentials)
    correct = [score_turn(answers[i], prompt.credentials[prompt.turn_order[i]]) for i in range(n)]
    return sum(correct) / n, correct


def main():
    global OUT
    check = "--check-determinism" in sys.argv
    out = OUT if not check else "results/random_span/determinism_check.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    OUT = out
    done = load_done()
    print(f"resuming: {len(done)} cells done -> {OUT}", flush=True)
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct", device="cuda")
    f = open(OUT, "a")
    t0 = time.time()
    n_prompts = 25 if check else N_PROMPTS
    budgets = [257] if check else BUDGETS
    arm_name = "2_protect_permanent_rerun" if check else "7_random_span_permanent"

    for i in range(n_prompts):
        seed = SEED_START + i
        prompt = make_multi_credential_prompt(
            seed=seed, n_credentials=CFG["n_credentials"], n_distractors=CFG["n_distractors"],
            words_per_paragraph=CFG["words_per_paragraph"], value_len=CFG["value_len"])
        turn_texts, body_offset = build_turn_texts(tokenizer, prompt)
        oracle_important, s_protected, s_groups, _ = compute_positions(
            tokenizer, turn_texts[0], body_offset, prompt, CFG["protect_after_chars"])
        r_protected, r_groups = build_random_protection(
            tokenizer, turn_texts[0], body_offset, prompt, s_groups, seed)
        prot, groups = (s_protected, s_groups) if check else (r_protected, r_groups)
        cred_cov = len(oracle_important & prot) / max(1, len(oracle_important))

        for budget in budgets:
            if (budget, arm_name, seed) in done:
                continue
            try:
                spec = BudgetSpec(total_budget=budget, full_fraction=0.5)
                policy = StructuralProtectionWrapper(PermanentEvictPolicy(spec), groups=groups)
                engine = CacheEngine(model, tokenizer, policy, device="cuda")
                frac, correct = run_arm(engine, turn_texts, prot, oracle_important, prompt)
                f.write(json.dumps({
                    "budget": budget, "iso_condition": "iso_token", "arm": arm_name, "seed": seed,
                    "frac_retrieved": frac, "per_credential_correct": correct,
                    "n_protected_tokens": len(prot), "n_structural_protected_tokens": len(s_protected),
                    "n_random_protected_groups": len(set(r_groups.values())),
                    "credential_token_coverage": cred_cov,
                }) + "\n")
                f.flush()
            except Exception as e:
                print(f"ERROR budget={budget} seed={seed}: {e}", flush=True)
                traceback.print_exc()
                torch.cuda.empty_cache()
        if (i + 1) % 5 == 0:
            print(f"[{i+1}/{n_prompts}] elapsed={(time.time()-t0)/60:.1f}min", flush=True)
    f.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
