"""QUEUE 1 — decompose the −0.683 interaction at budget 514, 4-bit.

Is the effect information loss, or generation collapse? Compares protection
alone (arm 2, permanent eviction, no cold tier) against protection+tiered
(arm 4) and classifies every answer into mutually exclusive buckets, then
computes accuracy CONDITIONAL on the output being non-degenerate.

If overall accuracy is 0.281 but conditional accuracy is high, the claim
cannot be "recoverable tiering costs 68 points of retrieval quality" — it
becomes "at this configuration the tiered system degenerates, so the
experiment cannot isolate recoverability from quantization-induced
generation failure."

Run: python -m kvcache_harness.q1_decompose_514
"""
from __future__ import annotations

import json
import os
import re
from statistics import mean

from . import engine as engine_mod
from .models import load_model_and_tokenizer
from .engine import CacheEngine
from .cache.base_policy import BudgetSpec
from .cache.permanent_evict import PermanentEvictPolicy
from .cache.recoverable_tier import RecoverableTierPolicy
from .cache.structural_protection import StructuralProtectionWrapper
from .tasks.multi_credential import make_multi_credential_prompt, score_turn
from .run_phase0r_calibration import build_turn_texts, compute_positions
from .run_bitwidth_degeneracy import WELL_FORMED, repetition_run

BUDGET = 514
SEEDS = list(range(3000, 3050))
OUT = "results/q1_decompose_514/answers.jsonl"
CFG = dict(n_credentials=6, value_len=14, n_distractors=20, words_per_paragraph=50,
           protect_after_chars=2, max_answer_tokens=22, rebalance_every=1,
           recency_window=64, keep_sink=True)


def classify(ans, correct):
    empty = not ans.strip()
    malformed = not bool(WELL_FORMED.match(ans))
    repetitive = repetition_run(ans)
    degenerate = malformed or repetitive
    if empty:
        return "empty"
    if degenerate:
        return "malformed"
    return "normal_correct" if correct else "normal_wrong"


def main():
    engine_mod.QUANT_BITS = 4
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    model, tok = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct", device="cuda")
    f = open(OUT, "w")

    arms = {
        "2_protect_permanent": lambda spec, g: StructuralProtectionWrapper(
            PermanentEvictPolicy(spec), groups=g),
        "4_recoverable_protected": lambda spec, g: StructuralProtectionWrapper(
            RecoverableTierPolicy(spec), groups=g),
    }

    rows = []
    for seed in SEEDS:
        p = make_multi_credential_prompt(seed=seed, n_credentials=CFG["n_credentials"],
                                          n_distractors=CFG["n_distractors"],
                                          words_per_paragraph=CFG["words_per_paragraph"],
                                          value_len=CFG["value_len"])
        tt, bo = build_turn_texts(tok, p)
        oi, prot, grp, _ = compute_positions(tok, tt[0], bo, p, CFG["protect_after_chars"])
        spec = BudgetSpec(total_budget=BUDGET, full_fraction=0.5)
        for arm, factory in arms.items():
            eng = CacheEngine(model, tok, factory(spec, grp), device="cuda")
            _, answers, _ = eng.generate_multi_turn(
                tt, prot, oi, max_answer_tokens=CFG["max_answer_tokens"],
                rebalance_every=CFG["rebalance_every"], eos_token_id=tok.eos_token_id,
                recency_window=CFG["recency_window"], keep_sink=CFG["keep_sink"])
            firsts = getattr(eng, "first_answer_tokens", [])
            for j, a in enumerate(answers):
                c = p.credentials[p.turn_order[j]]
                ok = bool(score_turn(a, c))
                row = {"arm": arm, "seed": seed, "turn": j, "answer": a, "correct": ok,
                        "bucket": classify(a, ok),
                        "first_tok_eos": (firsts[j] == tok.eos_token_id) if j < len(firsts) else None}
                rows.append(row)
                f.write(json.dumps(row) + "\n")
            f.flush()
    f.close()
    engine_mod.QUANT_BITS = 8

    print(f"\nBudget {BUDGET}, 4-bit, n={len(SEEDS)} prompts x 6 turns\n")
    hdr = (f"{'arm':>26s} {'acc':>7s} {'empty':>7s} {'EOS@0':>7s} {'malformed':>10s} "
           f"{'norm_wrong':>11s} {'norm_ok':>8s} {'cond_acc':>9s}")
    print(hdr)
    for arm in arms:
        sub = [r for r in rows if r["arm"] == arm]
        n = len(sub)
        buckets = {b: sum(1 for r in sub if r["bucket"] == b) / n
                   for b in ("empty", "malformed", "normal_wrong", "normal_correct")}
        eos = mean(1 if r["first_tok_eos"] else 0 for r in sub if r["first_tok_eos"] is not None)
        nondegen = [r for r in sub if r["bucket"] in ("normal_wrong", "normal_correct")]
        cond = mean(r["correct"] for r in nondegen) if nondegen else float("nan")
        print(f"{arm:>26s} {mean(r['correct'] for r in sub):7.3f} {buckets['empty']:7.3f} "
              f"{eos:7.3f} {buckets['malformed']:10.3f} {buckets['normal_wrong']:11.3f} "
              f"{buckets['normal_correct']:8.3f} {cond:9.3f}   (n_nondegen={len(nondegen)}/{n})")


if __name__ == "__main__":
    main()
