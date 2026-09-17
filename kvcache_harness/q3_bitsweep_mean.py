"""QUEUE 3 — bit-width sweep under MEAN attention accumulation.

The original sweep (results/bitwidth_degeneracy/) was measured under the sum
defect. This reruns it with SCORE_MODE="mean" so the corrected curve sits
alongside the original. The question is whether the non-monotonic shape —
whose central feature (the 6-bit collapse) already fails to replicate on the
independent implementation — survives the one defect we know about.

Run: python -m kvcache_harness.q3_bitsweep_mean
"""
from __future__ import annotations

import json
import os
from statistics import mean

from . import engine as engine_mod
from .models import load_model_and_tokenizer
from .engine import CacheEngine
from .cache.base_policy import BudgetSpec
from .cache.protected_split_tier import ProtectedSplitSignalTierPolicy
from .cache.promotion_signals import AttentionPromotion
from .tasks.multi_credential import make_multi_credential_prompt, score_turn
from .run_phase0r_calibration import build_turn_texts, compute_positions
from .run_bitwidth_degeneracy import is_degenerate, WELL_FORMED, repetition_run

BUDGET = 257
SEEDS = list(range(3000, 3050))
BITS = [8, 7, 6, 5, 4, 3]
REL_ERR = {8: 1.45, 7: 2.67, 6: 4.50, 5: 8.62, 4: 17.62, 3: 37.31}  # score-mode independent
OUT = "results/q3_bitsweep_mean/answers.jsonl"
CFG = dict(n_credentials=6, value_len=14, n_distractors=20, words_per_paragraph=50,
           protect_after_chars=2, max_answer_tokens=22, rebalance_every=1,
           recency_window=64, keep_sink=True)


def main():
    engine_mod.SCORE_MODE = "mean"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    model, tok = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct", device="cuda")
    f = open(OUT, "w")

    print(f"Bit-width sweep under SCORE_MODE=mean, budget {BUDGET}, n={len(SEEDS)} prompts")
    print("(sum-mode reference: 8 -> 0.410, 7 -> 0.293, 6 -> 0.000, 5 -> 0.000, 4 -> 0.123, 3 -> 0.000)")
    print()
    header = "{:>5s} {:>8s} {:>9s} {:>11s} {:>8s} {:>7s} {:>16s}".format(
        "bits", "rel_err", "accuracy", "degeneracy", "not_wf", "repet", "prompts_any_deg")
    print(header, flush=True)

    for bits in BITS:
        engine_mod.QUANT_BITS = bits
        per_prompt_any = []
        all_rows = []
        for seed in SEEDS:
            p = make_multi_credential_prompt(
                seed=seed, n_credentials=CFG["n_credentials"], n_distractors=CFG["n_distractors"],
                words_per_paragraph=CFG["words_per_paragraph"], value_len=CFG["value_len"])
            tt, bo = build_turn_texts(tok, p)
            oi, prot, grp, _ = compute_positions(tok, tt[0], bo, p, CFG["protect_after_chars"])
            pol = ProtectedSplitSignalTierPolicy(
                BudgetSpec(total_budget=BUDGET, full_fraction=0.5),
                AttentionPromotion(), groups=grp)
            eng = CacheEngine(model, tok, pol, device="cuda")
            _, answers, _ = eng.generate_multi_turn(
                tt, prot, oi, max_answer_tokens=CFG["max_answer_tokens"],
                rebalance_every=CFG["rebalance_every"], eos_token_id=tok.eos_token_id,
                recency_window=CFG["recency_window"], keep_sink=CFG["keep_sink"])
            rs = []
            for j, a in enumerate(answers):
                c = p.credentials[p.turn_order[j]]
                r = {"bits": bits, "seed": seed, "turn": j, "answer": a,
                     "correct": bool(score_turn(a, c)), "degenerate": is_degenerate(a),
                     "well_formed": bool(WELL_FORMED.match(a)), "repetitive": repetition_run(a)}
                rs.append(r)
                all_rows.append(r)
                f.write(json.dumps(r) + "\n")
            per_prompt_any.append(any(r["degenerate"] for r in rs))
        f.flush()
        line = "{:5d} {:7.2f}% {:9.3f} {:11.3f} {:8.3f} {:7.3f} {:16.3f}".format(
            bits, REL_ERR[bits],
            mean(r["correct"] for r in all_rows),
            mean(r["degenerate"] for r in all_rows),
            mean(not r["well_formed"] for r in all_rows),
            mean(r["repetitive"] for r in all_rows),
            mean(per_prompt_any))
        print(line, flush=True)

    f.close()
    engine_mod.SCORE_MODE = "sum"
    engine_mod.QUANT_BITS = 8


if __name__ == "__main__":
    main()
