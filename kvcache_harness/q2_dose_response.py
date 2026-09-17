"""QUEUE 2 — dose-response on lossy tokens.

Budget held at 514, 4-bit, arm 4 (protection + recoverable tier). Only the
number of tokens placed in the cold tier varies: 0, 50, 100, 150, 225.

Competitive budget at 514 is 514 - 64 (recency) - 1 (sink) = 449, so
full_fraction = (449 - Q) / 449 puts exactly Q tokens in the quantized tier.

If damage tracks the quantized-token count monotonically, that is the
mechanism behind the -0.683 and makes the Q1 reading conclusive. If it does
not track, the proposed mechanism is wrong and must be stated as such.

Run: python -m kvcache_harness.q2_dose_response
"""
from __future__ import annotations

import json
import os
from statistics import mean

from . import engine as engine_mod
from .models import load_model_and_tokenizer
from .engine import CacheEngine
from .cache.base_policy import BudgetSpec, TokenStatus
from .cache.recoverable_tier import RecoverableTierPolicy
from .cache.structural_protection import StructuralProtectionWrapper
from .tasks.multi_credential import make_multi_credential_prompt, score_turn
from .run_phase0r_calibration import build_turn_texts, compute_positions
from .run_bitwidth_degeneracy import is_degenerate

BUDGET = 514
COMPETITIVE = BUDGET - 64 - 1          # 449
DOSES = [0, 50, 100, 150, 225]
SEEDS = list(range(3000, 3050))
OUT = "results/q2_dose_response/raw.jsonl"
CFG = dict(n_credentials=6, value_len=14, n_distractors=20, words_per_paragraph=50,
           protect_after_chars=2, max_answer_tokens=22, rebalance_every=1,
           recency_window=64, keep_sink=True)


def main():
    engine_mod.QUANT_BITS = 4
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    model, tok = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct", device="cuda")
    f = open(OUT, "w")

    print(f"Budget {BUDGET} (competitive {COMPETITIVE}), 4-bit, arm 4, n={len(SEEDS)} prompts\n")
    print(f"{'quant_tokens':>13s} {'full_frac':>10s} {'accuracy':>9s} {'degeneracy':>11s} "
          f"{'realized_QUANT':>15s}")

    for Q in DOSES:
        ff = (COMPETITIVE - Q) / COMPETITIVE
        accs, degens, realized = [], [], []
        for seed in SEEDS:
            p = make_multi_credential_prompt(seed=seed, n_credentials=CFG["n_credentials"],
                                              n_distractors=CFG["n_distractors"],
                                              words_per_paragraph=CFG["words_per_paragraph"],
                                              value_len=CFG["value_len"])
            tt, bo = build_turn_texts(tok, p)
            oi, prot, grp, _ = compute_positions(tok, tt[0], bo, p, CFG["protect_after_chars"])
            spec = BudgetSpec(total_budget=BUDGET, full_fraction=ff)
            pol = StructuralProtectionWrapper(RecoverableTierPolicy(spec), groups=grp)
            eng = CacheEngine(model, tok, pol, device="cuda")
            trace, answers, _ = eng.generate_multi_turn(
                tt, prot, oi, max_answer_tokens=CFG["max_answer_tokens"],
                rebalance_every=CFG["rebalance_every"], eos_token_id=tok.eos_token_id,
                recency_window=CFG["recency_window"], keep_sink=CFG["keep_sink"])
            n = len(p.credentials)
            correct = [score_turn(answers[j], p.credentials[p.turn_order[j]]) for j in range(n)]
            accs.append(sum(correct) / n)
            degens.append(mean(is_degenerate(a) for a in answers))
            realized.append(sum(1 for v in trace.final_statuses.values() if v == TokenStatus.QUANT))
            f.write(json.dumps({"quant_tokens_target": Q, "full_fraction": ff, "seed": seed,
                                 "frac_retrieved": accs[-1], "degeneracy": degens[-1],
                                 "realized_quant": realized[-1]}) + "\n")
        f.flush()
        print(f"{Q:13d} {ff:10.4f} {mean(accs):9.3f} {mean(degens):11.3f} {mean(realized):15.1f}",
              flush=True)

    f.close()
    engine_mod.QUANT_BITS = 8


if __name__ == "__main__":
    main()
