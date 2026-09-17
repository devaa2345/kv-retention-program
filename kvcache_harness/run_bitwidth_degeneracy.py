"""Is the non-monotonic bit-width/accuracy curve a threshold effect, and is
it one basin or noise?

Reconstruction error is monotone in bit-width and the policy behaves
identically at every width (verified: level counts on intercepted calls,
error on real KV in the bfloat16 path, tier composition identical to three
decimals). Yet accuracy is not monotone -- 5 and 6 bit collapse while 4 bit
still copies strings. If that is greedy decoding crossing a threshold, it
should be PROMPT-DEPENDENT: some prompts degenerate at a given width and
others do not, and the fraction that do need not be ordered in bit-width.

Records every generated answer so we can report:
  * fraction of answers that are DEGENERATE (not the mean accuracy)
  * whether degenerate outputs at different widths are the SAME failure
    (one basin) or different ones (noise)

Run: python -m kvcache_harness.run_bitwidth_degeneracy
"""
from __future__ import annotations

import json
import os
import re
import time

from . import engine as em
from .models import load_model_and_tokenizer
from .engine import CacheEngine
from .cache.base_policy import BudgetSpec
from .cache.protected_split_tier import ProtectedSplitSignalTierPolicy
from .cache.promotion_signals import AttentionPromotion
from .tasks.multi_credential import make_multi_credential_prompt, score_turn
from .run_phase0r_calibration import build_turn_texts, compute_positions

BUDGET = 257
SEEDS = list(range(3000, 3050))          # n=50, disjoint seed set from the n=25 sweep's 3000-3024 tail
BITS = [8, 7, 6, 5, 4, 3]
OUT = "results/bitwidth_degeneracy/answers.jsonl"
CFG = dict(n_credentials=6, value_len=14, n_distractors=20, words_per_paragraph=50,
           protect_after_chars=2, max_answer_tokens=22, rebalance_every=1,
           recency_window=64, keep_sink=True)

WELL_FORMED = re.compile(r"^\s*sk-[0-9a-f]{8,}", re.I)


def repetition_run(text: str, k: int = 2, threshold: int = 4) -> bool:
    """True if some k-gram repeats >= threshold times consecutively."""
    for size in range(1, k + 1):
        for i in range(len(text) - size):
            unit = text[i:i + size]
            if not unit.strip():
                continue
            n = 1
            j = i + size
            while text[j:j + size] == unit:
                n += 1
                j += size
            if n >= threshold:
                return True
    return False


def is_degenerate(ans: str) -> bool:
    return (not WELL_FORMED.match(ans)) or repetition_run(ans)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
                done.add((r["bits"], r["seed"], r["turn"]))
            except Exception:
                pass

    model, tok = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct", device="cuda")
    f = open(OUT, "a")
    t0 = time.time()

    for bits in BITS:
        em.QUANT_BITS = bits
        for seed in SEEDS:
            if (bits, seed, 0) in done:
                continue
            p = make_multi_credential_prompt(
                seed=seed, n_credentials=CFG["n_credentials"], n_distractors=CFG["n_distractors"],
                words_per_paragraph=CFG["words_per_paragraph"], value_len=CFG["value_len"])
            tt, bo = build_turn_texts(tok, p)
            oi, prot, grp, _ = compute_positions(tok, tt[0], bo, p, CFG["protect_after_chars"])
            pol = ProtectedSplitSignalTierPolicy(
                BudgetSpec(total_budget=BUDGET, full_fraction=0.01), AttentionPromotion(), groups=grp)
            eng = CacheEngine(model, tok, pol, device="cuda")
            _, answers, _ = eng.generate_multi_turn(
                tt, prot, oi, max_answer_tokens=CFG["max_answer_tokens"],
                rebalance_every=CFG["rebalance_every"], eos_token_id=tok.eos_token_id,
                recency_window=CFG["recency_window"], keep_sink=CFG["keep_sink"])
            for j, ans in enumerate(answers):
                cred = p.credentials[p.turn_order[j]]
                f.write(json.dumps({
                    "bits": bits, "seed": seed, "turn": j,
                    "answer": ans,
                    "correct": score_turn(ans, cred),
                    "degenerate": is_degenerate(ans),
                    "well_formed": bool(WELL_FORMED.match(ans)),
                    "repetitive": repetition_run(ans),
                }) + "\n")
            f.flush()
        print(f"bits={bits} done, elapsed={(time.time()-t0)/60:.1f}min", flush=True)

    f.close()
    em.QUANT_BITS = 8
    print("bit-width degeneracy sweep complete", flush=True)


if __name__ == "__main__":
    main()
