"""Bracket N for the c~19 LEDGER-C anchor on Qwen2.5-14B-Instruct.

Only N (n_records) changes. n_fields=3 (achieved c=18.92 at N=40, PREREG_P3-identical fact
structure), H=4 bindings, target_tokens=2048 (filler collapses toward 0 as N grows -- the
achieved context length L is measured and reported, not silently forced), full_cache arm,
scoring identical to ledger_c.score_instance. Single process, single model load.

Strategy (set by the user): start at N=200, n=50. Double N while accuracy > 0.97. If a step
drops accuracy < 0.55, bisect between the last in-range-or-above N and the below-band N until
a value lands inside [0.55, 0.97], or until the bracket collapses to adjacent integers.
"""
from __future__ import annotations

import json
import statistics as st
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_stage0 as R  # noqa: E402
from p3 import keys3, runner  # noqa: E402
from p3.tasks import ledger_c  # noqa: E402

OUT = R.OUT
N_FIELDS_19 = 3
BAND = R.BAND


def build_at_N(N, i, tok, prefix):
    iid = f"{prefix}_N{N}_{i:05d}"
    sd = keys3.instance_seed(task="ledger_c", instance_id=iid, model=R.MODEL_NAME,
                             model_revision=R.MODEL_REV, n_fields=N_FIELDS_19,
                             n_records=N, layout=None, **R.ENV)
    inst = ledger_c.build(sd, iid, n_fields=N_FIELDS_19, n_records=N,
                          target_tokens=2048, tokenizer=tok)
    pre, _ = runner.templated_parts(tok, inst.context, "")
    facts, n_ctx = runner.facts_and_ctx(inst, tok, pre)
    posts = [runner.templated_parts(tok, inst.context, v.query)[1] for v in inst.variants]
    max_new = [len(tok(v.answer, add_special_tokens=False)["input_ids"]) + runner.MAX_NEW_SLACK
               for v in inst.variants]
    return inst, pre, posts, max_new, n_ctx


def measure_N(model, tok, N, n):
    scores, ctxs, times = [], [], []
    for i in range(n):
        inst, pre, posts, max_new, n_ctx = build_at_N(N, i, tok, "bracket")
        ctxs.append(n_ctx)
        t0 = time.perf_counter()
        texts, flags = runner.generate_with(model, tok, pre, posts, None, max_new)
        times.append(time.perf_counter() - t0)
        scores.append(ledger_c.score_instance(texts, inst))
    acc = st.fmean(scores)
    return dict(N=N, n=n, accuracy=acc, in_band=BAND[0] <= acc <= BAND[1],
                mean_n_ctx=st.fmean(ctxs), mean_sec_per_record=st.fmean(times))


def main():
    print("loading model...", flush=True)
    model, tok = R.load_model()
    print("loaded.", flush=True)

    history = []
    N = 200
    result = measure_N(model, tok, N, 50)
    history.append(result)
    print(f"N={N:5d}: acc={result['accuracy']:.4f} L~{result['mean_n_ctx']:.0f} "
          f"{result['mean_sec_per_record']:.2f}s/rec {'IN BAND' if result['in_band'] else ''}",
          flush=True)

    # Phase 1: double while above band ceiling.
    lo_above = None  # largest N still known to be ABOVE 0.97 (too easy)
    hi_below = None  # smallest N known to be BELOW 0.55 (too hard)
    if result["accuracy"] > BAND[1]:
        lo_above = N
        while result["accuracy"] > BAND[1]:
            N *= 2
            result = measure_N(model, tok, N, 50)
            history.append(result)
            print(f"N={N:5d}: acc={result['accuracy']:.4f} L~{result['mean_n_ctx']:.0f} "
                  f"{result['mean_sec_per_record']:.2f}s/rec "
                  f"{'IN BAND' if result['in_band'] else ''}", flush=True)
            if result["in_band"]:
                break
            if result["accuracy"] < BAND[0]:
                hi_below = N
                break
            lo_above = N
    elif result["accuracy"] < BAND[0]:
        hi_below = N
        while result["accuracy"] < BAND[0] and N > 40:
            N = max(40, N // 2)
            result = measure_N(model, tok, N, 50)
            history.append(result)
            print(f"N={N:5d}: acc={result['accuracy']:.4f} L~{result['mean_n_ctx']:.0f} "
                  f"{result['mean_sec_per_record']:.2f}s/rec "
                  f"{'IN BAND' if result['in_band'] else ''}", flush=True)
            if result["in_band"]:
                break
            if result["accuracy"] > BAND[1]:
                lo_above = N
                break
            hi_below = N

    # Phase 2: bisect if we have both an above-ceiling and a below-floor bound.
    while not result["in_band"] and lo_above is not None and hi_below is not None \
            and abs(hi_below - lo_above) > 5:
        N = (lo_above + hi_below) // 2
        result = measure_N(model, tok, N, 50)
        history.append(result)
        print(f"N={N:5d}: acc={result['accuracy']:.4f} L~{result['mean_n_ctx']:.0f} "
              f"{result['mean_sec_per_record']:.2f}s/rec "
              f"{'IN BAND' if result['in_band'] else ''}", flush=True)
        if result["in_band"]:
            break
        if result["accuracy"] > BAND[1]:
            lo_above = N
        else:
            hi_below = N

    (OUT / "bracket_n19_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if result["in_band"]:
        print(f"\nCHOSEN N = {result['N']} (c~19 acc={result['accuracy']:.4f}, "
              f"L~{result['mean_n_ctx']:.0f})")
        (OUT / "bracket_n19_chosen.json").write_text(json.dumps(result, indent=2),
                                                      encoding="utf-8")
    else:
        print(f"\nBRACKET DID NOT CONVERGE. Last: N={result['N']} acc={result['accuracy']:.4f}. "
              "Stopping without a chosen N -- report and ask before spending more.")


if __name__ == "__main__":
    main()
