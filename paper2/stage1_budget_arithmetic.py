"""Stage 1 — budget arithmetic (design plan v1 §3.2, §7). NO GPU.

Measures, per (model tokenizer, task):
    L        realised context length in that tokenizer
    k_gold   tokens needed to answer          (prescient oracle)
    K_all    tokens over all candidate facts  (causal oracle)

and classifies every (model, task, budget) cell:

    VOID        C <  k_gold             -> excluded before any run
    PARTIAL     k_gold <= C < 2*k_gold  -> run, reported separately, never pooled
    COMPETITIVE C >= 2*k_gold
    SATURATED   C >= 0.5*L              -> excluded (no compression pressure)

Gate to pass (v1 §7 Stage 1): **zero VOID cells remain in the planned grid.**

Every length is measured in the model's own tokenizer, never in characters or a reference
tokenizer -- the trap that silently zeroed 7 of 13 subtasks in the published audit.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer

from harness.keys import seed_key_without_seed
from harness.tasks import ledger

# v1 §3.2 pre-registered ladder
N_SINK, N_WINDOW = 8, 64
# Two tighter cells added 2026-09-06 to recover range for the information share I,
# which collapsed to ~0 on the one-hop task because its candidates are cheap.
LADDER = {"b-2": 16, "b-1": 32, "b0": 64, "b1": 128, "b2": 256, "b3": 512}
TARGET_L = 2048
TOLERANCE = 16

MODELS = {
    "M1": "Qwen/Qwen2.5-1.5B-Instruct",
    "M2": "Qwen/Qwen2.5-3B-Instruct",
    "M3": "meta-llama/Llama-3.2-3B-Instruct",
}

N_PROBE = 24  # instances per (model, task) for the length statistics


def classify(C: int, k_gold: float, L: float) -> str:
    if C >= 0.5 * L:
        return "SATURATED"
    if C < k_gold:
        return "VOID"
    if C < 2 * k_gold:
        return "PARTIAL"
    return "COMPETITIVE"


def measure_ledger(tok, model_id: str) -> dict:
    Ls, kg, ka, ka_raw, ans_toks = [], [], [], [], []
    for i in range(N_PROBE):
        iid = f"probe{i:04d}"
        seed = seed_key_without_seed(
            task="ledger", instance_id=iid, model=model_id, model_revision="probe",
            arm="probe", B=0 or 1, protocol="agnostic", device="nvidia", backend="none",
            torch_version="na", transformers_version="na", kvpress_version="na", dtype="na",
        )
        inst = ledger.build(
            seed, iid, target_tokens=TARGET_L, tokenizer=tok, tolerance=TOLERANCE
        )
        enc = tok(inst.context, add_special_tokens=False, return_offsets_mapping=True)
        off = enc["offset_mapping"]; n = len(off)
        Ls.append(n)
        # Tokens inside the mandatory floors (sink + recency window) are retained by EVERY
        # arm for free, so an oracle only pays for candidate tokens outside them. Comparing
        # C against the raw K_all overstates the demand and misclassified b1 as VOID.
        floors = set(range(min(N_SINK, n))) | set(range(max(0, n - N_WINDOW), n))
        def _idx(spans):
            return {ti for sp in spans for ti, (a, b) in enumerate(off)
                    if b > a and a < sp.end and b > sp.start}
        kg.append(len(_idx(inst.gold) - floors))
        ka.append(len(_idx(inst.candidates) - floors))
        ka_raw.append(len(_idx(inst.candidates)))
        ans_toks.append(len(tok(inst.answer, add_special_tokens=False)["input_ids"]))
    return {
        "L_median": statistics.median(Ls),
        "L_min": min(Ls), "L_max": max(Ls),
        "k_gold_median": statistics.median(kg),
        "k_gold_max": max(kg),
        "K_all_median": statistics.median(ka),
        "K_all_max": max(ka),
        "K_all_raw_median": statistics.median(ka_raw),
        "K_all_free_in_floors_median": statistics.median(ka_raw) - statistics.median(ka),
        "answer_tokens_median": statistics.median(ans_toks),
        "n_probe": N_PROBE,
    }


def main() -> int:
    out: dict = {
        "ladder": {k: {"C": v, "B": v + N_SINK + N_WINDOW} for k, v in LADDER.items()},
        "n_sink": N_SINK, "n_window": N_WINDOW, "target_L": TARGET_L,
        "models": {},
    }
    print("ladder (B = C + %d):" % (N_SINK + N_WINDOW))
    for name, spec in out["ladder"].items():
        print(f"  {name}  {spec['C']:5d} {spec['B']:5d}")
    print()

    for mid, repo in MODELS.items():
        try:
            tok = AutoTokenizer.from_pretrained(repo)
        except Exception as e:
            out["models"][mid] = {"model": repo, "error": f"{type(e).__name__}: {e}"[:200]}
            print(f"[{mid}] {repo}\n  UNAVAILABLE -- {type(e).__name__}\n")
            continue

        m = measure_ledger(tok, repo)
        cells = {}
        for bname, spec in out["ladder"].items():
            C = spec["C"]
            cells[bname] = {
                "C": C, "B": spec["B"],
                "B_over_L": round(spec["B"] / m["L_median"], 4),
                "zone_prescient": classify(C, m["k_gold_median"], m["L_median"]),
                "zone_causal": classify(C, m["K_all_median"], m["L_median"]),
                "causal_fits_all_candidates": C >= m["K_all_max"],
            }
        out["models"][mid] = {"model": repo, "ledger": m, "cells": cells}

        print(f"[{mid}] {repo}")
        print(f"  L      median {m['L_median']}  range [{m['L_min']}, {m['L_max']}]")
        print(f"  k_gold median {m['k_gold_median']}  max {m['k_gold_max']}   (1 gold span, one-hop)")
        print(f"  K_all  median {m['K_all_median']}  max {m['K_all_max']}   (H candidate spans, "
          f"payable; raw {m['K_all_raw_median']}, {m['K_all_free_in_floors_median']} free in floors)")
        print(f"  answer median {m['answer_tokens_median']} tokens")
        print(f"  {'cell':5s} {'C':>5s} {'B':>5s} {'B/L':>7s}  {'zone(gold)':>12s} {'zone(K_all)':>13s}  causal fits all")
        for bname, c in cells.items():
            print(f"  {bname:5s} {c['C']:5d} {c['B']:5d} {c['B_over_L']*100:6.1f}%  "
                  f"{c['zone_prescient']:>12s} {c['zone_causal']:>13s}  "
                  f"{'yes' if c['causal_fits_all_candidates'] else 'NO -- degrades'}")
        print()

    # Stage 1 gate
    voids = [
        (mid, b) for mid, mm in out["models"].items() if "cells" in mm
        for b, c in mm["cells"].items() if c["zone_prescient"] == "VOID"
    ]
    sats = [
        (mid, b) for mid, mm in out["models"].items() if "cells" in mm
        for b, c in mm["cells"].items() if c["zone_prescient"] == "SATURATED"
    ]
    out["gate"] = {
        "void_cells": [f"{m}/{b}" for m, b in voids],
        "saturated_cells": [f"{m}/{b}" for m, b in sats],
        "pass": len(voids) == 0,
    }
    print("=" * 68)
    print(f"STAGE 1 GATE (zero VOID cells in the planned grid): "
          f"{'PASS' if out['gate']['pass'] else 'FAIL'}")
    if voids:
        print(f"  VOID: {out['gate']['void_cells']}")
    if sats:
        print(f"  SATURATED (excluded, no compression pressure): {out['gate']['saturated_cells']}")

    p = Path(__file__).resolve().parent / "gates" / "nvidia" / "stage1_budget_arithmetic.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
