"""Decision 7 — the RESOLVED packing, and its structural consequence for A_causal and I. NO GPU.

Decision 7 is settled: oracle_causal packs complete span-pairs optimally (greedy knapsack,
ascending token cost, no partials), and every instance is scored as the mean over all H query
variants. This reports what that packing achieves; it no longer compares candidate rules.

The mechanism that makes this computable without a GPU:

    One-hop LEDGER: a fact is answerable iff its record line is retained whole.

    The query selects uniformly among the H=4 queried records. So the *structural ceiling* on
    oracle_causal's accuracy is

        A_causal_max = P(queried fact is complete in the keep-set)
                     = E[n_facts_complete] / H

    and, since oracle_prescient always retains the queried fact whole,

        A_presc_max  = 1.0

This is an upper bound on accuracy, not a prediction of it -- the model must still read what
was retained. Under the adopted packing it is also EXACT rather than approximate, because the
score is the mean over all H variants: expected accuracy is precisely (complete candidates)/H.

Implied information share, using the structural bounds:

        I = (A_presc - A_causal) / (A_presc - A_floor)

reported here at A_floor = 0 (the floor is near-zero at tight budgets on this task; any
positive floor only makes I larger, so this is the conservative reading).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer

from harness import ladder
from harness.keys import seed_key_without_seed
from harness.tasks import ledger

N_SINK, N_WINDOW = 8, 64
# Two tighter cells added 2026-09-06 to recover range for the information share I,
# which collapsed to ~0 on the one-hop task because its candidates are cheap.
LADDER = {"b-2": 16, "b-1": 32, "b0": 64, "b1": 128, "b2": 256, "b3": 512}
TARGET_L = 2048
N_INSTANCES = 40

MODELS = {
    "M1": "Qwen/Qwen2.5-1.5B-Instruct",
    "M2": "Qwen/Qwen2.5-3B-Instruct",
    "M3": "meta-llama/Llama-3.2-3B-Instruct",
}


def span_token_indices(context: str, tok) -> callable:
    enc = tok(context, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]

    def idx(start: int, end: int) -> tuple[int, ...]:
        return tuple(
            ti for ti, (a, b) in enumerate(offsets)
            if b > a and a < end and b > start
        )
    return idx, len(offsets)


def facts_for(inst: ledger.LedgerInstance, tok):
    """Group the H candidate spans into H FactSpans, in token indices.

    One-hop task: each fact is a single record-line span, so the knapsack packs single spans.
    The decision-7 rule is unchanged -- complete candidates only, maximise how many fit.
    """
    idx, n_ctx = span_token_indices(inst.context, tok)
    by_id: dict[str, list[tuple[int, ...]]] = {}
    for sp in inst.candidates:
        by_id.setdefault(sp.rec_id, []).append(idx(sp.start, sp.end))
    facts = [ladder.FactSpans(rid, tuple(sp)) for rid, sp in by_id.items()]
    gold = next(f for f in facts if f.fact_id == inst.gold[0].rec_id)
    return facts, gold, n_ctx


def main() -> int:
    out: dict = {"n_instances": N_INSTANCES, "H": ledger.N_BINDINGS, "models": {}}

    for mid, repo in MODELS.items():
        try:
            tok = AutoTokenizer.from_pretrained(repo)
        except Exception as e:
            out["models"][mid] = {"error": f"{type(e).__name__}"}
            print(f"[{mid}] UNAVAILABLE\n")
            continue

        insts = []
        for i in range(N_INSTANCES):
            iid = f"d7probe{i:04d}"
            seed = seed_key_without_seed(
                task="ledger", instance_id=iid, model=repo, model_revision="probe",
                arm="probe", B=1, protocol="agnostic", device="nvidia", backend="none",
                torch_version="na", transformers_version="na", kvpress_version="na",
                dtype="na",
            )
            inst = ledger.build(seed, iid, target_tokens=TARGET_L, tokenizer=tok)
            insts.append((inst, seed))

        rows: dict = {}
        for bname, C in LADDER.items():
            per_rule: dict = {}

            # reference: the prescient oracle always completes the queried fact
            presc_ok = []
            void = False
            for inst, seed in insts:
                facts, gold, n_ctx = facts_for(inst, tok)
                # prescient retains whichever variant is asked; averaged over H it is 1.0
                ok = []
                for v in inst.variants:
                    gf = next(f for f in facts if f.fact_id == v.rec_id)
                    try:
                        sel = ladder.oracle_prescient(n_ctx, C, gf, N_SINK, N_WINDOW, facts)
                    except ValueError:
                        # C < k_gold: the cell is VOID (v1 §3.2) and is excluded before any
                        # run. The ladder refuses rather than silently truncating the gold.
                        void = True
                        break
                    ok.append(1.0 if gf.is_complete_in(set(sel.kept)) else 0.0)
                if void:
                    break
                presc_ok.append(statistics.fmean(ok))
            if void:
                rows[bname] = {"C": C, "A_presc_max": None, "zone": "VOID",
                               "rules": {"pair_knapsack_optimal": {
                                   "A_causal_max": None, "facts_complete_mean": None,
                                   "I_implied": None, "degraded": None,
                                   "note": "VOID: C < k_gold, excluded before any run"}}}
                continue
            a_presc = statistics.fmean(presc_ok)

            exp_acc, n_complete, degraded_any = [], [], False
            for inst, seed in insts:
                facts, gold, n_ctx = facts_for(inst, tok)
                sel = ladder.oracle_causal(n_ctx, C, facts,
                                           n_sink=N_SINK, n_window=N_WINDOW)
                degraded_any |= sel.degraded
                # Decision 7: score is the mean over all H variants, so the oracle's expected
                # accuracy is exactly (complete pairs)/H -- not a lottery on the queried one.
                exp_acc.append(sel.notes["expected_accuracy"])
                n_complete.append(sel.n_facts_complete)
            a_causal = statistics.fmean(exp_acc)
            I = (a_presc - a_causal) / a_presc if a_presc > 0 else float("nan")
            per_rule["pair_knapsack_optimal"] = {
                "A_causal_max": round(a_causal, 4),
                "facts_complete_mean": round(statistics.fmean(n_complete), 3),
                "I_implied": round(I, 4),
                "degraded": degraded_any,
            }
            rows[bname] = {"C": C, "A_presc_max": round(a_presc, 4), "zone": "ok",
                           "rules": per_rule}
        out["models"][mid] = {"model": repo, "budgets": rows}

        print(f"\n[{mid}] {repo}   (n={N_INSTANCES}, H={ledger.N_BINDINGS})")
        print(f"  {'cell':4s} {'C':>4s} {'rule':18s} {'degraded':>9s} "
              f"{'facts done':>10s} {'A_causal_max':>13s} {'I implied':>10s}")
        for bname, r in rows.items():
            for rn, v in r["rules"].items():
                if v["A_causal_max"] is None:
                    print(f"  {bname:4s} {r['C']:4d} {'-- VOID: C < k_gold, excluded --':>55s}")
                    continue
                print(f"  {bname:4s} {r['C']:4d} {rn:18s} {str(v['degraded']):>9s} "
                      f"{v['facts_complete_mean']:10.2f} {v['A_causal_max']:13.3f} "
                      f"{v['I_implied']:10.3f}")

    p = Path(__file__).resolve().parent / "gates" / "nvidia" / "decision7_evidence.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {p}")
    print("\nNOTE: structural upper bounds on A_causal, not measured accuracy. "
          "Decision 7 remains OPEN.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
