"""Final anchor gate at N=200 (chosen by bracket_n19.py: c~19 acc=0.89, L~4154, IN BAND).

Only N changed from the N=40 run. Same n_fields per c (1, 3, 8 for c=8/19/40), same H=4,
same target_tokens=2048 (filler collapses as N/c grow; achieved L reported per cell, not
forced), full_cache arm, same scorer. c=1 stays on MARK-1 (unaffected by N) and is carried
forward from results/anchor_cell_c1.json rather than re-run.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_stage0 as R  # noqa: E402
from p3.tasks import ledger_c  # noqa: E402

N = 200
CELLS_LEDGER = (8, 19, 40)


def anchor_cell_N(model, tok, c_tag, n_fields, N, n=100):
    scores, ctxs, flags_all = [], [], []
    for i in range(n):
        inst, pre, posts, max_new, n_ctx = build_at_N_for_nf(N, n_fields, i, tok, "gateN200")
        ctxs.append(n_ctx)
        texts, flags = R.runner.generate_with(model, tok, pre, posts, None, max_new)
        scores.append(ledger_c.score_instance(texts, inst))
        flags_all.extend(flags)
    acc = st.fmean(scores)
    return dict(c_tag=c_tag, n_fields=n_fields, N=N, n=n, accuracy=acc, task="ledger_c",
                mean_n_ctx=st.fmean(ctxs), in_band=R.BAND[0] <= acc <= R.BAND[1],
                eos_rate=sum(1 for f in flags_all if f == "eos") / len(flags_all))


def build_at_N_for_nf(N, n_fields, i, tok, prefix):
    """Like bracket_n19.build_at_N but for an arbitrary n_fields (not just the c~19 one)."""
    iid = f"{prefix}_c{n_fields}_N{N}_{i:05d}"
    sd = R.keys3.instance_seed(task="ledger_c", instance_id=iid, model=R.MODEL_NAME,
                               model_revision=R.MODEL_REV, n_fields=n_fields,
                               n_records=N, layout=None, **R.ENV)
    inst = ledger_c.build(sd, iid, n_fields=n_fields, n_records=N,
                          target_tokens=2048, tokenizer=tok)
    pre, _ = R.runner.templated_parts(tok, inst.context, "")
    facts, n_ctx = R.runner.facts_and_ctx(inst, tok, pre)
    posts = [R.runner.templated_parts(tok, inst.context, v.query)[1] for v in inst.variants]
    max_new = [len(tok(v.answer, add_special_tokens=False)["input_ids"]) + R.runner.MAX_NEW_SLACK
               for v in inst.variants]
    return inst, pre, posts, max_new, n_ctx


def main():
    calib = json.loads((R.OUT / "calibration.json").read_text(encoding="utf-8"))
    c1 = json.loads((R.OUT / "anchor_cell_c1.json").read_text(encoding="utf-8"))

    print("loading model...", flush=True)
    model, tok = R.load_model()
    print("loaded.", flush=True)

    gate = {"1": c1}
    print(f"c=  1 (MARK-1, carried forward, N/A): accuracy={c1['accuracy']:.4f}  "
          f"{'IN BAND' if c1['in_band'] else 'OUT OF BAND'}", flush=True)

    for t in CELLS_LEDGER:
        nf = calib["chosen"][str(t)]["n_fields"]
        cell = anchor_cell_N(model, tok, t, nf, N, n=100)
        gate[str(t)] = cell
        band_txt = "IN BAND" if cell["in_band"] else "OUT OF BAND"
        print(f"c={t:3d} (n_fields={nf}, N={N}): accuracy={cell['accuracy']:.4f} "
              f"L~{cell['mean_n_ctx']:.0f}  {band_txt}", flush=True)
        (R.OUT / f"anchor_cell_c{t}_N{N}.json").write_text(json.dumps(cell, indent=2),
                                                            encoding="utf-8")

    (R.OUT / f"anchor_gate_N{N}.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    print("\n=== GATE SUMMARY (N=200 for c=8/19/40 LEDGER-C; c=1 MARK-1 unaffected by N) ===")
    any_out = False
    for t in (1,) + CELLS_LEDGER:
        c = gate[str(t)]
        print(f"c={t:3d}: acc={c['accuracy']:.4f} n={c['n']} "
              f"{'IN' if c['in_band'] else 'OUT OF'} band {R.BAND}")
        if not c["in_band"]:
            any_out = True
    if any_out:
        print("AT LEAST ONE ANCHOR IS OUT OF BAND -- stop, do not start grid work.")
    else:
        print("All four anchors in band. STOP HERE per instructions -- do not start grid work.")


if __name__ == "__main__":
    main()
