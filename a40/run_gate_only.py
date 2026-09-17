"""Rerun of step 6 only (anchor gate) -- calibration and the rate benchmark already saved.

The first run crashed on a print-formatting bug right after computing the c=1 cell (nf=None
for MARK-1 formatted with :2d), before anything in step 6 was written to disk. That GPU work
is not recoverable (never saved), so this reruns the full 4-cell gate cleanly, saving each
cell's result to disk immediately after it completes.
"""
from __future__ import annotations

import json

import run_stage0 as R


def main():
    calib = json.loads((R.OUT / "calibration.json").read_text(encoding="utf-8"))
    print("loading model...", flush=True)
    model, tok = R.load_model()
    print("loaded.", flush=True)

    print("\nstep 6: full_cache competence anchor gate, LEDGER-C, n=100 per cell", flush=True)
    gate = {}
    for t in R.TARGETS:
        nf = None if t == 1 else calib["chosen"][str(t)]["n_fields"]
        cell = R.anchor_cell(model, tok, t, nf, n=100)
        gate[str(t)] = cell
        band_txt = "IN BAND" if cell["in_band"] else "OUT OF BAND"
        nf_txt = "None" if nf is None else f"{nf:2d}"
        print(f"  c={t:3d} (n_fields={nf_txt}): accuracy={cell['accuracy']:.4f}  {band_txt}",
              flush=True)
        (R.OUT / f"anchor_cell_c{t}.json").write_text(json.dumps(cell, indent=2),
                                                       encoding="utf-8")

    gate_summary = {k: {kk: vv for kk, vv in v.items() if kk != "per_instance_scores"}
                    for k, v in gate.items()}
    (R.OUT / "anchor_gate.json").write_text(json.dumps(gate_summary, indent=2), encoding="utf-8")
    (R.OUT / "anchor_gate_full.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    print("\n=== GATE SUMMARY ===")
    any_out = False
    for t in R.TARGETS:
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
