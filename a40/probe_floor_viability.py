"""Check floor_pos viability at candidate budgets BEFORE running the Paper 2 ladder at 14B.

c=40's cell degenerated because the anchor gate checked only full_cache competence, not
floor_pos viability at the operating N. This time floor_pos is checked directly, at each
candidate budget, before spending a full n=100 ladder run on it. c=19 LEDGER-C, N=200
(achieved c~18.92 at C=512 already measured: floor_pos=0.1025 -- close to the 0.05 floor
even at the widest budget in the p3curve grid, so tighter budgets are a real risk).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_p3_curve as P  # noqa: E402
import run_stage0 as R  # noqa: E402
from harness import press  # noqa: E402
from p3 import runner  # noqa: E402
from p3.tasks import ledger_c  # noqa: E402

CANDIDATES = (16, 32, 64, 128)
N_PROBE = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-records", type=int, default=200)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    calib = json.loads((R.OUT / "calibration.json").read_text(encoding="utf-8"))
    sp = P.spec_for(19, calib)
    sp["n_records"] = args.n_records
    print(f"loading model... (floor_pos viability probe, c=19, {sp}, n={N_PROBE})", flush=True)
    model, tok = R.load_model()
    print("loaded.", flush=True)

    recs = []
    for i in range(N_PROBE):
        rec = P.build_instance(19, sp, i, tok)
        rec["iid"] = f"floorprobe_c19_N{args.n_records}_{i:05d}"
        recs.append(rec)
    print(f"  achieved n_ctx (mean) = {st.fmean(r['n_ctx'] for r in recs):.0f}", flush=True)

    results = {}
    for C in CANDIDATES:
        t0 = time.time()
        scores = []
        for rec in recs:
            p, _ = press.build_arm("floor_pos", n_ctx=rec["n_ctx"], C=C,
                                   n_sink=P.N_SINK, n_window=P.N_WINDOW,
                                   facts=rec["facts"], seed=rec["seed"])
            outs, flags = runner.generate_with(model, tok, rec["pre"], rec["posts"], p,
                                               rec["max_new"])
            scores.append(ledger_c.score_instance(outs, rec["inst"]))
        acc = st.fmean(scores)
        el = time.time() - t0
        results[C] = dict(accuracy=acc, degenerate=acc < 0.05, n=N_PROBE, wall_s=el)
        print(f"  C={C:4d}  B={C + P.N_SINK + P.N_WINDOW:4d}  floor_pos acc={acc:.4f}  "
              f"{'DEGENERATE' if acc < 0.05 else 'viable'}  ({el:.1f}s, {el / N_PROBE:.2f} s/row)",
              flush=True)

    (P.OUT / f"probe_floor_viability_c19_N{args.n_records}.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n=== FLOOR VIABILITY SUMMARY (c=19, N={args.n_records}) ===")
    for C, r in results.items():
        print(f"  C={C:4d}: floor_pos={r['accuracy']:.4f}  "
              f"{'-- DEGENERATE, exclude from budget set' if r['degenerate'] else '-- OK'}")


if __name__ == "__main__":
    raise SystemExit(main())
