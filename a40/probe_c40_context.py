"""Is the c=40 oracle ceiling (0.1675) driven by fact cost c, or by context length L?

c=40 was the only p3curve cell at L~8358 (N=200). This reruns the oracle_causal arm at the
SAME n_fields=8 (same fact cost, same achieved c~39.5) but N=80 instead of 200, dropping L
toward ~3200-3400. If accuracy recovers toward the ~0.88-0.93 seen at c=1/8/19, the ceiling
is a long-context effect; if it stays near 0.17, it's fact size. n=50, single arm
(oracle_causal), C=512 -- everything else identical to the p3curve cells.
"""
from __future__ import annotations

import json
import statistics as st
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_p3_curve as P  # noqa: E402
import run_stage0 as R  # noqa: E402

N_PROBE = 80
N_INST = 50
ARM = "oracle_causal"


def main():
    print(f"loading model... (probe: c=40 oracle at N={N_PROBE}, n={N_INST}, arm={ARM})",
          flush=True)
    model, tok = R.load_model()
    print("loaded.", flush=True)

    sp = dict(task="ledger_c", n_fields=8, n_records=N_PROBE)
    scores, n_ctx_seen = [], []
    for i in range(N_INST):
        rec = P.build_instance(40, sp, i, tok)
        # distinguish these instances from the N=200 c=40 run (different iid prefix)
        rec["iid"] = f"probeN80_c40_{i:05d}"
        n_ctx_seen.append(rec["n_ctx"])
        row, _ = P.run_arm(model, tok, rec, ARM)
        scores.append(row["score"])
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{N_INST}  running mean acc={st.fmean(scores):.4f}  "
                  f"mean L={st.fmean(n_ctx_seen):.0f}", flush=True)

    mean_acc, ci = P.mean_ci(scores, seed=zlib.crc32(b"probe_c40_context_N80") & 0xFFFFFFFF)
    result = dict(arm=ARM, c_tag=40, n_fields=8, n_records=N_PROBE, n=N_INST,
                 accuracy=mean_acc, accuracy_ci=ci, mean_n_ctx=st.fmean(n_ctx_seen),
                 comparison_N200_accuracy=0.1675, comparison_N200_mean_n_ctx=8358)
    (P.OUT / "probe_c40_context_N80.json").write_text(json.dumps(result, indent=2),
                                                       encoding="utf-8")
    print(f"\n=== PROBE RESULT ===")
    print(f"oracle_causal @ c=40, N={N_PROBE}, L~{result['mean_n_ctx']:.0f}: "
          f"acc={mean_acc:.4f} [{ci[0]:.4f},{ci[1]:.4f}]  (n={N_INST})")
    print(f"vs oracle_causal @ c=40, N=200, L~8358: acc=0.1675  (n=100)")
    if mean_acc > 0.5:
        print("RECOVERED toward c=1/8/19 levels -> ceiling looks like a LONG-CONTEXT effect")
    elif mean_acc < 0.30:
        print("STAYED LOW -> ceiling looks like a FACT-SIZE (c) effect, not context length")
    else:
        print("INTERMEDIATE -> partial recovery, likely a mix of both")


if __name__ == "__main__":
    raise SystemExit(main())
