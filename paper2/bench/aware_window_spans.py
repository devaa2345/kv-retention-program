"""Item 1: does the query's token span actually sit inside each method's SCORING window?

If the question is in the compressed prefix but outside the window a press scores from, the
aware arm is a no-op for that method and the non-reproduction is a harness fault, not a finding.
Tokenizer-only (no GPU) so it can run alongside N9.
"""
from __future__ import annotations
import statistics, sys, inspect
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer
from harness import methods
from harness.keys import seed_key_without_seed
from harness.tasks import ledger
from run_grid_aware import aware_parts

N_WINDOW = 64
MODELS = ["Qwen/Qwen2.5-3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"]
ARMS = ["snapkv", "tova", "expected_attn", "keydiff", "adakv_snapkv"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20

# --- what window does each press actually score from? -------------------------
print("=== declared scoring windows (kvpress 0.5.4, ratio=0.5) ===")
wins = {}
for a in ARMS:
    p = methods.build_method(a, 0.5)
    inner = getattr(p, "press", p)
    w = getattr(inner, "window_size", None)
    wins[a] = w
    extra = {k: v for k, v in vars(inner).items()
             if k in ("window_size", "kernel_size", "n_sink", "n_future_positions")}
    print(f"  {a:16s} class={type(inner).__name__:24s} window_size={w}  {extra}")
print(f"  floor constraint pins the last {N_WINDOW} positions (n_window)")

# --- where does the query sit? ------------------------------------------------
for M in MODELS:
    tok = AutoTokenizer.from_pretrained(M)
    rows = []
    for i in range(N):
        iid = f"grid_{i:05d}"
        sd = seed_key_without_seed(
            task="ledger", instance_id=iid, model=M, model_revision="x", arm="grid", B=1,
            protocol="agnostic", device="nvidia", backend="cuda-12.8", torch_version="t",
            transformers_version="5.2.0", kvpress_version="0.5.4", dtype="bfloat16")
        inst = ledger.build(sd, iid, target_tokens=2048, tokenizer=tok)
        for v in inst.variants:
            pre, post = aware_parts(tok, inst.context, v.query)
            enc = tok(pre, add_special_tokens=False, return_offsets_mapping=True)
            off = enc["offset_mapping"]
            n_ctx = len(off)
            cs = pre.rindex(v.query); ce = cs + len(v.query)
            qt = [ti for ti, (x, y) in enumerate(off) if y > x and x < ce and y > cs]
            q0, q1 = qt[0], qt[-1] + 1
            rows.append((n_ctx, q0, q1, n_ctx - q1, q1 - q0))
    n_ctx = statistics.fmean(r[0] for r in rows)
    q0 = statistics.fmean(r[1] for r in rows)
    q1 = statistics.fmean(r[2] for r in rows)
    tail = statistics.fmean(r[3] for r in rows)
    qlen = statistics.fmean(r[4] for r in rows)
    print(f"\n=== {M}  (n={len(rows)} prefixes) ===")
    print(f"  n_ctx {n_ctx:.1f}   query tokens [{q0:.1f}, {q1:.1f})   length {qlen:.1f}")
    print(f"  tokens AFTER the query in the prefix (template tail): {tail:.1f}")
    print(f"  query START is {n_ctx-q0:.1f} tokens from the end of the prefix")
    print(f"  {'method':16s} {'window':>7s} {'query fully inside window?':>28s} {'covered':>9s}")
    for a in ARMS:
        w = wins[a]
        if w is None:
            print(f"  {a:16s} {'n/a':>7s} {'query-agnostic by design':>28s} {'-':>9s}")
            continue
        # window covers positions [n_ctx-w, n_ctx)
        ins = [1.0 if (r[0] - w) <= r[1] else 0.0 for r in rows]
        cov = [max(0, r[2] - max(r[1], r[0] - w)) / (r[2] - r[1]) for r in rows]
        print(f"  {a:16s} {w:7d} {statistics.fmean(ins)*100:27.1f}% "
              f"{statistics.fmean(cov)*100:8.1f}%")
