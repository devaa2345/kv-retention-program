"""Dilution probe: SnapKV scored from a window sized to the QUERY, not to 64 tokens.

Item 1 established that the query (18-20 tokens) sits fully inside SnapKV's 64-token
observation window, so the attenuated aware delta is not a window-ALIGNMENT fault. But only
~31% of that window is question; the remaining ~44 tokens are ledger filler, against ~100%
question in the LongBench-style setups the +0.20 reference comes from. If that ~3x dilution is
the cause, sizing the window to the query should recover a positive delta.

  positive delta  -> attenuated SnapKV result is a TASK PROPERTY (short query, long synthetic
                     context), not a harness fault
  still ~zero     -> dilution is not the explanation and the harness question stays open

**This says nothing about KeyDiff's +0.3375.** That is a separate mechanism -- answering without
retaining gold (0.4364 aware vs 0.0545 agnostic on non-retained instances) -- and needs its own
treatment. Do not read this probe as addressing it.

Baseline (window_size=64) and the agnostic arm are read from the canonical n=200 grids on disk
rather than re-run: identical instances, seeds and code path, so the contrast is paired.
Records are written under protocol="aware_windowprobe" so they can never collide with a
canonical cell.
"""
from __future__ import annotations
import json, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import methods, press, stats
from harness.keys import RecordKey, assert_owned_by, seed_key_without_seed
from harness.tasks import ledger
from stage5_ladder_validation import generate_with
from run_grid_aware import aware_parts, spans_over

M = "Qwen/Qwen2.5-3B-Instruct"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
BUDGETS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "128,512").split(",")]
N_SINK, N_WINDOW = 8, 64
OUT = Path("runs/nvidia/diag_snapkv_window_M2_aware.jsonl")
ENV = dict(backend="cuda-12.8", transformers_version="5.2.0", kvpress_version="0.5.4",
           dtype="bfloat16")

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
rev = getattr(model.config, "_commit_hash", None) or "x"


def load(path, arm):
    by = {}
    p = Path(path)
    if not p.exists():
        return by
    for line in p.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r["key"]["arm"] == arm:
            by[(r["C"], r["key"]["instance_id"])] = r["score"]
    return by


base_aware = load("runs/nvidia/grid_M2_ledger_aware.jsonl", "snapkv")
base_agn = load("runs/nvidia/grid_M2_ledger_agnostic.jsonl", "snapkv")
print(f"  baseline on disk: aware {len(base_aware)} cells, agnostic {len(base_agn)} cells")

done = set()
if OUT.exists():
    for line in OUT.open(encoding="utf-8"):
        try: done.add(json.loads(line)["key_digest"])
        except Exception: pass
print(f"  resuming: {len(done)} probe records present")

probe = {C: {} for C in BUDGETS}
qlens = []
for i in range(N):
    iid = f"grid_{i:05d}"
    sd = seed_key_without_seed(
        task="ledger", instance_id=iid, model=M, model_revision=rev, arm="grid", B=1,
        protocol="agnostic", device="nvidia", torch_version=torch.__version__,
        seed=0, **ENV)
    inst = ledger.build(sd, iid, target_tokens=2048, tokenizer=tok)

    for C in BUDGETS:
        k = RecordKey(task="ledger", instance_id=iid, model=M, model_revision=rev,
                      arm="snapkv_wq", B=C + N_SINK + N_WINDOW,
                      protocol="aware_windowprobe", device="nvidia",
                      torch_version=torch.__version__, seed=sd, batch_size=1, **ENV)
        assert_owned_by(k, "nvidia")
        if k.digest() in done:
            continue
        sc = []
        for v in inst.variants:
            pre, post = aware_parts(tok, inst.context, v.query)
            _, n_ctx = spans_over(inst, tok, pre)
            qlen = len(tok(v.query, add_special_tokens=False)["input_ids"])
            qlens.append(qlen)
            ratio = press._ratio_for(C + N_SINK + N_WINDOW, n_ctx)
            sk = methods.build_method("snapkv", ratio)
            sk.window_size = qlen                      # <-- the only change
            p = methods.make_floor_constrained(sk, n_ctx, N_SINK, N_WINDOW)
            p.compression_ratio = ratio
            o = generate_with(model, tok, pre, [post], p)
            sc.append(1.0 if v.answer in o[0] else 0.0)
        score = statistics.fmean(sc)
        probe[C][iid] = score
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key_digest": k.digest(), "key": k.as_dict(),
                                "score": score, "C": C,
                                "window_size": qlens[-1]}) + "\n")
        done.add(k.digest())
    if (i + 1) % 25 == 0:
        print(f"  [{i+1}/{N}]", flush=True)

# reload so a resumed run analyses everything
probe = {C: {} for C in BUDGETS}
for line in OUT.open(encoding="utf-8"):
    r = json.loads(line)
    if r["C"] in probe:
        probe[r["C"]][r["key"]["instance_id"]] = r["score"]

print(f"\n{M}  aware protocol   SnapKV window_size: 64 (baseline) vs query length "
      f"({statistics.fmean(qlens):.1f} tokens) " if qlens else "")
print(f"{'C':>5} {'arm':14s} {'agnostic':>9s} {'aware':>8s} {'delta':>9s} {'95% CI':>20s}")
for C in BUDGETS:
    ids = sorted(set(probe[C]) & {i for (c, i) in base_aware if c == C}
                 & {i for (c, i) in base_agn if c == C})
    if not ids:
        print(f"{C:5d}  no paired instances"); continue
    ag = [base_agn[(C, i)] for i in ids]
    w64 = [base_aware[(C, i)] for i in ids]
    wq = [probe[C][i] for i in ids]
    d64 = stats.paired_contrast(w64, ag, seed=C, n_boot=20000)
    dwq = stats.paired_contrast(wq, ag, seed=C, n_boot=20000)
    dpr = stats.paired_contrast(wq, w64, seed=C + 7, n_boot=20000)
    print(f"{C:5d} {'snapkv w=64':14s} {statistics.fmean(ag):9.4f} "
          f"{statistics.fmean(w64):8.4f} {d64.mean_diff:+9.4f} "
          f"{str([round(d64.ci_low,4), round(d64.ci_high,4)]):>20s}   (n={len(ids)})")
    print(f"{C:5d} {'snapkv w=query':14s} {statistics.fmean(ag):9.4f} "
          f"{statistics.fmean(wq):8.4f} {dwq.mean_diff:+9.4f} "
          f"{str([round(dwq.ci_low,4), round(dwq.ci_high,4)]):>20s}")
    verdict = ("DILUTION EXPLAINS IT (delta turns positive)" if dwq.ci_low > 0
               else "dilution does NOT explain it (delta still not positive)")
    print(f"      probe - baseline = {dpr.mean_diff:+.4f} "
          f"[{dpr.ci_low:.4f}, {dpr.ci_high:.4f}]   -> {verdict}\n")
print("  NOTE: this probe does not address KeyDiff's +0.3375; that is a separate mechanism.")
