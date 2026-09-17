"""Paper 4 Stage 2 runner. PINNED ENV (WSL /opt/p2venv), single process, RTX 5070.

Packages:
  verify   prefill only, real scores. Blocking gate before any generation:
             (a) X keep-set aggregates reproduce Paper 3 Stage 4's stored captures;
             (b) U-X with all-singleton units == X up to exact score ties (the identity);
             (c) at c=40, U-X touches fewer distinct units and completes more, at matched budget.
           Arms X and U-X for all four methods, c in {40, 1}, C=512, n=50.
  pilot    generation. c in {40, 1}, C=512, n=50, arms floor_pos, snapkv, adakv_snapkv,
           U-snapkv, U-adakv_snapkv. Keep-sets are captured DURING the generating prefill, so
           parity, floors and completion describe exactly the cache that answered.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import methods
from p3 import keys3, runner
from p4 import common as CM
from p4 import unitwrap as UW

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "nvidia"
C = 512
NL = "\n"
PILOT_ARMS = ("floor_pos", "snapkv", "adakv_snapkv", "U-snapkv", "U-adakv_snapkv")
VERIFY_ARMS = CM.X_METHODS + tuple("U-" + a for a in CM.X_METHODS)
IDENTITY_INSTANCES = (0, 1)


def load_done(path):
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["key_digest"])
                except Exception:
                    continue
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("package", choices=("verify", "pilot"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--c", type=int, choices=(40, 1), nargs="+", default=[40, 1])
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    tag = CM.TAGS[args.model]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    rev = getattr(model.config, "_commit_hash", None) or "unresolved"
    out = RUNS / f"p4_{args.package}_{tag}.jsonl"
    fail = out.with_suffix(".failures.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    arms = VERIFY_ARMS if args.package == "verify" else PILOT_ARMS
    B = C + CM.N_SINK + CM.N_WINDOW
    total = args.n * len(args.c) * len(arms)
    print(f"  {args.package} {tag} c={args.c} n={args.n}: {total} rows ({len(done)} present) "
          f"on {CM.PRODUCED_ON}", flush=True)

    t0 = time.time()
    written = skipped = failed = 0
    for c_tag in args.c:
        for i in range(args.n):
            iid = "s4_%05d" % i
            b = CM.build(tag, args.model, rev, tok, c_tag, iid)
            caps_x = {}
            for arm in arms:
                kd = dict(task=b["sp"]["task"], instance_id=iid, model=args.model,
                          model_revision=rev, arm=arm, B=B, C=C, seed=b["sd"],
                          n_fields=b["sp"]["n_fields"], n_records=b["sp"]["k"],
                          layout=f"p4_{args.package}_oracle_units", matched_to=None,
                          max_new=max(b["mns"]), **CM.ENV)
                dg = keys3.digest(kd)
                need_identity = (args.package == "verify" and i in IDENTITY_INSTANCES)
                if dg in done and not need_identity:
                    skipped += 1
                    continue
                t1 = time.time()
                try:
                    cap, stats = methods.Capture(), {}
                    p = CM.build_press(arm, b["n_ctx"], C, b["ui"], cap, stats)
                    base = CM.base_arm(arm)
                    if args.package == "verify":
                        CM.prefill(model, tok, b["pre"], p)
                        gen = {}
                    else:
                        outs, flags = CM.generate_checked(
                            model, tok, b["pre"], b["posts"], p, b["mns"], b["n_ctx"], B,
                            headwise=base.startswith("adakv"))
                        gen = dict(score=CM.S4.SCORERS[b["sp"]["task"]](outs, b["inst"]),
                                   per_variant=[CM.S4.SCORE_ONE[b["sp"]["task"]](o, v.answer)
                                                for v, o in zip(b["inst"].variants, outs)],
                                   gen=outs, answers=[v.answer for v in b["inst"].variants],
                                   stop_flags=flags)
                    parity = runner.assert_budget_parity(cap, C, b["n_ctx"], base)
                    keeps = [cap.per_head[h] for h in cap.heads()]
                    met = CM.keep_metrics(keeps, b["facts"], b["line_units"], b["n_ctx"])
                    if not met["floor_ok"]:
                        raise AssertionError(f"{arm}: mandatory floors not retained")
                    if arm.startswith("U-") and stats.get("score_calls") != len(
                            {li for li, _ in cap.per_head}):
                        raise AssertionError(f"{arm}: score() not called exactly once per layer")
                    if not arm.startswith("U-"):
                        caps_x[arm] = cap
                    if dg not in done:
                        row = dict(key_digest=dg, key=kd, produced_on=CM.PRODUCED_ON,
                                   model_tag=tag, package=args.package, task=b["sp"]["task"],
                                   label="c=%d" % c_tag, c=b["sp"]["c"], C=C, B=B, arm=arm,
                                   instance=i, n_ctx=b["n_ctx"], B_asserted=parity,
                                   units_taken=stats.get("units_taken"),
                                   fallback=stats.get("fallback"),
                                   wall_s=time.time() - t1, **met, **gen)
                        with out.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(row) + NL)
                        done.add(dg)
                        written += 1
                    else:
                        skipped += 1
                    if need_identity and arm.startswith("U-"):
                        ui0 = UW.UnitIndex(b["n_ctx"], [], CM.N_SINK, CM.N_WINDOW)
                        cap0, st0 = methods.Capture(), {"store_scores": True}
                        p0 = CM.build_press(arm, b["n_ctx"], C, ui0, cap0, st0)
                        CM.prefill(model, tok, b["pre"], p0)
                        idr = CM.identity_check(caps_x[base], cap0, st0["scores"], arm)
                        row = dict(produced_on=CM.PRODUCED_ON, model_tag=tag, package="identity",
                                   label="c=%d" % c_tag, arm=arm, instance=i, **idr)
                        with (RUNS / f"p4_identity_{tag}.jsonl").open("a", encoding="utf-8") as f:
                            f.write(json.dumps(row) + NL)
                except Exception as e:
                    failed += 1
                    with fail.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(dict(key_digest=dg, key=kd,
                                                error="%s: %s" % (type(e).__name__, e))) + NL)
                    print("  FAIL", iid, c_tag, arm, type(e).__name__, e, flush=True)
            if (i + 1) % 10 == 0 or i == args.n - 1:
                el = time.time() - t0
                print("  [%s/%s c=%d] inst %d/%d written %d skipped %d failed %d  %.1f min"
                      % (args.package, tag, c_tag, i + 1, args.n, written, skipped, failed,
                         el / 60), flush=True)
    print("  DONE %s/%s: written %d skipped %d failed %d in %.1f min"
          % (args.package, tag, written, skipped, failed, (time.time() - t0) / 60), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
