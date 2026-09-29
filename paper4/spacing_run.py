"""PREREG_SPACING.md (31f318d9...) generation runner. Verifies the prereg hash before running.

Generates `spaced-U-snapkv` for the SAME instances/seeds/c/C as the existing unconstrained
`U-snapkv` rows in `runs/nvidia/p4_s3_{tag}.jsonl` (c=40, C=512), so the comparison is paired.
Per instance, the target `units_complete` is read from that stored row (K=24 relaxation target).
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import methods, press
from p3 import runner
from p4 import common as CM
from p4 import probes as PR
from p4.unitwrap import UnitIndex

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "nvidia"
PREREG_SHA = "893ca6618f2ede10e7e173939a29efdb15810d8cc7f80a929735061ea271a67a"
K = 24
C_TAG, C_BUDGET, N = 40, 512, 50
NL = "\n"


def check_prereg():
    from hash_file import digest
    got = digest(HERE / "PREREG_SPACING.md")
    if got != PREREG_SHA:
        raise SystemExit(f"PREREG_SPACING.md changed since freezing{NL}  recorded {PREREG_SHA}"
                         f"{NL}  actual   {got}")
    print(f"  PREREG_SPACING.md verified at {got}", flush=True)


_TARGET_CACHE = {}


def stored_target(tag, instance, base_method):
    """units_complete of the existing unconstrained U-<base_method> row for this instance.
    Checks the frozen Stage 3 grid file (instances 0-49) first, then the spacing test's own
    extra-unconstrained file (instances >= 50, from a raised-n scale-up) -- never writes to or
    depends on Stage 3's file having more than its own frozen n=50 rows."""
    u_arm = "U-" + base_method
    if tag not in _TARGET_CACHE:
        cache = {}
        for fname in (f"p4_s3_{tag}.jsonl", f"p4_spacing_extra_unconstrained_{tag}.jsonl"):
            p = RUNS / fname
            if not p.exists():
                continue
            for line in p.open(encoding="utf-8"):
                r = json.loads(line)
                if r.get("c_tag") == C_TAG and r.get("C") == C_BUDGET and "arm" in r:
                    cache[(r["arm"], r["instance"])] = r["units_complete"]
        _TARGET_CACHE[tag] = cache
    key = (u_arm, instance)
    if key not in _TARGET_CACHE[tag]:
        raise SystemExit(f"no stored {u_arm} row for {tag} instance {instance} at c={C_TAG} C={C_BUDGET}")
    return _TARGET_CACHE[tag][key]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--base", default="snapkv", choices=("snapkv", "expected_attn", "adakv_snapkv"),
                    help="base scorer/press to wrap (adakv_snapkv uses the S2 cross-head allocator)")
    a = ap.parse_args()
    check_prereg()
    session_id = uuid.uuid4().hex
    tag = CM.TAGS[a.model]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    rev = getattr(model.config, "_commit_hash", None) or "unresolved"
    out = RUNS / f"p4_spacing_{a.base}_{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    B = C_BUDGET + CM.N_SINK + CM.N_WINDOW
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            done.add(json.loads(line)["instance"])

    t0 = time.time()
    n_fail = 0
    for i in range(a.start, a.n):
        if i in done:
            continue
        iid = "s4_%05d" % i
        b = CM.build(tag, a.model, rev, tok, C_TAG, iid)
        target = stored_target(tag, i, a.base)
        t1 = time.time()
        try:
            cap, stats = methods.Capture(), {}
            ratio = press._ratio_for(B, b["n_ctx"])
            base = methods.make_floor_constrained(
                methods.build_method(a.base, ratio), b["n_ctx"], CM.N_SINK, CM.N_WINDOW)
            target_box = {"target": target}
            headwise = a.base.startswith("adakv")
            if headwise:
                p = PR.make_spaced_adakv_press(base, b["ui"], C_BUDGET, K, b["line_units"],
                                               target_box, capture=cap, stats=stats)
            else:
                p = PR.make_spaced_press(base, b["ui"], C_BUDGET, K, b["line_units"], target_box,
                                         capture=cap, stats=stats)
            p.compression_ratio = ratio
            outs, flags = CM.generate_checked(model, tok, b["pre"], b["posts"], p, b["mns"],
                                              b["n_ctx"], B, headwise=headwise)
            parity = runner.assert_budget_parity(cap, C_BUDGET, b["n_ctx"], a.base)
            keeps = [cap.per_head[h] for h in cap.heads()]
            met = CM.keep_metrics(keeps, b["facts"], b["line_units"], b["n_ctx"])
            if not met["floor_ok"]:
                raise AssertionError("mandatory floors not retained")
            arm_name = "spaced-U-" + a.base
            row = dict(key=dict(task=b["sp"]["task"], instance_id=iid, model=a.model,
                                model_revision=rev, arm=arm_name, B=B, C=C_BUDGET,
                                seed=b["sd"], K=K), session_id=session_id,
                       produced_on=CM.PRODUCED_ON, model_tag=tag, c_tag=C_TAG, C=C_BUDGET, B=B,
                       arm=arm_name, instance=i, n_ctx=b["n_ctx"], B_asserted=parity,
                       target_units_complete=target, n_relaxations=stats.get("n_relaxations"),
                       violations_relaxed=stats.get("violations_relaxed"),
                       layers_fully_relaxed=stats.get("layers_fully_relaxed"),
                       score=CM.S4.SCORERS[b["sp"]["task"]](outs, b["inst"]),
                       per_variant=[CM.S4.SCORE_ONE[b["sp"]["task"]](o, v.answer)
                                    for v, o in zip(b["inst"].variants, outs)],
                       gen=outs, answers=[v.answer for v in b["inst"].variants],
                       stop_flags=flags, wall_s=time.time() - t1, **met)
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + NL)
        except Exception as e:
            n_fail += 1
            print(f"  FAIL inst {i}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"  [{tag}] inst {i+1}/{a.n}  {(time.time()-t0)/60:.1f} min  fails {n_fail}",
                  flush=True)
    print(f"  DONE spacing {a.base} {tag}: {(time.time()-t0)/60:.1f} min, {n_fail} failures",
          flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
