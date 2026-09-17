"""Paper 3 curve replication at 14B, single cell C=512, one c_tag per invocation.

c=1 via MARK-1 (N unaffected), c in {8,19,40} via LEDGER-C at N=200 -- the value the anchor
gate found in-band for Qwen2.5-14B-Instruct (N=40 saturated near ceiling for this model).
Arms: full ladder (full_cache, null, random, floor_pos, oracle_causal, oracle_prescient) plus
the four admitted methods (snapkv, adakv_snapkv, expected_attn, keydiff). n=100, single
process, single C=512 cell (B = C + 72 = 584).

Reuses Paper 3's harness unmodified: harness.{ladder,methods,press}, p3.{keys3,runner},
p3.tasks.{ledger_c,mark1}. Rows are appended to a resumable JSONL (dedup by key_digest, same
convention as paper3/stage4_run.py) so a crash never loses completed arm-instances.

Completion accounting (p_g, q_complete, q_any) is computed uniformly for every arm from its
keep-set(s): a single global set for full_cache/null/random/floor_pos/oracle_causal, one set
per query for oracle_prescient (query-dependent by construction), and the real captured
per-(layer, KV-head) sets for the four scorer methods (captured during the SAME prefill used
for generation, via methods.make_capturing -- no separate capture pass). Budget parity is
asserted per instance from the captured keep-sets for every method arm before the row is
written.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_stage0 as R  # noqa: E402
from harness import ladder, methods, press  # noqa: E402
from p3 import keys3, runner  # noqa: E402
from p3.tasks import ledger_c, mark1  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = HERE / "results"
RUNS.mkdir(parents=True, exist_ok=True)

N_SINK, N_WINDOW = runner.N_SINK, runner.N_WINDOW
C = 512
B = C + N_SINK + N_WINDOW
N_LEDGER = 200
LADDER_ARMS = ("full_cache", "null", "random", "floor_pos", "oracle_causal", "oracle_prescient")
METHOD_ARMS = ("snapkv", "adakv_snapkv", "expected_attn", "keydiff")
ALL_ARMS = LADDER_ARMS + METHOD_ARMS
DEGENERATE_FLOOR = 0.05


def spec_for(c_tag, calib):
    if c_tag == 1:
        return dict(task="mark1", n_fields=None, n_records=mark1.N_ENTRIES)
    nf = calib["chosen"][str(c_tag)]["n_fields"]
    return dict(task="ledger_c", n_fields=nf, n_records=N_LEDGER)


def build_instance(c_tag, sp, i, tok):
    iid = f"p3curve_c{c_tag}_{i:05d}"
    sd = keys3.instance_seed(task=sp["task"], instance_id=iid, model=R.MODEL_NAME,
                             model_revision=R.MODEL_REV, n_fields=sp["n_fields"],
                             n_records=sp["n_records"], layout=None, **R.ENV)
    if sp["task"] == "mark1":
        inst = mark1.build(sd, iid, target_tokens=2048, tokenizer=tok)
        max_new = [8] * len(inst.variants)
    else:
        inst = ledger_c.build(sd, iid, n_fields=sp["n_fields"], n_records=sp["n_records"],
                              target_tokens=2048, tokenizer=tok)
        max_new = [len(tok(v.answer, add_special_tokens=False)["input_ids"]) + runner.MAX_NEW_SLACK
                   for v in inst.variants]
    pre, _ = runner.templated_parts(tok, inst.context, "")
    facts, n_ctx = runner.facts_and_ctx(inst, tok, pre)
    posts = [runner.templated_parts(tok, inst.context, v.query)[1] for v in inst.variants]
    return dict(iid=iid, seed=sd, inst=inst, pre=pre, facts=facts, n_ctx=n_ctx,
                posts=posts, max_new=max_new, task=sp["task"])


def keeps_for_arm(arm, n_ctx, facts, sd, inst, cap=None):
    if arm == "full_cache":
        return [set(range(n_ctx))]
    if arm == "null":
        return [set(ladder.null_arm(n_ctx, C, N_SINK, N_WINDOW).kept)]
    if arm == "random":
        return [set(ladder.random_arm(n_ctx, C, sd, N_SINK, N_WINDOW, facts).kept)]
    if arm == "floor_pos":
        return [set(ladder.floor_pos(n_ctx, C, N_SINK, N_WINDOW, facts).kept)]
    if arm == "oracle_causal":
        return [set(ladder.oracle_causal(n_ctx, C, facts, n_sink=N_SINK, n_window=N_WINDOW).kept)]
    if arm == "oracle_prescient":
        keeps = []
        for v in inst.variants:
            gf = next(f for f in facts if f.fact_id == v.rec_id)
            sel = ladder.oracle_prescient(n_ctx, C, gf, N_SINK, N_WINDOW, facts)
            keeps.append(set(sel.kept))
        return keeps
    return [cap.per_head[h] for h in cap.heads()]


def completion_metrics(keeps, gold, facts):
    pg = [len(gold & kk) / max(1, len(gold)) for kk in keeps]
    qc = [sum(1 for f in facts if f.is_complete_in(kk)) / len(facts) for kk in keeps]
    q_any = st.fmean([max(1.0 if f.is_complete_in(kk) else 0.0 for kk in keeps) for f in facts])
    return dict(p_g=st.fmean(pg), q_complete=st.fmean(qc), q_any=q_any, n_slots=len(keeps))


def run_arm(model, tok, rec, arm):
    pre, posts, max_new = rec["pre"], rec["posts"], rec["max_new"]
    n_ctx, facts, sd, inst = rec["n_ctx"], rec["facts"], rec["seed"], rec["inst"]
    cap = None
    t0 = time.perf_counter()
    if arm == "oracle_prescient":
        outs, flags = [], []
        for v, post, mn in zip(inst.variants, posts, max_new):
            gf = next(f for f in facts if f.fact_id == v.rec_id)
            p, _ = press.build_arm("oracle_prescient", n_ctx=n_ctx, C=C, n_sink=N_SINK,
                                   n_window=N_WINDOW, facts=facts, gold=gf, seed=sd)
            o, fl = runner.generate_with(model, tok, pre, [post], p, [mn])
            outs.append(o[0])
            flags.append(fl[0])
    elif arm in LADDER_ARMS:
        p, _ = press.build_arm(arm, n_ctx=n_ctx, C=C, n_sink=N_SINK, n_window=N_WINDOW,
                               facts=facts, seed=sd)
        outs, flags = runner.generate_with(model, tok, pre, posts, p, max_new)
    else:
        ratio = press._ratio_for(B, n_ctx)
        base = methods.make_floor_constrained(methods.build_method(arm, ratio),
                                              n_ctx, N_SINK, N_WINDOW)
        cap = methods.Capture()
        p = methods.make_capturing(base, cap)
        p.compression_ratio = ratio
        outs, flags = runner.generate_with(model, tok, pre, posts, p, max_new)
        runner.assert_budget_parity(cap, C, n_ctx, arm)
    wall = time.perf_counter() - t0

    keeps = keeps_for_arm(arm, n_ctx, facts, sd, inst, cap)
    gold = {t for f in facts for s in f.spans for t in s}
    cm = completion_metrics(keeps, gold, facts)
    scorer = mark1.score_instance if rec["task"] == "mark1" else ledger_c.score_instance
    one = mark1.score_one if rec["task"] == "mark1" else ledger_c.score_one
    score = scorer(outs, inst)
    per_variant = [one(o, v.answer) for o, v in zip(outs, inst.variants)]
    row = dict(arm=arm, score=score, per_variant=per_variant, stop_flags=flags,
               wall_s=wall, gen=outs, **cm)
    return row, (sorted(keeps[0]) if keeps else [])


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


def bootstrap_ratio_ci(method_scores, floor_scores, seed, n_boot=10000):
    rng = np.random.default_rng(seed)
    m = np.asarray(method_scores, dtype=float)
    f = np.asarray(floor_scores, dtype=float)
    n = len(m)
    idx = rng.integers(0, n, size=(n_boot, n))
    m_bs = m[idx].mean(axis=1)
    f_bs = f[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_bs = m_bs / f_bs
    ratio_bs = np.sort(ratio_bs[np.isfinite(ratio_bs)])
    if ratio_bs.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    lo = ratio_bs[int(0.025 * len(ratio_bs))]
    hi = ratio_bs[min(len(ratio_bs) - 1, int(0.975 * len(ratio_bs)))]
    point = m.mean() / f.mean() if f.mean() != 0 else float("nan")
    return float(point), (float(lo), float(hi))


def mean_ci(v, seed, n_boot=10000):
    rng = np.random.default_rng(seed)
    a = np.asarray(v, dtype=float)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    bs = np.sort(a[idx].mean(axis=1))
    return float(a.mean()), (float(bs[int(0.025 * n_boot)]), float(bs[int(0.975 * n_boot)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("c_tag", type=int, choices=(1, 8, 19, 40))
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    c_tag, n = args.c_tag, args.n

    calib = json.loads((R.OUT / "calibration.json").read_text(encoding="utf-8"))
    sp = spec_for(c_tag, calib)
    print(f"loading model... (c={c_tag}, task={sp['task']}, n_fields={sp['n_fields']}, "
          f"n_records={sp['n_records']}, n={n}, C={C}, B={B})", flush=True)
    model, tok = R.load_model()
    print("loaded.", flush=True)

    out_path = RUNS / f"p3curve_c{c_tag}.jsonl"
    done = load_done(out_path)
    keep_audit = {}
    n_ctx_seen = []
    t0 = time.time()
    written = skipped = 0
    for i in range(n):
        rec = build_instance(c_tag, sp, i, tok)
        n_ctx_seen.append(rec["n_ctx"])
        for arm in ALL_ARMS:
            kd = dict(task=sp["task"], instance_id=rec["iid"], model=R.MODEL_NAME,
                      model_revision=R.MODEL_REV, arm=arm, B=B, C=C, seed=rec["seed"],
                      n_fields=sp["n_fields"], n_records=sp["n_records"], layout="p3curve",
                      matched_to=None, max_new=max(rec["max_new"]), **R.ENV)
            dg = keys3.digest(kd)
            if dg in done:
                skipped += 1
                continue
            row, keep0 = run_arm(model, tok, rec, arm)
            full_row = dict(key_digest=dg, key=kd, c_tag=c_tag, task=sp["task"],
                            instance_id=rec["iid"], n_ctx=rec["n_ctx"], **row)
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(full_row) + "\n")
            done.add(dg)
            written += 1
            keep_audit[f"{rec['iid']}|{arm}"] = np.array(keep0, dtype=np.int32)
        if (i + 1) % 10 == 0 or i == n - 1:
            el = time.time() - t0
            rate = max(1e-9, (written + skipped) / el)
            remaining = (n * len(ALL_ARMS) - written - skipped) / rate if rate > 0 else 0
            print(f"  [c={c_tag}] inst {i + 1}/{n}  written {written} skipped {skipped}  "
                  f"{el / 60:.1f} min elapsed  {el / max(1, written):.2f} s/row  "
                  f"eta {remaining / 60:.1f} min", flush=True)

    audit_path = OUT / f"p3curve_c{c_tag}_keepset_audit.npz"
    if audit_path.exists():
        with np.load(audit_path) as prior:
            for k in prior.files:
                keep_audit.setdefault(k, prior[k])
    np.savez(audit_path, **keep_audit)

    # ---------------------------------------------------------------- aggregate
    rows = [json.loads(l) for l in out_path.open(encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["c_tag"] == c_tag]
    by_arm = {a: [r for r in rows if r["arm"] == a] for a in ALL_ARMS}
    floor_scores = [r["score"] for r in by_arm["floor_pos"]]
    floor_mean = st.fmean(floor_scores) if floor_scores else float("nan")
    degenerate = floor_mean < DEGENERATE_FLOOR

    summary = dict(c_tag=c_tag, n=n, C=C, B=B, task=sp["task"], n_fields=sp["n_fields"],
                   n_records=sp["n_records"], mean_n_ctx=st.fmean(n_ctx_seen),
                   floor_mean=floor_mean, degenerate_floor=degenerate, arms={})
    for arm in ALL_ARMS:
        rs = by_arm[arm]
        if not rs:
            continue
        scores = [r["score"] for r in rs]
        acc, acc_ci = mean_ci(scores, seed=zlib.crc32(f"acc|{c_tag}|{arm}".encode()) & 0xFFFFFFFF)
        entry = dict(n=len(rs), accuracy=acc, accuracy_ci=acc_ci,
                    p_g=st.fmean(r["p_g"] for r in rs),
                    q_complete=st.fmean(r["q_complete"] for r in rs),
                    q_any=st.fmean(r["q_any"] for r in rs),
                    eos_rate=sum(1 for r in rs for fl in r["stop_flags"] if fl == "eos") /
                             sum(len(r["stop_flags"]) for r in rs))
        if arm not in ("floor_pos",) and not degenerate and arm in METHOD_ARMS:
            ratio, ci = bootstrap_ratio_ci(scores, floor_scores,
                                           seed=zlib.crc32(f"ratio|{c_tag}|{arm}".encode())
                                           & 0xFFFFFFFF)
            entry["ratio_vs_floor"] = ratio
            entry["ratio_vs_floor_ci"] = ci
        summary["arms"][arm] = entry
    (OUT / f"p3curve_c{c_tag}_summary.json").write_text(json.dumps(summary, indent=2),
                                                        encoding="utf-8")

    print(f"\n=== c={c_tag} SUMMARY (n={n}, C={C}, N_records={sp['n_records']}, "
          f"L~{summary['mean_n_ctx']:.0f}) ===")
    print(f"floor_pos accuracy = {floor_mean:.4f} "
          f"{'-- DEGENERATE (<0.05), cell EXCLUDED' if degenerate else ''}")
    for arm in ALL_ARMS:
        if arm not in summary["arms"]:
            continue
        e = summary["arms"][arm]
        ratio_txt = ""
        if "ratio_vs_floor" in e:
            lo, hi = e["ratio_vs_floor_ci"]
            ratio_txt = f"  ratio={e['ratio_vs_floor']:.3f}x [{lo:.3f}, {hi:.3f}]"
        print(f"  {arm:16s} acc={e['accuracy']:.4f} [{e['accuracy_ci'][0]:.4f},"
              f"{e['accuracy_ci'][1]:.4f}]  p_g={e['p_g']:.3f} q_complete={e['q_complete']:.3f}"
              f"{ratio_txt}")

    el = time.time() - t0
    total_rows = n * len(ALL_ARMS)
    print(f"\nDONE c={c_tag}: {written} written, {skipped} skipped, {el / 60:.1f} min, "
          f"{el / max(1, written):.2f} s/row, {total_rows} total rows this cell")


if __name__ == "__main__":
    raise SystemExit(main())
