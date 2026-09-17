"""Paper 2's ladder at 14B: I (information share) and Delta_head, one budget per invocation.

c~19 via LEDGER-C, N=80 (dropped from the curve's N=200 -- Paper 2's I and Delta_head need a
viable floor_pos and a squeezed causal oracle, not the competence-matched N the curve needed;
N=80 restores floor_pos viability at every candidate budget while full_cache stays in band
[0.9400 @ n=50], confirmed by probe_floor_viability.py before this script was written).

Payable candidate cost = H(=4) x achieved_c(~18.93) ~= 75.7 tokens, measured directly (not
assumed) at N=40/80/200 alike (c is N-invariant; only L moves with N). Budgets C in
{32, 64, 128} bracket it: 32 and 64 squeeze the causal oracle (C < 75.7), 128 does not
(C > 75.7, expected I ~ 0 there, the ceiling reference point).

Arms: full_cache, null, random, floor_pos, oracle_causal, oracle_prescient,
oracle_causal_perhead (Qwen2.5-14B has 8 KV heads). n=100, single C per invocation.

I = (A_presc - A_causal) / (A_presc - A_floor), paired bootstrap through the ratio (every
arm mean recomputed on the SAME resample, never divided independently).
Delta_head = A(perhead) - A(causal), paired bootstrap, at equal budget and equal candidate
count (oracle_causal_perhead's own construction refuses a cell where heads hold different
candidate counts, so a completed row already satisfies that condition).
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_p3_curve as P  # noqa: E402
import run_stage0 as R  # noqa: E402
from harness import ladder, press  # noqa: E402
from p3 import keys3, runner  # noqa: E402
from p3.tasks import ledger_c  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = HERE / "results"

N_SINK, N_WINDOW = P.N_SINK, P.N_WINDOW
N_HEADS = 8                      # Qwen2.5-14B-Instruct num_key_value_heads
N_RECORDS = 80
N_FIELDS_19 = 3
PAYABLE_CANDIDATE_COST = 4 * 18.933   # H x achieved c, measured directly (N-invariant)
ARMS = ("full_cache", "null", "random", "floor_pos", "oracle_causal", "oracle_prescient",
        "oracle_causal_perhead")
N_INST = 100
DEGENERATE_FLOOR = 0.05


def keeps_for_arm(arm, n_ctx, C, facts, sd, inst, perhead_press=None):
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
    if arm == "oracle_causal_perhead":
        return [set(v) for v in perhead_press.keep_by_head.values()]
    raise ValueError(arm)


def run_arm(model, tok, rec, arm, C):
    pre, posts, max_new = rec["pre"], rec["posts"], rec["max_new"]
    n_ctx, facts, sd, inst = rec["n_ctx"], rec["facts"], rec["seed"], rec["inst"]
    perhead_press = None
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
    elif arm == "oracle_causal_perhead":
        perhead_press = press.build_arm("oracle_causal_perhead", n_ctx=n_ctx, C=C,
                                        n_sink=N_SINK, n_window=N_WINDOW, facts=facts,
                                        n_heads=N_HEADS)[0]
        outs, flags = runner.generate_with(model, tok, pre, posts, perhead_press, max_new)
    else:
        p, _ = press.build_arm(arm, n_ctx=n_ctx, C=C, n_sink=N_SINK, n_window=N_WINDOW,
                               facts=facts, seed=sd)
        outs, flags = runner.generate_with(model, tok, pre, posts, p, max_new)
    wall = time.perf_counter() - t0

    keeps = keeps_for_arm(arm, n_ctx, C, facts, sd, inst, perhead_press)
    gold = {t for f in facts for s in f.spans for t in s}
    cm = P.completion_metrics(keeps, gold, facts)
    score = ledger_c.score_instance(outs, inst)
    per_variant = [ledger_c.score_one(o, v.answer) for o, v in zip(outs, inst.variants)]
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


def bootstrap_I(presc, causal, floor, seed, n_boot=10000):
    rng = np.random.default_rng(seed)
    p, c, f = (np.asarray(x, dtype=float) for x in (presc, causal, floor))
    n = len(p)
    idx = rng.integers(0, n, size=(n_boot, n))
    p_bs, c_bs, f_bs = p[idx].mean(axis=1), c[idx].mean(axis=1), f[idx].mean(axis=1)
    denom = p_bs - f_bs
    with np.errstate(divide="ignore", invalid="ignore"):
        I_bs = (p_bs - c_bs) / denom
    I_bs = np.sort(I_bs[np.isfinite(I_bs)])
    if I_bs.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    lo = I_bs[int(0.025 * len(I_bs))]
    hi = I_bs[min(len(I_bs) - 1, int(0.975 * len(I_bs)))]
    d = p.mean() - f.mean()
    point = (p.mean() - c.mean()) / d if d != 0 else float("nan")
    return float(point), (float(lo), float(hi))


def bootstrap_diff(a, b, seed, n_boot=10000):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    d_bs = np.sort(a[idx].mean(axis=1) - b[idx].mean(axis=1))
    lo, hi = d_bs[int(0.025 * n_boot)], d_bs[int(0.975 * n_boot)]
    return float(a.mean() - b.mean()), (float(lo), float(hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("C", type=int)
    ap.add_argument("--n", type=int, default=N_INST)
    args = ap.parse_args()
    C, n = args.C, args.n
    B = C + N_SINK + N_WINDOW
    squeezed = C < PAYABLE_CANDIDATE_COST

    calib = json.loads((R.OUT / "calibration.json").read_text(encoding="utf-8"))
    sp = dict(task="ledger_c", n_fields=N_FIELDS_19, n_records=N_RECORDS)
    print(f"loading model... (P2 ladder, c=19, N={N_RECORDS}, C={C}, B={B}, n={n}, "
          f"payable_cost={PAYABLE_CANDIDATE_COST:.1f}, "
          f"{'SQUEEZED' if squeezed else 'NOT squeezed (ceiling reference)'})", flush=True)
    model, tok = R.load_model()
    print("loaded.", flush=True)

    out_path = RUNS / f"p2ladder_C{C}.jsonl"
    done = load_done(out_path)
    keep_audit = {}
    n_ctx_seen = []
    t0 = time.time()
    written = skipped = failed = 0
    for i in range(n):
        rec = P.build_instance(19, sp, i, tok)
        rec["iid"] = f"p2ladder_c19_N{N_RECORDS}_{i:05d}"
        n_ctx_seen.append(rec["n_ctx"])
        for arm in ARMS:
            kd = dict(task="ledger_c", instance_id=rec["iid"], model=R.MODEL_NAME,
                      model_revision=R.MODEL_REV, arm=arm, B=B, C=C, seed=rec["seed"],
                      n_fields=N_FIELDS_19, n_records=N_RECORDS, layout="p2ladder",
                      matched_to=None, max_new=max(rec["max_new"]), **R.ENV)
            dg = keys3.digest(kd)
            if dg in done:
                skipped += 1
                continue
            try:
                row, keep0 = run_arm(model, tok, rec, arm, C)
            except AssertionError as e:
                failed += 1
                with out_path.with_suffix(".failures.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(dict(key_digest=dg, key=kd, error=str(e))) + "\n")
                continue
            full_row = dict(key_digest=dg, key=kd, task="ledger_c", instance_id=rec["iid"],
                            n_ctx=rec["n_ctx"], **row)
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(full_row) + "\n")
            done.add(dg)
            written += 1
            keep_audit[f"{rec['iid']}|{arm}"] = np.array(keep0, dtype=np.int32)
        if (i + 1) % 10 == 0 or i == n - 1:
            el = time.time() - t0
            rate = max(1e-9, (written + skipped) / el)
            remaining = (n * len(ARMS) - written - skipped - failed) / rate if rate > 0 else 0
            print(f"  [C={C}] inst {i + 1}/{n}  written {written} skipped {skipped} "
                  f"failed {failed}  {el / 60:.1f} min  {el / max(1, written):.2f} s/row  "
                  f"eta {remaining / 60:.1f} min", flush=True)

    audit_path = OUT / f"p2ladder_C{C}_keepset_audit.npz"
    if audit_path.exists():
        with np.load(audit_path) as prior:
            for k in prior.files:
                keep_audit.setdefault(k, prior[k])
    np.savez(audit_path, **keep_audit)

    # ---------------------------------------------------------------- aggregate
    rows = [json.loads(l) for l in out_path.open(encoding="utf-8") if l.strip()]
    by_arm = {a: [r["score"] for r in rows if r["arm"] == a] for a in ARMS}
    floor_scores = by_arm["floor_pos"]
    floor_mean = st.fmean(floor_scores) if floor_scores else float("nan")
    degenerate = floor_mean < DEGENERATE_FLOOR

    summary = dict(C=C, B=B, n=n, task="ledger_c", n_fields=N_FIELDS_19, n_records=N_RECORDS,
                   mean_n_ctx=st.fmean(n_ctx_seen), payable_candidate_cost=PAYABLE_CANDIDATE_COST,
                   squeezed=squeezed, floor_mean=floor_mean, degenerate_floor=degenerate,
                   n_failed=failed, arms={})
    for arm in ARMS:
        s = by_arm[arm]
        if not s:
            continue
        acc, ci = P.mean_ci(s, seed=zlib.crc32(f"p2ladder_acc|{C}|{arm}".encode()) & 0xFFFFFFFF)
        summary["arms"][arm] = dict(n=len(s), accuracy=acc, accuracy_ci=ci)

    if not degenerate and by_arm["oracle_prescient"] and by_arm["oracle_causal"]:
        I_point, I_ci = bootstrap_I(by_arm["oracle_prescient"], by_arm["oracle_causal"],
                                    floor_scores,
                                    seed=zlib.crc32(f"p2ladder_I|{C}".encode()) & 0xFFFFFFFF)
        summary["I"] = dict(value=I_point, ci=I_ci)
    else:
        summary["I"] = dict(value=None, ci=None, reason="degenerate floor_pos" if degenerate
                            else "missing arm data")

    if by_arm["oracle_causal_perhead"] and by_arm["oracle_causal"]:
        # Delta_head needs the SAME instances on both sides (paired); use only instances
        # where BOTH arms succeeded (a perhead failure drops that instance from the pair).
        ph_by_iid = {r["instance_id"]: r["score"] for r in rows if r["arm"] == "oracle_causal_perhead"}
        oc_by_iid = {r["instance_id"]: r["score"] for r in rows if r["arm"] == "oracle_causal"}
        common = sorted(set(ph_by_iid) & set(oc_by_iid))
        ph_paired = [ph_by_iid[k] for k in common]
        oc_paired = [oc_by_iid[k] for k in common]
        dh_point, dh_ci = bootstrap_diff(ph_paired, oc_paired,
                                         seed=zlib.crc32(f"p2ladder_dhead|{C}".encode())
                                         & 0xFFFFFFFF)
        summary["delta_head"] = dict(value=dh_point, ci=dh_ci, n_paired=len(common))
    else:
        summary["delta_head"] = dict(value=None, ci=None, reason="missing arm data")

    (OUT / f"p2ladder_C{C}_summary.json").write_text(json.dumps(summary, indent=2),
                                                      encoding="utf-8")

    print(f"\n=== C={C} SUMMARY (n={n}, N={N_RECORDS}, L~{summary['mean_n_ctx']:.0f}, "
          f"{'SQUEEZED' if squeezed else 'not squeezed'}) ===")
    print(f"floor_pos = {floor_mean:.4f} "
          f"{'-- DEGENERATE (<0.05)' if degenerate else '-- viable'}")
    for arm in ARMS:
        if arm not in summary["arms"]:
            continue
        e = summary["arms"][arm]
        print(f"  {arm:22s} acc={e['accuracy']:.4f} [{e['accuracy_ci'][0]:.4f},"
              f"{e['accuracy_ci'][1]:.4f}]  n={e['n']}")
    if summary["I"]["value"] is not None:
        print(f"\nI = {summary['I']['value']:.4f} [{summary['I']['ci'][0]:.4f},"
              f"{summary['I']['ci'][1]:.4f}]")
    else:
        print(f"\nI = N/A ({summary['I'].get('reason')})")
    if summary["delta_head"]["value"] is not None:
        print(f"Delta_head = {summary['delta_head']['value']:.4f} "
              f"[{summary['delta_head']['ci'][0]:.4f},{summary['delta_head']['ci'][1]:.4f}] "
              f"(n_paired={summary['delta_head']['n_paired']})")
    else:
        print(f"Delta_head = N/A ({summary['delta_head'].get('reason')})")

    el = time.time() - t0
    print(f"\nDONE C={C}: {written} written, {skipped} skipped, {failed} failed, "
          f"{el / 60:.1f} min, {el / max(1, written):.2f} s/row")


if __name__ == "__main__":
    raise SystemExit(main())
