"""Paper 4 Stage 3 grid runner. Frozen prereg PREREG_P4_S3.md, verified at start-up.

Single process, RTX 5070, no pipes. One cell = (c, C) x 10 arms x n=50 instances; all rows of a
cell are produced by one process, and the analysis scores a cell only if its rows share one
session_id (prereg section 5.12). An interrupted cell is discarded and re-run whole.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import methods, press
from p3 import keys3, runner
from p4 import common as CM
from p4.floorcheck import assert_floor_pos

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "nvidia"
PREREG_SHA = "843c788cd8a2c99fe55c19e47639d2dbbe01b86c063452140c7ec6e7a0e1cbc2"
CONFOUND_WATCH = ("snapkv", "adakv_snapkv", "expected_attn", "keydiff")   # A3.4, widened to all

BUDGETS = (32, 64, 128, 256, 512)
ARMS = ("floor_pos", "oracle_causal", "snapkv", "adakv_snapkv", "expected_attn", "keydiff",
        "U-snapkv", "U-adakv_snapkv", "U-expected_attn", "U-keydiff")
N = 50
ENV3 = dict(CM.ENV, torch_version="2.11.0+cu128")   # dedup key records it (prereg 4); the
NL = "\n"                                           # instance seed does NOT (amendment A2)


def check_prereg():
    from hash_file import digest
    got = digest(HERE / "PREREG_P4_S3.md")
    if got != PREREG_SHA:
        raise SystemExit(f"PREREG_P4_S3.md has changed since freezing{NL}  recorded {PREREG_SHA}"
                         f"{NL}  actual   {got}")
    print(f"  PREREG_P4_S3.md verified at {got}", flush=True)


def metrics3(keeps, facts, line_units, n_ctx):
    """prereg 3.3: q_slot, q_any, q_maj together -- none is 'the' usability measure."""
    m = CM.keep_metrics(keeps, facts, line_units, n_ctx)
    n_slots = len(keeps)
    q_maj = sum(1 for f in facts
                if sum(1 for kk in keeps if f.is_complete_in(kk)) > n_slots / 2) / len(facts)
    m["q_slot"] = m.pop("q_complete")
    m["q_maj"] = q_maj
    return m


def build_press3(arm, b, C, cap, stats):
    if arm == "oracle_causal":
        p, _ = press.build_arm("oracle_causal", n_ctx=b["n_ctx"], C=C, n_sink=CM.N_SINK,
                               n_window=CM.N_WINDOW, facts=b["facts"], seed=b["sd"])
        p = methods.make_capturing(p, cap)
        p.compression_ratio = press._ratio_for(C + CM.N_SINK + CM.N_WINDOW, b["n_ctx"])
        return p
    return CM.build_press(arm, b["n_ctx"], C, b["ui"], cap, stats)


def load_state(path):
    """rows by cell, plus their session ids -- for the single-session resume rule."""
    rows = defaultdict(list)
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                rows[(r["c_tag"], r["C"])].append(r)
    done, drop = set(), set()
    for cell, rr in rows.items():
        sessions = {r.get("session_id") for r in rr}
        covered = {(r["arm"], r["instance"]) for r in rr}
        if len(sessions) == 1 and len(covered) == len(ARMS) * N and len(rr) == len(ARMS) * N:
            done.add(cell)
        else:
            drop.add(cell)
    return rows, done, drop


def prune(path, drop):
    """Remove rows of incomplete or session-mixed cells, so no cell can span a session."""
    if not path.exists() or not drop:
        return 0
    kept, removed = [], 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if (r["c_tag"], r["C"]) in drop:
                removed += 1
            else:
                kept.append(line)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(kept), encoding="utf-8", newline="")
    tmp.replace(path)
    return removed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--c", type=int, nargs="+", required=True, choices=(1, 8, 19, 40))
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--C", type=int, nargs="+", default=list(BUDGETS), choices=list(BUDGETS),
                    help="restrict to these budgets (A4 re-runs one claim-bearing cell at a time)")
    a = ap.parse_args()
    budgets = tuple(c for c in BUDGETS if c in a.C)
    check_prereg()
    session_id = uuid.uuid4().hex
    tag = CM.TAGS[a.model]
    out = RUNS / f"p4_s3_{tag}.jsonl"
    fail = out.with_suffix(".failures.jsonl")
    prog = HERE / "out" / "stage3_progress.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    _, done, drop = load_state(out)
    drop = {c for c in drop if c[0] in a.c}
    removed = prune(out, drop)
    print(f"  session {session_id[:8]} | {tag} | c={a.c} | complete cells {len(done)} | "
          f"discarded {removed} rows from {len(drop)} partial/mixed-session cells", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    rev = getattr(model.config, "_commit_hash", None) or "unresolved"
    t0 = time.time()
    n_fail_total = n_rows_total = 0

    for c_tag in a.c:
        for C in budgets:
            if (c_tag, C) in done:
                print(f"  [skip] c={c_tag} C={C} already complete in one session", flush=True)
                continue
            B = C + CM.N_SINK + CM.N_WINDOW
            t1 = time.time()
            bitmaps, cell_rows, n_fail = {}, [], 0
            for i in range(a.n):
                iid = "s4_%05d" % i
                b = CM.build(tag, a.model, rev, tok, c_tag, iid)
                for arm in ARMS:
                    kd = dict(task=b["sp"]["task"], instance_id=iid, model=a.model,
                              model_revision=rev, arm=arm, B=B, C=C, seed=b["sd"],
                              n_fields=b["sp"]["n_fields"], n_records=b["sp"]["k"],
                              layout="p4_stage3_oracle", matched_to=None,
                              max_new=max(b["mns"]), **ENV3)
                    t2 = time.time()
                    try:
                        cap, stats = methods.Capture(), {}
                        p = build_press3(arm, b, C, cap, stats)
                        base = CM.base_arm(arm)
                        outs, flags = CM.generate_checked(
                            model, tok, b["pre"], b["posts"], p, b["mns"], b["n_ctx"], B,
                            headwise=base.startswith("adakv"))
                        parity = runner.assert_budget_parity(cap, C, b["n_ctx"], base)
                        keeps = [cap.per_head[h] for h in cap.heads()]
                        if arm == "floor_pos":
                            for kk in keeps:
                                assert_floor_pos(kk, b["n_ctx"], C)
                        met = metrics3(keeps, b["facts"], b["line_units"], b["n_ctx"])
                        if not met["floor_ok"]:
                            raise AssertionError(f"{arm}: mandatory floors not retained")
                        if arm.startswith("U-") and stats.get("score_calls") != len(
                                {li for li, _ in cap.per_head}):
                            raise AssertionError(f"{arm}: score() not once per layer")
                        bm = np.zeros((len(keeps), b["n_ctx"]), dtype=bool)
                        for s, kk in enumerate(keeps):
                            bm[s, sorted(kk)] = True
                        bitmaps[f"{i}|{arm}"] = np.packbits(bm, axis=1)
                        cell_rows.append(dict(
                            key_digest=keys3.digest(kd), key=kd, session_id=session_id,
                            gather="score",          # A4; rows written before A4 are "ascending"
                            produced_on=CM.PRODUCED_ON, model_tag=tag, package="stage3",
                            task=b["sp"]["task"], c_tag=c_tag, c=b["sp"]["c"], C=C, B=B, arm=arm,
                            instance=i, n_ctx=b["n_ctx"], B_asserted=parity,
                            units_taken=stats.get("units_taken"), fallback=stats.get("fallback"),
                            score=CM.S4.SCORERS[b["sp"]["task"]](outs, b["inst"]),
                            per_variant=[CM.S4.SCORE_ONE[b["sp"]["task"]](o, v.answer)
                                         for v, o in zip(b["inst"].variants, outs)],
                            gen=outs, answers=[v.answer for v in b["inst"].variants],
                            stop_flags=flags, wall_s=time.time() - t2, **met))
                    except Exception as e:
                        n_fail += 1
                        with fail.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(dict(key=kd, session_id=session_id,
                                                    error=f"{type(e).__name__}: {e}")) + NL)
                        print(f"  FAIL c={c_tag} C={C} {iid} {arm}: {type(e).__name__}: {e}",
                              flush=True)
                if (i + 1) % 10 == 0:
                    print(f"    c={c_tag} C={C} inst {i+1}/{a.n} rows {len(cell_rows)} "
                          f"fails {n_fail} {(time.time()-t1)/60:.1f} min", flush=True)

            n_rows_total += len(cell_rows)
            n_fail_total += n_fail
            if n_fail or len(cell_rows) != len(ARMS) * a.n:
                print(f"  CELL INCOMPLETE c={c_tag} C={C}: {len(cell_rows)} rows, {n_fail} "
                      f"failures -- NOT written (cell must be whole and single-session)",
                      flush=True)
                continue
            with out.open("a", encoding="utf-8") as f:
                for r in cell_rows:
                    f.write(json.dumps(r) + NL)
            np.savez_compressed(RUNS / f"p4_s3_keepsets_{tag}_c{c_tag}_C{C}.npz", **bitmaps)
            acc = defaultdict(list)
            for r in cell_rows:
                acc[r["arm"]].append(r["score"])
            summ = {k: float(np.mean(v)) for k, v in acc.items()}
            line = dict(model_tag=tag, c_tag=c_tag, C=C, session_id=session_id,
                        n=a.n, minutes=(time.time() - t1) / 60, acc=summ,
                        floor=summ["floor_pos"], degenerate=summ["floor_pos"] < 0.05,
                        d_snapkv=summ["U-snapkv"] - summ["snapkv"],
                        d_adakv=summ["U-adakv_snapkv"] - summ["adakv_snapkv"],
                        d_expected=summ["U-expected_attn"] - summ["expected_attn"],
                        d_keydiff=summ["U-keydiff"] - summ["keydiff"])
            with prog.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line) + NL)
            if c_tag >= 8:      # A3.4 watch: flag a c>=8 confound immediately, not at the report
                import zlib
                per = defaultdict(dict)
                for r in cell_rows:
                    per[r["arm"]][r["instance"]] = r["score"]
                for x, u in (("snapkv", "U-snapkv"), ("adakv_snapkv", "U-adakv_snapkv"),
                             ("expected_attn", "U-expected_attn"), ("keydiff", "U-keydiff")):
                    d = np.array([per[u][i] - per[x][i] for i in range(a.n)], dtype=float)
                    rng = np.random.default_rng(
                        zlib.crc32(f"p4|s3|{tag}|{c_tag}|{C}|{x}".encode()) & 0xFFFFFFFF)
                    m = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
                    lo, hi = np.percentile(m, [2.5, 97.5])
                    if x in CONFOUND_WATCH and lo > 0 and not line["degenerate"]:
                        print(f"  *** CONFOUND WATCH (A3.4): c={c_tag} C={C} {tag} {u} - {x} "
                              f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] CI excludes zero, positive "
                              f"-- the c=1 confound may extend past c=1", flush=True)
            print(f"  CELL DONE c={c_tag} C={C} in {line['minutes']:.1f} min | floor "
                  f"{line['floor']:.3f}{' DEGENERATE' if line['degenerate'] else ''} | "
                  f"dU snapkv {line['d_snapkv']:+.3f} adakv {line['d_adakv']:+.3f} "
                  f"expected {line['d_expected']:+.3f} keydiff {line['d_keydiff']:+.3f}",
                  flush=True)

    print(f"  DONE {tag} c={a.c}: {n_rows_total} rows, {n_fail_total} failures, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)
    return 1 if n_fail_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
