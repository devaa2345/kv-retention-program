"""Experiment runner: append-only JSONL checkpointing, resume, and failure logging.

Required row fields (spec section 6): nominal_budget, effective_tokens, iso_condition,
quant_bits, model, context_length. Different bit-widths go to different directories.

Failure policy (spec section 9 / brief): a run that raises is written to a separate failures
file, never silently dropped -- the paired analysis needs identical seed sets across arms and
silent dropout breaks it without complaining.
"""

from __future__ import annotations

import json
import os
import time
import traceback

from .arms import ARMS, PROMO_IDS, make_cfg
from .cache_engine import QuantAudit
from .model import MODEL_ID
from .task import build_prompt


def cell_key(row: dict) -> str:
    return "|".join(str(row[k]) for k in
                    ("nominal_budget", "iso_condition", "arm", "seed", "quant_bits",
                     "promotion", "context_target"))


def load_done(path: str) -> set[str]:
    """Read completed keys for resume, across EVERY result file in the same directory.

    Scanning only this stage's own file would re-run cells another stage already completed --
    Phase 2's P1 arm is configurationally identical to Phase 1's arm-4/budget-257 cell, and the
    sweep's 4-bit point is that same cell again. Since a cell key fully determines the result,
    reusing the existing row is correct and avoids redundant compute.

    Membership-only use; the set is never iterated into output.
    """
    import glob as _glob
    done: set[str] = set()
    d = os.path.dirname(path) or "."
    for f_ in sorted(_glob.glob(os.path.join(d, "*.jsonl"))):
        if f_.endswith(".failures.jsonl"):
            continue
        try:
            fh = open(f_)
        except OSError:
            continue
        with fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(cell_key(json.loads(line)))
                except Exception:
                    continue      # a torn final line from a kill; it will simply be redone
    return done


def run_cells(eng, cells: list[dict], out_dir: str, tag: str, verbose_every: int = 25) -> dict:
    """cells: list of dicts with keys arm, budget, seed, iso_condition, quant_bits, promotion,
    quant_byte_cost, recency_window, context_target, n_filler_paragraphs."""
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, f"{tag}.jsonl")
    failures_path = os.path.join(out_dir, f"{tag}.failures.jsonl")
    done = load_done(results_path)

    n_done = n_new = n_fail = 0
    t0 = time.time()
    with open(results_path, "a") as fout, open(failures_path, "a") as ffail:
        for i, c in enumerate(cells):
            probe = {"nominal_budget": c["budget"], "iso_condition": c["iso_condition"],
                     "arm": c["arm"], "seed": c["seed"], "quant_bits": c["quant_bits"],
                     "promotion": c["promotion"], "context_target": c.get("context_target", 1029)}
            key = cell_key(probe)
            if key in done:
                n_done += 1
                continue
            try:
                cfg = make_cfg(c["arm"], c["budget"], iso_condition=c["iso_condition"],
                               quant_bits=c["quant_bits"],
                               quant_byte_cost=c.get("quant_byte_cost", 0.25),
                               promotion=c["promotion"],
                               recency_window=c.get("recency_window", 64))
                cfg.assert_budget_valid()
                prompt = build_prompt(c["seed"],
                                      n_filler_paragraphs=c.get("n_filler_paragraphs", 7))
                audit = QuantAudit()
                r = eng.run_prompt(prompt, cfg, seed=c["seed"], audit=audit,
                                   collect_dormancy=c.get("collect_dormancy", False))
                row = dict(probe)
                row.update({
                    "arm_label": ARMS[c["arm"]]["label"],
                    "protection": cfg.protection,
                    "eviction": cfg.eviction,
                    "promotion_id": PROMO_IDS.get(c["promotion"], c["promotion"]),
                    "effective_budget": cfg.total_budget,
                    "effective_tokens": r["effective_tokens"],
                    "n_full": r["n_full"], "n_quant": r["n_quant"], "bytes": r["bytes"],
                    "quant_byte_cost": cfg.quant_byte_cost,
                    "physical_byte_cost": c["quant_bits"] / 16.0,   # [GAP-N]
                    "full_fraction": cfg.full_fraction,
                    "recency_window": cfg.recency_window, "sink": cfg.sink,
                    "model": getattr(eng, "model_id", MODEL_ID),
                    "context_length": r["context_length"],
                    "accuracy": r["accuracy"], "k": r["k"], "n_turns": r["n_turns"],
                    "outputs": r["outputs"],
                    "oracle_oversubscribed": r["oracle_oversubscribed"],
                    "n_protected_lines": r["n_protected_lines"],
                    "dormancy_events": r["dormancy_events"],
                    "dormancy_max_run": r["dormancy_max_run"],
                    "quant_violations": len(audit.violations),
                    "quant_calls": audit.calls,
                    "quant_max_levels": audit.max_levels_seen,
                    "first_token_eos": r["first_token_eos"],
                })
                fout.write(json.dumps(row) + "\n")
                fout.flush()
                os.fsync(fout.fileno())
                n_new += 1
            except Exception as e:
                ffail.write(json.dumps({**probe, "error": repr(e),
                                        "trace": traceback.format_exc()[-800:]}) + "\n")
                ffail.flush()
                os.fsync(ffail.fileno())
                n_fail += 1
            if verbose_every and (i + 1) % verbose_every == 0:
                el = time.time() - t0
                rate = n_new / el if el > 0 else 0
                left = (len(cells) - i - 1) / rate if rate > 0 else 0
                print(f"  [{tag}] {i+1}/{len(cells)} new={n_new} skip={n_done} fail={n_fail} "
                      f"{rate:.2f}/s eta={left/60:.1f}min", flush=True)
    return {"tag": tag, "new": n_new, "skipped": n_done, "failed": n_fail,
            "elapsed_s": time.time() - t0, "results": results_path,
            "failures": failures_path}
