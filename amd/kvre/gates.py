"""Calibration gates G1-G5. Spec defines only G5; G1-G4 are defined in SPEC_QUESTIONS [GAP-A].

Refuse to proceed to measurement if any gate fails.
"""
from __future__ import annotations
import json, statistics
from .task import build_prompt, question_for, N_CREDENTIALS
from .arms import make_cfg
from .cache_engine import error_table, assert_monotone_error, QuantAudit
from .policy import PolicyConfig
from .model import kv_tensors_from_cache


def gate_G1(tok, seeds, target=1029, tol=0.05):
    """Task validity: context length band, credential uniqueness and single occurrence."""
    lens, problems = [], []
    for s in seeds:
        p = build_prompt(s)   # single source of truth: build_prompt's own defaults
        text = tok.apply_chat_template(
            [{"role": "user", "content": p.context + "\n\n" + question_for(p, 0)}],
            tokenize=False, add_generation_prompt=True)
        lens.append(len(tok(text)["input_ids"]))
        vals = [c.value for c in p.credentials]
        if len(vals) != N_CREDENTIALS:
            problems.append(f"seed {s}: {len(vals)} credentials")
        if len(set(vals)) != len(vals):
            problems.append(f"seed {s}: duplicate credential values")
        for c in p.credentials:
            if len(c.value) != 17:
                problems.append(f"seed {s}: value {c.value!r} not 17 chars")
            if p.context.count(c.value) != 1:
                problems.append(f"seed {s}: {c.label} appears {p.context.count(c.value)}x")
    med = statistics.median(lens)
    if abs(med - target) / target > tol:
        problems.append(f"median context {med} outside +/-{tol:.0%} of {target}")
    return {"gate": "G1", "pass": not problems, "median_len": med,
            "range": [min(lens), max(lens)], "problems": problems[:10]}


def gate_G2(eng, seeds, min_acc=0.70):
    """Model competence ceiling: full_cache_ref must leave headroom for eviction to damage."""
    accs = [eng.run_prompt(build_prompt(s), make_cfg(6, 10**9), seed=s)["accuracy"]
            for s in seeds]
    m = statistics.mean(accs)
    return {"gate": "G2", "pass": m >= min_acc, "full_cache_mean_acc": m,
            "n": len(accs), "min_required": min_acc}


def gate_G3(eng, seeds, budgets, recency_window=64, sink=1):
    """Budget arithmetic: void-cell refusal, effective_tokens honesty, budget invariance."""
    problems = []
    void_refused = []
    for b in [51, 102] + list(budgets):
        cfg = PolicyConfig(total_budget=b, recency_window=recency_window, sink=sink)
        room = cfg.competitive_room()
        try:
            cfg.assert_budget_valid()
            if room <= 0:
                problems.append(f"budget {b}: room={room} but was NOT refused")
        except ValueError:
            void_refused.append(b)
            if room > 0:
                problems.append(f"budget {b}: room={room}>0 but was refused")
    # full_cache_ref must be budget-invariant
    inv = {}
    for b in budgets:
        accs = [eng.run_prompt(build_prompt(s), make_cfg(6, b), seed=s)["accuracy"]
                for s in seeds]
        inv[b] = statistics.mean(accs)
    if len(set(round(v, 9) for v in inv.values())) != 1:
        problems.append(f"full_cache_ref varies with budget: {inv}  <-- budget is leaking")
    # effective_tokens must equal realised retained count, never the nominal budget
    for b in budgets:
        r = eng.run_prompt(build_prompt(seeds[0]), make_cfg(1, b), seed=seeds[0])
        if r["effective_tokens"] > b:
            problems.append(f"budget {b}: effective_tokens {r['effective_tokens']} > budget")
    return {"gate": "G3", "pass": not problems, "void_refused": void_refused,
            "full_cache_by_budget": inv, "problems": problems}


def gate_G4(eng, tok, seed=9000, bit_list=(8, 4, 2)):
    """Quantizer integrity on tensors actually written during generation."""
    from .engine import Engine
    p = build_prompt(seed)
    text = tok.apply_chat_template(
        [{"role": "user", "content": p.context + "\n\n" + question_for(p, 0)}],
        tokenize=False, add_generation_prompt=True)
    import torch
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(eng.device)
    with torch.no_grad():
        out = eng.model(**ids, use_cache=True)
    kvs = kv_tensors_from_cache(out.past_key_values)
    tbl = error_table(kvs, list(bit_list))
    mono = assert_monotone_error(tbl)

    # level counts on tensors written during real generation (audit hooked into apply_tiers)
    audits = {}
    for bits in bit_list:
        au = QuantAudit()
        eng.run_prompt(build_prompt(seed), make_cfg(4, 257, quant_bits=bits), seed=seed, audit=au)
        audits[bits] = {"calls": au.calls, "max_levels": au.max_levels_seen,
                        "limit": 2 ** bits, "violations": au.violations[:5]}
    problems = list(mono)
    for bits, a in audits.items():
        if a["violations"]:
            problems.append(f"{bits}-bit level violations: {a['violations']}")
        if a["calls"] == 0:
            problems.append(f"{bits}-bit: quantizer never called during generation")
    return {"gate": "G4", "pass": not problems, "error_table": tbl,
            "generation_audit": audits, "problems": problems}


def gate_G5(eng, seeds, budget=257, min_frac=0.80):
    """Dormancy, as specified in section 2. Measured in the unprotected attention-ranked arm."""
    weak, strict = 0, 0
    per = []
    for s in seeds:
        r = eng.run_prompt(build_prompt(s), make_cfg(1, budget), seed=s, collect_dormancy=True)
        per.append((r["dormancy_events"], r["dormancy_max_run"]))
        if r["dormancy_events"] >= 1:
            weak += 1
        if r["dormancy_max_run"] >= 8:
            strict += 1
    n = len(seeds)
    return {"gate": "G5", "pass": weak / n >= min_frac, "frac_weak": weak / n,
            "frac_strict_8step": strict / n, "n": n, "min_required": min_frac}
