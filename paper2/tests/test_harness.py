"""Unit tests for the freeze-independent harness modules.

Run: python -m pytest tests/ -q      (or: python tests/test_harness.py)

These test invariants that no decision in PREREG_P2.md §3 can change. Nothing here
presupposes decision 7 -- the oracle_causal tests assert that it REFUSES to choose.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from harness import ladder, stats
from harness.keys import RecordKey, seed_for, seed_key_without_seed, assert_owned_by
from harness.tasks import ledger

FAILED: list[str] = []


def check(cond: bool, label: str) -> None:
    if cond:
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label}")
        FAILED.append(label)


BASE = dict(
    task="ledger", instance_id="i0007", model="Qwen/Qwen2.5-1.5B-Instruct",
    model_revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306", arm="floor_pos", B=328,
    protocol="agnostic", device="nvidia", backend="cuda-12.8",
    torch_version="2.10.0+cu128", transformers_version="5.2.0",
    kvpress_version="0.5.4+6d965557a5b9", dtype="bfloat16",
)


def test_keys() -> None:
    print("\n[keys]")
    s = seed_key_without_seed(**BASE)
    k = RecordKey(**BASE, seed=s)
    check(RecordKey(**BASE, seed=s).digest() == k.digest(), "digest reproducible")
    check(dataclasses.replace(k, arm="oracle_causal").digest() != k.digest(), "arm changes key")
    check(dataclasses.replace(k, batch_size=4).digest() != k.digest(), "batch_size changes key")
    check(dataclasses.replace(k, backend="rocm-7.2").digest() != k.digest(), "backend changes key")
    check(seed_for(k, "task") != seed_for(k, "policy"), "named streams differ")
    check(seed_for(k, "task") == seed_for(k, "task"), "streams deterministic")
    check(k.runs_dir() == "runs/nvidia", "runs_dir follows device")
    for bad in (dict(model_revision="main"), dict(device="rocm"), dict(B=0), dict(batch_size=0)):
        try:
            RecordKey(**{**BASE, **bad, "seed": s}); check(False, f"rejects {bad}")
        except ValueError:
            check(True, f"rejects {list(bad)[0]}={list(bad.values())[0]!r}")
    try:
        assert_owned_by(dataclasses.replace(k, device="amd"), "nvidia"); check(False, "rule 2")
    except PermissionError:
        check(True, "rule 2: blocks cross-device write")


def test_ledger() -> None:
    print("\n[ledger]")
    inst = ledger.build(12345, "t0001")
    check(inst.answer in inst.context, "answer present in context")
    check(len(inst.variants) == ledger.N_BINDINGS, "H query variants emitted")
    check(len({v.answer for v in inst.variants}) == ledger.N_BINDINGS, "variant answers distinct")
    check(not any(v.answer in ledger.EXEMPLARS for v in inst.variants),
          "2-shot exemplars leak no instance answer")
    check(inst.context.startswith("Worked examples"), "exemplars are in the context")
    check(ledger.score_instance([v.answer for v in inst.variants], inst) == 1.0,
          "score_instance = 1.0 when every variant is right")
    check(ledger.score_instance([inst.variants[0].answer, "x", "y", "z"], inst) == 0.25,
          "score_instance = 1/H when one variant is right")
    check(len(inst.gold) == 1, "one gold span (one-hop task)")
    g = inst.gold[0]
    check(g.kind == "record", "gold span is a record line")
    check(inst.answer in g.text, "answer lives in the gold record line")
    check(inst.variants[0].rec_id in g.text, "gold line carries the queried record id")
    check(all(v.rec_id in v.query and v.answer in v.gold[0].text for v in inst.variants),
          "every variant is consistent with its gold span")
    check(len({v.rec_id for v in inst.variants}) == ledger.N_BINDINGS,
          "H distinct record ids queried")
    check("responsible for record" not in inst.context, "no binding sentences remain")
    check(len(inst.candidates) == ledger.N_BINDINGS, "candidate set = H spans")
    check({s.text for s in inst.gold} <= {c.text for c in inst.candidates},
          "gold is a subset of candidates")
    check(abs(inst.meta["chance"] - 1 / ledger.N_RECORDS) < 1e-12,
          f"chance = 1/{ledger.N_RECORDS}")
    check(ledger.score(f"the value is {inst.answer}.", inst), "scorer accepts correct")
    check(not ledger.score("the value is 000000.", inst), "scorer rejects wrong")
    check(ledger.build(12345, "t0001").context == inst.context, "generation deterministic")
    check(ledger.build(999, "t0001").context != inst.context, "seed changes instance")
    for sp in inst.candidates:
        check(inst.context[sp.start:sp.end] == sp.text, f"span offsets exact ({sp.kind})") if sp is inst.candidates[0] else None
    check(all(inst.context[s.start:s.end] == s.text for s in inst.candidates),
          "all candidate span offsets exact")


def _toy_facts(n_ctx=1000, n_facts=4, span_len=14, start=200, gap=97, spans_per_fact=2):
    """Toy facts. `spans_per_fact=1` is the one-hop shape, 2 the old two-hop shape.

    The decision-7 knapsack must behave identically for both -- the rule is about completeness,
    not about how many spans a fact happens to have.
    """
    facts = []
    p = start
    for i in range(n_facts):
        sp = []
        for _ in range(spans_per_fact):
            sp.append(tuple(range(p, p + span_len))); p += gap
        facts.append(ladder.FactSpans(f"f{i}", tuple(sp)))
    return facts


def test_ladder() -> None:
    print("\n[ladder]")
    n_ctx, C, ns, nw = 1000, 128, 8, 64
    B = C + ns + nw
    facts = _toy_facts(n_ctx)

    for name, sel in [
        ("floor_pos", ladder.floor_pos(n_ctx, C, ns, nw, facts)),
        ("random", ladder.random_arm(n_ctx, C, 7, ns, nw, facts)),
        ("oracle_prescient", ladder.oracle_prescient(n_ctx, C, facts[0], ns, nw, facts)),
    ]:
        check(sel.n_kept == B, f"{name} retains exactly B={B} (got {sel.n_kept})")
    check(ladder.null_arm(n_ctx, C, ns, nw).n_kept == B, f"null retains exactly B={B}")

    fp = ladder.floor_pos(n_ctx, C, ns, nw, facts)
    check(set(range(ns)) <= set(fp.kept), "floor_pos keeps the sink")
    check(set(range(n_ctx - nw, n_ctx)) <= set(fp.kept), "floor_pos keeps the recency window")

    op = ladder.oracle_prescient(n_ctx, C, facts[0], ns, nw, facts)
    check(op.n_facts_complete >= 1, "prescient oracle completes the queried fact")

    # --- decision 7 RESOLVED: optimal complete-pair packing --------------------
    big = _toy_facts(n_ctx, n_facts=4, span_len=30)  # payable > C = 128
    sel = ladder.oracle_causal(n_ctx, C, big, n_sink=ns, n_window=nw)
    check(sel.n_kept == B, "oracle_causal retains exactly B")
    check(sel.degraded, "flagged degraded when the candidate set does not fit")
    check(sel.degradation_rule == "pair_knapsack_optimal", "packing recorded on the record")
    # no partial pairs, ever
    check(sel.n_facts_complete == sel.notes["n_pairs_taken"], "no partial pair retained")
    check(sel.n_candidate_spans_complete == 2 * sel.n_facts_complete,
          "every retained span belongs to a complete pair")

    # one-hop shape: the same rule, single spans, optimality assertion must still hold
    single = _toy_facts(n_ctx, n_facts=4, span_len=45, spans_per_fact=1)  # payable > C
    s1sel = ladder.oracle_causal(n_ctx, C, single, n_sink=ns, n_window=nw)
    check(s1sel.n_kept == B, "one-hop: retains exactly B")
    check(s1sel.n_facts_complete == s1sel.notes["n_pairs_taken"], "one-hop: no partial retained")
    check(s1sel.n_candidate_spans_complete == s1sel.n_facts_complete,
          "one-hop: one span per complete fact")
    c1 = [len({t for sp in f.spans for t in sp}) for f in single]
    check(ladder._optimal_pair_count(c1, C) == s1sel.notes["n_pairs_taken"],
          "one-hop: packing matches the exhaustive optimum")
    # optimality is asserted internally; confirm the assertion actually fires when violated
    costs = [len({t for sp in f.spans for t in sp}) for f in big]
    check(ladder._optimal_pair_count(costs, C) == sel.notes["n_pairs_taken"],
          "packing matches the exhaustive optimum")
    check(sel.notes["expected_accuracy"] == sel.notes["n_pairs_taken"] / sel.notes["H"],
          "expected accuracy = complete pairs / H")

    # a wider budget must complete at least as many pairs (monotone in C)
    prev = -1
    for c in (64, 128, 256, 512):
        s2 = ladder.oracle_causal(n_ctx, c, big, n_sink=ns, n_window=nw)
        check(s2.n_facts_complete >= prev, f"pairs complete monotone in C (C={c})")
        prev = s2.n_facts_complete

    small = _toy_facts(n_ctx, n_facts=2, span_len=8)
    ok = ladder.oracle_causal(n_ctx, C, small, n_sink=ns, n_window=nw)
    check(not ok.degraded, "not flagged degraded when everything fits")
    check(ok.n_facts_complete == 2, "all facts complete when it fits")


def test_stats() -> None:
    print("\n[stats]")
    rng = np.random.default_rng(0)
    n = 300
    floor = rng.random(n) * 0.2
    causal = floor + 0.6 + rng.random(n) * 0.1
    method = floor + 0.3 + rng.random(n) * 0.1

    g = stats.capture_ratio(method, floor, causal, seed=1, n_boot=2000)
    check(not g.refused, "G_m computed when headroom is wide")
    check(g.ci_low < g.value < g.ci_high, "G_m point estimate inside its CI")
    check(0.3 < g.value < 0.7, f"G_m plausible ({g.value:.3f})")

    # refusal-to-normalise
    tight = floor + 0.05
    gr = stats.capture_ratio(method, floor, tight, seed=1, n_boot=500)
    check(gr.refused and gr.value is None, "refusal-to-normalise fires below 0.15")
    check("degenerate headroom" in (gr.reason or ""), "refusal reason recorded")

    # unclipped
    over = stats.capture_ratio(causal + 0.1, floor, causal, seed=1, n_boot=500)
    check(over.value is not None and over.value > 1, "G_m reported unclipped above 1")
    check(over.audit_required, "G_m > 1 triggers ceiling-validity audit")

    presc = causal + 0.2
    i = stats.information_share(presc, causal, floor, seed=1, n_boot=2000)
    check(0 < i.value < 1, f"I in (0,1) ({i.value:.3f})")

    a = np.array([1.0, 1, 1, 0, 0]); b = np.array([1.0, 1, 1, 1, 0])
    pc = stats.paired_contrast(a, b, seed=1, n_boot=500)
    check(pc.n_differ == 1, "n_differ counts differing instances")

    # Paper 1's two kinds of null, both equivalent within margin but distinguishable by
    # how often the arms actually disagree. n=200 so the CI is tight enough to declare.
    same = rng.random(200); other = same.copy(); other[:4] = 1 - other[:4]
    p_same = stats.paired_contrast(same, other, seed=2, n_boot=3000, equivalence_margin=0.10)
    check(p_same.null_kind == "same behaviour",
          f"null kind: same behaviour (differ {p_same.frac_differ:.2f})")
    noisy = same + rng.normal(0, 0.02, 200)
    p_diff = stats.paired_contrast(same, noisy, seed=3, n_boot=3000, equivalence_margin=0.10)
    check(p_diff.null_kind == "different behaviour, same outcome",
          f"null kind: different behaviour same outcome (differ {p_diff.frac_differ:.2f})")

    o = stats.check_ladder_ordering(null=0.01, random=0.02, floor_pos=0.30,
                                    oracle_causal=0.80, oracle_prescient=0.90, full_cache=0.95)
    check(o.ok, "valid ladder ordering passes")
    bad = stats.check_ladder_ordering(null=0.01, random=0.02, floor_pos=0.30,
                                      oracle_causal=0.97, oracle_prescient=0.90, full_cache=0.95)
    check(not bad.ok and bad.denoising, "denoising ceiling caught (Paper 1 budget-514 case)")


if __name__ == "__main__":
    test_keys(); test_ledger(); test_ladder(); test_stats()
    print("\n" + "=" * 60)
    if FAILED:
        print(f"FAILED {len(FAILED)}:"); [print("  -", f) for f in FAILED]; raise SystemExit(1)
    print("ALL TESTS PASS")
