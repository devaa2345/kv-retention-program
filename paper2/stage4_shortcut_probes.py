"""Task B shortcut probes (v1 §4.2) — the Stage 4 admission gate for LEDGER.

Task B is not admitted until these pass. It carries the paper's main claim precisely because
Task A admits a surface-pattern shortcut (Paper 1's structural protection jumps 0.150 -> 0.907
when distractors stop matching the `sk-` pattern), so LEDGER's freedom from shortcuts has to be
demonstrated, not asserted.

Five probes in v1 §4.2. Three are model-free and are run here:

    1. BM25 retrieval over records with the query hidden      <= chance + 0.02
    2. Regex value-extractor (no query)                       <= chance + 0.02
    3. Position prior (always answer from the last record)    <= chance + 0.02

Two require a forward pass and are deferred to the pinned WSL toolchain, because they produce
accuracy numbers that gate admission and every such number must come from the frozen
environment (v1 §5.3: backend is a stratum, not an implementation detail):

    4. full_cache with the target record deleted              ~ 0
       (the "bindings deleted" probe is retired -- there are no bindings in a one-hop task)

chance = 1/N_records = 1/40 = 0.025, so the pass threshold is 0.045.

A fourth diagnostic is included and is NON-GATING but now expected to be HIGH: BM25 *with*
the real query, over records only. The one-hop query names the record id verbatim, so a
lexical retriever should find it. That is not a shortcut -- it is the task. It is reported so
the contrast with the query-HIDDEN probe is explicit: knowing the query makes the record
trivially findable; not knowing it leaves you at chance, which is exactly the property a
query-agnostic compression benchmark needs.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from harness.keys import seed_key_without_seed
from harness.tasks import ledger

# Instances are built at the TIGHTEST realistic setting -- targeted to L=2048 in the Qwen
# tokenizer, which yields the fewest filler sentences of any admitted model and therefore the
# fewest surname decoys. That is the worst case for the BM25 probe, so passing here passes
# everywhere. The probes themselves use no model.
TIGHTEST_TOKENIZER = "Qwen/Qwen2.5-1.5B-Instruct"

N_INSTANCES = 200
TOLERANCE = 0.02
VALUE_RE = re.compile(r"\b\d{6}\b")
_WORD = re.compile(r"[A-Za-z0-9]+")


def toks(s: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(s)]


class BM25:
    """Standard Okapi BM25. Written out rather than pulled in, so the probe has no dependency."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(1, self.N)
        self.tf = [Counter(d) for d in docs]
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()
        }

    def score(self, query: list[str], i: int) -> float:
        tf, dl, s = self.tf[i], len(self.docs[i]), 0.0
        for t in query:
            if t not in tf:
                continue
            f = tf[t]
            s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (
                f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            )
        return s

    def top1(self, query: list[str]) -> int:
        return max(range(self.N), key=lambda i: self.score(query, i))


def record_lines(inst: ledger.LedgerInstance) -> list[str]:
    return [l for l in inst.context.split("\n") if re.match(r"^R\d{3} \| ", l)]


def main() -> int:
    tok = AutoTokenizer.from_pretrained(TIGHTEST_TOKENIZER)
    insts = []
    for i in range(N_INSTANCES):
        iid = f"probe{i:05d}"
        seed = seed_key_without_seed(
            task="ledger", instance_id=iid, model="probe", model_revision="probe",
            arm="probe", B=1, protocol="agnostic", device="nvidia", backend="none",
            torch_version="na", transformers_version="na", kvpress_version="na", dtype="na",
        )
        insts.append(ledger.build(seed, iid, target_tokens=2048, tokenizer=tok))

    chance = 1.0 / ledger.N_RECORDS
    thr = chance + TOLERANCE
    hits = {k: 0 for k in ("bm25_query_hidden", "regex_value", "position_prior", "bm25_with_query")}
    bm25_narrowed = 0

    for inst in insts:
        recs = record_lines(inst)
        gold_line = inst.gold[0].text
        gold_i = recs.index(gold_line)
        bm = BM25([toks(r) for r in recs])

        # 1. BM25, query hidden: rank records against the rest of the document.
        non_record = "\n".join(l for l in inst.context.split("\n") if not re.match(r"^R\d{3} \| ", l))
        if bm.top1(toks(non_record)) == gold_i:
            hits["bm25_query_hidden"] += 1
        # diagnostic: does the hidden-query retriever at least narrow to the 4 bound records?
        bound = {c.text for c in inst.candidates}
        if recs[bm.top1(toks(non_record))] in bound:
            bm25_narrowed += 1

        # 2. Regex value extractor, no query: take the first 6-digit value in the context.
        vals = VALUE_RE.findall(inst.context)
        if vals and vals[0] == inst.answer:
            hits["regex_value"] += 1

        # 3. Position prior: always answer from the last record line.
        if VALUE_RE.findall(recs[-1])[-1] == inst.answer:
            hits["position_prior"] += 1

        # 4. Diagnostic (non-gating): BM25 with the real query, records only.
        if bm.top1(toks(inst.query)) == gold_i:
            hits["bm25_with_query"] += 1

    n = len(insts)
    out = {
        "n_instances": n, "n_records": ledger.N_RECORDS,
        "chance": chance, "tolerance": TOLERANCE, "threshold": thr,
        "probes": {}, "deferred": [
            "full_cache with bindings deleted (needs forward pass, pinned WSL toolchain)",
            "full_cache with target record deleted (needs forward pass, pinned WSL toolchain)",
        ],
    }
    gating = ("bm25_query_hidden", "regex_value", "position_prior")
    print(f"LEDGER shortcut probes — n={n}, chance=1/{ledger.N_RECORDS}={chance:.4f}, "
          f"threshold={thr:.4f}\n")
    print(f"  {'probe':24s} {'score':>8s} {'thr':>8s}  result")
    all_pass = True
    for k in gating:
        sc = hits[k] / n
        ok = sc <= thr
        all_pass &= ok
        out["probes"][k] = {"score": round(sc, 4), "threshold": round(thr, 4),
                            "pass": ok, "gating": True}
        print(f"  {k:24s} {sc:8.4f} {thr:8.4f}  {'PASS' if ok else '** FAIL **'}")

    sc = hits["bm25_with_query"] / n
    out["probes"]["bm25_with_query"] = {"score": round(sc, 4), "gating": False}
    out["probes"]["bm25_hidden_narrows_to_bound"] = {
        "score": round(bm25_narrowed / n, 4), "gating": False,
        "note": "fraction of hidden-query BM25 top-1 hits landing on any of the H queried "
                "records; H/N would be chance for that weaker target",
    }
    print(f"\n  diagnostics (non-gating):")
    print(f"  {'bm25_with_query':24s} {sc:8.4f}           "
          f"(one-hop: expected HIGH -- the query names the id; not a shortcut)")
    print(f"  {'bm25_hidden->bound record':24s} {bm25_narrowed / n:8.4f}           "
          f"(chance for that target = {4 / ledger.N_RECORDS:.4f})")

    out["gate"] = {"pass": all_pass, "note": "3 of 5 probes; 2 deferred to the pinned toolchain"}
    print("\n" + "=" * 62)
    print(f"STAGE 4 SHORTCUT GATE (model-free probes): "
          f"{'PASS' if all_pass else 'FAIL'}   [2 of 5 probes still deferred]")

    p = Path(__file__).resolve().parent / "gates" / "nvidia" / "stage4_shortcut_probes.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
