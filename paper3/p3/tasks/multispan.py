"""MULTISPAN — the `k` axis: one fact split into `k` ADJACENT labelled parts.

Stage 2.3 established that SCATTERED placement is unusable at this model scale: `full_cache`
falls 0.910 -> 0.400 on M2 purely by moving one fact's two halves apart, with no compression at
all. So the `k` axis is run with the parts ADJACENT, on consecutive lines, which anchored in
band on both models (0.906 / 0.938 at k = 2).

    R011 /A | Fenorson | Treasury
    R011 /B | 280294 | Northgate

Every part carries the record id and a part label, so a part is identifiable on its own but not
sufficient: the query asks for all fields in order, so all `k` parts must survive. `k` = 1 is
the same construction with a single part, so the ONLY thing that changes across `k` is how many
line-breaks divide the same fields.

**`c` is held fixed across `k`, not `n_fields`.** Adding parts adds label tokens, so holding
`n_fields` fixed would confound `k` with fact cost -- the very confound Stage 1 spent a whole
stage disentangling for `c` and `C`. `p3/calibrate_k.py` solves `n_fields` per (model, `k`) so
that the achieved `c` lands near 19, and the achieved value is what enters any analysis.

This is a separate module rather than a parameter on `scatter.py` so that Stage 2's committed
artifact and its scorer stay exactly as they were recorded.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Sequence

from p3.tasks.ledger_c import (DEPTS, SITES, STATUS, SURNAMES, Span, Variant, _FIELD_KINDS)

N_RECORDS = 40
N_BINDINGS = 4
PART_LABELS = ("A", "B", "C", "D", "E", "F")

PREAMBLE = ("INTERNAL LEDGER EXTRACT. Each record is recorded in consecutive parts, marked "
            "/A, /B and so on. Governance notes are interleaved between records.\n")

_MENTION = (
    "Correspondence concerning record {rec_id} was logged and filed without further action.",
    "A handover note covering record {rec_id} was circulated to the desk.",
    "Record {rec_id} was reviewed at the quarterly walkthrough.",
    "The standing delegation for record {rec_id} remains unchanged this period.",
    "Filing details for record {rec_id} were refreshed in the directory.",
    "Record {rec_id} appeared on the routine attendance sheet for the session.",
)


@dataclass
class MultiSpanInstance:
    instance_id: str
    context: str
    variants: tuple[Variant, ...]
    candidates: tuple[Span, ...]
    meta: dict = field(default_factory=dict)

    @property
    def query(self) -> str: return self.variants[0].query
    @property
    def answer(self) -> str: return self.variants[0].answer
    @property
    def gold(self) -> tuple[Span, ...]: return self.variants[0].gold


def _fields_for(rng, n_fields, names, idx):
    out = []
    for j in range(n_fields):
        kind = _FIELD_KINDS[j % len(_FIELD_KINDS)]
        if kind == "surname":
            out.append(names[idx])
        elif kind == "dept":
            out.append(rng.choice(DEPTS))
        elif kind == "site":
            out.append(rng.choice(SITES))
        elif kind == "status":
            out.append(rng.choice(STATUS))
        else:
            out.append("%d" % rng.randrange(100000, 1000000))
    return out


def _split(seq, k):
    """Split `seq` into k contiguous chunks as evenly as possible, every chunk non-empty."""
    n = len(seq)
    base, rem = divmod(n, k)
    out, i = [], 0
    for j in range(k):
        take = base + (1 if j < rem else 0)
        out.append(seq[i:i + take])
        i += take
    return out


def _exemplar(k, n_fields):
    a = ["Aldermill", "Treasury", "471203", "Northgate", "Final", "882640"]
    fa = [a[j % len(a)] for j in range(n_fields)]
    parts = _split(fa, k)
    lines = ["R900 /%s | %s" % (PART_LABELS[j], " | ".join(p)) for j, p in enumerate(parts)]
    return ("Worked example (same format as below).\n" + "\n".join(lines)
            + "\nQ: reproduce record R900. A: " + " | ".join(fa)
            + "\nFind every part of the given record and reproduce its fields in order.\n\n")


def build(seed: int, instance_id: str, *, k: int, n_fields: int,
          n_records: int = N_RECORDS, n_bindings: int = N_BINDINGS,
          target_tokens: int | None = None, tokenizer=None,
          tolerance: int = 16) -> MultiSpanInstance:
    if not 1 <= k <= len(PART_LABELS):
        raise ValueError("k out of range")
    if n_fields < k:
        raise ValueError("need at least one field per part")

    rng = random.Random(seed)
    names = rng.sample(SURNAMES, n_records)
    rec_ids = ["R%03d" % (i + 1) for i in range(n_records)]
    fields = [_fields_for(rng, n_fields, names, i) for i in range(n_records)]
    answers = [" | ".join(f) for f in fields]
    parts = [["%s /%s | %s" % (rec_ids[i], PART_LABELS[j], " | ".join(p))
              for j, p in enumerate(_split(fields[i], k))] for i in range(n_records)]
    queried_idx = rng.sample(range(n_records), n_bindings)
    exemplar = _exemplar(k, n_fields)

    def assemble(n_filler: int) -> str:
        r2 = random.Random(seed ^ 0x3C1F)
        pool = list(rec_ids)
        r2.shuffle(pool)
        filler = [_MENTION[r2.randrange(len(_MENTION))].format(rec_id=pool[j % len(pool)])
                  for j in range(n_filler)]
        # Each record is placed as ONE BLOCK of k lines, then blocks are interleaved uniformly
        # among the filler. Placing k lines individually and walking forward on collision --
        # which is what `ledger_c` can safely do because its records are single lines -- breaks
        # adjacency as soon as blocks are long enough to collide: at k = 4 it produced parts
        # 162 lines apart, caught by the assertion below.
        n_slots = n_records + len(filler)
        slots: list[object] = [None] * n_slots
        taken: set[int] = set()
        for i in range(n_records):
            pos = int(round((i + 0.5) * n_slots / n_records))
            pos = min(max(pos, 0), n_slots - 1)
            while pos in taken:
                pos = (pos + 1) % n_slots
            taken.add(pos)
            slots[pos] = parts[i]
        fi = iter(filler)
        body: list[str] = []
        for s in slots:
            if s is None:
                body.append(next(fi))
            else:
                body.extend(s)
        return exemplar + PREAMBLE + "\n".join(body) + "\n"

    n_filler = 24
    context = assemble(n_filler)
    if target_tokens is not None and tokenizer is not None:
        lo, hi = 0, 2048
        for _ in range(28):
            n_tok = len(tokenizer(context, add_special_tokens=False)["input_ids"])
            if abs(n_tok - target_tokens) <= tolerance:
                break
            if n_tok < target_tokens:
                lo = n_filler
                n_filler = (n_filler + hi) // 2 if hi > n_filler else n_filler + 8
            else:
                hi = n_filler
                n_filler = (lo + n_filler) // 2
            if n_filler <= 0:
                n_filler = 0
                context = assemble(0)
                break
            context = assemble(n_filler)

    lines = context.split("\n")
    line_of = {t: i for i, t in enumerate(lines)}
    seps = []
    for i in range(n_records):
        idxs = [line_of.get(p) for p in parts[i]]
        if any(x is None for x in idxs):
            raise RuntimeError("a part is missing from the assembled body")
        seps.append(max(idxs) - min(idxs))
    if k > 1 and max(seps) != k - 1:
        raise AssertionError(
            "parts are not adjacent: max span %d, expected %d" % (max(seps), k - 1))

    def _span(text: str, rid: str) -> Span:
        at = context.find(text)
        if at < 0:
            raise RuntimeError("part not found: %r" % text)
        return Span("record", text, at, at + len(text), rid)

    cand = []
    for i in queried_idx:
        for p in parts[i]:
            cand.append(_span(p, rec_ids[i]))

    variants = tuple(
        Variant(rec_ids[i],
                ("Reproduce record %s in full. It is recorded in %d part%s. Answer with the "
                 "fields only, in order, separated by vertical bars."
                 % (rec_ids[i], k, "" if k == 1 else "s")),
                answers[i], tuple(_span(p, rec_ids[i]) for p in parts[i]))
        for i in queried_idx)

    return MultiSpanInstance(instance_id, context, variants, tuple(cand),
                             meta=dict(task="multispan", k=k, n_fields=n_fields,
                                       n_records=n_records, n_filler=n_filler,
                                       max_part_span_lines=max(seps)))


_WORD = re.compile(r"[A-Za-z0-9]+")
_REC_ID = re.compile(r"^r\d{3}$")


def field_list(s: str) -> list[str]:
    """Content words, with record ids and single-letter part labels dropped.

    Self-contained rather than shared with `ledger_c`, because this task uses part labels A-F
    while Stage 2's scorer only strips A and B, and Stage 2's scorer is part of a committed
    record that is not being edited. No field value in this family is a single letter or of
    the form R###, so nothing gradable is removed.
    """
    ws = [w.lower() for w in _WORD.findall(s or "")]
    return [w for w in ws if not _REC_ID.match(w) and len(w) > 1]


def score_one(generated: str, answer: str) -> float:
    gold, gen = field_list(answer), field_list(generated)
    if not gold:
        return 0.0
    n = len(gold)
    return 1.0 if any(gen[i:i + n] == gold for i in range(len(gen) - n + 1)) else 0.0


def score_instance(generated_texts: Sequence[str], inst: MultiSpanInstance) -> float:
    if len(generated_texts) != len(inst.variants):
        raise ValueError("expected %d generations, got %d"
                         % (len(inst.variants), len(generated_texts)))
    return sum(score_one(g, v.answer)
               for g, v in zip(generated_texts, inst.variants)) / len(inst.variants)


__all__ = ["build", "score_one", "score_instance", "field_list",
           "MultiSpanInstance", "N_RECORDS", "N_BINDINGS"]
