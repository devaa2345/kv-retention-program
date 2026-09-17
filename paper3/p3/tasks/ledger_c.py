"""LEDGER-C — Paper 2's LEDGER with **fact cost `c` as a first-class parameter**.

Paper 2's record line was `R011 | Surname | Dept | 280294`, ~18.9 tokens on M2 and ~12.9 on M3,
and `c` was never varied. Stage 2 probe 2.1 needs it varied, and Stage 1 showed why: the
implied `c_eff` rises monotonically with budget, so `c` and `C` are confounded in everything
Paper 2 measured.

**Making `c` the FUNCTIONAL fact cost, not just the span length.**

Paper 2 measured completeness two ways -- the whole LINE and the (id, value) pair -- precisely
because a policy keeping the id and the value while dropping the surname and department was
being scored as "record not retained" when it could in fact still answer. Padding a record line
to make it longer would reproduce that problem at every `c`: the nominal cost would grow while
the minimal answerable set stayed at id+value, and the cost axis would measure nothing.

So the query here asks the model to **reproduce the whole record line**:

    "Reproduce record R011 exactly, in full, in the format shown."   ->   the line's fields

Every token of the line is then required, `c` is the functional fact cost by construction, and
the two completeness units of Paper 2 collapse into one. The task stays a pure COPY -- no
composition, no arithmetic -- which is the property that let Paper 2's one-hop LEDGER reach a
competence anchor at all after the two-hop design failed.

`c` is varied by the number of fields. Field values are natural words and 6-digit numbers, so
a longer line is a longer copy rather than a harder one; whether that survives at c = 40 is an
empirical question the anchor answers, not an assumption made here.

**Held fixed from Paper 2, deliberately:** N = 40 records, H = 4 queried, L ~ 2048 in the
model's own tokenizer, uniform interleaving of records through filler (the v2 layout), the
2-shot exemplar frame, and the surname pool. Chance is 1/40 on record identity.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Sequence

N_RECORDS = 40
N_BINDINGS = 4

DEPTS = ("Procurement", "Actuarial", "Custody", "Clearing", "Treasury", "Compliance",
         "Settlement", "Registry", "Underwriting", "Reconciliation", "Valuation", "Payments")

SITES = ("Northgate", "Harborline", "Westmoor", "Kingsfell", "Ashcombe", "Draymouth",
         "Elmsworth", "Fairhollow", "Granthill", "Larkmede", "Oakstead", "Pinebury")

STATUS = ("Open", "Closed", "Pending", "Review", "Cleared", "Held", "Draft", "Final")

_SYL_A = ("Bar", "Cald", "Dorn", "Ell", "Fen", "Gar", "Hal", "Ing", "Jar", "Kel",
          "Lund", "Mor", "Nass", "Orr", "Pell", "Quist", "Rand", "Sel", "Tor", "Vane")
_SYL_B = ("ac", "en", "ir", "ol", "un", "am", "eth", "id", "or", "us")
_SYL_C = ("ker", "son", "dahl", "mont", "well", "ridge", "stad", "hardt", "field", "wick")


def surname_pool() -> list[str]:
    pool: list[str] = []
    for a in _SYL_A:
        for b in _SYL_B:
            for c in _SYL_C:
                pool.append(a + b + c)
                if len(pool) >= 400:
                    return pool
    return pool


SURNAMES = surname_pool()

PREAMBLE = ("INTERNAL LEDGER EXTRACT. The lines below mix account records with governance notes "
            "for the current audit period. Each record has an identifier followed by its "
            "fields, separated by vertical bars.\n")

_MENTION_TEMPLATES = (
    "Correspondence concerning record {rec_id} was logged and filed without further action.",
    "A handover note covering record {rec_id} was circulated to the desk.",
    "Record {rec_id} was reviewed at the quarterly walkthrough.",
    "The standing delegation for record {rec_id} remains unchanged this period.",
    "Filing details for record {rec_id} were refreshed in the directory.",
    "Record {rec_id} appeared on the routine attendance sheet for the session.",
)

# Field generators, cycled in this order. Exemplar values sit outside every generated pool.
_FIELD_KINDS = ("surname", "dept", "value", "site", "status", "value")


@dataclass(frozen=True)
class Span:
    kind: str
    text: str
    start: int
    end: int
    rec_id: str | None = None


@dataclass(frozen=True)
class Variant:
    rec_id: str
    query: str
    answer: str
    gold: tuple[Span, ...]


@dataclass
class LedgerCInstance:
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


def _fields_for(rng: random.Random, n_fields: int, names: list[str], idx: int) -> list[str]:
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
            out.append(f"{rng.randrange(100000, 1000000)}")
    return out


def _exemplars(n_fields: int) -> str:
    a = ["Aldermill", "Treasury", "471203", "Northgate", "Final", "882640"]
    b = ["Brackenhold", "Registry", "318574", "Westmoor", "Open", "204917"]
    fa = [a[j % len(a)] for j in range(n_fields)]
    fb = [b[j % len(b)] for j in range(n_fields)]
    sa, sb = " | ".join(fa), " | ".join(fb)
    return ("Worked examples (same format as below).\n"
            f"R900 | {sa}\n"
            f"Q: reproduce record R900. A: {sa}\n"
            f"R901 | {sb}\n"
            f"Q: reproduce record R901. A: {sb}\n"
            "Find the line with the given record id and reproduce its fields in order.\n\n")


def build(seed: int, instance_id: str, *, n_fields: int,
          n_records: int = N_RECORDS, n_bindings: int = N_BINDINGS,
          target_tokens: int | None = None, tokenizer=None,
          tolerance: int = 16) -> LedgerCInstance:
    """One instance with `n_fields` fields per record.

    `n_fields` sets the fact cost; the achieved `c` in tokens is measured, never assumed --
    see `p3/calibrate_c.py`, which solves `n_fields` per model for a target `c`.
    """
    rng = random.Random(seed)
    names = rng.sample(SURNAMES, n_records)
    rec_ids = [f"R{i + 1:03d}" for i in range(n_records)]
    fields = [_fields_for(rng, n_fields, names, i) for i in range(n_records)]
    records = [f"{rec_ids[i]} | " + " | ".join(fields[i]) for i in range(n_records)]
    answers = [" | ".join(fields[i]) for i in range(n_records)]

    queried_idx = rng.sample(range(n_records), n_bindings)
    exemplars = _exemplars(n_fields)

    def assemble(n_filler: int) -> str:
        pool = list(rec_ids)
        rng.shuffle(pool)
        filler = [_MENTION_TEMPLATES[rng.randrange(len(_MENTION_TEMPLATES))].format(
            rec_id=pool[j % len(pool)]) for j in range(n_filler)]
        total = len(records) + len(filler)
        slots: list[str | None] = [None] * total
        taken: set[int] = set()
        for i in range(len(records)):
            pos = int(round((i + 0.5) * total / len(records)))
            pos = min(max(pos, 0), total - 1)
            while pos in taken:
                pos = (pos + 1) % total
            taken.add(pos)
            slots[pos] = records[i]
        fi = iter(filler)
        body = [s if s is not None else next(fi) for s in slots]
        return exemplars + PREAMBLE + "\n".join(body) + "\n"

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
            rng = random.Random(seed + 10_000 + n_filler)
            context = assemble(n_filler)

    def _span(text: str, rid: str) -> Span:
        start = context.find(text)
        if start < 0:
            raise RuntimeError(f"span not found: {text!r}")
        return Span("record", text, start, start + len(text), rid)

    cand = [_span(records[i], rec_ids[i]) for i in queried_idx]
    variants = tuple(
        Variant(rec_ids[i],
                (f"Reproduce record {rec_ids[i]} exactly, in full, in the format shown. "
                 "Answer with the fields only, separated by vertical bars."),
                answers[i], (_span(records[i], rec_ids[i]),))
        for i in queried_idx)

    return LedgerCInstance(instance_id, context, variants, tuple(cand),
                           meta=dict(n_fields=n_fields, n_records=n_records,
                                     n_filler=n_filler, task="ledger_c"))


_WS = re.compile(r"\s+")
_WORD = re.compile(r"[A-Za-z0-9]+")
_REC_ID = re.compile(r"^r\d{3}$")


def field_list(s: str) -> list[str]:
    """The sequence of CONTENT WORDS in a string, with record ids and part markers dropped.

    Grading must test retrieval, not obedience to an output format. Asked to reproduce a
    two-part record, the model reliably answers

        R034 /A | Barenwell | Payments
        R034 /B | 711505 | Granthill

    which is the correct content, correctly ordered, with the line markers kept. A substring
    test on the joined fields scores that 0 and reads a working retrieval as a retention
    failure -- the same class of error as Paper 2's B9, where a pinned `max_new` measured
    instruction-following and was reported as retrieval.

    Comparison is at WORD level rather than field level so that a model which answers in
    prose ("the value is 280294") is not penalised either; that matters because at
    `n_fields = 1` the answer is a single field and there is no ordering left to enforce.
    Every field value in this task family is a single word -- a surname, a one-word
    department, site or status, or a digit string -- so word order carries exactly the field
    order and nothing is lost.

    Dropped as markers: record ids (`r000`-form) and the bare part letters `a` and `b`. No
    field value is a single letter or of the form R###, so nothing gradable is removed.
    """
    ws = [w.lower() for w in _WORD.findall(s or "")]
    return [w for w in ws if not _REC_ID.match(w) and w not in ("a", "b")]


def normalise(s: str) -> str:
    s = _WS.sub(" ", (s or "").strip())
    return " | ".join(p.strip() for p in s.split("|")).lower()


def score_one(generated: str, answer: str) -> float:
    """1.0 iff the gold word sequence appears as a CONTIGUOUS run in the generation."""
    gold, gen = field_list(answer), field_list(generated)
    if not gold:
        return 0.0
    n = len(gold)
    return 1.0 if any(gen[i:i + n] == gold for i in range(len(gen) - n + 1)) else 0.0


def score_instance(generated_texts: Sequence[str], inst: LedgerCInstance) -> float:
    if len(generated_texts) != len(inst.variants):
        raise ValueError(f"expected {len(inst.variants)} generations, got {len(generated_texts)}")
    return sum(score_one(g, v.answer) for g, v in zip(generated_texts, inst.variants)) / len(
        inst.variants)


__all__ = ["build", "score_instance", "score_one", "normalise", "field_list",
           "LedgerCInstance", "Span", "Variant", "N_RECORDS", "N_BINDINGS"]
