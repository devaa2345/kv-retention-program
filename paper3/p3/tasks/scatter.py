"""SPLIT-LEDGER — the same fact, laid out ADJACENT or SCATTERED (probe 2.3).

Probe 2.3 asks whether the recency floor's advantage survives when facts are deliberately
scattered, so that contiguity cannot help either policy. That only works as a control if the
scattered and unscattered conditions are **identical in content** and differ only in where the
tokens sit. So both layouts come from this one module, from the same seed, with the same
fields, the same records, the same queries and the same answers.

Each record is split into two labelled halves:

    R011 /A | Fenorson | Treasury
    R011 /B | 280294 | Northgate

and the query asks for both halves in order:

    "Record R011 is split across two lines, /A and /B. Reproduce its fields in order,
     part A then part B."

So the fact is genuinely two spans that must BOTH survive -- the label carries the binding, so
a half on its own is identifiable but not sufficient.

    layout="adjacent"   the two halves are consecutive lines
    layout="scattered"  the two halves are placed at independent positions, far apart

Under `adjacent`, a recency block of width W that reaches the record captures both halves
together, so P(fact complete) ~ W/L as in Paper 2. Under `scattered`, the halves are
independent, so P(both inside the block) ~ (W/L)^2 and the floor's positional advantage should
collapse. A pointwise scorer has no such structural dependence on layout -- it keeps whatever
it scores highly, wherever it is. That asymmetry is the probe.

**Minimum separation is enforced**, not left to chance: under `scattered` the two halves are
placed in different halves of the body and at least `MIN_SEP_LINES` apart, so no scattered
instance is accidentally adjacent.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Sequence

from p3.tasks.ledger_c import (DEPTS, SITES, STATUS, SURNAMES, Span, Variant, _FIELD_KINDS,
                               score_one as _score_one)

N_RECORDS = 40
N_BINDINGS = 4
MIN_SEP_LINES = 12

PREAMBLE = ("INTERNAL LEDGER EXTRACT. Each record is recorded in two parts, marked /A and /B, "
            "which may appear anywhere in the extract. Governance notes are interleaved.\n")

_MENTION = (
    "Correspondence concerning record {rec_id} was logged and filed without further action.",
    "A handover note covering record {rec_id} was circulated to the desk.",
    "Record {rec_id} was reviewed at the quarterly walkthrough.",
    "The standing delegation for record {rec_id} remains unchanged this period.",
    "Filing details for record {rec_id} were refreshed in the directory.",
    "Record {rec_id} appeared on the routine attendance sheet for the session.",
)

EXEMPLARS = ("Worked examples (same format as below).\n"
             "R900 /A | Aldermill | Treasury\n"
             "R900 /B | 471203 | Northgate\n"
             "Q: reproduce record R900. A: Aldermill | Treasury | 471203 | Northgate\n"
             "Find both parts of the given record and reproduce their fields in order.\n\n")


@dataclass
class SplitInstance:
    instance_id: str
    context: str
    variants: tuple[Variant, ...]
    candidates: tuple[Span, ...]        # one Span per HALF: 2 per queried record
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
            out.append(f"{rng.randrange(100000, 1000000)}")
    return out


def build(seed: int, instance_id: str, *, layout: str, n_fields: int = 4,
          n_records: int = N_RECORDS, n_bindings: int = N_BINDINGS,
          target_tokens: int | None = None, tokenizer=None,
          tolerance: int = 16) -> SplitInstance:
    if layout not in ("adjacent", "scattered", "scattered_uniform"):
        raise ValueError(layout)
    if n_fields < 2:
        raise ValueError("need at least one field per half")

    rng = random.Random(seed)
    names = rng.sample(SURNAMES, n_records)
    rec_ids = [f"R{i + 1:03d}" for i in range(n_records)]
    fields = [_fields_for(rng, n_fields, names, i) for i in range(n_records)]
    half = n_fields // 2
    halves_a = [f"{rec_ids[i]} /A | " + " | ".join(fields[i][:half]) for i in range(n_records)]
    halves_b = [f"{rec_ids[i]} /B | " + " | ".join(fields[i][half:]) for i in range(n_records)]
    answers = [" | ".join(fields[i]) for i in range(n_records)]
    queried_idx = rng.sample(range(n_records), n_bindings)

    def assemble(n_filler: int) -> str:
        r2 = random.Random(seed ^ 0x5F3A)
        pool = list(rec_ids)
        r2.shuffle(pool)
        filler = [_MENTION[r2.randrange(len(_MENTION))].format(rec_id=pool[j % len(pool)])
                  for j in range(n_filler)]
        total = 2 * n_records + len(filler)
        slots: list[str | None] = [None] * total
        taken: set[int] = set()

        def place(pos, text):
            pos = min(max(pos, 0), total - 1)
            while pos in taken:
                pos = (pos + 1) % total
            taken.add(pos)
            slots[pos] = text

        if layout == "adjacent":
            # both halves consecutive, at the same uniform positions LEDGER-C would use
            for i in range(n_records):
                pos = int(round((i + 0.5) * total / n_records))
                place(pos, halves_a[i])
                place(pos + 1, halves_b[i])
        elif layout == "scattered":
            # DEFECTIVE, KEPT FOR THE RECORD. /A uniformly through the FIRST half of the body,
            # /B through the SECOND. This guarantees separation, but it also guarantees that
            # the recency block -- the last B of L tokens, ~28% of the body at C=512 -- can
            # contain ONLY /B lines. `floor_pos` can then never complete a fact, and scores
            # 0.0000 by construction rather than by scattering. Measured: every compressed arm
            # at 0.0000 on both models, which is the vacuous-gate failure Paper 2 documented
            # in its own Stage 4 ablations. Superseded by `scattered_uniform`.
            for i in range(n_records):
                pa = int(round((i + 0.5) * (total // 2) / n_records))
                pb = total // 2 + int(round((i + 0.5) * (total - total // 2) / n_records))
                place(pa, halves_a[i])
                place(pb, halves_b[i])
        else:
            # scattered_uniform -- THE CONTROL AS INTENDED. Both halves are drawn from the
            # SAME uniform distribution over the whole body, so neither half is positionally
            # privileged and a recency block can hold either, both, or neither. Separation is
            # enforced as a MINIMUM by resampling, never as a structural partition.
            # Constructive, not rejection-sampled. An earlier version shuffled and retried,
            # falling back to a fixed pairing when 64 shuffles failed; that fallback paired
            # neighbouring slots and 3 of 50 M2 instances then tripped the separation
            # assertion and killed the run. This builds a valid assignment directly: take a
            # free slot, pair it with a uniformly chosen free slot at least MIN_SEP_LINES
            # away, and if none qualifies take the furthest one available. Both halves still
            # come from the same uniform grid, so neither is positionally privileged.
            # Separation is guaranteed by CONSTRUCTION, not by sampling. Two earlier attempts
            # failed here and both failed the same way: a rejection sampler with a fallback,
            # and a greedy pairing, each ran out of distant slots for the last few records and
            # tripped the separation assertion on 3 and 8 of 50 M2 instances respectively --
            # killing the run rather than degrading it.
            #
            # Slot j is paired with slot j + n_records, so every pair sits half the body apart.
            # Which of the two halves takes the earlier slot is randomised per record, so /A is
            # uniformly spread over the whole body rather than confined to the front -- which
            # is the single property the first (defective) layout lacked.
            r3 = random.Random(seed ^ 0xA17C)
            grid = [int(round((j + 0.5) * total / (2 * n_records)))
                    for j in range(2 * n_records)]
            pairs = [(grid[j], grid[j + n_records]) for j in range(n_records)]
            r3.shuffle(pairs)
            for i in range(n_records):
                lo, hi = pairs[i]
                if r3.random() < 0.5:
                    place(lo, halves_a[i])
                    place(hi, halves_b[i])
                else:
                    place(hi, halves_a[i])
                    place(lo, halves_b[i])

        fi = iter(filler)
        body = [s if s is not None else next(fi) for s in slots]
        return EXEMPLARS + PREAMBLE + "\n".join(body) + "\n"

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

    def _span(text: str, rid: str) -> Span:
        at = context.find(text)
        if at < 0:
            raise RuntimeError(f"half not found: {text!r}")
        return Span("record", text, at, at + len(text), rid)

    # enforce the separation the layout claims
    seps = []
    for i in range(n_records):
        la, lb = line_of.get(halves_a[i]), line_of.get(halves_b[i])
        if la is None or lb is None:
            raise RuntimeError("half missing from assembled body")
        seps.append(abs(lb - la))
    if layout == "adjacent" and max(seps) != 1:
        raise AssertionError(f"adjacent layout has non-adjacent halves: max sep {max(seps)}")
    if layout.startswith("scattered") and min(seps) < MIN_SEP_LINES:
        raise AssertionError(
            f"scattered layout has halves only {min(seps)} lines apart "
            f"(minimum {MIN_SEP_LINES}); the control would not be controlling anything")

    cand = []
    for i in queried_idx:
        cand.append(_span(halves_a[i], rec_ids[i]))
        cand.append(_span(halves_b[i], rec_ids[i]))

    variants = tuple(
        Variant(rec_ids[i],
                (f"Record {rec_ids[i]} is split across two lines, /A and /B. Reproduce its "
                 "fields in order, part A then part B. Answer with the fields only, "
                 "separated by vertical bars."),
                answers[i], (_span(halves_a[i], rec_ids[i]), _span(halves_b[i], rec_ids[i])))
        for i in queried_idx)

    return SplitInstance(instance_id, context, variants, tuple(cand),
                         meta=dict(task="split_ledger", layout=layout, n_fields=n_fields,
                                   n_filler=n_filler, sep_lines_min=min(seps),
                                   sep_lines_mean=sum(seps) / len(seps)))


def score_one(generated: str, answer: str) -> float:
    """Shared with LEDGER-C: the gold field sequence as a contiguous run, ids and /A /B
    markers stripped. Both layouts and both tasks are graded identically, which is what
    makes the adjacent/scattered contrast a contrast."""
    return _score_one(generated, answer)


def score_instance(generated_texts: Sequence[str], inst: SplitInstance) -> float:
    if len(generated_texts) != len(inst.variants):
        raise ValueError(f"expected {len(inst.variants)} generations, got {len(generated_texts)}")
    return sum(score_one(g, v.answer) for g, v in zip(generated_texts, inst.variants)) / len(
        inst.variants)


__all__ = ["build", "score_instance", "score_one", "SplitInstance", "N_RECORDS", "N_BINDINGS"]
