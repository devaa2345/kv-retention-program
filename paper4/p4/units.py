"""Oracle units: the TRUE fact boundaries, read from the task generators' own line formats.

Query-blind by construction. Units are EVERY record (LEDGER-C) or EVERY entry (MARK-1) in the
context, never only the H queried ones -- restricting units to the queried facts would hand U-X
the identity of the answer and turn a fragmentation test into a prescient oracle.

LEDGER-C: a unit is a whole record line `R0dd | f1 | ... | fk` -- the fact the query asks the
model to reproduce, and the span `ledger_c.build` itself uses as the fact. The two worked-example
record lines (R900, R901) are also record lines and are included, matching Paper 3's
`units_complete` accounting (42 units).

MARK-1: a unit is an entry WORD (`ledger`'s `_span` convention: the word, not the shared "- "
prefix). The four queried category words are one token each, so at c = 1 unit-awareness cannot
make a queried fact more complete; the 36 surnames are multi-token units, so U-X still reallocates
among distractors. That is what makes c = 1 a CONTROL rather than an identity: any accuracy gain
there cannot come from completing the fact. (The strict identity -- all units singletons -- is
checked separately and exactly.)

Line offsets are walked, not searched: `context.index(line)` can match a line as a prefix of an
earlier, longer line.
"""
from __future__ import annotations

import re

from p4.unitwrap import UnitIndex

REC = re.compile(r"^R(\d{3})\b")


def _lines(context: str):
    at = 0
    for line in context.split("\n"):
        yield line, at
        at += len(line) + 1


def char_spans(inst, task: str):
    out = []
    for line, at in _lines(inst.context):
        if task == "ledger_c":
            if REC.match(line):
                out.append((at, at + len(line)))
        elif task == "mark1":
            if line.startswith("- ") and len(line) > 2:
                out.append((at + 2, at + len(line)))
        else:
            raise ValueError(task)
    return out


def token_units(inst, task: str, tok, pre: str):
    enc = tok(pre, add_special_tokens=False, return_offsets_mapping=True)
    off = enc["offset_mapping"]
    base = pre.index(inst.context)
    units = []
    for a, b in char_spans(inst, task):
        a, b = a + base, b + base
        units.append([ti for ti, (x, y) in enumerate(off) if y > x and x < b and y > a])
    return units, len(off)


def oracle_unit_index(inst, task: str, tok, pre: str, n_sink: int, n_window: int) -> UnitIndex:
    units, n_ctx = token_units(inst, task, tok, pre)
    return UnitIndex(n_ctx, units, n_sink, n_window)


__all__ = ["char_spans", "token_units", "oracle_unit_index"]
