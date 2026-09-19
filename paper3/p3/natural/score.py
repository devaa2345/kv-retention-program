"""Scorer for the natural-text ladder. Unit-tested in test_nat_scorer.py BEFORE any verdict.

(v2, after the M3 audit: only the first answer block is scored.)
A variant scores 1.0 iff EVERY required element appears in the generation as a whole-token match
(case-insensitive; thousands commas and '$' stripped), AND the generation does not name any
OTHER person from the instance's fact set -- a model that lists every name in the context must
not score by shotgun. An instance scores the MEAN over its H variants.

Conventions that already cost this project false failures, and how they are handled here:
  * substring matching ('14' inside '2014')  -> boundary-anchored, not substring
  * aggregation ('landed on any of H')       -> mean over H variants, chance = 1/(H+D)
  * a model echoing the worked example       -> exemplar values are disjoint from every
                                                generated pool (asserted at build time)
"""
from __future__ import annotations

import re

_NUM = re.compile(r"(?<=\d),(?=\d{3})")


def norm(s: str) -> str:
    s = (s or "").replace("$", " ")
    s = _NUM.sub("", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def contains(gen_norm: str, elem: str) -> bool:
    e = norm(elem)
    if not e:
        return False
    return re.search(r"(?<![a-z0-9])" + re.escape(e) + r"(?![a-z0-9])", gen_norm) is not None


_CONT = re.compile(r"\n\s*\n|\n\s*[B-Z]\s*[:.)]\s")


def first_answer(generated: str) -> str:
    """The FIRST answer block. Llama-3.2 often answers correctly and then invents a further
    'B: ...' record; that continuation is not the answer and must not trigger the shotgun rule.
    Cut at the first blank line or at a line that opens a new lettered item (B:, C:, ...).
    Single newlines inside one answer (numbered or one-field-per-line lists) are kept."""
    m = _CONT.search((generated or "").lstrip())
    g = (generated or "").lstrip()
    return g[:m.start()] if m else g


def score_one(generated: str, elements: list, all_persons: list) -> float:
    g = norm(first_answer(generated))
    if not all(contains(g, e) for e in elements):
        return 0.0
    gold = norm(elements[0])
    for p in all_persons:
        if norm(p) != gold and contains(g, p):
            return 0.0
    return 1.0


def score_instance(gens: list, variants: list, all_persons: list) -> float:
    if len(gens) != len(variants):
        raise ValueError(f"expected {len(variants)} generations, got {len(gens)}")
    return sum(score_one(g, v["elements"], all_persons)
               for g, v in zip(gens, variants)) / len(variants)
