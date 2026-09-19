"""Fact-sentence family for the natural-text ladder.

    The audit of the {prog} programme was completed by {person} of the {dept} in {year}
    at a cost of ${amount}, and it covered the {loc} depot.

Five elements after the cue, in FIXED sentence order, so that the elements a query requires are
always a contiguous PREFIX of the sentence. That gives a nested cost axis with contiguous gold:

    level 1  person                                    (gold = sentence start .. end of person)
    level 3  person, department, year                  (gold = sentence start .. end of year)
    level 5  person, department, year, amount, depot   (gold = whole sentence)

This is natural surrounding prose with SYSTEMATICALLY GENERATED fact sentences. It is not found
data, and the paper must say so.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from p3.tasks.ledger_c import SURNAMES

ELEMENTS = ("person", "dept", "year", "amount", "loc")
LEVELS = {1: ELEMENTS[:1], 3: ELEMENTS[:3], 5: ELEMENTS}

_ON = ("Hal", "Bren", "Cor", "Dun", "Ed", "Fen", "Gar", "Har", "Kel", "Lan", "Mar", "Nor",
       "Os", "Pen", "Quen", "Rav", "Sel", "Tor", "Ul", "Wen", "Ash", "Bal", "Cal", "Dal",
       "Elm", "Fal", "Gil", "Hen", "Ing", "Jar", "Kes", "Lor", "Mel", "Nev", "Orm", "Pel")
_MID = ("wor", "dle", "mor", "ver", "brin", "ston", "ley", "hol", "car", "thorn")
_END = ("th", "by", "ford", "wick", "ton", "dale", "mere", "ham")
DEPTS = ("Fisheries", "Public Works", "Forestry", "Transport", "Water Supply", "Housing",
         "Agriculture", "Customs", "Education", "Health", "Mines", "Ports", "Records",
         "Revenue", "Sanitation", "Surveys", "Trade", "Lands", "Posts", "Energy", "Rivers",
         "Standards", "Roads", "Archives")
LOCS = ("Ravensmoor", "Kingsbridge", "Thornbury", "Aldergrove", "Marlowe", "Ashcombe",
        "Fairhaven", "Stonebridge", "Oakhurst", "Redcliffe", "Westmere", "Larkfield",
        "Cresswell", "Dunmore", "Elderton", "Foxley", "Greywater", "Hollins", "Ironbridge",
        "Juniper Bay", "Kilbride", "Lowmoor", "Millbrook", "Northgate", "Oldcastle",
        "Pinewood", "Quarry Hill", "Rushden", "Saltmarsh", "Tanfield")
FIRST = ("Marisa", "Tobias", "Helena", "Gareth", "Imogen", "Rafael", "Nadia", "Cormac",
         "Priya", "Leopold", "Sunniva", "Anselm", "Odette", "Bertram", "Yara", "Desmond",
         "Freya", "Ignatius", "Jocasta", "Konrad", "Lucinda", "Magnus", "Noor", "Osric",
         "Petra", "Quentin", "Rosalind", "Silas", "Thea", "Ulrich", "Vera", "Wilfred",
         "Xenia", "Yusuf", "Zelda", "Alistair", "Beatrix", "Cyrus", "Delia", "Emeric",
         "Fenella", "Gideon", "Hattie", "Ivor", "Juliet", "Kasper", "Linnea", "Matthias",
         "Nerys", "Orla", "Percival", "Rhona", "Stellan", "Tamsin", "Ursula", "Viktor")
EXEMPLAR = dict(prog="Wexbourne", person="Dara Quill", dept="Department of Ports", year="2011",
                amount="20,500", loc="Alderney")
_RESERVED = {"Wexbourne", "Dara Quill", "Alderney"}


@dataclass
class Fact:
    fact_id: str
    prog: str
    person: str
    dept: str
    year: str
    amount: str
    loc: str

    def segments(self):
        """(text, element-or-None) pieces; their concatenation is the sentence."""
        return [("The audit of the ", None), (self.prog, "prog"),
                (" programme was completed by ", None), (self.person, "person"),
                (" of the ", None), (self.dept, "dept"), (" in ", None),
                (self.year, "year"), (" at a cost of $", None), (self.amount, "amount"),
                (", and it covered the ", None), (self.loc, "loc"), (" depot.", None)]

    def text(self) -> str:
        return "".join(t for t, _ in self.segments())

    def spans(self):
        """Element -> (start, end) offsets within the sentence."""
        out, at = {}, 0
        for t, k in self.segments():
            if k:
                out[k] = (at, at + len(t))
            at += len(t)
        return out


PROGS = sorted({a + b + c for a in _ON for b in _MID for c in _END} - _RESERVED)


def make_facts(rng: random.Random, n: int, avoid_words: set) -> list:
    """`n` facts: unique programme names and unique surnames; every other element is drawn from
    the SAME pools for queried and distractor facts alike (lexically matched by construction)."""
    progs = rng.sample([p for p in PROGS if p.lower() not in avoid_words], n)
    surn = rng.sample([s for s in SURNAMES if s.lower() not in avoid_words and s != "Quill"], n)
    facts = []
    for i in range(n):
        person = f"{rng.choice(FIRST)} {surn[i]}"
        amt = rng.randrange(120, 960) * 100
        facts.append(Fact(f"F{i:02d}", progs[i], person, "Department of " + rng.choice(DEPTS),
                          str(rng.randrange(2004, 2024)), f"{amt:,}", rng.choice(LOCS)))
    return facts


def query_text(level: int, prog: str) -> str:
    if level == 1:
        return f"Who completed the audit of the {prog} programme? Answer with the name only."
    items = {3: "the person who completed it | their department | the year",
             5: "the person who completed it | their department | the year | "
                "the cost in dollars | the depot it covered"}[level]
    n = {3: "three", 5: "five"}[level]
    return (f"For the audit of the {prog} programme, give: {items}. Answer with those {n} "
            "items only, in that order, separated by ' | '.")


def element_answers(f: Fact, level: int) -> list:
    vals = dict(person=f.person, dept=f.dept, year=f.year, amount=f.amount, loc=f.loc)
    return [vals[e] for e in LEVELS[level]]


def preamble() -> str:
    e = EXEMPLAR
    sent = Fact("X", e["prog"], e["person"], e["dept"], e["year"], e["amount"], e["loc"]).text()
    a1 = e["person"]
    a3 = e["person"] + " | " + e["dept"] + " | " + e["year"]
    a5 = a3 + " | $" + e["amount"] + " | " + e["loc"]
    return ("Below is a compilation of passages from several books. Scattered among the passages "
            "are audit records, each a single sentence beginning 'The audit of the'. Answer "
            "questions about the audit records using only the compilation.\n\n"
            "Worked example. Record: " + sent + "\n"
            "Q: " + query_text(1, e["prog"]) + "\nA: " + a1 + "\n"
            "Q: " + query_text(3, e["prog"]) + "\nA: " + a3 + "\n"
            "Q: " + query_text(5, e["prog"]) + "\nA: " + a5 + "\n\n---\n\n")
