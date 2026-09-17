"""MARK-1 — single-token facts, `c` = 1. The sharpest test in the programme (probe 2.4).

**Why LEDGER cannot reach c = 1.** A retrieval fact needs a key and a value: the id locates the
record and the value answers. That is two spans and, in either tokenizer, at least six tokens.
Paper 2's minimum was 12.9. So `c` = 1 needs a task where the token that must survive is
simultaneously the cue and the answer.

**The construction.** The context is a list of N = 40 single-word entries. Thirty-six are
surnames; four are the odd ones out, one from each of four disjoint categories, and each is a
word that tokenises to exactly ONE token in **both** model tokenizers (verified, not assumed).
Each of the H = 4 queries names a category:

    "Exactly one entry in the list is a colour. Which one? Answer with that word only."

The minimal token set that must be retained to answer is the single token carrying that word.
Nothing has to be co-retained with it: there is no id to bind, and the category is a property
the model reads off the token itself. So `p_g ** c_eff` collapses to `p_g` exactly, and a
pointwise scorer -- which is strictly better informed than position -- should beat a recency
floor. If it does not, co-retention is not what is limiting these methods.

**Why this design and not a yes/no probe.** A forced choice puts chance at 0.5 and would make
the competence anchor meaningless. Here the model must emit the exact word from an open
vocabulary; the empirical chance level is measured by the `null` and `random` arms rather than
asserted, exactly as in Paper 2.

**Held fixed from LEDGER:** N = 40 entries, H = 4 queries scored as their mean, L ~ 2048 in the
model's own tokenizer, uniform interleaving of entries through filler, a 2-shot exemplar frame.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Sequence

from p3.tasks.ledger_c import SURNAMES, Span, Variant

N_ENTRIES = 40
N_BINDINGS = 4

# Every word below is exactly one token in BOTH Qwen2.5 and Llama-3.2 in the "- {w}\n" frame.
# Checked by out/_tokcheck.py; `metal` was dropped because silver/gold overlap `colour`.
CATEGORIES = {
    "colour": ("red", "green", "blue", "black", "white", "grey", "brown", "pink",
               "purple", "orange", "yellow"),
    "month": ("January", "February", "March", "April", "June", "July", "August",
              "September", "October", "November", "December"),
    "animal": ("horse", "sheep", "goat", "wolf", "bear", "deer", "fox", "hawk",
               "crow", "mouse", "tiger", "camel"),
    "fruit": ("apple", "pear", "plum", "grape", "lemon", "peach", "cherry", "mango"),
}
CAT_ORDER = ("colour", "month", "animal", "fruit")

PREAMBLE = ("INVENTORY TAG LIST. The lines below are tags recorded during the audit, one per "
            "line. Most are surnames. A few are ordinary words of other kinds.\n")

EXEMPLARS = ("Worked examples (same format as below).\n"
             "- Aldermill\n- violet\n- Brackenhold\n"
             "Q: which entry is a colour? A: violet\n"
             "Q: which entry is a surname? A: Aldermill\n"
             "Find the single entry of the named kind and report that word only.\n\n")

_MENTION = (
    "The tag sheet was countersigned at the close of the session.",
    "Tags were transcribed from the desk log without amendment.",
    "This page of the tag list was checked against the register.",
    "Entries on this sheet were confirmed by the duty officer.",
    "The sheet was filed with the period's audit papers.",
    "Transcription of this page completed without exception.",
)


@dataclass
class Mark1Instance:
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


def build(seed: int, instance_id: str, *, n_entries: int = N_ENTRIES,
          target_tokens: int | None = None, tokenizer=None,
          tolerance: int = 16) -> Mark1Instance:
    rng = random.Random(seed)
    n_odd = len(CAT_ORDER)
    names = rng.sample(SURNAMES, n_entries - n_odd)
    odd = {cat: rng.choice(CATEGORIES[cat]) for cat in CAT_ORDER}

    entries = list(names) + [odd[c] for c in CAT_ORDER]
    rng.shuffle(entries)
    lines = [f"- {e}" for e in entries]

    def assemble(n_filler: int) -> str:
        filler = [_MENTION[rng.randrange(len(_MENTION))] for _ in range(n_filler)]
        total = len(lines) + len(filler)
        slots: list[str | None] = [None] * total
        taken: set[int] = set()
        for i in range(len(lines)):
            pos = int(round((i + 0.5) * total / len(lines)))
            pos = min(max(pos, 0), total - 1)
            while pos in taken:
                pos = (pos + 1) % total
            taken.add(pos)
            slots[pos] = lines[i]
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
            rng = random.Random(seed + 10_000 + n_filler)
            context = assemble(n_filler)

    def _span(word: str) -> Span:
        """The span of the WORD ITSELF, not of its line -- the fact is the token, and the
        leading '- ' is shared by every entry and carries nothing."""
        needle = f"- {word}\n"
        at = context.find(needle)
        if at < 0:
            raise RuntimeError(f"entry not found: {word!r}")
        start = at + 2
        return Span("entry", word, start, start + len(word), word)

    cand = tuple(_span(odd[c]) for c in CAT_ORDER)
    variants = tuple(
        Variant(odd[c],
                (f"Exactly one entry in the list is a {c}. Which one? "
                 "Answer with that word only."),
                odd[c], (_span(odd[c]),))
        for c in CAT_ORDER)

    return Mark1Instance(instance_id, context, variants, cand,
                         meta=dict(task="mark1", n_entries=n_entries, n_filler=n_filler,
                                   odd=dict(odd)))


_WORD = re.compile(r"[A-Za-z]+")


def score_one(generated: str, answer: str) -> float:
    """1.0 iff the answer word is the FIRST word emitted, case-insensitively.

    A substring test would be wrong here: the categories are small closed sets, so a model
    that lists several colours would score a hit by accident. Requiring the first word makes
    the arms comparable and keeps the guessing floor at the level the `random` arm measures.
    """
    m = _WORD.search(generated or "")
    return 1.0 if (m and m.group(0).lower() == answer.lower()) else 0.0


def score_instance(generated_texts: Sequence[str], inst: Mark1Instance) -> float:
    if len(generated_texts) != len(inst.variants):
        raise ValueError(f"expected {len(inst.variants)} generations, got {len(generated_texts)}")
    return sum(score_one(g, v.answer) for g, v in zip(generated_texts, inst.variants)) / len(
        inst.variants)


__all__ = ["build", "score_instance", "score_one", "Mark1Instance",
           "N_ENTRIES", "N_BINDINGS", "CATEGORIES", "CAT_ORDER"]
