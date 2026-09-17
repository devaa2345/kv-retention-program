"""Task B — LEDGER, ONE-HOP (design plan v1 §4.2, amended 2026-09-06).

Per instance:

    CONTEXT (target L tokens in the model's OWN tokenizer)
      2-shot exemplars              worked lookup + bare-value answer
      preamble                      fixed boilerplate
      body                          N = 40 record lines INTERLEAVED UNIFORMLY through the
                                    filler (v2 layout), identical schema:
                                      "R{iii} | {surname} | {dept} | {6-digit value}"
                                    filler is record-id mentions, lexically matched, to L

    QUERY (appended AFTER compression -- agnostic protocol, v1 §5.2)
      "What is the value on record R047? Answer with the 6-digit value only."

    GOLD SPAN (1): the record line with that id.
    ANSWER = the 6-digit value. Gold-substring match.

**Why one hop.** Three two-hop variants (surname-link at N=96 and N=80, id-link at N=40), plus
2-shot exemplars and `max_new_tokens=32`, all left the v1 §4.5 competence anchor between 0.08
and 0.23 against a required [0.55, 0.97]. The two-hop chain is beyond this model class. Dropping
to one hop is a **design decision, not tuning**: the binding sentence is removed and the query
names the record id directly, so the task is retrieval under compression rather than
retrieval-plus-composition.

**What one hop costs, stated rather than hidden.** The interaction property is gone. The old
construction could show a method retaining `s2` while dropping `s1` — the "set-valued vs
pointwise scoring" failure mode the Stage 8 decomposition was built to see (v1 §4.2). A
single-span task cannot express it. What survives is the selection question itself: which of N
format-identical lines a policy keeps under budget.

**What `oracle_causal` knows here.** The candidate set is the H record lines that this
instance's queries will ask about. The oracle knows those H, not which one is asked. Note this
is a weaker and stranger notion than before: with no binding sentence, nothing *in the context*
marks which records are queryable, so the oracle's knowledge is about the evaluation's query
distribution rather than about readable content. That is a real shift and is flagged in
PREREG_P2.md rather than absorbed.

Decision 7 is unchanged in rule: retain complete candidate spans only, packed to maximise how
many fit in C, ascending token cost, remainder from `floor_pos`, verified per instance against
an exhaustive optimum. With one span per fact the knapsack is simply simpler.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

# N is an explicit Stage 4 tuning knob (v1 §4.2): 96 -> 80 -> 40. chance = 1/40 = 0.025.
N_RECORDS = 40
# H queries per instance, each naming a different record id. Decision 7 scores their mean.
N_BINDINGS = 4

DEPTS = (
    "Procurement", "Actuarial", "Custody", "Clearing", "Treasury", "Compliance",
    "Settlement", "Registry", "Underwriting", "Reconciliation", "Valuation", "Payments",
)

EXEMPLARS = (
    "Worked examples (same format as below).\n"
    "R900 | Aldermill | Treasury | 471203\n"
    "Q: value on record R900? A: 471203\n"
    "R901 | Brackenhold | Registry | 882640\n"
    "Q: value on record R901? A: 882640\n"
    "Find the line with the given record id and report its value.\n\n"
)

PREAMBLE = (
    "INTERNAL LEDGER EXTRACT. The lines below mix account records with governance notes for "
    "the current audit period. Each record has an identifier, a responsible party, a "
    "department, and a posted value.\n"
)

# Exemplar ids R900/R901 and surnames Aldermill/Brackenhold sit outside every generated pool
# (ids are R001..R{N}; surnames come from the syllable pool below), so they cannot collide with
# or leak any instance. Asserted in tests/test_harness.py.

_SYL_A = ("Bar", "Cald", "Dorn", "Ell", "Fen", "Gar", "Hal", "Ing", "Jar", "Kel",
          "Lund", "Mor", "Nass", "Orr", "Pell", "Quist", "Rand", "Sel", "Tor", "Vane")
_SYL_B = ("ac", "en", "ir", "ol", "un", "am", "eth", "id", "or", "us")
_SYL_C = ("ker", "son", "dahl", "mont", "well", "ridge", "stad", "hardt", "field", "wick")


def surname_pool() -> list[str]:
    """400 deterministic, lexically homogeneous surnames."""
    pool: list[str] = []
    for a in _SYL_A:
        for b in _SYL_B:
            for c in _SYL_C:
                pool.append(a + b + c)
                if len(pool) >= 400:
                    return pool
    return pool


SURNAMES = surname_pool()

_MENTION_TEMPLATES = (
    "Correspondence concerning record {rec_id} was logged and filed without further action.",
    "A handover note covering record {rec_id} was circulated to the desk.",
    "Record {rec_id} was reviewed at the quarterly walkthrough.",
    "The standing delegation for record {rec_id} remains unchanged this period.",
    "Filing details for record {rec_id} were refreshed in the directory.",
    "Record {rec_id} appeared on the routine attendance sheet for the session.",
)


@dataclass(frozen=True)
class Span:
    """A contiguous character span of the context, with its role in the task."""
    kind: str        # "record"
    text: str
    start: int
    end: int
    rec_id: str | None = None


@dataclass(frozen=True)
class Variant:
    """One of the H queries answerable from this context.

    Decision 7 scores every instance as the MEAN over all H variants, because the causal
    oracle's information is "content but not query" and the query is uniform over H.
    """
    rec_id: str
    query: str
    answer: str
    gold: tuple[Span, ...]     # exactly one span in the one-hop task


@dataclass
class LedgerInstance:
    instance_id: str
    context: str
    variants: tuple[Variant, ...]        # all H -- scored as their mean
    candidates: tuple[Span, ...]         # the H queried record lines -- causal oracle
    meta: dict = field(default_factory=dict)

    # Convenience views onto variant 0. The full evaluation uses `variants`.
    @property
    def query(self) -> str: return self.variants[0].query
    @property
    def answer(self) -> str: return self.variants[0].answer
    @property
    def gold(self) -> tuple[Span, ...]: return self.variants[0].gold


def _fmt_record(idx: int, surname: str, dept: str, value: str) -> str:
    return f"R{idx:03d} | {surname} | {dept} | {value}"


def build(
    seed: int,
    instance_id: str,
    *,
    n_records: int = N_RECORDS,
    n_bindings: int = N_BINDINGS,
    target_tokens: int | None = None,
    tokenizer=None,
    tolerance: int = 16,
) -> LedgerInstance:
    """Construct one instance.

    If `target_tokens` and `tokenizer` are given, filler is grown/trimmed until the context is
    `target_tokens +/- tolerance` in **that tokenizer** (v1 §3.2: every length is measured in
    the model's own tokenizer, never in characters or a reference tokenizer).
    """
    rng = random.Random(seed)

    names = rng.sample(SURNAMES, n_records)
    values = [f"{rng.randrange(100000, 1000000)}" for _ in range(n_records)]
    depts = [rng.choice(DEPTS) for _ in range(n_records)]
    rec_ids = [f"R{i + 1:03d}" for i in range(n_records)]
    records = [_fmt_record(i + 1, names[i], depts[i], values[i]) for i in range(n_records)]

    # The H records this instance's queries will ask about.
    queried_idx = rng.sample(range(n_records), n_bindings)

    def assemble(n_filler: int) -> str:
        # Filler mentions record ids drawn uniformly from ALL records, queried and not alike.
        # Drawing only from non-queried ids would invert the old leak: a queried record would
        # be the one that appears exactly once, which is just as informative as appearing
        # twice. Uniform draw makes mention count carry no information in either direction.
        pool = list(rec_ids)
        rng.shuffle(pool)
        filler = [
            _MENTION_TEMPLATES[rng.randrange(len(_MENTION_TEMPLATES))].format(
                rec_id=pool[j % len(pool)]
            )
            for j in range(n_filler)
        ]

        # v2 LAYOUT: records are INTERLEAVED uniformly through the filler rather than sitting
        # as one contiguous block. Content is identical; only ordering changes.
        #
        # Why: with a contiguous mid-context block, gold occupied token positions ~206-955
        # while `floor_pos` retains [0, n_sink) + the most recent B-n_sink -- i.e. [1487, 2063)
        # even at C=512. Gold could never be in the floor at ANY admitted budget, so the
        # registered denominator was structurally absorbing at exactly 0.0000 and G_m collapsed
        # to A_m / A_causal. Uniform interleaving makes gold positionally uniform, so a
        # recency-shaped floor retains gold at roughly its budget fraction -- graded and
        # non-absorbing, which is what decision 2 assumed.
        total = len(records) + len(filler)
        slots: list[str | None] = [None] * total
        taken: set[int] = set()
        for i in range(len(records)):
            pos = int(round((i + 0.5) * total / len(records)))
            pos = min(max(pos, 0), total - 1)
            while pos in taken:                 # resolve collisions deterministically
                pos = (pos + 1) % total
            taken.add(pos)
            slots[pos] = records[i]
        fi = iter(filler)
        body_lines = [s if s is not None else next(fi) for s in slots]
        return EXEMPLARS + PREAMBLE + "\n".join(body_lines) + "\n"

    n_filler = 24
    context = assemble(n_filler)
    if target_tokens is not None and tokenizer is not None:
        lo, hi = 0, 1024
        for _ in range(24):
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
            raise RuntimeError(f"span not found in context: {text!r}")
        return Span(kind="record", text=text, start=start, end=start + len(text), rec_id=rid)

    cand = [_span(records[i], rec_ids[i]) for i in queried_idx]

    variants = []
    for i in queried_idx:
        variants.append(Variant(
            rec_id=rec_ids[i],
            query=(f"What is the value on record {rec_ids[i]}? "
                   "Answer with the 6-digit value only."),
            answer=values[i],
            gold=(_span(records[i], rec_ids[i]),),
        ))

    return LedgerInstance(
        instance_id=instance_id,
        context=context,
        variants=tuple(variants),
        candidates=tuple(cand),
        meta={
            "n_records": n_records,
            "n_bindings": n_bindings,
            "n_filler": n_filler,
            "H": n_bindings,
            "hops": 1,
            "chance": 1.0 / n_records,
        },
    )


def score(generated_text: str, inst: LedgerInstance, variant: int = 0) -> bool:
    """Gold-substring match: True if the gold value appears anywhere in the generation.

    Paper 1's scorer convention (`credential.value in generated_text`), unchanged. A correct
    value inside a preamble counts -- "The value is 703226." scores.

    Credit is given ONLY for this variant's gold. Accepting any of the H gold values would
    credit a generation for answering a different question, which is why that reading was not
    adopted.
    """
    return inst.variants[variant].answer in generated_text


def score_instance(generated_texts: Sequence[str], inst: LedgerInstance) -> float:
    """Mean over all H query variants (decision 7). One generation per variant.

    This is a MEAN with partial credit (0, 1/H, ... 1), not a conjunction.
    """
    if len(generated_texts) != len(inst.variants):
        raise ValueError(
            f"expected {len(inst.variants)} generations (one per variant), "
            f"got {len(generated_texts)}"
        )
    return sum(v.answer in g for g, v in zip(generated_texts, inst.variants)) / len(inst.variants)


def token_len_of_spans(spans: Sequence[Span], context: str, tokenizer) -> int:
    """Context tokens overlapping these spans, in the model's own tokenizer."""
    enc = tokenizer(context, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    keep = set()
    for sp in spans:
        for ti, (a, b) in enumerate(offsets):
            if b > a and a < sp.end and b > sp.start:
                keep.add(ti)
    return len(keep)


__all__ = ["build", "score", "score_instance", "Span", "Variant", "LedgerInstance",
           "token_len_of_spans", "SURNAMES", "EXEMPLARS"]
