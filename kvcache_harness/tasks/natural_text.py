"""Adapter: Paper 3's natural-text fact task (Gutenberg prose + templated 'The audit of the ...'
fact sentences) presented to THIS harness as a multi-turn retrieval task (Paper 1, Option B item 2).

Mapping onto the credential experiment:
  credential lines            -> the H=4 QUERIED fact sentences (level 1: answer = the person)
  distractor lines            -> the D other fact sentences (same template, indistinguishable)
  structural pattern          -> regex on the cue the task instruction itself states,
                                 'The audit of the ... depot.' (N=H+D sentences + the worked
                                 example), one atomic group per matched sentence
  oracle_important            -> gold token span of each queried fact at level 1
                                 (sentence start .. end of the person name)
  turn order / dormancy       -> H turns, one per queried fact, shuffled per instance
Scoring is Paper 3's scorer (whole-token, case-insensitive, no shotgun naming) via score_one.
"""
from __future__ import annotations

import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper3"))
from p3.natural import score as S        # noqa: E402

from .credential_retrieval import char_spans_to_token_positions, char_spans_to_token_groups

FACT_RE = re.compile(r"The audit of the [^\n]*?depot\.")


@dataclass
class NatPrompt:
    iid: str
    seed: int
    context: str
    queried: list          # list[dict] fact records with role == 'queried'
    turn_order: list       # indices into `queried`
    persons: list
    queries: dict          # fact_id -> level-1 query string
    elements: dict         # fact_id -> level-1 element list


def load(path, n=None, tag="M2"):
    out = []
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        rd = rec["renderings"][tag]
        queried = [f for f in rd["facts"] if f["role"] == "queried"]
        rng = random.Random(rec["seed"] + 424242)
        order = list(range(len(queried)))
        rng.shuffle(order)
        vs = {v["fact_id"]: v for v in rec["variants"] if v["level"] == 1}
        out.append(NatPrompt(rec["instance_id"], rec["seed"], rd["context"], queried, order, rec["persons"],
                             {k: v["query"] for k, v in vs.items()}, {k: v["elements"] for k, v in vs.items()}))
        if n and len(out) >= n:
            break
    return out


def build_turn_texts(tokenizer, p: NatPrompt):
    q0 = p.queries[p.queried[p.turn_order[0]]["fact_id"]]
    first = p.context + "\n" + q0
    turn0 = tokenizer.apply_chat_template([{"role": "user", "content": first}], tokenize=False,
                                          add_generation_prompt=True)
    body_offset = turn0.find(p.context)
    assert body_offset >= 0
    texts = [turn0]
    for k in range(1, len(p.turn_order)):
        q = p.queries[p.queried[p.turn_order[k]]["fact_id"]]
        texts.append(f"<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n")
    return texts, body_offset


def compute_positions(tokenizer, turn0_text, body_offset, p: NatPrompt):
    om = tokenizer(turn0_text, return_offsets_mapping=True, add_special_tokens=True)["offset_mapping"]
    oracle = set()
    for f in p.queried:
        a, b = f["gold_char"]["1"]
        oracle |= char_spans_to_token_positions(om, [(a + body_offset, b + body_offset)], protect_after_chars=0)
    spans = [(m.start() + body_offset, m.end() + body_offset) for m in FACT_RE.finditer(p.context)]
    groups = char_spans_to_token_groups(om, spans, protect_after_chars=0)
    return oracle, set(groups), groups, len(spans)


def score_turn(answer: str, p: NatPrompt, turn: int) -> bool:
    f = p.queried[p.turn_order[turn]]
    return S.score_one(answer, p.elements[f["fact_id"]], p.persons) == 1.0
