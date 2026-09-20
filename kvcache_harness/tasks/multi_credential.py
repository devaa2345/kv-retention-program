"""Phase 0-R instrument: multi-credential retrieval with induced dormancy.

Replaces the single-credential binary task (credential_retrieval.py, kept
for reference/regression but no longer the primary instrument). N
credentials are embedded in one config dump; the model is then asked about
each of them, one per conversational turn, in a shuffled order. A
credential's *dormancy span* = how many other turns intervene between the
dump (where it was last genuinely attended) and its own question — that
span is the only regime where a recoverable-tiering policy's promotion
mechanism has anything to act on. Query order (and therefore span) is
randomized per prompt from `seed`, and is recorded per credential so it can
be used as a factor / covariate in analysis, not just a design fixed effect.

Score = fraction of the N credentials correctly answered (k/N) — replaces
the old binary exact-match, and is a graded metric that doesn't compress at
the top the way edit-distance does on long strings.
"""

from __future__ import annotations

import random
import re
import string
from dataclasses import dataclass, field

from .credential_retrieval import FILLER_WORDS, DISTRACTOR_KEYS, _rand_value, LABEL_PATTERN


AMD_DISTRACTOR_LABELS = [
    "SESSION", "BUILD", "TRACE", "SHARD", "ROUTE", "BATCH", "NODE", "QUEUE",
    "REGION", "BUCKET", "STREAM", "WORKER", "DIGEST", "MIRROR", "TENANT",
    "CHANNEL", "SEGMENT", "REPLICA", "MANIFEST", "CHECKPOINT",
]   # copied from amd/kvre/task.py DISTRACTOR_LABELS


@dataclass
class Credential:
    key: str
    value: str
    label_char_span: tuple    # (start,end) of "key:" in dump_text
    value_char_span: tuple    # (start,end) of the value substring in dump_text


@dataclass
class MultiCredentialPrompt:
    dump_text: str
    credentials: list           # list[Credential], in DUMP order (index = credential id)
    turn_order: list            # list[int], credential ids in QUESTION order (the dormancy manipulation)
    seed: int
    value_len: int


def make_multi_credential_prompt(
    seed: int,
    n_credentials: int = 8,
    n_distractors: int = 20,
    words_per_paragraph: int = 40,
    value_len: int = 14,
    distractor_format: str = "ours",
) -> MultiCredentialPrompt:
    """distractor_format:
      "ours" (default, used by every result before 2026-09-20): distractors are
          `cache_ttl: 37.18` / `log_level: ugiwtb`, short values in a different format.
      "amd": distractors share the credentials' full surface shape,
          `SESSION_1_ID: sk-<value_len hex>`, as in the independent reimplementation
          (amd/kvre/task.py, [GAP-V]). Label set and label pattern are theirs; the hex alphabet
          is OURS (string.hexdigits.lower(), the same as our credentials) so that ONLY the
          distractor format differs between the two cells of the H-ORTH factorial.
    """
    rng = random.Random(seed)

    cred_keys = [f"CRED_{i}_KEY" for i in range(n_credentials)]
    cred_values = ["sk-" + "".join(rng.choice(string.hexdigits.lower()) for _ in range(value_len))
                    for _ in range(n_credentials)]

    if distractor_format == "ours":
        distractor_key_pool = rng.sample(DISTRACTOR_KEYS, min(n_distractors, len(DISTRACTOR_KEYS)))
        distractor_lines = [(k, _rand_value(rng, rng.choice(["int", "float", "str"]))) for k in distractor_key_pool]
    elif distractor_format == "amd":
        distractor_lines = [
            (f"{AMD_DISTRACTOR_LABELS[i % len(AMD_DISTRACTOR_LABELS)]}_{i + 1}_ID",
             "sk-" + "".join(rng.choice(string.hexdigits.lower()) for _ in range(value_len)))
            for i in range(n_distractors)
        ]
    else:
        raise ValueError(distractor_format)

    all_lines = [(cred_keys[i], cred_values[i]) for i in range(n_credentials)] + distractor_lines
    rng.shuffle(all_lines)

    parts = ["System configuration dump follows.\n"]
    credentials = [None] * n_credentials
    for i, (k, v) in enumerate(all_lines):
        if words_per_paragraph and i % 2 == 0:
            para = " ".join(rng.choice(FILLER_WORDS) for _ in range(words_per_paragraph))
            parts.append(para + "\n")

        prefix = "".join(parts)
        line = f"{k}: {v}\n"
        label_start = len(prefix)
        label_end = label_start + len(f"{k}:")

        if k.startswith("CRED_"):
            cred_idx = int(k.split("_")[1])
            value_start = label_start + len(f"{k}: ")
            value_end = value_start + len(v)
            credentials[cred_idx] = Credential(
                key=k, value=v,
                label_char_span=(label_start, label_end),
                value_char_span=(value_start, value_end),
            )
        parts.append(line)

    dump_text = "".join(parts)
    assert all(c is not None for c in credentials)

    turn_order = list(range(n_credentials))
    rng.shuffle(turn_order)

    return MultiCredentialPrompt(
        dump_text=dump_text, credentials=credentials, turn_order=turn_order,
        seed=seed, value_len=value_len,
    )


def question_for(prompt: MultiCredentialPrompt, credential_idx: int) -> str:
    key = prompt.credentials[credential_idx].key
    return f"What is the value of {key}? Reply with ONLY the value, nothing else."


def dormancy_span(prompt: MultiCredentialPrompt, credential_idx: int) -> int:
    """Number of *other* turns asked before this credential's own turn —
    the dormancy manipulation's realized value for this credential in this
    prompt (0 = asked first, N-1 = asked last)."""
    return prompt.turn_order.index(credential_idx)


def score_turn(generated_text: str, credential: Credential) -> bool:
    return credential.value in generated_text


def find_structural_label_spans(dump_text: str) -> list:
    spans = []
    for m in LABEL_PATTERN.finditer(dump_text):
        line_end = dump_text.find("\n", m.end())
        if line_end == -1:
            line_end = len(dump_text)
        spans.append((m.start(), line_end))
    return spans
