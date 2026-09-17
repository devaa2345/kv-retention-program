"""Synthetic credential-retrieval / dormant-token task, reimplementing the
setup described in Transactional Attention (arXiv:2604.11288): a long
context with scattered `key: value`-style config/credential lines, one of
which is the target, buried under filler text, with a cache budget forced
far below the context length. The eviction policy under test never sees
which line is the target — only the task scorer and `oracle_static` do.
"""

from __future__ import annotations

import random
import re
import string
from dataclasses import dataclass


FILLER_WORDS = (
    "the system processes incoming requests through a series of validation "
    "steps before routing them to the appropriate handler which then "
    "consults internal state and external services to assemble a response "
    "that is eventually serialized and returned to the caller after logging "
    "relevant metadata for later analysis and monitoring purposes across "
    "the distributed deployment"
).split()

DISTRACTOR_KEYS = [
    "timeout_ms", "retry_count", "cache_ttl", "batch_size", "worker_pool",
    "log_level", "region", "endpoint_id", "shard_count", "queue_depth",
    "max_connections", "poll_interval", "buffer_size", "thread_count",
    "session_ttl", "rate_limit", "heartbeat_ms", "compression", "replica_count",
    "flush_interval",
]


@dataclass
class CredentialPrompt:
    text: str
    target_value: str
    target_char_span: tuple      # (start, end) of the *value* substring
    label_char_spans: list       # list of (start, end) for every "label:" span (all distractor + target lines)
    seed: int


def _rand_value(rng: random.Random, kind: str) -> str:
    if kind == "int":
        return str(rng.randint(1, 65535))
    if kind == "float":
        return f"{rng.uniform(0, 100):.2f}"
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(4, 8)))


def make_credential_prompt(
    seed: int,
    n_distractors: int = 20,
    filler_paragraphs_between: int = 6,
    words_per_paragraph: int = 40,
    target_key: str = "API_KEY",
) -> CredentialPrompt:
    rng = random.Random(seed)

    target_value = "sk-" + "".join(rng.choice(string.hexdigits.lower()) for _ in range(24))

    keys = rng.sample(DISTRACTOR_KEYS, min(n_distractors, len(DISTRACTOR_KEYS)))
    lines = [(k, _rand_value(rng, rng.choice(["int", "float", "str"]))) for k in keys]
    target_idx = rng.randint(max(1, len(lines) // 4), max(1, 3 * len(lines) // 4))
    lines.insert(target_idx, (target_key, target_value))

    parts = ["System configuration dump follows.\n"]
    label_char_spans = []
    target_char_span = None
    for i, (k, v) in enumerate(lines):
        if filler_paragraphs_between and i % 2 == 0:
            para = " ".join(rng.choice(FILLER_WORDS) for _ in range(words_per_paragraph))
            parts.append(para + "\n")

        prefix = "".join(parts)
        line = f"{k}: {v}\n"
        label_start = len(prefix)
        label_end = label_start + len(f"{k}:")
        label_char_spans.append((label_start, label_end))

        if k == target_key:
            value_start = label_start + len(f"{k}: ")
            value_end = value_start + len(v)
            target_char_span = (value_start, value_end)

        parts.append(line)

    parts.append(
        f"\nQuestion: What is the value of {target_key}? "
        f"Reply with ONLY the value and nothing else."
    )
    text = "".join(parts)

    assert target_char_span is not None
    return CredentialPrompt(
        text=text,
        target_value=target_value,
        target_char_span=target_char_span,
        label_char_spans=label_char_spans,
        seed=seed,
    )


def score_response(generated_text: str, target_value: str) -> bool:
    """Exact-match scorer: does the target credential value appear verbatim
    in the generated continuation?"""
    return target_value in generated_text


LABEL_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*:\s")


def find_structural_label_spans(text: str) -> list:
    """Cheap structural heuristic a real sponsorship policy would use:
    regex-match `word:` label patterns, generically, with no knowledge of
    which one is the true target, and sponsor the *whole line* (label
    through end-of-line) rather than a fixed character count — a fixed
    small window would (and in an early pilot run, did) truncate longer
    values like the 27-char credential used here, making "protection"
    fail even though it protected the label. Real sponsorship heuristics
    protect a structurally meaningful unit (the line/field), not an
    arbitrary character count."""
    spans = []
    for m in LABEL_PATTERN.finditer(text):
        line_end = text.find("\n", m.end())
        if line_end == -1:
            line_end = len(text)
        spans.append((m.start(), line_end))
    return spans


def build_chat_prompt(tokenizer, prompt: "CredentialPrompt"):
    """Wrap `prompt.text` (the config dump + question) as a user turn via
    the model's chat template, and return (input_ids, offset_mapping,
    body_offset) where body_offset is how many characters into the
    rendered chat text `prompt.text` itself begins — needed to shift the
    task's char spans (computed relative to `prompt.text` alone) onto the
    tokenization of the full chat-formatted string. Instruct models (Qwen2.5
    included) degrade badly on raw completion-style prompts, so this is not
    optional.
    """
    messages = [{"role": "user", "content": prompt.text}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    body_offset = chat_text.find(prompt.text)
    assert body_offset >= 0, "prompt.text not found verbatim in the rendered chat template"

    enc = tokenizer(chat_text, return_offsets_mapping=True, return_tensors="pt")
    return enc["input_ids"], enc["offset_mapping"][0].tolist(), body_offset


def char_spans_to_token_groups(offset_mapping, char_spans, protect_after_chars: int = 12) -> dict:
    """Like char_spans_to_token_positions, but returns {token_pos: group_id}
    (group_id = index into char_spans) instead of a flat set, so a
    protection policy can keep or drop a whole span atomically instead of
    letting individual tokens within it compete separately — token-level
    competition can leave *some* characters of a value surviving while
    others don't, corrupting it mid-string even though "protection" is
    nominally active."""
    groups = {}
    for gid, (start, end) in enumerate(char_spans):
        end = end + protect_after_chars
        for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start is None:
                continue
            if tok_end > start and tok_start < end:
                groups[tok_idx] = gid
    return groups


def char_spans_to_token_positions(offset_mapping, char_spans, protect_after_chars: int = 12) -> set:
    """Map a list of (start, end) char spans to the set of token indices
    (into `offset_mapping`, a list of (start,end) per token) that overlap
    [start, end + protect_after_chars) — extending slightly past the label
    so the adjacent value tokens are covered too, matching "sponsor nearby
    tokens" semantics."""
    positions = set()
    for (start, end) in char_spans:
        end = end + protect_after_chars
        for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start is None:
                continue
            if tok_end > start and tok_start < end:
                positions.add(tok_idx)
    return positions
