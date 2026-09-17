"""Task generation and scoring: multi-credential retrieval with induced dormancy.

Spec ref: SPEC_REIMPL_v1.md section 2.
Resolutions: [SPEC-GAP 1] turn order, [SPEC-GAP 2] exact substring, [GAP-P] turn-level credit.

Determinism rules (spec section 1): no iteration or selection over an unordered set anywhere in
this module. Every collection that influences output is a list with an explicit order.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

N_CREDENTIALS = 6
N_DISTRACTORS = 20
WORDS_PER_PARAGRAPH = 50
PAD_WORDS = 26          # calibrated to the spec's ~1029-token target; see [GAP-T]
VALUE_HEX_CHARS = 14          # value is "sk-" + 14 hex = 17 chars
MAX_ANSWER_TOKENS = 22

HEX = "0123456789abcdef"

# Distractor labels: same surface shape as credentials, non-credential semantics.
DISTRACTOR_LABELS = [
    "SESSION", "BUILD", "TRACE", "SHARD", "ROUTE", "BATCH", "NODE", "QUEUE",
    "REGION", "BUCKET", "STREAM", "WORKER", "DIGEST", "MIRROR", "TENANT",
    "CHANNEL", "SEGMENT", "REPLICA", "MANIFEST", "CHECKPOINT",
]

# [GAP-S] Filler is coherent prose, not sampled word-soup. Both satisfy "filler paragraphs of
# 50 words"; coherent prose is the more natural reading, and it measurably removes baseline
# wrong-line retrieval errors (5/48 -> 1/48) without changing accuracy, so that eviction-induced
# retrieval damage is measured against a clean background rather than a noisy one.
FILLER_SENTENCES = [
    "The service operates across several regions and each request is routed through a load balancer.",
    "Operators should note that throughput varies with payload size during peak traffic hours.",
    "Retries are capped at three attempts to avoid cascading failure under sustained load.",
    "Logs are rotated nightly and metrics are sampled at one second intervals.",
    "Configuration changes propagate to all edge nodes within roughly two minutes.",
    "Storage is replicated to three availability zones before a write is acknowledged.",
    "Background compaction runs hourly and may briefly increase read latency.",
    "Cache entries expire after ten minutes of inactivity unless they are explicitly refreshed.",
    "Clients must handle partial responses gracefully during staged rollout windows.",
    "Capacity is temporarily reduced while instances are drained for scheduled maintenance.",
    "Connection state is held for the duration of the session and then released.",
    "The scheduler prefers nodes with the lowest observed queue depth at dispatch time.",
    "Health checks run every thirty seconds and mark a node unavailable after two failures.",
    "Compressed payloads are decoded at the edge to reduce backend processing cost.",
    "Audit records are written synchronously and retained for ninety days.",
    "Rate limits are applied per tenant rather than per connection.",
    "Warm standby replicas are promoted automatically when the primary stops responding.",
    "Index rebuilds are deferred until off peak windows to protect query latency.",
    "Telemetry is aggregated at one minute resolution before being forwarded upstream.",
    "Deployment proceeds one zone at a time with an automatic rollback on error budget burn.",
]



@dataclass
class Credential:
    index: int          # 1-based label index
    label: str          # e.g. "CRED_1_KEY"
    value: str          # e.g. "sk-a1b2c3d4e5f6a7"
    line: str           # full line as it appears in context
    depth_rank: int     # 0 = earliest credential in the context


@dataclass
class Prompt:
    seed: int
    context: str
    credentials: list[Credential]
    turn_order: list[int] = field(default_factory=list)  # indices into credentials, per turn

    @property
    def n_turns(self) -> int:
        return len(self.turn_order)


def _make_value(rng: random.Random) -> str:
    return "sk-" + "".join(rng.choice(HEX) for _ in range(VALUE_HEX_CHARS))


def _make_paragraph(rng: random.Random) -> str:
    """Coherent prose of about WORDS_PER_PARAGRAPH words, assembled from whole sentences."""
    words: list[str] = []
    while len(words) < WORDS_PER_PARAGRAPH:
        words.extend(rng.choice(FILLER_SENTENCES).split())
    # exactly WORDS_PER_PARAGRAPH words, as the spec fixes that constant; this also removes the
    # length variance that whole-sentence assembly would introduce.
    words = words[:WORDS_PER_PARAGRAPH]
    return " ".join(words).rstrip(".") + "."


def build_prompt(seed: int, n_filler_paragraphs: int = 7,
                 pad_words: int = PAD_WORDS,
                 n_credentials: int = N_CREDENTIALS,
                 n_distractors: int = N_DISTRACTORS) -> Prompt:
    """Construct one context. Deterministic in `seed` alone.

    Layout: 26 key-like lines (6 credential, 20 distractor) laid out in a fixed sequence with
    credentials at varied, non-clustered depths, and filler paragraphs interleaved so that no
    two credential lines are adjacent.
    """
    rng = random.Random(seed)

    # --- unique values, no collisions between credentials and distractors ---
    used_values: list[str] = []

    def fresh_value() -> str:
        while True:
            v = _make_value(rng)
            if v not in used_values:
                used_values.append(v)
                return v

    # --- credential line depths: spread across the 26 slots, jittered, never adjacent ---
    n_lines = n_credentials + n_distractors
    # Even spacing gives slots 2, 6, 10, 14, 18, 22 for 6 credentials in 26 slots.
    stride = n_lines // n_credentials
    base_slots = [2 + i * stride for i in range(n_credentials)]
    cred_slots: list[int] = []
    for b in base_slots:
        jitter = rng.randint(-1, 1)
        s = min(max(b + jitter, 0), n_lines - 1)
        while s in cred_slots:            # list membership, deterministic
            s = (s + 1) % n_lines
        cred_slots.append(s)
    cred_slots.sort()                     # explicit sort; depth_rank follows context order

    # --- assign credential labels to slots ---
    # Label index is decoupled from depth so that label order does not leak depth order.
    label_indices = list(range(1, n_credentials + 1))
    rng.shuffle(label_indices)

    credentials: list[Credential] = []
    for depth_rank, label_idx in enumerate(label_indices):
        label = f"CRED_{label_idx}_KEY"
        value = fresh_value()
        credentials.append(
            Credential(
                index=label_idx,
                label=label,
                value=value,
                line=f"{label}: {value}",
                depth_rank=depth_rank,
            )
        )

    # --- distractor lines ---
    distractor_lines: list[str] = []
    for i in range(n_distractors):
        lab = f"{DISTRACTOR_LABELS[i % len(DISTRACTOR_LABELS)]}_{i + 1}_ID"
        distractor_lines.append(f"{lab}: {fresh_value()}")

    # --- weave the 26 lines ---
    lines: list[str] = []
    di = 0
    for slot in range(n_lines):
        if slot in cred_slots:
            lines.append(credentials[cred_slots.index(slot)].line)
        else:
            lines.append(distractor_lines[di])
            di += 1

    # --- interleave filler paragraphs so credential lines are separated ---
    # Place a paragraph after roughly every (n_lines / (n_filler+1)) lines.
    body_parts: list[str] = ["Configuration dump follows.", ""]
    if n_filler_paragraphs > 0:
        every = max(1, n_lines // (n_filler_paragraphs + 1))
    else:
        every = n_lines + 1
    placed = 0
    for i, ln in enumerate(lines):
        body_parts.append(ln)
        if placed < n_filler_paragraphs and (i + 1) % every == 0:
            body_parts.append("")
            body_parts.append(_make_paragraph(rng))
            body_parts.append("")
            placed += 1
    while placed < n_filler_paragraphs:       # any remainder goes at the end
        body_parts.append("")
        body_parts.append(_make_paragraph(rng))
        placed += 1

    # Length calibration knob. A filler paragraph is a ~62-token quantum, too coarse to land on
    # the spec's stated ~1029-token target, so a final short filler passage of `pad_words` words
    # trims the remainder. Calibrated once against the tokenizer; see SPEC_QUESTIONS [GAP-T].
    if pad_words > 0:
        words: list[str] = []
        while len(words) < pad_words:
            words.extend(rng.choice(FILLER_SENTENCES).split())
        body_parts.append("")
        body_parts.append(" ".join(words[:pad_words]).rstrip(".") + ".")

    context = "\n".join(body_parts)

    # --- [SPEC-GAP 1] turn order: reverse depth. Earliest-placed credential asked last. ---
    turn_order = list(range(n_credentials - 1, -1, -1))   # depth_rank 5,4,3,2,1,0

    return Prompt(seed=seed, context=context, credentials=credentials, turn_order=turn_order)


def question_for(prompt: Prompt, turn: int) -> str:
    cred = prompt.credentials[prompt.turn_order[turn]]
    return f"What is the value of {cred.label}? Reply with the value only."


def target_for(prompt: Prompt, turn: int) -> Credential:
    return prompt.credentials[prompt.turn_order[turn]]


def score_turn(prompt: Prompt, turn: int, generated_text: str) -> bool:
    """[SPEC-GAP 2] exact substring on raw generated text. [GAP-P] credit only on the asking turn."""
    return target_for(prompt, turn).value in generated_text


def score_prompt(prompt: Prompt, generated_texts: list[str]) -> float:
    """Fraction retrieved k/N over the prompt's turns. N = this prompt's credential count."""
    k = sum(1 for t, g in enumerate(generated_texts) if score_turn(prompt, t, g))
    return k / len(prompt.credentials)


def dormancy_gap(prompt: Prompt, turn: int) -> int:
    """Turns elapsed between the credential's context depth rank and the turn that queries it."""
    return turn - target_for(prompt, turn).depth_rank
