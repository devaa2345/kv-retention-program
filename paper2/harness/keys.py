"""The dedup key — single source of truth (repo rule 3).

Never reconstruct a record key inline. Build a `RecordKey`, and derive everything
(the dedup digest, the RNG seed) from it.

Rule 5: seeds come from CRC32 of the canonical key. Python's builtin `hash()` is salted
per process (PYTHONHASHSEED) and is irreproducible across runs and machines.
"""

from __future__ import annotations

import hashlib
import json
import zlib
from dataclasses import asdict, dataclass, fields
from typing import Any

# Every field below participates in the dedup key. Order is fixed and load-bearing:
# the canonical form is order-independent (sorted), but this list is what `validate()`
# checks completeness against.
KEY_FIELDS = (
    # --- what was run ---
    "task",           # task identifier, e.g. "secret_2048" | "ledger"
    "instance_id",    # per-task instance identifier (the prompt), stable across arms
    "model",          # HF repo id, e.g. "Qwen/Qwen2.5-1.5B-Instruct"
    "model_revision", # resolved commit sha of the model snapshot, never "main"
    "arm",            # arm/method identifier, e.g. "floor_pos" | "oracle_causal" | "snapkv"
    "B",              # retained-token budget, in tokens
    "protocol",       # evaluation protocol identifier (e.g. iso-token accounting variant)
    # --- where and with what it was run ---
    "device",         # "nvidia" | "amd"  -- device ownership, asserted against runs/ dir
    "backend",        # e.g. "cuda-12.8" | "rocm-7.2"
    "torch_version",
    "transformers_version",
    "kvpress_version",  # version string + git sha, or "absent" if the arm does not use kvpress
    "dtype",          # e.g. "bfloat16"
    "seed",           # the instance seed actually used (see seed_for)
    "batch_size",     # always present; 1 when unbatched. Changing it changes the key by design.
)

_DEVICES = ("nvidia", "amd")


@dataclass(frozen=True, slots=True)
class RecordKey:
    task: str
    instance_id: str
    model: str
    model_revision: str
    arm: str
    B: int
    protocol: str
    device: str
    backend: str
    torch_version: str
    transformers_version: str
    kvpress_version: str
    dtype: str
    seed: int
    batch_size: int = 1

    # -- construction -------------------------------------------------------

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Fail loudly on an incomplete or malformed key (rule 3: assert it in code)."""
        declared = {f.name for f in fields(self)}
        missing = set(KEY_FIELDS) - declared
        if missing:
            raise ValueError(f"RecordKey is missing dedup fields: {sorted(missing)}")
        extra = declared - set(KEY_FIELDS)
        if extra:
            raise ValueError(
                f"RecordKey has fields outside KEY_FIELDS: {sorted(extra)}. "
                "Every field on the key must participate in the dedup digest."
            )
        for name in KEY_FIELDS:
            v = getattr(self, name)
            if v is None or (isinstance(v, str) and not v.strip()):
                raise ValueError(f"RecordKey.{name} is empty; every dedup field must be set")
        if self.device not in _DEVICES:
            raise ValueError(f"device must be one of {_DEVICES}, got {self.device!r}")
        if self.model_revision in ("main", "master", "HEAD"):
            raise ValueError(
                "model_revision must be a resolved commit sha, not a moving ref "
                f"(got {self.model_revision!r}). A moving ref silently changes the model mid-run."
            )
        if not isinstance(self.B, int) or self.B <= 0:
            raise ValueError(f"B must be a positive int token budget, got {self.B!r}")
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError(f"batch_size must be a positive int, got {self.batch_size!r}")
        if not isinstance(self.seed, int):
            raise ValueError(f"seed must be an int, got {self.seed!r}")

    # -- canonical forms ----------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical(self) -> str:
        """Deterministic, stable across processes, machines and Python versions."""
        return json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )

    def digest(self) -> str:
        """The dedup digest. Two records with the same digest are the same cell."""
        return hashlib.sha256(self.canonical().encode("ascii")).hexdigest()

    def short(self) -> str:
        return self.digest()[:16]

    # -- provenance ---------------------------------------------------------

    def runs_dir(self) -> str:
        """The only directory this record may be written to (rule 1)."""
        return f"runs/{self.device}"


def seed_for(key: RecordKey | str, stream: str = "") -> int:
    """CRC32-derived seed (rule 5). Never `hash()`.

    `stream` names an independent RNG stream for the same cell (e.g. "task", "policy"),
    so that two consumers inside one record do not share a sequence.
    """
    canon = key.canonical() if isinstance(key, RecordKey) else str(key)
    payload = f"{canon}|{stream}".encode("ascii", errors="strict")
    return zlib.crc32(payload) & 0xFFFFFFFF


def seed_key_without_seed(**kw: Any) -> int:
    """Derive the instance seed for a cell *before* the seed field is known.

    The `seed` field is part of the dedup key, which makes it circular to seed a record
    from its own key. Resolve it by CRC32 over the key with `seed` held at a fixed
    sentinel, then construct the real RecordKey with the result.
    """
    probe = dict(kw)
    probe["seed"] = 0
    probe.setdefault("batch_size", 1)
    canon = json.dumps(probe, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return zlib.crc32(canon.encode("ascii")) & 0xFFFFFFFF


def assert_owned_by(key: RecordKey, device: str) -> None:
    """Rule 1 + rule 2 guard. Call before every write and every refill."""
    if key.device != device:
        raise PermissionError(
            f"Record belongs to device {key.device!r} but this session owns {device!r}. "
            "Never refill a hole on the other machine (repo rule 2) — it reintroduces the "
            "device confound the design exists to avoid."
        )


__all__ = [
    "KEY_FIELDS",
    "RecordKey",
    "seed_for",
    "seed_key_without_seed",
    "assert_owned_by",
]
