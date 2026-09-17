"""Stage 2 record keys.

Paper 2's `harness/keys.py` has a fixed 15-field key that cannot express Stage 2's new axes
(`task`, `n_fields`, `layout`, and the matched-arm source). Rather than quietly widening a
frozen key, Stage 2 defines its own, keeps every Paper 2 field, and adds the new ones. The
instance seed is still derived by CRC32 over the key with `seed` held at a sentinel, per
Paper 2's repo rule 5, so seeding stays reproducible and non-circular.
"""
from __future__ import annotations

import hashlib
import json
import zlib

FIELDS = ("task", "instance_id", "model", "model_revision", "arm", "B", "C", "protocol",
          "device", "backend", "torch_version", "transformers_version", "kvpress_version",
          "dtype", "seed", "batch_size", "n_fields", "n_records", "layout", "matched_to",
          "max_new")


def canon(d: dict) -> str:
    return json.dumps({k: d.get(k) for k in FIELDS}, sort_keys=True,
                      separators=(",", ":"), ensure_ascii=True)


def digest(d: dict) -> str:
    return hashlib.sha256(canon(d).encode("ascii")).hexdigest()


def instance_seed(**kw) -> int:
    """CRC32 over the key with `seed` at a sentinel -- the instance seed cannot depend on
    itself. Deliberately independent of `arm`, `B`, `C` and `max_new` so that every arm at
    every budget sees the SAME instance; an instance that changed with the arm would make
    every contrast a different-data contrast."""
    probe = dict(kw)
    probe["seed"] = 0
    for k in ("arm", "B", "C", "max_new", "matched_to"):
        probe[k] = None
    return zlib.crc32(canon(probe).encode("ascii")) & 0xFFFFFFFF


__all__ = ["FIELDS", "canon", "digest", "instance_seed"]
