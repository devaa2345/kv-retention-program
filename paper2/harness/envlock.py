"""Environment fingerprint + lockfile writer.

Run as `python -m harness.envlock` from the repo root. Writes `env/<device>.lock`.

A version drift on either box mid-run invalidates every record produced after it
(repo CLAUDE.md, "Environment pinning"), and the dedup key is what lets you find them —
so the values captured here are exactly the ones that appear in `RecordKey`.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

TRACKED = (
    "torch",
    "transformers",
    "kvpress",
    "accelerate",
    "tokenizers",
    "safetensors",
    "numpy",
    "scipy",
    "datasets",
)


def _ver(pkg: str) -> str:
    try:
        return md.version(pkg)
    except Exception:
        return "absent"


def _git_sha(pkg: str) -> str | None:
    """kvpress is pinned by git sha as well as version (repo rule 3)."""
    try:
        mod = __import__(pkg)
        root = Path(mod.__file__).resolve().parent
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def kvpress_version() -> str:
    """The value that goes in RecordKey.kvpress_version."""
    v = _ver("kvpress")
    if v == "absent":
        return "absent"
    sha = _git_sha("kvpress")
    return f"{v}+{sha[:12]}" if sha else v


def fingerprint() -> dict[str, Any]:
    fp: dict[str, Any] = {
        "device": None,
        "backend": None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {p: _ver(p) for p in TRACKED},
        "kvpress_pinned": kvpress_version(),
    }
    try:
        import torch

        fp["packages"]["torch"] = torch.__version__
        if torch.cuda.is_available():
            is_rocm = getattr(torch.version, "hip", None) is not None
            if is_rocm:
                fp["device"] = "amd"
                fp["backend"] = f"rocm-{torch.version.hip}"
            else:
                fp["device"] = "nvidia"
                fp["backend"] = f"cuda-{torch.version.cuda}"
            fp["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
                "total_mem_mib": round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**2
                ),
                "count": torch.cuda.device_count(),
            }
        fp["cudnn"] = getattr(torch.backends.cudnn, "version", lambda: None)()
    except Exception as e:  # pragma: no cover
        fp["torch_error"] = repr(e)
    fp["env"] = {
        k: os.environ.get(k)
        for k in ("HF_HOME", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "PYTHONHASHSEED")
        if os.environ.get(k)
    }
    return fp


def main() -> int:
    fp = fingerprint()
    dev = fp.get("device")
    if dev not in ("nvidia", "amd"):
        print("ERROR: could not identify an accelerator; refusing to write a lockfile.")
        print(json.dumps(fp, indent=2))
        return 1
    out = Path(__file__).resolve().parent.parent / "env" / f"{dev}.lock"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fp, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(fp, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
