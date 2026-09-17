"""Pre-flight repair for Stage 4 JSONL files after an interrupted run.

A power loss mid-append can leave a half-written final line. The runner's resume logic already
skips unparseable lines when rebuilding its done-set, so that key is re-run -- but the broken
bytes stay in the file and break every downstream `json.loads`. This removes them.

Only lines that fail to parse are removed; a duplicated key digest is reported but kept, since
Stage 2 showed duplicates from concurrent writers are byte-identical and dedup belongs in
analysis, not in a repair step. Every removal is printed, and a `.bak` of the original is kept
whenever anything changes.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs" / "nvidia"


def main() -> int:
    changed = 0
    for f in sorted(RUNS.glob("stage4_*.jsonl")):
        if f.name.endswith(".failures.jsonl"):
            continue
        raw = f.read_bytes()
        good, dropped = [], []
        for i, line in enumerate(raw.split(b"\n")):
            if not line.strip():
                continue
            try:
                json.loads(line)
                good.append(line)
            except Exception:
                dropped.append((i, line[:80]))
        dups = sum(v - 1 for v in Counter(json.loads(l)["key_digest"] for l in good).values()
                   if v > 1)
        fixed = b"\n".join(good) + b"\n"
        if dropped or fixed != raw:
            shutil.copy2(f, f.with_suffix(f.suffix + ".bak"))
            f.write_bytes(fixed)
            changed += 1
        print("  %-28s ok=%6d dropped=%d dup_digests=%d%s"
              % (f.name, len(good), len(dropped), dups,
                 "  REPAIRED (backup kept)" if (dropped or fixed != raw) else ""))
        for i, head in dropped:
            print("      dropped line %d: %r" % (i, head))
    print("  repair: %d file(s) changed" % changed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
