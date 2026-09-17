"""kvpress smoke test — every Tier-A press in the roster must import and construct.

Roster from PREREG_P2.md §4.2 (decision 5). LU-KV is included because kvpress v0.5.4 ships
`lukv_press`, which moved it from "our port" to "upstream press + our profiling" (§4.3).
"""

from __future__ import annotations

import json
import sys

import kvpress

ROSTER = {
    "snapkv": "SnapKVPress",
    "tova": "TOVAPress",
    "expected_attn": "ExpectedAttentionPress",
    "keydiff": "KeyDiffPress",
    "adakv_snapkv": "AdaKVPress",
    "lukv": "LUKVPress",
    # ladder-adjacent presses used as tripwires / floor reference
    "random": "RandomPress",
    "knorm": "KnormPress",
    "streaming_llm": "StreamingLLMPress",
}

out: dict = {"kvpress_version": getattr(kvpress, "__version__", "unknown"), "presses": {}}
ok = True
for arm, cls_name in ROSTER.items():
    cls = getattr(kvpress, cls_name, None)
    if cls is None:
        out["presses"][arm] = {"class": cls_name, "found": False}
        ok = False
        continue
    entry = {"class": cls_name, "found": True}
    try:
        if cls_name == "AdaKVPress":
            inst = cls(kvpress.SnapKVPress(compression_ratio=0.5))
        elif cls_name == "LUKVPress":
            # constructed but NOT exercised: no budget curve exists for any of our models
            # (BUDGET_CURVE_URLS ships only Llama-3.1-8B x ExpectedAttention), and it fetches
            # over the network at runtime -- see PREREG_P2.md §4.3.
            inst = cls
            entry["note"] = "class present; requires a profiled budget curve we must generate"
            entry["constructed"] = "deferred"
            out["presses"][arm] = entry
            continue
        else:
            inst = cls(compression_ratio=0.5)
        entry["constructed"] = True
    except Exception as e:
        entry["constructed"] = False
        entry["error"] = f"{type(e).__name__}: {e}"[:160]
        ok = False
    out["presses"][arm] = entry

# does the shipped LU-KV curve table cover any of our models?
try:
    from kvpress.presses.lukv_press import BUDGET_CURVE_URLS
    out["lukv_budget_curves"] = [list(k) for k in BUDGET_CURVE_URLS]
except Exception as e:
    out["lukv_budget_curves"] = f"unreadable: {type(e).__name__}"

out["all_ok"] = ok
print(json.dumps(out, indent=2))
sys.exit(0 if ok else 1)
