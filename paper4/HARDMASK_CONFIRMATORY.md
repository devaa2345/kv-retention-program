# Hard-mask confirmatory test — running record

## Step 1 — the 15 E3 fabrication cases: source and hit rate (recorded)

11 snapkv, 4 expected_attn (non-adakv subset of the original 25-case fabrication pool, first 15).
**Max-attention = queried record: 4/15 (27%)** — dragged down by SnapKV's own low rate (~25% at
scale, per the earlier 102-case check); expected_attn's 3/4 here is consistent with its ~54%
population rate. **The one conversion happened on one of the 4 matched cases (1/4 = 25% within
the matched subpopulation, 0/11 on mismatched ones).** The earlier 1/15 conversion figure
reflects a population where the top-attended record was usually wrong to begin with, not a
verdict on the mechanism itself.

## Step 2 (part 1) — the 14 unconverted fabrication outputs, classified

| Category | n |
|---|---|
| Verbatim copy of a different, wrong record | 8 |
| Degenerate/empty despite the CORRECT record being identified | 3 |
| Truncated and wrong record | 2 |
| Partial/abbreviated fields, wrong record | 1 |

Of the 3 cases where attention *did* find the right record but still failed, all three
degenerated into empty/truncated output rather than producing coherent wrong content — a
distinct residual weakness from simple mistargeting, addressed further below.
