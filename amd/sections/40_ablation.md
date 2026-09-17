## Structural protection degenerates into an oracle when distractors stop matching

Structural protection matches lines by surface pattern. Rewriting distractor values to a
non-credential form (`ref_<hex>` instead of `sk-<hex>`), n=50, budget 257:

| arm | shared shape | distinct shape | diff | 95% CI | p |
|---|---|---|---|---|---|
| arm2 | 0.1500 | 0.9067 | **+0.7567** | [+0.7133, +0.8000] | 0.0001 |
| arm4 | 0.1167 | 0.3400 | **+0.2233** | [+0.1567, +0.2900] | 0.0001 |
| arm6 | 0.7433 | 0.8800 | **+0.1367** | [+0.0533, +0.2300] | 0.0053 |

`arm2` (structural/permanent) jumps from 0.150 to **0.907 — above the full-cache ceiling
of 0.880 measured under the same condition**. With only 6 lines matching its pattern
instead of 26, structural protection stops competing and simply retains every credential:
it *becomes* `oracle_static`. The full-cache arm gains only +0.137, so most of arm 2's
+0.757 is degeneration of the mechanism, not the task becoming easier.

**Implication for pattern-based KV protection generally:** its strength is set by the ratio
of target lines to pattern-matching non-target lines. This is a property of the mechanism,
not a quirk of this task — and it means the spec's phrase "same surface shape" ([GAP-V])
is load-bearing in a way the document never states. The *direction* of the protection
effect should transfer to other designs; its *magnitude* should not be quoted out of the
6:26 ratio it was measured at.

