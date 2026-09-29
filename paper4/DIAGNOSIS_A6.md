# A6 — diagnosed: a deterministic KV-ORDER effect, not nondeterminism

Grid still paused. Nothing resumed. Evidence: `out/a6_determinism_M2.json`,
`out/a6_determinism_M3.json`, `runs/nvidia/p4_probe_{M2,M3}.jsonl`.

## 1. It is NOT nondeterminism

Same process, same instance, same arm, generated twice (n=10 instances, c=8, C=512):

| arm | M2 repeat-run identical | M3 repeat-run identical |
|---|---|---|
| floor_pos | 10/10 | 10/10 |
| snapkv | 10/10 | 10/10 |
| U-snapkv | 10/10 | 10/10 |
| ascend-snapkv | 10/10 | 10/10 |

Keep-sets identical 10/10 in every case. Generation on this harness is **exactly reproducible**,
so the A1–A5 CIs from this screen are not inflated by run-to-run noise, and those verdicts stand
as screened (subject to §3).

## 2. It IS a real, deterministic gather-order effect

`ascend-snapkv` differs from `snapkv` **only** in the order in which the identical retained tokens
are written into the cache (`topk` score order vs ascending position order):

| | M2 | M3 |
|---|---|---|
| keep-set metrics identical (p_g, q_complete, q_any, units_complete, units_touched), n=20 | **20/20** | **20/20** |
| keep-sets identical in the determinism run, n=10 | **10/10** | **10/10** |
| generations byte-identical, n=20 screen | 10/20 | 11/20 |
| generations byte-identical, n=10 determinism run | 3/10 | 5/10 |

**Named precisely:** with RoPE already applied before compression, attention is permutation-
invariant in exact arithmetic, so this cannot change the attended content. What changes is the
**order of bf16 accumulation in the sdpa kernel over the key axis**. Reordering the summands
perturbs logits at the last bits; where the top-2 logits are near-tied, greedy decoding flips, and
the generation diverges from that token on. This is the same class of numerical fragility this
programme has already recorded twice — Paper 2's batch-order disagreement (83/100 generated-token
agreement against batch-1) and Paper 3's session-boundary tie-flips (M3 pre/post power-loss).

Example (M2, instance 0, identical keep-set): score-order emits
`Record R006|The standing delegation for record R011 remains…`, ascending-order emits
`Barethhardt|Barethhardt` — both wrong, but divergent from the first token, which is the signature
of a tie flip rather than a content difference.

## 3. Why this matters beyond A6 — it is a confound in the paper's headline contrast

`X` arms gather in **score order** (`ScorerPress.compress`). `U-X` arms gather in **ascending
position order** (`p4/unitwrap.py`). So every `X` vs `U-X` comparison in Stage 2 and Stage 3 differs
in cache order as well as in allocation, and the order component is not fragmentation.

Measured size of the order component alone (identical keep-sets, n=20, c=8 C=512):
**M2 +0.025 [+0.000, +0.062]; M3 +0.000 [+0.000, +0.000]** (accuracy, ascending minus score order).

On M3 the effect is zero to three decimals despite 9/20 generations differing — divergent
generations that are equally wrong. On M2 it is positive and of the **same order of magnitude as
Stage 2's headline `U-X − X` = +0.045**. It cannot be assumed negligible.

The clean comparator for `U-X` is an `X` arm gathered in the **same** order (`ascend-X`), which
isolates allocation from ordering. That comparison has not yet been run at full n.

## 4. Status

Reported, not resolved. No wrapper change, no prereg change, no resumption, no further probes.
