# Track 2 — reading the failures (c=40, C=512, both models, post-A4 wrapper)

CPU-only ground-truth lookups. No new arms, no allocation variants, no fix proposed. Grid still
paused; nothing GPU-side beyond data already on disk (the post-A4 pilot cell, `68dc570`).

## Step 1 — disagreement set (floor correct, U-X wrong, queried fact complete in >= 1 slot)

Filter as specified: (a) `floor_pos` correct, (b) the U-X arm wrong, (c) the queried fact complete
in **at least one** U-X slot (`q_any`-level; "complete in every slot" is unsatisfiable given
`q_slot` averages ~0.08 — that was tried first and returned zero cases on M2, reported and
corrected before proceeding, not silently swapped).

Raw counts (query-level, out of 200 queries per cell), before any filtering:

| model | pair | count |
|---|---|---|
| M2 | snapkv | 27 |
| M2 | adakv_snapkv | 23 |
| M2 | expected_attn | 32 |
| M3 | snapkv | 18 |
| M3 | adakv_snapkv | 20 |
| M3 | expected_attn | 21 |

All six cells clear 8; none low-n. **Total 141**, all satisfying completeness in >=1 slot (every
floor-correct/U-wrong case had the fact retained whole somewhere — consistent with the measured
`q_any` of 0.19-0.72 for these arms). Exceeds the ~25-30 cap, so sampled down.

## Sampling

24 cases (12 per model, 4 per pair), all with **zero gold-word overlap** in the U-X generation --
prioritizing confidently-wrong over near-misses, as specified.

## Step 2/3 — the table

For each case: which real record (searched across the **entire** context, not a narrow window) the
wrong answer's content actually matches, and how many records away that is from the one queried.
"Record distance" is the id-order distance between the queried record and the best-matching one;
token-position adjacency in the retained sequence was checked directly and is consistent with this
(gap to the nearest other retained token was 1 in 19 of 24 cases -- i.e. the queried record's
tokens are flanked immediately by another record's tokens in the same slot's cache).

| model | pair | inst | q | gold | tag | reasoning |
|---|---|---|---|---|---|---|
| M2 | snapkv | 3 | 1 | R029 | hallucination | numbers not present anywhere in context (432109, 563214); weak match |
| M2 | snapkv | 6 | 1 | R031 | hallucination | numbers not in context (512345, 654321 -- round/sequential, not the corpus's random 6-digit style) |
| M2 | snapkv | 6 | 3 | R032 | **contamination** | answer is (almost) exactly record R030, 2 records away |
| M2 | snapkv | 12 | 2 | R031 | hallucination | numbers not in context (123456, 345678) |
| M2 | adakv_snapkv | 6 | 1 | R031 | **contamination** | answer is (almost) exactly record R030, 1 record away |
| M2 | adakv_snapkv | 6 | 3 | R032 | **contamination** | answer is (almost) exactly record R030, 2 records away |
| M2 | adakv_snapkv | 21 | 2 | R036 | **contamination** | answer is (almost) exactly record R037, 1 record away |
| M2 | adakv_snapkv | 26 | 1 | R033 | misattribution (distant) | matches record R038, 5 records away |
| M2 | expected_attn | 2 | 2 | R039 | **contamination** | answer is (almost) exactly record R040, 1 record away |
| M2 | expected_attn | 2 | 3 | R028 | no visible pattern | degenerate/blank field output |
| M2 | expected_attn | 6 | 1 | R031 | **contamination** | answer is (almost) exactly record R029, 2 records away |
| M2 | expected_attn | 7 | 2 | R029 | hallucination | numbers not in context (321456, 757090) |
| M3 | snapkv | 0 | 3 | R030 | **refusal** | asserts record absent, though complete in this slot |
| M3 | snapkv | 1 | 2 | R032 | **refusal** | "there is no record R032 in the provided text" |
| M3 | snapkv | 5 | 0 | R028 | **refusal** | "records provided start from R001 and go up to R040" (false) |
| M3 | snapkv | 9 | 2 | R033 | **refusal** | "there is no record R033" |
| M3 | adakv_snapkv | 1 | 2 | R032 | misattribution (distant) | matches record R005, 27 records away |
| M3 | adakv_snapkv | 23 | 0 | R031 | no visible pattern | weak overlap (3), not confidently either way |
| M3 | adakv_snapkv | 24 | 3 | R037 | hallucination | number not in context (649374) |
| M3 | adakv_snapkv | 36 | 2 | R030 | misattribution (distant) | matches record R024, 6 records away |
| M3 | expected_attn | 0 | 3 | R030 | misattribution (distant) | matches record R001, 29 records away |
| M3 | expected_attn | 1 | 2 | R032 | **refusal** | "there is no record R032" |
| M3 | expected_attn | 9 | 2 | R033 | **refusal** | "I can't reproduce record R033" |
| M3 | expected_attn | 14 | 3 | R030 | **refusal** | "there is no record R030" |

## Step 4 — tag distribution and per-model claim

| model | contamination | hallucination | refusal | misattribution (distant) | no visible pattern |
|---|---|---|---|---|---|
| M2 (n=12) | **6** | 4 | 0 | 1 | 1 |
| M3 (n=12) | 0 | 1 | **7** | 3 | 1 |

**The two models fail in different, dominant, and mutually exclusive ways. Neither dominant mode
appears at all on the other model** (0 refusals on M2; 0 contaminations on M3).

**M2's candidate mechanism, stated falsifiably:** *generation degrades when a kept fact's
immediately-adjacent retained record (1-2 records away in the same slot's cache) is itself
complete, causing the model to answer with that neighboring record's content instead of the
queried one -- independent of the queried fact's own completeness.* All 6 contamination cases have
record-distance <= 2 and near-total field overlap (6-7 of ~8 fields) with a real, complete,
nearby record. This is the clean confusability signature the c=8/c=1 keep-set inspection in
`DIAGNOSIS_EA.md` gestured at but did not test directly: two complete, adjacent facts, and the
model picks the wrong one.

**M2's secondary mode (hallucination, 4/12)** is a different failure: numbers absent from the
entire context (654321, 123456-style), so these are not retrieval errors at all -- the model
invents a plausible record rather than copying any real one. Not explained by adjacency.

**M3's candidate mechanism, stated falsifiably:** *M3 asserts absence of a record that is present
and complete in the retained cache, rather than misretrieving a present one.* This is not a
fragmentation or adjacency story -- it looks like an instruction-following/confidence failure under
compression, closer to Paper 2's B9 (max_new_tokens as a format confound) than to anything in
Track 1. M3's remaining errors (3/12 "misattribution, distant") point AWAY from adjacency: the
matched records sit 6-29 positions away, not 1-2.

## What this does not establish

- n = 12 per model, hand-tagged from one cell (c=40, C=512). Not screened, not replicated, not a
  confirmed mechanism -- exactly as scoped.
- The `hallucination` and `misattribution (distant)` tags used fixed thresholds (record distance
  <=2 & overlap>=6 for contamination; overlap>=4 otherwise for distant misattribution; fabricated
  numbers otherwise) chosen after seeing a few cases, not preregistered -- this is Step 3's read,
  not a Step-4-onward test.
- "No visible pattern" appears once per model and is reported as such, not chased.

## What would need to happen before this is trusted

Per the standing instruction: the same cheap-screen-then-both-model-replication treatment as
A1-A9, with the mechanism claim split by model since the two failure modes are not the same
candidate. For M2's contamination claim, the natural falsifiable test is CPU-only and cheap: among
ALL floor-correct/U-wrong cases (not just the 24 sampled), does record-distance-to-nearest-complete-
neighbor predict contamination-vs-other, with a threshold fixed before scoring? For M3's refusal
claim, the natural test needs generation (does forcing a lower refusal-token logit or lengthening
the answer change the outcome?) and is out of scope for a CPU-only read.

No fix or new wrapper variant is proposed here.
