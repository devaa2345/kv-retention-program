# Track 2 — selection-effect check on the zero-overlap ("confidently wrong") filter

Full 141-case population (no sampling, no overlap filter), tagged for refusal-language and
contamination-pattern only. CPU-only, no GPU. Source: `out/_track2_full141_tagged.json`.

## The check

| model | subset | n | refusal | contamination |
|---|---|---|---|---|
| M2 | all | 82 | 0 (0%) | 14 (17%) |
| M2 | zero-overlap (sampled from) | 42 | 0 (0%) | 11 (26%) |
| M2 | nonzero-overlap (excluded) | 40 | 0 (0%) | 3 (8%) |
| M3 | all | 59 | 18 (31%) | 3 (5%) |
| M3 | zero-overlap (sampled from) | 28 | 18 (64%) | 0 (0%) |
| M3 | nonzero-overlap (excluded) | 31 | 0 (0%) | 3 (10%) |

## Verdict: the model split survives; one count needs correcting

**M2's "no refusal" finding is robust, not filter-dependent.** 0/82 in the full population, 0 in
both subsets. The zero-overlap filter did not hide any M2 refusals because there are none to hide.

**M3's refusal dominance is genuine, not an artifact — but its concentration in the zero-overlap
bucket is expected by construction, not evidence of anything.** Refusal text ("there is no record
R032") cannot share gold-answer words by construction, so refusals are *necessarily* zero-overlap.
The original sampling (drawn only from zero-overlap cases) was therefore the right place to find
M3's refusals, not a bucket that suppressed them elsewhere — there is nowhere else for them to be.
31% of M3's full population is refusal; that number stands.

**M3's contamination count in `TRACK2_FAILURE_READ.md` was wrong, and the filter caused it.** All
3 of M3's contamination cases live in the **excluded** nonzero-overlap bucket (10% of that bucket,
vs 0% of the sampled zero-overlap bucket). The original report's table said "0/12 contamination on
M3" — that was true of the 12 sampled cases, but the full population shows M3 does contaminate,
just rarely (3/59 ≈ 5%, against M2's 14/82 ≈ 17%) and its contamination cases apparently retain
enough of the wrong record's overlap with gold by chance (or partial corruption) to fall into the
near-miss bucket rather than the confident-miss one. **`TRACK2_FAILURE_READ.md`'s claim "0/12
contamination on M3" is corrected here to: M3 contaminates too, at roughly a third of M2's rate,
and it was invisible to the sampled 12 specifically because the sampling filter excluded the
bucket where it lives on this model.**

**M2's contamination finding survives, with its size revised.** 17% of the full M2 population
(14/82), not something conjured by the filter — present in both buckets, at 26% and 8%. The filter
enriched for it (as expected, since a full wrong-record swap should look confidently wrong) but did
not manufacture it.

## What changes for a prereg, if one is written

- The **refusal-vs-contamination split by model is confirmed at the full-population level**, not
  just in the 24-case sample: M2 0% refusal / 17% contamination; M3 31% refusal / 5% contamination.
- **M3's contamination rate is nonzero and should be included as a target rate**, not treated as
  absent. A prereg comparing "does record-distance predict contamination" should pool across both
  models rather than treating M3 as a model with no contamination to explain.
- Any future sampling for this kind of qualitative read should **stratify by overlap-bucket**
  rather than drawing only from the confidently-wrong bucket, since which failure mode ends up
  confidently-wrong versus near-miss is itself model-dependent and not knowable in advance.

No fix or wrapper variant proposed. Nothing prereg'd yet.
