# Where the cache should matter: a two-hop bridge task design (design only, no GPU)

## Motivation

Every real-text result in this exploratory thread (`REALTEXT_REINTERPRETATION.md`) and the
LEDGER-C proximity check point the same way: whenever the answer lives in **one** contiguous
span, a decent pointer (BM25-over-everything, or literal record-ID substring match for LEDGER-C)
finds it, and the cache barely matters — a plain sentence-only or record-only prompt gets nearly
the same accuracy as any compressed-cache method. **A single-span task structurally cannot show a
cache-management method any advantage**, because a single-span task is solvable by "find the one
right piece of text," which any reasonable pointer already does.

For the cache to matter, the task needs to require **combining evidence from two or more spans
that a single-hop pointer will not co-locate.** HotpotQA's bridge-question subset, with its own
annotated supporting-fact sentences, is a natural fit: this is real prose (not synthetic), has a
gold two-sentence evidence set per question, and is a well-known benchmark, so results are legible
against prior published KV-compression work if any exists (see Item 4).

## Task construction

- Source: HotpotQA distractor-setting validation split, `type == "bridge"` questions only (not
  `comparison`, which doesn't require chaining through an intermediate entity).
- For each question, take its two gold supporting-fact sentences (`supporting_facts`: title +
  sentence index, resolved against `context`). Verify programmatically that the two gold
  sentences are NOT part of the same paragraph/title (bridge questions usually satisfy this by
  construction, but confirm rather than assume).
- Concatenate the two gold paragraphs plus HotpotQA's own distractor paragraphs (already provided
  per-question, ~8 paragraphs total) into one long context, **placing the two gold paragraphs far
  apart** (one near the start, one near the end of the concatenated document) — this is the
  condition that makes contiguous-recency retention (floor) structurally unable to hold both at
  once at a realistic budget, unlike SnapKV/attention-based methods that can pick from anywhere.
- Target context length: 2000–3000 tokens (matching this session's established working range;
  avoids the transient-OOM zone found repeatedly tonight around 2200–2400+ tokens — pilot this
  explicitly before committing to a size).
- Extractive scope note: HotpotQA answers are often short spans but not always verbatim-copyable
  in the same clean way LEDGER-C or SQuAD's span-style answers are; some bridge answers require
  light synthesis (e.g., "the first entity's employer, which is mentioned in the second sentence
  as X"). This needs a CPU-only preflight check on a sample: confirm a workable fraction of
  bridge answers are extractive substrings of one of the two gold sentences, or the literal mask
  convention (which worked for LEDGER-C and SQuAD precisely because answers are copyable spans)
  won't apply cleanly here and would need its own design decision before any masked arm runs.

## Arms

1. **top-1 sentence only** — no context cache at all, just the single BM25/attention-top-ranked
   sentence inserted (mirrors this session's `sentence_only` ablation arm). Expected to fail on
   bridge questions specifically *because* one sentence alone cannot supply both hops — this is
   the arm that should show the task actually requires more than one span, unlike the SQuAD
   ablation where it didn't.
2. **top-k sentences only** (k=2 or 3, no cache) — same idea but with room for both gold
   sentences if the pointer ranks them both highly. Tests whether "just get more text via a
   pointer" (this session's `farthest_first`/whole-sentence retrieval spirit) is sufficient
   without any managed cache, i.e. whether this is fundamentally a *retrieval-count* problem
   rather than a *cache-management* problem.
3. **floor_pos** (contiguous recency, current session's own convention) — should struggle
   structurally when the two gold paragraphs are placed far apart and the compression budget
   can't span both.
4. **SnapKV** (token-level attention-guided, current session's own convention) — the real test:
   does its selection-from-anywhere property let it retain both gold spans where floor cannot?
5. **full_cache** (nothing evicted) — competence gate and ceiling, see below.

## Full-cache competence gate

Before ANY compressed-arm comparison is meaningful, full_cache accuracy on this task must clear a
predetermined floor (a specific number should be set once real bridge-answer format is inspected,
analogous to the 5070 real-text pilot's own `[.55,.97]` full-cache anchor gate in
`PREREG_REALTEXT_5070.md`). If full_cache itself cannot answer bridge questions reliably — because
the 3B models can't chain two hops even with perfect, complete evidence in front of them — then no
compressed-arm result on this task means anything: a failure to beat floor would be indistinguishable
from "the model can't do two-hop reasoning at all," not "the cache lost the second hop."

**This gate is not hypothetical for this model class.** Paper 2's own PREREG (`PREREG_P2.md`)
found two-hop LEDGER-style tasks collapse for these same-class 3B models: one-hop accuracy 0.4375
vs two-hop accuracy 0.100–0.1263 across multiple two-hop task variants, with diagnosis explicitly
naming "the binding constraint was the two-hop chain itself" — models "stopped after hop 1." This
is a synthetic task, not HotpotQA, but it's the closest available same-hardware, same-model-class
precedent, and it says plainly: **do not assume 3B models can complete two-hop reasoning even with
every piece of evidence directly in front of them.** The competence gate exists precisely to catch
this before spending GPU time on a comparison that can't be interpreted either way.

## Expected costs

Rough order-of-magnitude, extrapolating this session's own real-text timings (SQuAD-scale
prefill/generation at similar context lengths): a small development set (~20-30 instances) for
construction/gate-checking, CPU-only where possible (sentence-pair distance verification, format
preflight); then, only if the full-cache gate passes, a held-out confirmation set sized by
observed variance (likely 60-160 instances given this session's own established sizing pattern),
each instance needing 5 arms × up to 2 forward passes — a few GPU-hours at most, well within what
this session's other confirmatory runs have taken (M2 tight-KV confirmation cost ~13.6 GPU-minutes
for 160 instances × 4 arms; this design has more arms but similar per-arm cost).

## Failure risks

- **The competence gate fails outright** (most likely risk, per Paper 2's own two-hop precedent):
  the model can't reliably do two-hop synthesis even from full_cache. In this case the entire
  design should be shelved for 3B models, not silently downgraded to a weaker task, per this
  session's standing practice of reporting negative gates plainly rather than re-scoping after
  the fact.
- **Bridge answers aren't cleanly extractive**, undermining the literal-mask convention that made
  every measurement in this session interpretable; would need either a different task subset
  (single-sentence-extractive bridge questions only, if any exist) or a different scoring/masking
  convention, decided BEFORE generation, not adjusted afterward.
- **Distractor paragraphs dilute the signal**: HotpotQA's distractor setting includes ~8
  paragraphs of which only 2 are relevant; if floor's own accuracy is already very low due to
  irrelevant content dominating a small budget rather than due to genuinely losing the far
  paragraph, the placement manipulation (gold sentences far apart) may not be the dominant effect
  — needs a CPU-only check on how much of the 2000-3000 token budget is actually distractor
  content at the target compression ratio.
- **SnapKV's own known failure mode from tonight's work**: this session repeatedly found SnapKV's
  compressed-position attention pointer can be right for the wrong reason (retained union larger
  than any single head's budget, cross-head diffusion) — a two-hop task doubles the chances of a
  correct-looking retention masking an incorrect actual mechanism; any positive SnapKV result here
  would need the same kind of component ablation this session just ran on the SQuAD result before
  being trusted as a genuine two-hop retention win.

## Not run

This is a design document only. No GPU time has been spent on this task. Per instruction, nothing
here should be treated as a confirmatory or even exploratory result until a prereg is written and
the full-cache competence gate is checked.
