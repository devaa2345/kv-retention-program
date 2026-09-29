# Prior art: retrieval-only baselines vs long-context/KV-compression benchmarks

Web search only (2026-09 web search results). Not a systematic literature review -- a first pass
to establish whether this session's benchmark-retrieval critique is novel or already known.

## What exists

**1. The general critique that popular long-context benchmarks reward literal/lexical retrieval
rather than genuine reasoning is established and has produced follow-up benchmarks.**
Needle-in-a-Haystack (NIAH) is explicitly criticized in the literature for relying on "trivial
keyword matching" -- models can exploit literal overlap between the question and the needle
sentence to succeed without real comprehension. **NoLiMa** (Adobe Research,
[arxiv.org/abs/2502.05167](https://arxiv.org/abs/2502.05167)) was built specifically to close
this gap: it constructs needles with minimal lexical overlap to the question, forcing models to
use latent/semantic association rather than literal matching, and shows large accuracy drops for
models that looked strong on standard NIAH.

**2. A directly relevant, very recent audit paper on the KV-cache side**: "How Query Visibility
Changes KV-Cache Compression Rankings: A Matched-Budget Audit"
([arxiv.org/abs/2607.11942](https://arxiv.org/abs/2607.11942), 2026-07). Its core argument: most
KV-cache compression evidence is collected under a **query-aware** protocol, where the benchmark
question is visible to the compressor before it decides what to keep. This matches a deployment
where the document is re-compressed per question -- a deployment in which a compressed cache buys
almost nothing, because the expensive prefill is paid again anyway. The economically interesting
case is **query-agnostic**: compress once, query many times, where the question doesn't exist yet
at compression time. The paper reports compression-method rankings change substantially between
the two protocols. This is the closest existing work to this session's finding: it is also about
whether the query being available (directly or via search) undermines the significance of a
compression method's own measured gains -- though its framing is "which evaluation protocol" 
rather than "does a retrieval-only baseline with no compression at all beat every compression
variant, including uncompressed full cache."

**3. Adjacent but not identical: RAG-vs-compression comparisons.** "Beyond RAG: Task-Aware KV
Cache Compression for Comprehensive Knowledge Reasoning"
([arxiv.org/abs/2503.04973](https://arxiv.org/abs/2503.04973), 2025-03) directly compares RAG
(similarity-search retrieval into a fresh short context) against KV-cache compression, and reports
their proposed task-aware compression method beats plain RAG on LongBench v2 by up to 7 points.
This is the opposite conclusion from this session's finding (compression beating retrieval, not
retrieval beating compression) -- but it is evidence that RAG-vs-compression is an active,
recognized comparison axis in this literature, using a broadly similar retrieval-baseline
methodology to what this session's BM25-over-all-sentences ablation used.

**4. General observation on retrieval quality vs downstream accuracy** (from a broader
long-context-RAG search, not specific to KV-cache compression): reports that high retrieval recall
does not guarantee downstream LLM accuracy (one cited result: dense retrieval reaching 90% recall
but 0% downstream accuracy, vs BM25's 96.7% recall producing only 3.3% accuracy) -- a caution that
"the pointer found the right text" and "the model used it correctly" are separate questions, which
matches this session's own repeated finding (e.g. the SQuAD pilot's "71/162 gold-selected queries
actually answered exactly" gap, and the LEDGER-C proximity check's apparent tendency to sometimes
ignore inserted evidence).

## What does NOT clearly exist (based on this search)

No paper found in this pass that makes the SPECIFIC claim this session's ablation makes: that a
plain full-text-search retrieval baseline, inserted into a short/no-compression context with NO
KV-cache management at all, **matches or beats a purpose-built KV-cache-compression method's
measured accuracy gain on the same task** -- i.e., that the compression mechanism itself is
contributing ~nothing beyond what a working pointer already supplies. The closest analogues (item
2, the query-visibility audit; item 3, RAG-vs-compression) are adjacent framings of a similar
underlying concern (does the benchmark protocol let something other than the compression method
do the real work?) but neither is the same experiment or the same conclusion.

## Read

The *general category* of critique -- long-context/KV-cache benchmarks can be dominated by
retrieval/lookup rather than by the mechanism under test -- is not new; it has an established
line of work (NIAH's lexical-matching critique → NoLiMa; query-aware-vs-agnostic KV-cache
auditing). **The specific empirical demonstration in this session** (a full-text BM25 baseline
with zero compression beats a frozen, previously-confirmed KV-cache focus system by ~15 points,
and a component ablation showing the compressed cache itself contributes ~nothing once the
pointer is good) looks like a new, concrete instance of that general critique on this session's
own real-text tight-KV task -- not a previously-published result found in this search, but also
not a claim that the underlying concern is original to this session. Recommend citing the
query-visibility audit paper and the NIAH/NoLiMa line directly if this reinterpretation is written
up further, both to place the finding in context and to avoid overclaiming novelty of the general
critique (only the specific measurement here is new).

## Prior art on the mechanism itself: re-presenting attended evidence near the query

**Verified**: "Attention Sorting Combats Recency Bias In Long Context Language Models"
(Peysakhovich & Lerer, [arxiv.org/abs/2310.01427](https://arxiv.org/abs/2310.01427), 2023-10) is a
real paper. Its mechanism: decode one step, sort the context's documents by the attention they
just received (highest-attention document placed LAST, i.e. closest to the query/generation
point), repeat, then generate the final answer from the resorted context. The motivating claim is
that models under-attend to relevant information earlier in a long context (a recency/position
prior baked in from pretraining), and that moving high-attention content physically closer to the
query compensates for it.

**This is the closest existing precedent to this session's proximity-insertion finding** (both
the real-text "focus" insertion and the LEDGER-C proximity check): both work by taking whatever
the model already attends to and physically relocating/duplicating it next to the query, and both
report that this alone recovers most of a masking/retrieval method's measured gain. The mechanism
is not identical -- Attention Sorting reorders the EXISTING context in place and iterates;
this session's checks insert an EXTRA COPY of the identified evidence right before the query,
leaving the original context (compressed or not) untouched -- but the underlying claim (proximity
to the query, not the compression/masking machinery, is what drives the gain) is the same
insight, verified in a 2023 paper on off-the-shelf long-context models.

**Related, more recent work in the same family** (not independently verified beyond search
snippets): "In-Context Re-ranking" (ICR) uses attention as an implicit relevance signal to reorder
retrieved passages within the context window, and "ReContext: Recursive Evidence Replay as LLM
Harness for Long-Context Reasoning" ([arxiv.org/abs/2607.02509](https://arxiv.org/abs/2607.02509))
maintains and replays an accumulating evidence pool alongside the original context and question,
conditioning each new selection round on what was already replayed -- both are instances of the
same general pattern (use attention/relevance to find evidence, then re-present it favorably
relative to the query).

**Plain assessment of what, if anything, remains new**: the *general mechanism* -- re-presenting
attended-to evidence near the query recovers a position/attention bias, independent of any
compression or masking -- is not new; it has a specific, named 2023 paper (Attention Sorting) and
an active surrounding literature (ICR, ReContext). What this session adds, if anything, is narrower
and more specific: (1) demonstrating this same effect INSIDE a KV-cache-compression evaluation
specifically (showing a frozen, previously-confirmed compression+mask system's gain is largely
reproduced by simple proximity insertion, both on real text and on the LEDGER-C synthetic task),
and (2) the direct comparison against a full-text retrieval baseline (BM25-over-everything) on the
same task, which is not the framing Attention Sorting or ICR use (they reorder/replay what's
already retrieved; they don't set up a full-text-search-vs-compression contrast). The general
insight is not original to this session; the specific application -- using it to reinterpret a
compression-method's own confirmed benchmark result -- looks narrower and not found in this pass.

## Scope and caveats

Web search only, no paper full-text read beyond search-result summaries, no citation-graph or
venue/date verification beyond what the search snippets state. A proper literature review before
any external write-up should read the query-visibility audit paper in full and check its own
citations for closer prior art, and should search dedicated venues (ACL Anthology, OpenReview)
directly rather than relying on general web search alone.

Sources:
- [NoLiMa: Long-Context Evaluation Beyond Literal Matching](https://arxiv.org/abs/2502.05167)
- [How Query Visibility Changes KV-Cache Compression Rankings: A Matched-Budget Audit](https://arxiv.org/abs/2607.11942)
- [Beyond RAG: Task-Aware KV Cache Compression for Comprehensive Knowledge Reasoning](https://arxiv.org/abs/2503.04973)
- [KV Cache Compression, But What Must We Give in Return? A Comprehensive Benchmark of Long Context Capable Approaches](https://arxiv.org/pdf/2407.01527)
- [Attention Sorting Combats Recency Bias In Long Context Language Models](https://arxiv.org/abs/2310.01427)
- [ReContext: Recursive Evidence Replay as LLM Harness for Long-Context Reasoning](https://arxiv.org/abs/2607.02509)
