## Phase 1 as a three-factor design: protection × tiering × bit-width

Arms 1 and 2 are permanent-eviction (`n_quant = 0`) and therefore bit-width invariant;
they serve as the shared reference level rather than being re-run. Complete n=150 data
exists at both widths for the tiered arms.

| budget | bits | protection | tiering | interaction | 95% CI | n differ |
|---|---|---|---|---|---|---|
| 154 | **8** | +0.0272 | +0.0006 | **+0.0011** | [+0.0000, +0.0033] | 1 |
| 154 | 4 | +0.0233 | -0.0033 | -0.0067 | [-0.0122, -0.0022] | 6 |
| 257 | **8** | +0.1639 | -0.0006 | **-0.0011** | [-0.0056, +0.0033] | 5 |
| 257 | 4 | +0.1417 | -0.0239 | -0.0456 | [-0.0611, -0.0311] | 47 |
| 514 | **8** | +0.3128 | -0.0017 | **-0.0056** | [-0.0189, +0.0067] | 32 |
| 514 | 4 | +0.2578 | -0.1144 | -0.1156 | [-0.1444, -0.0856] | 112 |

**Both 8-bit interaction CIs contain zero; both 4-bit CIs exclude it.** The tiering main
effect at 8-bit is indistinguishable from zero at every budget.

### The bit-width factor, tested directly

| budget | tiering @8-bit | tiering @4-bit | difference | 95% CI |
|---|---|---|---|---|
| 154 | +0.0006 | -0.0033 | **-0.0039** | [-0.0067, -0.0017] |
| 257 | -0.0006 | -0.0239 | **-0.0233** | [-0.0311, -0.0161] |
| 514 | -0.0017 | -0.1144 | **-0.1128** | [-0.1278, -0.0983] |

The width effect is significant at every budget and grows with budget: larger budgets
retain more credential tokens and therefore expose more of them to quantization damage.

### Interpretation

The interaction the spec reports as ≈0 is **not a null result; it is a result conditional
on bit-width.** Protection's benefit survives near-lossless (8-bit) tiering essentially
intact and is progressively destroyed by aggressive (4-bit) tiering, in proportion to how
much benefit there was to destroy. The reference's ≈0 and this implementation's −0.116 are
two slices of the same surface; both are correct about their own condition.

The mechanism is established independently by the credential-survival instrumentation:
arms 2 and 4 evict **exactly the same 92.1 credential tokens**, so tiering adds no eviction
at all. Its entire effect is precision loss on tokens that were retained — which is why it
tracks reconstruction error (1.18% at 8-bit vs 19.98% at 4-bit) rather than anything about
the retention policy.

> **This condition was added post-hoc.** The 4-bit Phase 1 ran first, the disagreement with
> the reference was observed, and the 8-bit condition was then specified and run to test a
> stated mechanism. The prediction was made before the run and confirmed, but this is an
> exploratory inference, not a pre-registered confirmatory one.

