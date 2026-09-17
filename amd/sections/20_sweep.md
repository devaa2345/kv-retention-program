## Bit-width sweep — full table (common seed set, n=50 at every width)

8-bit and 4-bit had n=150 available; they are restricted to seeds 0–49 so all six widths
are measured on identical prompts. Full-n values: 8-bit 0.1644, 4-bit 0.1189.

A turn output is *well-formed* if it contains an `sk-<14 hex>` token. A prompt is *clean*
if all 6 turns are well-formed, *degenerate* if none are. `empty/EOS` is the fraction of
turn outputs empty after stripping. Reconstruction error is measured on tensors
**intercepted during live generation**.

| bits | n | mean acc | empty/EOS | repetitive | degenerate | clean | mixed | recon err | levels/limit | violations |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 50 | 0.1433 | 0.000 | 0.017 | 7 | 28 | 15 | 1.15% | 152/256 | 0 |
| 7 | 50 | 0.1600 | 0.000 | 0.043 | 7 | 23 | 20 | 2.28% | 102/128 | 0 |
| 6 | 50 | 0.1300 | 0.007 | 0.030 | 9 | 13 | 28 | 4.56% | 61/64 | 0 |
| **5** | 50 | **0.0000** | 0.223 | 0.127 | 48 | 0 | 2 | 9.36% | 32/32 | 0 |
| 4 | 50 | 0.1167 | 0.010 | 0.057 | 9 | 9 | 32 | 19.31% | 16/16 | 0 |
| **3** | 50 | **0.0000** | 0.513 | 0.050 | 46 | 0 | 4 | 39.92% | 8/8 | 0 |

### The headline claim reproduces

**Accuracy is not monotone in bit-width; reconstruction error is.** Error rises strictly
across all six widths (1.15 → 2.28 → 4.56 → 9.36 → 19.31 → 39.92 %), yet accuracy
collapses to exactly 0.0000 at 5-bit and **recovers to 0.1167 at 4-bit while carrying more
than twice the reconstruction error**, then collapses again at 3-bit.

The reference reports 4-bit recovery at 0.123; this implementation measures 0.1167 (n=50)
and 0.1189 (n=150). **The curve shape is independently confirmed.** The widths that
collapse differ — the reference reports 6- and 5-bit, here it is 5- and 3-bit with 6-bit
merely degraded — so the *phenomenon* replicates while the *specific widths* do not. That
is itself the substance of §9's caution against reading a sweep as a severity ordering.

**The quantizer is cleared.** Level counts never exceed 2^bits at any width (0 violations,
checked per group on tensors actually written during generation), occupancy rises smoothly
from 0.44 to 0.98 as width falls, and error is strictly monotone on those same intercepted
tensors. The collapse is how the model responds to quantization noise, not a quantizer bug.

The collapse signature is degenerate *output*: at 5-bit, 48 of 50 prompts produce no
well-formed value on any turn and empty/EOS jumps to 0.223; at 3-bit, 46 of 50 and 0.513.
At every working width empty/EOS is ≤0.018.

