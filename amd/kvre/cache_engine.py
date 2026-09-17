"""Three-tier KV cache: FULL / QUANT / EVICT, quantizer, and byte accounting.

Spec ref: SPEC_REIMPL_v1.md section 3.
Resolutions: [GAP-G] grouping, [SPEC-GAP 3]/[GAP-N] byte cost, [GAP-D] promotion is not lossless.

Quantizer contract (section 3, explicit): applied per position over the (kv_heads, head_dim)
slice, in bfloat16. NOT a float32 reimplementation -- the spec records that a float32 version
understated error by ~2x. Every arithmetic step below stays in bf16 for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

FULL, QUANT, EVICT = 0, 1, 2
TIER_NAMES = {FULL: "FULL", QUANT: "QUANT", EVICT: "EVICT"}


# --------------------------------------------------------------------------------------
# Quantizer
# --------------------------------------------------------------------------------------

QUANT_ARITH = "fp32_mid"   # see [GAP-Q]; "bf16_all" is the literal-text variant


def quantize_dequantize(x: torch.Tensor, bits: int, return_codes: bool = False,
                        arith: str | None = None):
    """Affine min/max quantize-dequantize round trip, in the tensor's own dtype.

    x: (B, H, S, D). One group per (batch, position): all H*D elements share a scale/zero.
    [GAP-G]: the spec's "over the (kv_heads, head_dim) slice" is read as one group covering
    both axes together.

    Returns dequantized tensor of the same shape/dtype (and integer codes if requested).
    """
    assert 1 <= bits <= 16, f"bits out of range: {bits}"
    arith = arith or QUANT_ARITH
    b, h, s, d = x.shape
    store_dt = x.dtype

    # (B, S, H*D) -- group axis last
    g = x.permute(0, 2, 1, 3).reshape(b, s, h * d)
    # [GAP-Q] arithmetic precision. "fp32_mid": compute in fp32, store result at the engine's
    # dtype (bf16) -- this is what a real int8-codes + bf16-scale cache physically does, and it
    # reproduces the section 3 reference table. "bf16_all" keeps every intermediate in bf16,
    # the literal reading of section 3's sentence; it overstates 8-bit error ~1.4x.
    g = g.float() if arith == "fp32_mid" else g
    dt = g.dtype

    lo = g.min(dim=-1, keepdim=True).values
    hi = g.max(dim=-1, keepdim=True).values
    qmax = float(2 ** bits - 1)

    scale = (hi - lo) / torch.tensor(qmax, dtype=dt, device=x.device)
    # Degenerate groups (constant slice) get scale 1 so codes are all zero and deq == lo == x.
    degenerate = scale == 0
    scale = torch.where(degenerate, torch.ones_like(scale), scale)

    codes = torch.round((g - lo) / scale)
    codes = torch.clamp(codes, 0.0, qmax)
    deq = (codes * scale + lo).to(store_dt)      # storage is always the engine dtype

    deq = deq.reshape(b, s, h, d).permute(0, 2, 1, 3).contiguous()
    if return_codes:
        return deq, codes.reshape(b, s, h, d).permute(0, 2, 1, 3).contiguous()
    return deq


def relative_error(orig: torch.Tensor, recon: torch.Tensor) -> float:
    """||recon - orig||_F / ||orig||_F, computed in float32 for a stable *measurement*.

    The quantization path itself is bf16 (above); only this norm is float32, because measuring
    the error in bf16 would quantize the measurement.
    """
    o = orig.float()
    r = recon.float()
    return (torch.linalg.vector_norm(r - o) / torch.linalg.vector_norm(o)).item()


# --------------------------------------------------------------------------------------
# Quantizer assertions (spec section 3 "Level-count assertion", gate G4)
# --------------------------------------------------------------------------------------

@dataclass
class QuantAudit:
    """Records every quantizer call made on tensors actually written during generation."""
    calls: int = 0
    checks: int = 0
    mean_levels_sum: float = 0.0
    max_levels_seen: int = 0
    violations: list[str] = field(default_factory=list)
    level_histogram: dict[int, int] = field(default_factory=dict)   # distinct-level count -> freq

    def check_levels(self, codes: torch.Tensor, bits: int, tag: str,
                     check_stride: int = 25, warmup: int = 20):
        """Assert distinct level count <= 2^bits, per group (one group per position).

        Checked per group, not tensor-wide: each group carries its own scale, so tensor-wide
        distinct values legitimately exceed 2^bits.

        Vectorised: sorts along the group axis and counts transitions, so ALL groups are
        verified in one kernel with a single host sync. The earlier form called torch.unique
        on 16 sampled groups, costing ~18 GPU synchronisations per call and ~65% of total
        runtime. This version is both cheaper and a strictly stronger assertion (every group,
        not a sample). It is additionally throttled after `warmup` calls, since the check is
        observational and does not affect any measured quantity.
        """
        self.calls += 1
        if self.calls > warmup and (self.calls % check_stride) != 0:
            return
        self.checks += 1
        limit = 2 ** bits
        b, h, s, d = codes.shape
        g = codes.permute(0, 2, 1, 3).reshape(b * s, h * d)
        gs, _ = torch.sort(g, dim=1)
        distinct = (gs[:, 1:] != gs[:, :-1]).sum(dim=1) + 1        # per-group level count
        stats = torch.stack([distinct.max().double(), distinct.double().mean(),
                             g.min().double(), g.max().double()])
        mx, mean_lv, cmin, cmax = stats.tolist()
        mx = int(mx)
        self.max_levels_seen = max(self.max_levels_seen, mx)
        self.level_histogram[mx] = self.level_histogram.get(mx, 0) + 1
        self.mean_levels_sum += mean_lv
        if mx > limit:
            self.violations.append(f"{tag}: group has {mx} distinct levels > 2^{bits}={limit}")
        if cmin < 0 or cmax > limit - 1:
            self.violations.append(f"{tag}: codes out of range [{cmin},{cmax}] for {bits} bits")

    def ok(self) -> bool:
        return not self.violations


def error_table(kv_tensors: list[tuple[torch.Tensor, torch.Tensor]],
                bit_list: list[int], arith: str | None = None) -> dict[int, dict[str, float]]:
    """Mean relative reconstruction error over layers, per bit-width, for keys and values.

    kv_tensors: per-layer (keys, values) as actually held in the cache, bf16.
    Reproduces the section 3 reference table.
    """
    out: dict[int, dict[str, float]] = {}
    for bits in bit_list:
        kerrs, verrs = [], []
        for k, v in kv_tensors:
            kerrs.append(relative_error(k, quantize_dequantize(k, bits, arith=arith)))
            verrs.append(relative_error(v, quantize_dequantize(v, bits, arith=arith)))
        out[bits] = {
            "keys": 100.0 * sum(kerrs) / len(kerrs),
            "values": 100.0 * sum(verrs) / len(verrs),
        }
    return out


def assert_monotone_error(table: dict[int, dict[str, float]]) -> list[str]:
    """Gate G4: relative error must increase strictly as bit-width falls."""
    problems: list[str] = []
    widths = sorted(table.keys(), reverse=True)          # explicit sort, high bits first
    for which in ("keys", "values"):
        prev_w, prev_e = None, None
        for w in widths:
            e = table[w][which]
            if prev_e is not None and not (e > prev_e):
                problems.append(
                    f"non-monotone {which}: {prev_w}bit={prev_e:.4f}% -> {w}bit={e:.4f}%"
                )
            prev_w, prev_e = w, e
    return problems


# --------------------------------------------------------------------------------------
# Byte accounting
# --------------------------------------------------------------------------------------

def cache_bytes(n_full: int, n_quant: int, quant_byte_cost: float = 0.25) -> float:
    """Cost in units of one bf16 position. EVICT contributes 0."""
    return n_full * 1.0 + n_quant * quant_byte_cost


def iso_memory_tokens(total_budget: int, full_fraction: float = 0.5,
                      quant_byte_cost: float = 0.25) -> int:
    """Section 6: raw token count for a tiered arm funded to match permanent-arm bytes.

    bytes_per_token = f*1.0 + (1-f)*quant_byte_cost;  T = total_budget / bytes_per_token.
    With f=0.5, cost=0.25 -> 0.625 -> T = 1.6 * total_budget, matching the spec's worked example.
    """
    per_token = full_fraction * 1.0 + (1.0 - full_fraction) * quant_byte_cost
    return int(round(total_budget / per_token))


# --------------------------------------------------------------------------------------
# Cache surgery
# --------------------------------------------------------------------------------------

def cache_length(cache) -> int:
    return cache.layers[0].keys.shape[-2]


def apply_tiers(cache, keep_positions: list[int], quant_positions_mask: list[bool],
                bits: int, audit: QuantAudit | None = None, tag: str = "") -> None:
    """Rewrite the cache in place to hold exactly `keep_positions`, quantizing where masked.

    keep_positions: ordered list of cache-relative indices to retain (must be sorted ascending;
                    order is the new cache order, so ascending keeps RoPE/causal order intact).
    quant_positions_mask: same length; True -> that retained position is QUANT tier.

    [GAP-D]: quantization is destructive. A position later promoted to FULL keeps its dequantized
    values; the bf16 originals are gone.
    """
    assert keep_positions == sorted(keep_positions), "keep_positions must be ascending"
    assert len(keep_positions) == len(quant_positions_mask)
    if not keep_positions:
        raise ValueError("refusing to empty the cache")

    dev = cache.layers[0].keys.device
    idx = torch.tensor(keep_positions, dtype=torch.long, device=dev)
    qmask = torch.tensor(quant_positions_mask, dtype=torch.bool, device=dev)

    for li, layer in enumerate(cache.layers):
        k = layer.keys.index_select(-2, idx)
        v = layer.values.index_select(-2, idx)
        if qmask.any():
            qi = torch.nonzero(qmask, as_tuple=True)[0]
            ksub = k.index_select(-2, qi)
            vsub = v.index_select(-2, qi)
            kdq, kcodes = quantize_dequantize(ksub, bits, return_codes=True)
            vdq, vcodes = quantize_dequantize(vsub, bits, return_codes=True)
            if audit is not None:
                audit.check_levels(kcodes, bits, f"{tag}L{li}K")
                audit.check_levels(vcodes, bits, f"{tag}L{li}V")
            k = k.index_copy(-2, qi, kdq)
            v = v.index_copy(-2, qi, vdq)
        layer.keys = k.contiguous()
        layer.values = v.contiguous()
