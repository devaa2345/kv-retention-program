"""sm_120 verification by ACTUAL kernel launch, not by torch.cuda.is_available().

The failure this guards against is specific and silent-ish: cu126 wheels carry SASS for
sm_50..sm_90 only, so on a Blackwell 5070 every launch dies with cudaErrorNoKernelImage even
though `is_available()` returns True and the device name prints correctly (v1 §6.1). So the
check has to run real kernels in the dtypes and ops the grid actually uses.
"""

from __future__ import annotations

import json
import sys

import torch

out: dict = {}
out["torch"] = torch.__version__
out["cuda"] = torch.version.cuda
out["arch_list"] = torch.cuda.get_arch_list()
out["device"] = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
out["capability"] = f"{cap[0]}.{cap[1]}"
out["sm_120_in_arch_list"] = any("120" in a for a in out["arch_list"])

checks: dict = {}

# 1. bf16 matmul at grid scale
a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
b = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
c = (a @ b).float()
torch.cuda.synchronize()
checks["bf16_matmul_2048"] = {"ok": bool(torch.isfinite(c).all()), "sum": float(c.sum())}

# 2. sdpa causal -- the pinned backend (decision 6)
q = torch.randn(1, 12, 2048, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 12, 2048, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 12, 2048, 128, device="cuda", dtype=torch.bfloat16)
o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
torch.cuda.synchronize()
checks["sdpa_causal_2048"] = {"ok": bool(torch.isfinite(o).all()), "shape": list(o.shape)}

# 3. softmax + gather, the ops a scorer press leans on
s = torch.softmax(torch.randn(1, 12, 64, 2048, device="cuda", dtype=torch.float32), dim=-1)
idx = s.mean(dim=(0, 1, 2)).topk(256).indices
torch.cuda.synchronize()
checks["softmax_topk"] = {"ok": bool(idx.numel() == 256)}

# 4. an actual backward pass (ForesightKV training touches this on Machine A, but the
#    kernel-image failure mode is per-arch, so prove it here too)
x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
y = (x @ x.T).float().sum()
y.backward()
torch.cuda.synchronize()
checks["backward"] = {"ok": bool(x.grad is not None and torch.isfinite(x.grad).all())}

out["checks"] = checks
out["peak_alloc_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2)
out["all_ok"] = all(c["ok"] for c in checks.values()) and out["sm_120_in_arch_list"]

print(json.dumps(out, indent=2))
sys.exit(0 if out["all_ok"] else 1)
