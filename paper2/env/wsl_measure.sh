#!/usr/bin/env bash
# Re-measure §2 throughput under the pinned WSL toolchain (plan v2 §1 / v1 §10).
# 200 timed records per admitted model at 2048 prefill, sdpa (decision 6), bf16, batch 1.
set -euo pipefail
cd /mnt/d/INNOCREW/Blockage/paper2
export HF_HOME=/mnt/d/hf_cache
PY=/opt/p2venv/bin/python

echo "=== env lock ==="
"$PY" -m harness.envlock

for M in Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3.2-3B-Instruct; do
  echo "=== $M (sdpa, 200 records) ==="
  "$PY" bench/bench_throughput.py --model "$M" --attn sdpa --skip-attn --iters 200 --warmup 5
done
