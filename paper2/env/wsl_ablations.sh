#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/INNOCREW/Blockage/paper2
export HF_HOME=/mnt/d/hf_cache
PY=/opt/p2venv/bin/python
for M in Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3.2-3B-Instruct; do
  echo "=== $M ==="
  "$PY" stage4_ablations.py --model "$M" --n 100
done
