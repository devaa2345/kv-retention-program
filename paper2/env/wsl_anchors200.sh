#!/usr/bin/env bash
# Stage 4 anchors, n=200, TRUE mean over all H query variants, max_new=32, N=40.
set -euo pipefail
cd /mnt/d/INNOCREW/Blockage/paper2
export HF_HOME=/mnt/d/hf_cache
PY=/opt/p2venv/bin/python
for M in Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3.2-3B-Instruct; do
  "$PY" stage4_competence_sweep.py --model "$M" --n 200 --records 40 2>&1 \
    | grep -E '^(Qwen|meta|  [0-9]|  N|wrote)' || true
done
