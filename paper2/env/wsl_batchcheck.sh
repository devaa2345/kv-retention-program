#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/INNOCREW/Blockage/paper2
export HF_HOME=/mnt/d/hf_cache
PY=/opt/p2venv/bin/python
for M in Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3.2-3B-Instruct; do
  "$PY" bench/batch_identity_check.py --model "$M" --n 25 2>&1 \
    | grep -vE 'Loading weights|it/s' || true   # exit 1 means 'differs', not a script failure
done
