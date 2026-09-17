#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/INNOCREW/Blockage/paper2
export HF_HOME=/mnt/d/hf_cache
PY=/opt/p2venv/bin/python

echo "########## BATCH-vs-BATCH-1 BIT-IDENTITY CHECK (v1 6.1) ##########"
for M in Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3.2-3B-Instruct; do
  "$PY" bench/batch_identity_check.py --model "$M" --n 25 2>&1 \
    | grep -E '^(Qwen|meta|  )' || true
done

echo "########## STAGE 4 ANCHORS  n=200, mean over H, N=40, max_new=32 ##########"
for M in Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3.2-3B-Instruct; do
  "$PY" stage4_competence_sweep.py --model "$M" --n 200 --records 40 2>&1 \
    | grep -E '^(Qwen|meta|  [0-9]|  N|wrote)'
done
