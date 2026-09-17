#!/usr/bin/env bash
# Post-decision re-measurement: competence anchors (H-variant mean, max_new=32) and
# 200-record throughput at 32 decode tokens.
set -euo pipefail
cd /mnt/d/INNOCREW/Blockage/paper2
export HF_HOME=/mnt/d/hf_cache
PY=/opt/p2venv/bin/python

echo "########## COMPETENCE ANCHORS (N=80, H=4, max_new=32, mean over H variants) ##########"
for M in Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3.2-3B-Instruct; do
  "$PY" stage4_competence_sweep.py --model "$M" --n 40 --records 80 2>&1 \
    | grep -E '^(Qwen|meta|  [0-9]|  N|wrote)'
done

echo "########## THROUGHPUT 200 records @ 32 decode tokens ##########"
for M in Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3.2-3B-Instruct; do
  "$PY" bench/bench_throughput.py --model "$M" --attn sdpa --skip-attn --iters 200 --warmup 5 2>&1 \
    | grep -E 's/record|revision'
done
