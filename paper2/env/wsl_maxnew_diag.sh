#!/usr/bin/env bash
# DIAGNOSTIC ONLY. v1 §5.4 pins max_new_tokens=12 for Task B. This measures how much of the
# competence-anchor failure is a *format* artifact of that pin (models spending all 12 tokens
# on a restatement preamble) rather than a capability limit. It does not change the pin.
set -euo pipefail
cd /mnt/d/INNOCREW/Blockage/paper2
export HF_HOME=/mnt/d/hf_cache
PY=/opt/p2venv/bin/python
for M in meta-llama/Llama-3.2-3B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-1.5B-Instruct; do
  "$PY" stage4_competence_sweep.py --model "$M" --n 40 --records 96 --max-new 32 2>&1 \
    | grep -E '^(meta|Qwen|  [0-9]|  N|wrote)'
done
