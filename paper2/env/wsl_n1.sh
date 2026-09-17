#!/usr/bin/env bash
# N1 - Stage 3 random-span control, the last hard gate.
# Same instance set Stage 5 validated on. Admitted budgets only (M2: b-2 is VOID).
set -uo pipefail
cd /mnt/d/INNOCREW/Blockage/paper2
export HF_HOME=/mnt/d/hf_cache
PY=/opt/p2venv/bin/python

echo "########## M2 Qwen2.5-3B  (C = 32,64,128,256,512) ##########"
"$PY" n1_random_span_control.py --model Qwen/Qwen2.5-3B-Instruct --n 48 \
  --budgets 32,64,128,256,512 2>&1 | grep -vE 'Loading weights|it/s' || true

echo "########## M3 Llama-3.2-3B  (C = 16,32,64,128,256,512) ##########"
"$PY" n1_random_span_control.py --model meta-llama/Llama-3.2-3B-Instruct --n 48 \
  --budgets 16,32,64,128,256,512 2>&1 | grep -vE 'Loading weights|it/s' || true
