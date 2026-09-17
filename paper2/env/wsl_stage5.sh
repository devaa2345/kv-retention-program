#!/usr/bin/env bash
# Stage 5 ladder validation on the v2 interleaved layout, n=48, admitted budgets only.
# M2: b-2 (C=16) is VOID (k_gold 19 > 16) and is excluded before any run.
set -uo pipefail
cd /mnt/d/INNOCREW/Blockage/paper2
export HF_HOME=/mnt/d/hf_cache
PY=/opt/p2venv/bin/python

echo "########## M2 Qwen2.5-3B  (C = 32,64,128,256,512) ##########"
"$PY" stage5_ladder_validation.py --model Qwen/Qwen2.5-3B-Instruct --n 48 \
  --budgets 32,64,128,256,512 2>&1 | grep -vE 'Loading weights|it/s' || true

echo "########## M3 Llama-3.2-3B  (C = 16,32,64,128,256,512) ##########"
"$PY" stage5_ladder_validation.py --model meta-llama/Llama-3.2-3B-Instruct --n 48 \
  --budgets 16,32,64,128,256,512 2>&1 | grep -vE 'Loading weights|it/s' || true
