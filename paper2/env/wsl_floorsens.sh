#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/INNOCREW/Blockage/paper2
export HF_HOME=/mnt/d/hf_cache PYTHONUNBUFFERED=1
PY=/opt/p2venv/bin/python
L=runs/nvidia/logs; mkdir -p "$L"
echo "########## floor sensitivity M2 (Reading B: keep C) ##########"; date +%H:%M
"$PY" bench/floor_sensitivity.py --model Qwen/Qwen2.5-3B-Instruct --tag M2 \
  --budgets 32,64,128,256,512 --n 200 --out runs/nvidia/floor_readingC_M2.jsonl 2>&1 \
  | grep -vE 'Loading weights|it/s' | tee "$L/floorsens_M2.log" || true
echo "########## floor sensitivity M3 (Reading B: keep C) ##########"; date +%H:%M
"$PY" bench/floor_sensitivity.py --model meta-llama/Llama-3.2-3B-Instruct --tag M3 \
  --budgets 16,32,64,128,256,512 --n 200 --out runs/nvidia/floor_readingC_M3.jsonl 2>&1 \
  | grep -vE 'Loading weights|it/s' | tee "$L/floorsens_M3.log" || true
echo "########## done ##########"; date +%H:%M
