#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/INNOCREW/Blockage/paper3
export HF_HOME=/mnt/d/hf_cache PYTHONPATH=/mnt/d/INNOCREW/Blockage/paper3:/mnt/d/INNOCREW/Blockage/paper2 TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1
PY=/opt/p2venv/bin/python
N=${N:-24}
echo "########## ChunkKV G2/G3  M2 Qwen2.5-3B ##########"
"$PY" chunkkv_admission.py --model Qwen/Qwen2.5-3B-Instruct --n $N --budgets 32,128,512 2>&1 | grep -vE 'Loading weights|it/s'
echo "########## ChunkKV G2/G3  M3 Llama-3.2-3B ##########"
"$PY" chunkkv_admission.py --model meta-llama/Llama-3.2-3B-Instruct --n $N --budgets 32,128,512 2>&1 | grep -vE 'Loading weights|it/s'
