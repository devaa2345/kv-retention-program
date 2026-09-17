#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/INNOCREW/Blockage/paper2
export HF_HOME=/mnt/d/hf_cache
PY=/opt/p2venv/bin/python
echo "########## N4a  M2 Qwen2.5-3B ##########"
"$PY" n4a_admission_gates.py --model Qwen/Qwen2.5-3B-Instruct --n 24 --budgets 32,128,512 2>&1 | grep -vE 'Loading weights|it/s' || true
echo "########## N4a  M3 Llama-3.2-3B ##########"
"$PY" n4a_admission_gates.py --model meta-llama/Llama-3.2-3B-Instruct --n 24 --budgets 32,128,512 2>&1 | grep -vE 'Loading weights|it/s' || true
