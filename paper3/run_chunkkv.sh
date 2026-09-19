#!/bin/sh
cd /mnt/d/INNOCREW/Blockage/paper3
export HF_HOME=/mnt/d/hf_cache PYTHONPATH=/mnt/d/INNOCREW/Blockage/paper3:/mnt/d/INNOCREW/Blockage/paper2 TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1
N=${1:-100}
/opt/p2venv/bin/python chunkkv_run.py --model Qwen/Qwen2.5-3B-Instruct --n $N >> out/chunkkv_M2.log 2>&1
/opt/p2venv/bin/python chunkkv_run.py --model meta-llama/Llama-3.2-3B-Instruct --n $N >> out/chunkkv_M3.log 2>&1
echo "CHUNKKV_DONE" >> out/chunkkv_progress.log
