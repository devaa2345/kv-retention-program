#!/bin/sh
cd /mnt/d/INNOCREW/Blockage/paper3
export HF_HOME=/mnt/d/hf_cache PYTHONPATH=/mnt/d/INNOCREW/Blockage/paper3:/mnt/d/INNOCREW/Blockage/paper2 TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1
rm -f runs/nvidia/natC_smoke.jsonl
/opt/p2venv/bin/python nat_expC.py --tag M3 --C 256 --n 1 --out runs/nvidia/natC_smoke.jsonl > out/natC_smoke.log 2>&1
