#!/bin/sh
cd /mnt/d/INNOCREW/Blockage/paper3
export HF_HOME=/mnt/d/hf_cache PYTHONPATH=/mnt/d/INNOCREW/Blockage/paper3:/mnt/d/INNOCREW/Blockage/paper2 TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1
/opt/p2venv/bin/python nat_gate.py --tag M3 --only_full --levels 3,5 --slack 128 --out runs/nvidia/nat_gate_v2rerun_M3.jsonl > out/nat_rerun_M3.log 2>&1
