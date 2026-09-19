#!/bin/sh
cd /mnt/d/INNOCREW/Blockage/paper3
export HF_HOME=/mnt/d/hf_cache PYTHONPATH=/mnt/d/INNOCREW/Blockage/paper3:/mnt/d/INNOCREW/Blockage/paper2 TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1
/opt/p2venv/bin/python nat_expC.py --tag M3 --C 512 >> out/natC_M3_C512.log 2>&1
echo "BUDGET_CELL_DONE M3 512" >> out/natC_progress.log
echo "BUDGET_DONE 512 (run stopped by user request; C=1024 not run)" >> out/natC_progress.log
