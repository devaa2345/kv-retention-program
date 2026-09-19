#!/bin/sh
cd /mnt/d/INNOCREW/Blockage/paper3
export HF_HOME=/mnt/d/hf_cache PYTHONPATH=/mnt/d/INNOCREW/Blockage/paper3:/mnt/d/INNOCREW/Blockage/paper2 TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1
for C in 256 512; do for t in M2 M3; do
  /opt/p2venv/bin/python nat_expC.py --tag $t --C $C --arms chunkkv --out runs/nvidia/natCK_${t}_C${C}.jsonl >> out/natCK_${t}_C${C}.log 2>&1
  echo "NATCK_DONE $t $C" >> out/natCK_progress.log
done; done
