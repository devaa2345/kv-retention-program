#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/INNOCREW/Blockage/paper2
export HF_HOME=/mnt/d/hf_cache PYTHONUNBUFFERED=1
PY=/opt/p2venv/bin/python
L=runs/nvidia/logs
echo "########## item 4 RERUN: both guards relaxed ##########"; date +%H:%M
"$PY" bench/dhead_guard_sensitivity.py 200 2>&1 \
  | grep -vE 'Loading weights|it/s' | tee "$L/dhead_guard2.log" || true
echo "########## done ##########"; date +%H:%M
