#!/usr/bin/env bash
# Items 1-4. Item 5 (LU-KV) is BLOCKED: env/lukv_curves/ absent.
set -uo pipefail
cd /mnt/d/INNOCREW/Blockage/paper2
export HF_HOME=/mnt/d/hf_cache
export PYTHONUNBUFFERED=1
PY=/opt/p2venv/bin/python
L=runs/nvidia/logs
mkdir -p "$L"

echo "########## [1/5] item 3: KeyDiff exactly-zero subgroup ##########"; date +%H:%M
"$PY" bench/keydiff_zero_subgroup.py 200 2>&1 \
  | grep -vE 'Loading weights|it/s' | tee "$L/keydiff_zero.log" || true

echo "########## [2/5] item 4: Delta_head guard sensitivity M2 C=32 ##########"; date +%H:%M
"$PY" bench/dhead_guard_sensitivity.py 200 2>&1 \
  | grep -vE 'Loading weights|it/s' | tee "$L/dhead_guard.log" || true

echo "########## [3/5] items 1+2: slot-aware completeness M2 ##########"; date +%H:%M
"$PY" bench/slot_aware_completeness.py --model Qwen/Qwen2.5-3B-Instruct --tag M2 --n 200 2>&1 \
  | grep -vE 'Loading weights|it/s' | tee "$L/slotaware_M2.log" || true

echo "########## [4/5] items 1+2: slot-aware completeness M3 ##########"; date +%H:%M
"$PY" bench/slot_aware_completeness.py --model meta-llama/Llama-3.2-3B-Instruct --tag M3 --n 200 2>&1 \
  | grep -vE 'Loading weights|it/s' | tee "$L/slotaware_M3.log" || true

echo "########## [5/5] LU-KV admission (runs only if curves present) ##########"; date +%H:%M
if [ -d env/lukv_curves ]; then
  echo "curves found -- G2/G3 admission would run here"
else
  echo "BLOCKED: env/lukv_curves/ absent -- LU-KV not registered, not admitted, not run"
fi
echo "########## chain complete ##########"; date +%H:%M
