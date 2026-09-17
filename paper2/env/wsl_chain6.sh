#!/usr/bin/env bash
# GPU queue v6 -- picks up AFTER M3-aware. Slot-predicate validity check first (human-requested),
# then the remaining two stages. Everything resumes from disk.
set -uo pipefail
cd /mnt/d/INNOCREW/Blockage/paper2
export HF_HOME=/mnt/d/hf_cache
export PYTHONUNBUFFERED=1
PY=/opt/p2venv/bin/python
L=runs/nvidia/logs
mkdir -p "$L"

echo "########## [1/4] slot-predicate validity: leak population (M2 aware C=512) ##########"; date +%H:%M
"$PY" bench/slot_validity.py --mode leak --n 200 2>&1 \
  | grep -vE 'Loading weights|it/s' | tee "$L/slot_leak.log" || true

echo "########## [2/4] slot-predicate validity: completeness M2 + M3 ##########"; date +%H:%M
"$PY" bench/slot_validity.py --mode completeness --model Qwen/Qwen2.5-3B-Instruct --n 100 2>&1 \
  | grep -vE 'Loading weights|it/s' | tee "$L/slot_completeness_M2.log" || true
"$PY" bench/slot_validity.py --mode completeness --model meta-llama/Llama-3.2-3B-Instruct --n 100 2>&1 \
  | grep -vE 'Loading weights|it/s' | tee "$L/slot_completeness_M3.log" || true

echo "########## [3/4] SnapKV window dilution probe (M2, aware) ##########"; date +%H:%M
"$PY" bench/snapkv_window_probe.py 200 128,512 2>&1 \
  | grep -vE 'Loading weights|it/s' | tee "$L/snapkv_probe.log" || true

echo "########## [4/4] N9 M2 Qwen2.5-3B (2 KV heads) ##########"; date +%H:%M
"$PY" n9_decomposition.py --model Qwen/Qwen2.5-3B-Instruct \
  --budgets 16,32,64,128,256,512 --n 200 \
  --out runs/nvidia/n9_M2_ledger.jsonl 2>&1 \
  | grep -vE 'Loading weights|it/s' | tee "$L/n9_m2.log" || true
echo "########## chain complete ##########"; date +%H:%M
