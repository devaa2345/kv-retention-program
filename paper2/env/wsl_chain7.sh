#!/usr/bin/env bash
# GPU queue v7. Finishes M3-aware (killed 363 records short by a bad watcher threshold:
# 4000 hardcoded, but M3 has 11 arms x 2 budgets x 200 = 4400), writes its analysis, then
# runs the slot-predicate validity checks and the remaining two stages.
set -uo pipefail
cd /mnt/d/INNOCREW/Blockage/paper2
export HF_HOME=/mnt/d/hf_cache
export PYTHONUNBUFFERED=1
PY=/opt/p2venv/bin/python
LADDER=full_cache,null,random,floor_pos,oracle_causal,oracle_prescient
L=runs/nvidia/logs
mkdir -p "$L"

echo "########## [1/6] N8 M3 aware -- FINISH (resume, 4037/4400) ##########"; date +%H:%M
"$PY" run_grid_aware.py --model meta-llama/Llama-3.2-3B-Instruct --budgets 128,512 \
  --arms "$LADDER",snapkv,tova,expected_attn,keydiff,adakv_snapkv \
  --n 200 --out runs/nvidia/grid_M3_ledger_aware.jsonl 2>&1 \
  | grep -vE 'Loading weights|it/s' | tee "$L/m3_aware_run.log" || true
"$PY" analyze_aware.py --agnostic runs/nvidia/grid_M3_ledger_agnostic.jsonl \
  --aware runs/nvidia/grid_M3_ledger_aware.jsonl --label "N8 M3 Llama-3.2-3B aware" \
  --out runs/nvidia/analysis_aware_M3.json 2>&1 | tee "$L/m3_aware_analysis.log" || true

echo "########## [2/6] slot validity: leak population (M2 aware C=512) ##########"; date +%H:%M
"$PY" bench/slot_validity.py --mode leak --n 200 2>&1 \
  | grep -vE 'Loading weights|it/s' | tee "$L/slot_leak.log" || true

echo "########## [3/6] slot validity: completeness M2 ##########"; date +%H:%M
"$PY" bench/slot_validity.py --mode completeness --model Qwen/Qwen2.5-3B-Instruct --n 100 2>&1 \
  | grep -vE 'Loading weights|it/s' | tee "$L/slot_completeness_M2.log" || true

echo "########## [4/6] slot validity: completeness M3 ##########"; date +%H:%M
"$PY" bench/slot_validity.py --mode completeness --model meta-llama/Llama-3.2-3B-Instruct --n 100 2>&1 \
  | grep -vE 'Loading weights|it/s' | tee "$L/slot_completeness_M3.log" || true

echo "########## [5/6] SnapKV window dilution probe (M2, aware) ##########"; date +%H:%M
"$PY" bench/snapkv_window_probe.py 200 128,512 2>&1 \
  | grep -vE 'Loading weights|it/s' | tee "$L/snapkv_probe.log" || true

echo "########## [6/6] N9 M2 Qwen2.5-3B (2 KV heads) ##########"; date +%H:%M
"$PY" n9_decomposition.py --model Qwen/Qwen2.5-3B-Instruct \
  --budgets 16,32,64,128,256,512 --n 200 \
  --out runs/nvidia/n9_M2_ledger.jsonl 2>&1 \
  | grep -vE 'Loading weights|it/s' | tee "$L/n9_m2.log" || true
echo "########## chain complete ##########"; date +%H:%M
