#!/usr/bin/env bash
# GPU queue v5 -- same order as v4, but every stage tees to its own log under runs/nvidia/logs/
# so results are durable instead of sitting in a pipe buffer until the chain ends.
set -uo pipefail
cd /mnt/d/INNOCREW/Blockage/paper2
export HF_HOME=/mnt/d/hf_cache
export PYTHONUNBUFFERED=1
PY=/opt/p2venv/bin/python
LADDER=full_cache,null,random,floor_pos,oracle_causal,oracle_prescient
L=runs/nvidia/logs
mkdir -p "$L"

echo "########## [1/4] KeyDiff aware-leak investigation (M2, C=512, n=200) ##########"; date +%H:%M
"$PY" bench/keydiff_leak_investigation.py 512 200 2>&1 \
  | grep -vE 'Loading weights|it/s' | tee "$L/keydiff_leak.log" || true

echo "########## [2/4] N8 M3 Llama-3.2-3B aware (finish) ##########"; date +%H:%M
"$PY" run_grid_aware.py --model meta-llama/Llama-3.2-3B-Instruct --budgets 128,512 \
  --arms "$LADDER",snapkv,tova,expected_attn,keydiff,adakv_snapkv \
  --n 200 --out runs/nvidia/grid_M3_ledger_aware.jsonl 2>&1 \
  | grep -vE 'Loading weights|it/s' | tee "$L/m3_aware_run.log" || true
"$PY" analyze_aware.py --agnostic runs/nvidia/grid_M3_ledger_agnostic.jsonl \
  --aware runs/nvidia/grid_M3_ledger_aware.jsonl --label "N8 M3 Llama-3.2-3B aware" \
  --out runs/nvidia/analysis_aware_M3.json 2>&1 | tee "$L/m3_aware_analysis.log" || true

echo "########## [3/4] SnapKV window dilution probe (M2, aware) ##########"; date +%H:%M
"$PY" bench/snapkv_window_probe.py 200 128,512 2>&1 \
  | grep -vE 'Loading weights|it/s' | tee "$L/snapkv_probe.log" || true

echo "########## [4/4] N9 M2 Qwen2.5-3B (2 KV heads) ##########"; date +%H:%M
"$PY" n9_decomposition.py --model Qwen/Qwen2.5-3B-Instruct \
  --budgets 16,32,64,128,256,512 --n 200 \
  --out runs/nvidia/n9_M2_ledger.jsonl 2>&1 \
  | grep -vE 'Loading weights|it/s' | tee "$L/n9_m2.log" || true
echo "########## chain complete ##########"; date +%H:%M
