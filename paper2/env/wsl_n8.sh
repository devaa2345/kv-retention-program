#!/usr/bin/env bash
# N8 - query-AWARE calibration sub-grid, b1 (C=128) and b3 (C=512), both models.
set -uo pipefail
cd /mnt/d/INNOCREW/Blockage/paper2
export HF_HOME=/mnt/d/hf_cache
PY=/opt/p2venv/bin/python
LADDER=full_cache,null,random,floor_pos,oracle_causal,oracle_prescient
echo "########## N8 M2 Qwen2.5-3B (aware) ##########"; date +%H:%M
"$PY" run_grid_aware.py --model Qwen/Qwen2.5-3B-Instruct --budgets 128,512 \
  --arms "$LADDER",snapkv,expected_attn,keydiff,adakv_snapkv \
  --n 200 --out runs/nvidia/grid_M2_ledger_aware.jsonl 2>&1 | grep -vE 'Loading weights|it/s' || true
echo "########## N8 M3 Llama-3.2-3B (aware) ##########"; date +%H:%M
"$PY" run_grid_aware.py --model meta-llama/Llama-3.2-3B-Instruct --budgets 128,512 \
  --arms "$LADDER",snapkv,tova,expected_attn,keydiff,adakv_snapkv \
  --n 200 --out runs/nvidia/grid_M3_ledger_aware.jsonl 2>&1 | grep -vE 'Loading weights|it/s' || true
date +%H:%M
