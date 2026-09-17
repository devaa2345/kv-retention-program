#!/usr/bin/env bash
# GPU queue v3. Order per the user's standing sequence, with two additions inserted:
# a smoke step ahead of the fragmentation re-run, and the SnapKV dilution probe after
# M3-aware. Every stage resumes from disk via key_digest / line-count dedup.
set -uo pipefail
cd /mnt/d/INNOCREW/Blockage/paper2
export HF_HOME=/mnt/d/hf_cache
PY=/opt/p2venv/bin/python
LADDER=full_cache,null,random,floor_pos,oracle_causal,oracle_prescient

echo "########## [1/7] N9 M3 Llama-3.2-3B (resume) ##########"; date +%H:%M
"$PY" n9_decomposition.py --model meta-llama/Llama-3.2-3B-Instruct \
  --budgets 16,32,64,128,256,512 --n 200 \
  --out runs/nvidia/n9_M3_ledger.jsonl 2>&1 | grep -vE 'Loading weights|it/s' || true

echo "########## [2/7] fragmentation SMOKE (N=2, C=512) ##########"; date +%H:%M
"$PY" bench/fragmentation_units.py Qwen/Qwen2.5-3B-Instruct 2 512 \
  2>&1 | grep -vE 'Loading weights|it/s'
if [ ${PIPESTATUS[0]} -ne 0 ]; then
  echo "!!! SMOKE FAILED -- skipping both fragmentation runs, continuing queue"
  SKIP_FRAG=1
else
  SKIP_FRAG=0
fi

if [ "$SKIP_FRAG" -eq 0 ]; then
  echo "########## [3/7] fragmentation units M2 (N=200) ##########"; date +%H:%M
  "$PY" bench/fragmentation_units.py Qwen/Qwen2.5-3B-Instruct 200 16,32,64,128,256,512 \
    2>&1 | grep -vE 'Loading weights|it/s' || true
  echo "########## [4/7] fragmentation units M3 (N=200) ##########"; date +%H:%M
  "$PY" bench/fragmentation_units.py meta-llama/Llama-3.2-3B-Instruct 200 16,32,64,128,256,512 \
    2>&1 | grep -vE 'Loading weights|it/s' || true
fi

echo "########## [5/7] N8 M3 Llama-3.2-3B aware (finish) ##########"; date +%H:%M
"$PY" run_grid_aware.py --model meta-llama/Llama-3.2-3B-Instruct --budgets 128,512 \
  --arms "$LADDER",snapkv,tova,expected_attn,keydiff,adakv_snapkv \
  --n 200 --out runs/nvidia/grid_M3_ledger_aware.jsonl 2>&1 | grep -vE 'Loading weights|it/s' || true
"$PY" analyze_aware.py --agnostic runs/nvidia/grid_M3_ledger_agnostic.jsonl \
  --aware runs/nvidia/grid_M3_ledger_aware.jsonl --label "N8 M3 Llama-3.2-3B aware" \
  --out runs/nvidia/analysis_aware_M3.json 2>&1 | grep -v 'it/s' || true

echo "########## [6/7] SnapKV window dilution probe (M2, aware) ##########"; date +%H:%M
"$PY" bench/snapkv_window_probe.py 200 128,512 2>&1 | grep -vE 'Loading weights|it/s' || true

echo "########## [7/7] N9 M2 Qwen2.5-3B (2 KV heads) ##########"; date +%H:%M
"$PY" n9_decomposition.py --model Qwen/Qwen2.5-3B-Instruct \
  --budgets 16,32,64,128,256,512 --n 200 \
  --out runs/nvidia/n9_M2_ledger.jsonl 2>&1 | grep -vE 'Loading weights|it/s' || true
echo "########## chain complete ##########"; date +%H:%M
