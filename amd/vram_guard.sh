#!/bin/bash
# Halts the suite if VRAM use threatens the desktop. Checkpointing means a halt loses
# at most one in-flight run; the suite resumes from the JSONL.
LIMIT=${1:-21000000000}    # 21 GB of 25.75 GB
cd "/home/kxrx26/research test"
while pgrep -f run_all.sh > /dev/null; do
  U=$(rocm-smi --showmeminfo vram 2>/dev/null | grep -i "VRAM Total Used" | grep -oE "[0-9]+" | tail -1)
  if [ -n "$U" ] && [ "$U" -gt "$LIMIT" ]; then
    echo "[$(date +%H:%M:%S)] VRAM $U > $LIMIT - HALTING SUITE to protect desktop"
    pkill -f run_phase; pkill -f run_sweep; pkill -f run_context; pkill -f run_all.sh
    break
  fi
  sleep 20
done
echo "[$(date +%H:%M:%S)] vram guard exiting (suite finished or halted)"
