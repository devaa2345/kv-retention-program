#!/bin/bash
# Runs one job under supervision, restarting it if the GPU faults. All work is checkpointed,
# so a restart resumes from the last committed row. Guards against hipErrorLaunchFailure,
# which killed all processes at 04:02 when four ROCm contexts were live at once.
cd "/home/kxrx26/research test"
export PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/kxrx26/quant-rocm/bin/python
SCRIPT="$1"; ARGS="$2"; FILE="$3"; TARGET="$4"; LOG="$5"
for attempt in $(seq 1 40); do
  N=$(wc -l < "$FILE" 2>/dev/null || echo 0)
  if [ "$N" -ge "$TARGET" ]; then echo "[$(date +%H:%M:%S)] COMPLETE $SCRIPT ($N/$TARGET)"; break; fi
  echo "[$(date +%H:%M:%S)] attempt $attempt: $SCRIPT at $N/$TARGET"
  $PY $SCRIPT $ARGS >> "$LOG" 2>&1
  rc=$?
  N2=$(wc -l < "$FILE" 2>/dev/null || echo 0)
  if [ "$N2" -ge "$TARGET" ]; then echo "[$(date +%H:%M:%S)] COMPLETE $SCRIPT ($N2/$TARGET)"; break; fi
  echo "[$(date +%H:%M:%S)] $SCRIPT exited rc=$rc at $N2 rows; cooling 45s then resuming"
  sleep 45
done
