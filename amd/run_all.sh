#!/bin/bash
set -u
cd "/home/kxrx26/research test"
export PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=0
PY=/home/kxrx26/quant-rocm/bin/python
log(){ echo "[$(date +%H:%M:%S)] $*"; }
log "=== PHASE1 START ==="
$PY run_phase1.py 150 2>&1 | grep -viE "^loading|warning" || log "PHASE1 NONZERO EXIT"
log "=== PHASE2 START ==="
$PY run_phase2.py 150 2>&1 | grep -viE "^loading|warning" || log "PHASE2 NONZERO EXIT"
log "=== SWEEP START ==="
$PY run_sweep.py 50 2>&1 | grep -viE "^loading|warning" || log "SWEEP NONZERO EXIT"
log "=== CONTEXT CALIBRATION START ==="
$PY calibrate_context.py 2>&1 | grep -viE "^loading|warning" || log "CALIB NONZERO EXIT"
log "=== CONTEXT RUNS START ==="
$PY run_context.py 100 2>&1 | grep -viE "^loading|warning" || log "CONTEXT NONZERO EXIT"
log "=== REPORT ==="
$PY make_report.py 2>&1 | tail -5
log "=== ALL DONE ==="
