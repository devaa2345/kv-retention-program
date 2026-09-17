#!/bin/bash
cd "/home/kxrx26/research test"
PY=/home/kxrx26/quant-rocm/bin/python
export PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=0
log(){ echo "[$(date +%H:%M:%S)] $*"; }
busy(){ pgrep -f "python run_phase2.py|python run_phase1_8bit.py|python run_context.py|python run_rho.py|python run_band.py" > /dev/null; }
while busy; do sleep 20; done
log "=== PHASE2 8-BIT (residual test) ==="
$PY run_phase2_8bit.py 150 2>&1 | grep -viE "^loading|warning" | tail -5
log "=== POST-HOC ==="
$PY run_posthoc.py 2>&1 | grep -viE "^loading|warning"
log "=== DISTRACTOR ABLATION ==="
$PY run_ablation_distractor.py 50 2>&1 | grep -viE "^loading|warning"
log "=== REPORTS ==="
$PY make_report.py 2>&1 | tail -3
$PY make_agreement.py 2>&1 | head -2
log "=== PIPELINE COMPLETE ==="
