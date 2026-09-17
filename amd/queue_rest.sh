#!/bin/bash
cd "/home/kxrx26/research test"
export PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=0
PY=/home/kxrx26/quant-rocm/bin/python
lg(){ echo "[$(date +%H:%M:%S)] $*"; }
nrun(){ ps -eo cmd | grep -c "bin/python run_[a-z]" ; }
# hold until a slot frees (<=3 heavy jobs), then run rho, posthoc, ablation, reports
while [ "$(nrun)" -gt 3 ]; do sleep 20; done
lg "RHO"; $PY run_rho.py 2>&1 | grep -viE "^loading|warning"
while [ "$(nrun)" -gt 3 ]; do sleep 20; done
lg "ABLATION"; $PY run_ablation_distractor.py 50 2>&1 | grep -viE "^loading|warning"
while [ "$(nrun)" -gt 2 ]; do sleep 20; done
lg "POSTHOC"; $PY run_posthoc.py 2>&1 | grep -viE "^loading|warning"
while [ "$(nrun)" -gt 1 ]; do sleep 20; done
lg "REPORTS"; $PY make_report.py 2>&1 | tail -2; $PY make_agreement.py 2>&1 | head -2
lg "PIPELINE COMPLETE"
