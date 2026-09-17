#!/bin/bash
cd "/home/kxrx26/research test"
while pgrep -f "python run_phase2.py" > /dev/null; do sleep 15; done
echo "[$(date +%H:%M:%S)] phase2 done - running rho manipulation check"
PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=0 /home/kxrx26/quant-rocm/bin/python run_rho.py 2>&1 | grep -viE "^loading|warning"
echo "[$(date +%H:%M:%S)] rho done"
