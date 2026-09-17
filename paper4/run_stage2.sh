#!/bin/sh
# Paper 4 Stage 2 driver. Sequential, single-process, no pipes on any runner.
# verify (blocking) -> pilot c=40 both models -> pilot c=1 both models -> analysis.
cd /d/INNOCREW/Blockage/paper4
Q=Qwen/Qwen2.5-3B-Instruct
L=meta-llama/Llama-3.2-3B-Instruct

guard () {
  if MSYS_NO_PATHCONV=1 wsl -d Ubuntu-24.04 -e bash -lc "pgrep -f 'stage2_run.py|stage4_run.py'" >/dev/null 2>&1; then
    echo "ABORT: a runner is already running -- refusing to start a second writer"
    exit 1
  fi
}

run () {   # package model c
  guard
  log="out/_p4_$1_c$3_$(echo $2 | tr '/' '_').log"
  echo "=== $(date +%H:%M:%S)  $1  $2  c=$3 ==="
  MSYS_NO_PATHCONV=1 ./wsl.sh stage2_run.py "$1" --model "$2" --c $3 --n 50 > "$log" 2>&1
  rc=$?
  tail -2 "$log"
  if [ $rc -ne 0 ]; then echo "ABORT: $1 $2 c=$3 exited $rc"; exit $rc; fi
}

run verify "$Q" "40 1"
run verify "$L" "40 1"
MSYS_NO_PATHCONV=1 ./wsl.sh stage2_analyse.py verify > out/_p4_verify_analysis.log 2>&1
if [ $? -ne 0 ]; then echo "=== VERIFY GATE FAILED -- no generation run ==="; exit 3; fi
echo "=== VERIFY GATE PASS $(date +%H:%M:%S) ==="

run pilot "$Q" 40
run pilot "$L" 40
run pilot "$Q" 1
run pilot "$L" 1
MSYS_NO_PATHCONV=1 ./wsl.sh stage2_analyse.py pilot > out/_p4_pilot_analysis.log 2>&1
echo "=== STAGE 2 COMPLETE $(date +%H:%M:%S) ==="
