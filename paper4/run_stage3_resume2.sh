#!/bin/sh
# Paper 4 Stage 3, second resume. Runs c=19 and c=40 on both models AS FROZEN (10 arms, n=50),
# then closes the M3 c=8 gap left by the confound halt. Per-cost reports after each level.
# Sequential, single process, no pipes. Confound watch armed for all four methods.
cd /d/INNOCREW/Blockage/paper4
Q=Qwen/Qwen2.5-3B-Instruct
L=meta-llama/Llama-3.2-3B-Instruct

guard () {
  if MSYS_NO_PATHCONV=1 wsl -d Ubuntu-24.04 -e bash -lc "pgrep -f 'stage[0-9]_run.py|stage3_preflight.py'" >/dev/null 2>&1; then
    echo "ABORT: a runner is already running -- refusing to start a second writer"; exit 1
  fi
}

run () {  # model c
  guard
  log="out/_p4_s3_c$2_$(echo $1 | tr '/' '_').log"
  echo "=== $(date +%H:%M:%S) grid $1 c=$2 ==="
  MSYS_NO_PATHCONV=1 ./wsl.sh stage3_run.py --model "$1" --c $2 --n 50 > "$log" 2>&1
  rc=$?
  grep -E "CONFOUND WATCH" "$log"
  grep -E "CELL DONE|CELL INCOMPLETE|DONE " "$log" | tail -6
  if [ $rc -ne 0 ]; then echo "ABORT: grid $1 c=$2 exited $rc"; exit $rc; fi
}

report () {  # c
  MSYS_NO_PATHCONV=1 ./wsl.sh stage3_budget_report.py --c "$1" > "out/_p4_s3_level_c$1.log" 2>&1
  echo "=== LEVEL REPORT c=$1 written ==="
}

run "$Q" 19
run "$L" 19
report 19

run "$Q" 40
run "$L" 40
report 40

# gap left by the confound halt: M3 c=8 never ran. Flagged, run last.
run "$L" 8
report 8

MSYS_NO_PATHCONV=1 ./wsl.sh stage3_analyse.py control > out/_p4_s3_c1_diagnostic.log 2>&1
MSYS_NO_PATHCONV=1 ./wsl.sh stage3_analyse.py report > out/_p4_s3_report.log 2>&1
echo "=== STAGE 3 COMPLETE $(date +%H:%M:%S) ==="
