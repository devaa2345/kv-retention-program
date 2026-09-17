#!/usr/bin/env bash
# Machine N toolchain install, WSL2 Ubuntu 24.04.
# Run as: wsl -d Ubuntu-24.04 -u root -- bash /mnt/d/INNOCREW/Blockage/paper2/env/wsl_install.sh
#
# Written as a file rather than passed on the command line because Git-Bash on the Windows
# side rewrites bare "https://host/path" arguments into Windows paths before wsl.exe sees
# them, which silently corrupted the first attempt (--index-url became "download.pytorch.orgwhlcu128").
set -euo pipefail

VENV=/opt/p2venv
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

echo "=== torch (cu128, sm_120) ==="
"$VENV/bin/pip" install --index-url "$TORCH_INDEX" torch

echo "=== kvpress 0.5.4 (pins transformers <5.3,>=4.56.0) ==="
"$VENV/bin/pip" install "kvpress==0.5.4"

echo "=== supporting packages ==="
"$VENV/bin/pip" install "numpy<3" "scipy<2" pandas datasets accelerate

echo "=== versions ==="
"$VENV/bin/pip" list 2>/dev/null | grep -Ei '^(torch|transformers|kvpress|accelerate|numpy|scipy|pandas|datasets|tokenizers|safetensors) '
