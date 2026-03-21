#!/usr/bin/env bash
# Setup script for KG+Text Experiment environment
# Tested on: Ubuntu 24.04 (g4dn.xlarge with Tesla T4)
set -euo pipefail

echo "=== KG+Text Experiment Environment Setup ==="

# 1. Install python3-venv (needed for venv creation)
echo "[1/5] Installing python3-venv..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv

# 2. Install NVIDIA driver (for GPU instances)
echo "[2/5] Installing NVIDIA driver..."
if lspci | grep -qi nvidia; then
    if ! nvidia-smi &>/dev/null; then
        sudo apt-get install -y -qq nvidia-driver-550
        sudo modprobe nvidia
        sudo modprobe nvidia-uvm
        echo "  NVIDIA driver installed."
    else
        echo "  NVIDIA driver already working."
    fi
    nvidia-smi
else
    echo "  No NVIDIA GPU detected, skipping driver install."
fi

# 3. Create Python venv (remove stale one if present)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"

echo "[3/5] Creating Python venv at ${VENV_DIR}..."
if [ -d "$VENV_DIR" ]; then
    # Check if existing venv is functional
    if ! "${VENV_DIR}/bin/python3" --version &>/dev/null; then
        echo "  Removing stale venv..."
        rm -rf "$VENV_DIR"
    else
        echo "  Venv already exists and is functional."
    fi
fi
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

# 4. Install Python dependencies
echo "[4/5] Installing Python packages (torch+CUDA, numpy, tqdm)..."
source "${VENV_DIR}/bin/activate"
pip install --quiet numpy tqdm torch --index-url https://download.pytorch.org/whl/cu124

# 5. Verify
echo "[5/5] Verifying installation..."
python -c "
import torch, numpy, tqdm
print(f'  Python:  {__import__(\"sys\").version.split()[0]}')
print(f'  PyTorch: {torch.__version__}')
print(f'  NumPy:   {numpy.__version__}')
print(f'  CUDA:    {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"})')
"

echo ""
echo "=== Setup complete ==="
echo "Activate with:  source ${VENV_DIR}/bin/activate"
echo "Smoke test:     python kg_text_experiment.py --models B \"B'\" --smoke --kg_as_text --exp 7a"
