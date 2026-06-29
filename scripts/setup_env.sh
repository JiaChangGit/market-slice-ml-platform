#!/usr/bin/env bash
set -euo pipefail

echo "=== market-slice-ml-platform environment setup ==="

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y \
    python3.12 python3.12-venv python3.12-dev \
    build-essential git curl wget ca-certificates graphviz libgomp1 language-pack-en
fi

if ! command -v python3.12 >/dev/null 2>&1; then
  echo "Python 3.12 is not installed. Install it with your OS package manager or uv."
  exit 1
fi

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel "setuptools<82"
python -m pip install -e ".[dev]"

if ! python -c "import torch" >/dev/null 2>&1; then
  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

echo "=== Optional PyG acceleration install ==="
bash scripts/install_pyg.sh || echo "WARNING: PyG acceleration unavailable; using pure PyTorch fallback."
python scripts/detect_ibkr_host.py || true
python scripts/doctor.py
bash scripts/run_all_checks.sh --phase0
echo "Environment setup complete. Activate with: source .venv/bin/activate"
