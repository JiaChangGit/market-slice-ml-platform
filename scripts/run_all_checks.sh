#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---full}"
export NO_NETWORK=1

python scripts/check_cuda.py
python scripts/verify_install.py
python scripts/verify_torch_stack.py
python scripts/doctor.py
python scripts/check_data_sources.py
python scripts/check_no_trading_api.py

if [[ "${MODE}" == "--phase0" ]]; then
  echo "PHASE0 checks passed."
  exit 0
fi

ruff check src/ tests/ scripts/
mypy src/
pytest tests/unit/ -q --tb=short

if [[ "${MODE}" == "--full" ]]; then
  pytest tests/property/ -q --tb=short
  pytest tests/integration/ -q --tb=short
  pytest tests/e2e/ -q --tb=short
  python scripts/smoke_local_pipeline.py
fi
echo "All checks passed."
