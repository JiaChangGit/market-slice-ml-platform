#!/usr/bin/env bash
set -euo pipefail

echo "=== PyG installer ==="

TORCH_VER=$(python - <<'PY'
try:
    import torch
    print(torch.__version__.split("+")[0])
except Exception:
    print("")
PY
)

if [[ -z "${TORCH_VER}" ]]; then
  echo "ERROR: PyTorch is not installed. Install PyTorch first."
  exit 1
fi

TORCH_MINOR=$(python - <<PY
parts="${TORCH_VER}".split(".")
print(f"{parts[0]}.{parts[1]}.0")
PY
)

TORCH_CUDA=$(python - <<'PY'
import torch
print(torch.version.cuda or "")
PY
)

if [[ -z "${TORCH_CUDA}" ]]; then
  CUDA_TAG="cpu"
else
  MAJOR=$(echo "${TORCH_CUDA}" | cut -d. -f1)
  MINOR=$(echo "${TORCH_CUDA}" | cut -d. -f2)
  CUDA_TAG="cu${MAJOR}${MINOR}"
fi

WHL_URL="https://data.pyg.org/whl/torch-${TORCH_MINOR}+${CUDA_TAG}.html"
echo "PyTorch      : ${TORCH_VER}"
echo "torch CUDA   : ${TORCH_CUDA:-cpu}"
echo "PyG wheel URL: ${WHL_URL}"

if ! python -m pip install torch_geometric; then
  echo "WARNING: torch_geometric could not be installed; pure PyTorch GNN fallback remains available."
  exit 0
fi

if ! python -m pip install pyg_lib torch_scatter torch_sparse -f "${WHL_URL}"; then
  echo "WARNING: optional PyG acceleration wheels are unavailable; pure PyTorch fallback remains available."
fi

python - <<'PY'
import torch
try:
    import torch_geometric
except Exception as exc:
    print(f"WARNING: torch_geometric verification failed: {exc}")
else:
    print(
        f"OK torch={torch.__version__}, torch_cuda={torch.version.cuda}, "
        f"pyg={torch_geometric.__version__}"
    )
PY
echo "PyG optional install attempt complete."
