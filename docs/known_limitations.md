# Known Limitations

本文記錄目前驗證時仍存在的限制，避免文件宣稱未驗證能力。

## Environment

- Host `python3` 可能是 Python 3.10；專案要求 Python `>=3.12,<3.13`。
- 已驗證環境使用 `.venv/bin/python` 3.12.13。
- CPU Torch baseline 已通過；CUDA runtime 未啟用。

## Optional Packages

以下套件在目前環境未安裝或未啟用，屬於 optional：

- `torch_geometric`
- `pyg_lib`
- `torch_scatter`
- `torch_sparse`
- `ib_insync`
- `akshare`

GNN 流程需保留 pure PyTorch fallback。

## Live Providers

測試與 smoke pipeline 不呼叫 live Provider。若要使用 Historical Provider，需設定 `NO_NETWORK=0`，並依 Provider 準備 credentials、network 與可用資料區間。

## Docker

`docker compose config` 已通過。Docker build context 已透過 `.dockerignore` 限制，但目前環境尚未在限定時間內完成 `docker compose build` 並產出 image。

## Repository State

目前 workspace 不是 Git repository，無法用 Git diff 或 Git status 追蹤變更。
