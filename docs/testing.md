# Testing and Verification

本文件整理目前專案使用的驗證流程。所有測試預設 `NO_NETWORK=1`，不得呼叫 live market-data Provider。

## Local Checks

```bash
source .venv/bin/activate
python -m compileall scripts src tests -q
python scripts/check_no_trading_api.py
python scripts/check_data_sources.py
ruff check src tests scripts
mypy src --ignore-missing-imports
pytest tests/unit/ -q --tb=short
pytest tests/property/ -q --tb=short
pytest tests/integration/ -q --tb=short
pytest tests/e2e/ -q --tb=short
python scripts/smoke_local_pipeline.py
bash scripts/run_all_checks.sh --full
```

成功輸出範例：

```text
No-trading static guard passed.
All checks passed!
Success: no issues found in 105 source files
Smoke pipeline passed: data/reports/smoke_report.html
All checks passed.
```

## Smoke Pipeline

`scripts/smoke_local_pipeline.py` 呼叫 `src/market_slice_ml/pipeline.py` 的 `run_synthetic_smoke()`。流程使用 deterministic synthetic Bars，不讀取 live Provider。

輸出：

```text
data/reports/smoke_report.html
```

Report 應為 self-contained HTML，不包含 CDN 或外部 URL。

## Static Guard

`scripts/check_no_trading_api.py` 掃描 `src/` 與 `tests/`，拒絕 execution-oriented API vocabulary。允許相關詞只存在於 guard 清單、文件說明或測試中。

## CI

`.github/workflows/ci.yml` 執行 `bash scripts/run_all_checks.sh --full`。CI 不需要 live Provider credentials。
