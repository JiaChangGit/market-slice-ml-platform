# Architecture

本文記錄 Market Slice ML Platform 目前已實作的架構。描述以程式碼、設定檔與測試可驗證的內容為準。

## System Architecture

```mermaid
flowchart LR
    CLI["CLI<br/>cli/main.py"]
    Web["FastAPI Web UI<br/>web/app.py"]
    JobManager["JobManager<br/>services/job_service.py"]
    ResearchService["ResearchService<br/>services/research_service.py"]
    ProviderRegistry["ProviderRegistry<br/>providers/provider_registry.py"]
    Pipeline["Pipeline Modules<br/>processing/features/labels/slicing/ml"]
    Storage["Local Storage<br/>Parquet + SQLite + DuckDB"]
    Report["HTML Report<br/>reporting/html_report.py"]

    CLI --> ResearchService
    Web --> ResearchService
    Web --> JobManager
    JobManager --> ResearchService
    ResearchService --> ProviderRegistry
    ResearchService --> Pipeline
    ResearchService --> Storage
    Pipeline --> Storage
    ResearchService --> Report
```

對應檔案：`src/market_slice_ml/cli/main.py`、`src/market_slice_ml/web/app.py`、`src/market_slice_ml/services/research_service.py`、`src/market_slice_ml/services/job_service.py`、`src/market_slice_ml/providers/provider_registry.py`、`src/market_slice_ml/reporting/html_report.py`。

## Data Flow

```mermaid
flowchart TD
    Provider["Historical Provider or synthetic fixture"]
    Raw["Raw Bars<br/>data/raw/"]
    Canonical["Canonical 5m<br/>data/canonical/"]
    Features["Features<br/>data/features/"]
    Labels["Labels<br/>data/labels/"]
    Slices["Synchronized Slices<br/>data/slices/"]
    Train["Train Model Runs<br/>data/models/runs/"]
    Predict["Prediction<br/>data/predictions/"]
    Report["Report<br/>data/reports/"]

    Provider --> Raw
    Raw --> Canonical
    Canonical --> Features
    Features --> Labels
    Labels --> Slices
    Slices --> Train
    Train --> Predict
    Predict --> Report
```

對應流程：`ResearchService.fetch()`、`build_canonical()`、`build_features()`、`build_labels()`、`build_slices()`、`train()`、`predict()`、`generate_report()`。`scripts/smoke_local_pipeline.py` 使用 synthetic fixture 走同一類資料轉換，但不呼叫 live Provider。

## Sequence Diagram

```mermaid
sequenceDiagram
    participant User
    participant CLI as market-ml
    participant Service as ResearchService
    participant Pipeline as Pipeline modules
    participant Storage as data/
    participant Report as HTML Report

    User->>CLI: python scripts/smoke_local_pipeline.py
    CLI->>Service: run_synthetic_smoke(data_root)
    Service->>Pipeline: canonical -> features -> labels -> slices
    Pipeline->>Storage: write local artifacts
    Service->>Pipeline: train tiny Tree / LSTM / GNN fallback
    Service->>Report: write_html_report()
    Report->>Storage: data/reports/smoke_report.html
    CLI-->>User: Smoke pipeline passed
```

對應檔案：`scripts/smoke_local_pipeline.py`、`src/market_slice_ml/pipeline.py`、`src/market_slice_ml/reporting/html_report.py`。

## Module Relationship

```mermaid
classDiagram
    class ResearchService
    class ModelService
    class ProviderRegistry
    class MetadataStore
    class ParquetStore
    class DuckDBStore
    class PathResolver
    class JobManager

    ResearchService --> ModelService
    ResearchService --> ProviderRegistry
    ResearchService --> MetadataStore
    ResearchService --> ParquetStore
    ResearchService --> DuckDBStore
    ResearchService --> PathResolver
    JobManager --> MetadataStore
```

對應檔案：`src/market_slice_ml/services/model_service.py`、`src/market_slice_ml/storage/metadata_store.py`、`src/market_slice_ml/storage/parquet_store.py`、`src/market_slice_ml/storage/duckdb_store.py`、`src/market_slice_ml/storage/path_resolver.py`。
