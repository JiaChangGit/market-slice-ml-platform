"""Local-only FastAPI application with token protection for LAN binding."""

from __future__ import annotations

import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from market_slice_ml.config.settings import Settings, get_settings
from market_slice_ml.services.job_service import JobManager
from market_slice_ml.services.research_service import ResearchService
from market_slice_ml.web.schemas import (
    ApiError,
    FetchJobRequest,
    PredictionRequest,
    SymbolsJobRequest,
    TrainJobRequest,
)

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
LOGGER = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    configured = settings or get_settings()
    if configured.web_host not in LOCAL_HOSTS and not configured.web_api_token:
        raise RuntimeError("Non-local Web binding requires WEB_API_TOKEN")
    service = ResearchService(configured)
    jobs = JobManager(service.metadata, service.data_root)
    web_root = Path(__file__).parent
    templates = Jinja2Templates(directory=web_root / "templates")

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        jobs.shutdown()

    app = FastAPI(
        title="Market Slice Web UI",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=web_root / "static"), name="static")

    @app.middleware("http")
    async def require_api_token(request: Request, call_next: Any) -> Any:
        token = configured.web_api_token
        if request.url.path.startswith("/api/") and token:
            supplied = request.headers.get("X-API-Token", "")
            if not secrets.compare_digest(supplied, token):
                error = ApiError(
                    code="invalid_api_token",
                    message="API token 缺少或不正確。",
                    suggested_action="在設定中輸入 WEB_API_TOKEN 後重試。",
                )
                return JSONResponse(status_code=401, content=error.model_dump(mode="json"))
        return await call_next(request)

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict):
            content = exc.detail
        else:
            content = ApiError(
                code="http_error",
                message=str(exc.detail),
                suggested_action="確認輸入與目前 pipeline 狀態後重試。",
            ).model_dump(mode="json")
        return JSONResponse(status_code=exc.status_code, content=content)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        warnings = tuple(
            f"{'.'.join(str(item) for item in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        error = ApiError(
            code="request_validation_error",
            message="輸入資料格式不正確，操作尚未送出。",
            suggested_action="依照欄位提示修正內容後重試。",
            warnings=warnings,
        )
        return JSONResponse(status_code=422, content=error.model_dump(mode="json"))

    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception("Unhandled Web request error", exc_info=exc)
        error = ApiError(
            code="internal_error",
            message="伺服器處理請求時發生未預期錯誤。",
            suggested_action="查看 server log；確認環境後再重試。",
        )
        return JSONResponse(status_code=500, content=error.model_dump(mode="json"))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"token_required": bool(configured.web_api_token)},
        )

    @app.get("/api/status")
    async def api_status() -> dict[str, object]:
        return {
            "environment": service.environment_summary(),
            "stages": [item.model_dump(mode="json") for item in service.pipeline_stages()],
            "providers": [item.model_dump(mode="json") for item in service.provider_readiness()],
            "jobs": [item.model_dump(mode="json") for item in jobs.list(20)],
            "slices": service.slice_manifests(),
            "runs": service.model_runs()[:20],
            "reports": _report_files(service.data_root),
            "token_required": bool(configured.web_api_token),
        }

    @app.get("/api/jobs")
    async def api_jobs() -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in jobs.list()]

    @app.get("/api/jobs/{job_id}")
    async def api_job(job_id: str) -> dict[str, Any]:
        record = jobs.get(job_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=ApiError(
                    code="job_not_found",
                    message="找不到指定的 Job。",
                    suggested_action="重新整理 Job 清單後再選取。",
                ).model_dump(mode="json"),
            )
        return record.model_dump(mode="json")

    @app.post("/api/jobs/fetch", status_code=202)
    async def api_fetch(payload: FetchJobRequest) -> dict[str, Any]:
        record = jobs.submit(
            "fetch",
            payload.model_dump(mode="json"),
            lambda: service.fetch(
                payload.symbols,
                payload.interval,
                payload.start_utc,
                payload.end_utc,
                payload.provider,
            ),
        )
        return record.model_dump(mode="json")

    @app.post("/api/jobs/canonical", status_code=202)
    async def api_canonical(payload: SymbolsJobRequest) -> dict[str, Any]:
        symbols = payload.symbols or _all_symbols(service)
        record = jobs.submit(
            "canonical",
            {"symbols": symbols},
            lambda: service.build_canonical(symbols),
        )
        return record.model_dump(mode="json")

    @app.post("/api/jobs/features", status_code=202)
    async def api_features(payload: SymbolsJobRequest) -> dict[str, Any]:
        symbols = payload.symbols or _all_symbols(service)
        record = jobs.submit(
            "features",
            {"symbols": symbols},
            lambda: service.build_features(symbols),
        )
        return record.model_dump(mode="json")

    @app.post("/api/jobs/labels", status_code=202)
    async def api_labels(payload: SymbolsJobRequest) -> dict[str, Any]:
        targets, _ = service.symbols()
        symbols = payload.symbols or targets
        record = jobs.submit(
            "labels",
            {"symbols": symbols},
            lambda: service.build_labels(symbols),
        )
        return record.model_dump(mode="json")

    @app.post("/api/jobs/slices", status_code=202)
    async def api_slices() -> dict[str, Any]:
        record = jobs.submit("slices", {}, service.build_slices)
        return record.model_dump(mode="json")

    @app.post("/api/jobs/train", status_code=202)
    async def api_train(payload: TrainJobRequest) -> dict[str, Any]:
        record = jobs.submit(
            "train",
            payload.model_dump(mode="json"),
            lambda: service.train(payload.horizon, payload.force),
        )
        return record.model_dump(mode="json")

    @app.post("/api/jobs/report", status_code=202)
    async def api_report() -> dict[str, Any]:
        record = jobs.submit("report", {}, service.generate_report)
        return record.model_dump(mode="json")

    @app.post("/api/predictions")
    async def api_prediction(payload: PredictionRequest) -> dict[str, Any]:
        try:
            prediction = service.predict(payload.symbol.upper(), payload.horizon)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(
                status_code=409,
                detail=ApiError(
                    code="prediction_unavailable",
                    message="目前無法產生 Prediction。",
                    suggested_action=(
                        "確認該 Symbol 的 Features 與 Horizon Model run 已完成；"
                        "詳細原因可查看 diagnostics。"
                    ),
                    warnings=(str(exc),),
                ).model_dump(mode="json"),
            ) from exc
        return prediction.model_dump(mode="json")

    @app.get("/reports/{name}")
    async def open_report(name: str) -> FileResponse:
        safe_name = Path(name).name
        path = service.data_root / "reports" / safe_name
        if safe_name != name or not path.exists() or path.suffix.lower() != ".html":
            raise HTTPException(status_code=404, detail="找不到指定的 Report。")
        return FileResponse(path, media_type="text/html; charset=utf-8")

    return app


def _all_symbols(service: ResearchService) -> list[str]:
    targets, context = service.symbols()
    return targets + context


def _report_files(data_root: Path) -> list[dict[str, str]]:
    root = data_root / "reports"
    if not root.exists():
        return []
    return [
        {"name": path.name, "url": f"/reports/{path.name}"}
        for path in sorted(root.glob("*.html"), reverse=True)
    ]
