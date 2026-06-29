"""Single-writer background jobs with SQLite state and UTF-8 logs."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from uuid import uuid4

from market_slice_ml.domain.enums import JobStatus, OperationStatus
from market_slice_ml.services.models import JobRecord, OperationReport
from market_slice_ml.storage.metadata_store import MetadataStore


class JobManager:
    def __init__(self, store: MetadataStore, data_root: str | Path, max_workers: int = 1) -> None:
        self.store = store
        self.log_root = Path(data_root) / "logs" / "jobs"
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="market-job")
        self._submit_lock = Lock()
        self.store.interrupt_running_jobs()

    def submit(
        self,
        job_type: str,
        parameters: dict[str, object],
        operation: Callable[[], OperationReport],
    ) -> JobRecord:
        with self._submit_lock:
            job_id = str(uuid4())
            log_path = self.log_root / f"{job_id}.log"
            payload = {
                "job_id": job_id,
                "job_type": job_type,
                "status": JobStatus.QUEUED.value,
                "created_at_utc": datetime.now(UTC),
                "started_at_utc": None,
                "finished_at_utc": None,
                "message": "Job 正在等待本機 worker。",
                "suggested_action": "",
                "warnings": [],
                "parameters": parameters,
                "result": {},
                "log_path": str(log_path),
            }
            self.store.create_job(payload)
            self.executor.submit(self._run, job_id, log_path, operation)
            record = self.store.get_job(job_id)
            if record is None:
                raise RuntimeError("job persistence failed")
            return JobRecord.model_validate(record)

    def _run(
        self,
        job_id: str,
        log_path: Path,
        operation: Callable[[], OperationReport],
    ) -> None:
        started = datetime.now(UTC)
        self.store.update_job(
            job_id,
            {
                "status": JobStatus.RUNNING.value,
                "started_at_utc": started,
                "message": "Job 執行中。",
            },
        )
        self._append_log(log_path, f"{started.isoformat()} job started")
        try:
            report = operation()
            status = self._job_status(report.status, report.warnings)
            finished = datetime.now(UTC)
            self.store.update_job(
                job_id,
                {
                    "status": status.value,
                    "finished_at_utc": finished,
                    "message": report.message,
                    "suggested_action": report.suggested_action,
                    "warnings": list(report.warnings),
                    "result": report.result,
                },
            )
            self._append_log(
                log_path,
                f"{finished.isoformat()} job {status.value}: {report.message}",
            )
        except Exception as exc:
            finished = datetime.now(UTC)
            self._append_log(log_path, traceback.format_exc())
            self.store.update_job(
                job_id,
                {
                    "status": JobStatus.FAILED.value,
                    "finished_at_utc": finished,
                    "message": f"操作失敗（{type(exc).__name__}）。",
                    "suggested_action": ("開啟 Job log，修正輸入或環境問題後重試。"),
                },
            )

    @staticmethod
    def _job_status(status: OperationStatus, warnings: tuple[str, ...]) -> JobStatus:
        if status in {OperationStatus.FAILED, OperationStatus.DISABLED}:
            return JobStatus.FAILED
        if status in {OperationStatus.PARTIAL, OperationStatus.NO_DATA} or warnings:
            return JobStatus.COMPLETED_WITH_WARNINGS
        return JobStatus.COMPLETED

    @staticmethod
    def _append_log(path: Path, message: str) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(message.rstrip() + "\n")

    def get(self, job_id: str) -> JobRecord | None:
        payload = self.store.get_job(job_id)
        return None if payload is None else JobRecord.model_validate(payload)

    def list(self, limit: int = 50) -> list[JobRecord]:
        return [JobRecord.model_validate(item) for item in self.store.list_jobs(limit)]

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)
