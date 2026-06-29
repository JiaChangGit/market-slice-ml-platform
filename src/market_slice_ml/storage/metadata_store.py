"""SQLAlchemy-only SQLite metadata registry."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, DateTime, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    __abstract__ = True


class DatasetManifestRow(Base):
    __tablename__ = "dataset_manifest"

    dataset_id: Mapped[str] = mapped_column(String, primary_key=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class RegistryRow(Base):
    __tablename__ = "registry"

    registry_type: Mapped[str] = mapped_column(String, primary_key=True)
    registry_id: Mapped[str] = mapped_column(String, primary_key=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class JobRow(Base):
    __tablename__ = "job_runs"

    job_id: Mapped[str] = mapped_column(String, primary_key=True)
    job_type: Mapped[str] = mapped_column(String, index=True)
    status: Mapped[str] = mapped_column(String, index=True)
    created_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    started_at_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    message: Mapped[str] = mapped_column(String, default="")
    suggested_action: Mapped[str] = mapped_column(String, default="")
    warnings: Mapped[list[str]] = mapped_column(JSON, default=list)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    log_path: Mapped[str] = mapped_column(String, default="")


class MetadataStore:
    def __init__(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{destination}")
        Base.metadata.create_all(self.engine)

    def register_manifest(self, manifest: Any) -> None:
        payload = manifest.model_dump(mode="json")
        with Session(self.engine) as session:
            session.merge(DatasetManifestRow(dataset_id=manifest.dataset_id, payload=payload))
            session.commit()

    def get_manifest(self, dataset_id: str) -> Any | None:
        from market_slice_ml.versioning.dataset_manifest import DatasetManifest

        with Session(self.engine) as session:
            row = session.get(DatasetManifestRow, dataset_id)
            return None if row is None else DatasetManifest.model_validate(row.payload)

    def list_manifests(self) -> list[dict[str, Any]]:
        with Session(self.engine) as session:
            rows = session.scalars(select(DatasetManifestRow)).all()
            return [row.payload for row in rows]

    def save_registry(self, registry_type: str, registry_id: str, payload: dict[str, Any]) -> None:
        with Session(self.engine) as session:
            session.merge(
                RegistryRow(
                    registry_type=registry_type,
                    registry_id=registry_id,
                    payload=payload,
                )
            )
            session.commit()

    def list_registry(self, registry_type: str) -> list[dict[str, Any]]:
        with Session(self.engine) as session:
            statement = select(RegistryRow).where(RegistryRow.registry_type == registry_type)
            rows = session.scalars(statement).all()
            return [row.payload for row in rows]

    def create_job(self, payload: dict[str, Any]) -> None:
        with Session(self.engine) as session:
            session.add(JobRow(**payload))
            session.commit()

    def update_job(self, job_id: str, values: dict[str, Any]) -> None:
        allowed = {
            "status",
            "started_at_utc",
            "finished_at_utc",
            "message",
            "suggested_action",
            "warnings",
            "result",
            "log_path",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unsupported job fields: {sorted(unknown)}")
        with Session(self.engine) as session:
            row = session.get(JobRow, job_id)
            if row is None:
                raise KeyError(job_id)
            for name, value in values.items():
                setattr(row, name, value)
            session.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with Session(self.engine) as session:
            row = session.get(JobRow, job_id)
            return None if row is None else self._job_payload(row)

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        with Session(self.engine) as session:
            statement = select(JobRow).order_by(JobRow.created_at_utc.desc()).limit(limit)
            return [self._job_payload(row) for row in session.scalars(statement)]

    def interrupt_running_jobs(self) -> int:
        changed = 0
        with Session(self.engine) as session:
            statement = select(JobRow).where(JobRow.status.in_(["queued", "running"]))
            for row in session.scalars(statement):
                row.status = "interrupted"
                row.finished_at_utc = datetime.now(UTC)
                row.message = "前一個程序在 Job 完成前停止。"
                row.suggested_action = "檢查 Job log 後重新送出操作。"
                changed += 1
            session.commit()
        return changed

    @staticmethod
    def _job_payload(row: JobRow) -> dict[str, Any]:
        return {
            "job_id": row.job_id,
            "job_type": row.job_type,
            "status": row.status,
            "created_at_utc": row.created_at_utc,
            "started_at_utc": row.started_at_utc,
            "finished_at_utc": row.finished_at_utc,
            "message": row.message,
            "suggested_action": row.suggested_action,
            "warnings": row.warnings,
            "parameters": row.parameters,
            "result": row.result,
            "log_path": row.log_path,
        }
