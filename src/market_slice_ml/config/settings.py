"""Environment-backed application settings."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_root: Path = Path("data")
    data_start_date: date = date(2022, 1, 1)
    data_end_date: date | None = None
    ibkr_host: str = "auto"
    ibkr_port: int = 7497
    ibkr_client_id: int = 1
    alpha_vantage_api_key: str = ""
    akshare_enabled: bool = False
    schwab_enabled: bool = False
    log_level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    no_network: bool = True
    web_host: str = "127.0.0.1"
    web_port: int = Field(default=8000, ge=1, le=65535)
    web_api_token: str = ""


def get_settings() -> Settings:
    return Settings()
