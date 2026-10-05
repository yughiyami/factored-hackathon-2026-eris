"""Runtime settings, loaded from environment variables (prefix IRIS_) and an optional .env file."""
from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IRIS_", env_file=".env", extra="ignore")

    # LLM
    llm_mode: str = "auto"  # auto | mock | anthropic
    orchestrator_model: str = "claude-sonnet-5-5"
    classifier_model: str = "claude-haiku-4-5-20251001"
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    # Data
    data_backend: str = "demo"  # demo | silver
    demo_db: Path = ROOT / "demo" / "iris_demo.duckdb"
    silver_dir: Path = ROOT / "data" / "silver"
    runtime_db: Path = ROOT / "runtime" / "iris.sqlite"
    policy_path: Path = ROOT / "config" / "policy.yaml"
    intent_model_path: Path = ROOT / "models" / "intent.joblib"
    as_of_date: str | None = None  # override the bank "today"; default = latest data timestamp

    # Auth (mock trusted identity service)
    session_secret: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    session_ttl_minutes: int = 15
    test_otp: str = "246810"  # demo-only one-time code of the mock identity provider

    # Testing
    fault_inject: str = ""  # comma-separated tool names that always fail

    @property
    def fault_tools(self) -> set[str]:
        return {t.strip() for t in self.fault_inject.split(",") if t.strip()}

    def resolve(self, p: Path) -> Path:
        return p if p.is_absolute() else ROOT / p


@lru_cache
def get_settings() -> Settings:
    return Settings()
