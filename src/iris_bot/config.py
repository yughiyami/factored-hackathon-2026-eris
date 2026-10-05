"""Runtime settings, loaded from environment variables (prefix IRIS_) and an optional .env file."""
from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]

# provider -> (orchestrator model, classifier/summary model); override with IRIS_*_MODEL.
PROVIDER_MODELS: dict[str, tuple[str, str]] = {
    "deepseek": ("deepseek-v4-pro", "deepseek-flash"),
    "anthropic": ("claude-sonnet-5-5", "claude-haiku-4-5-20251001"),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IRIS_", env_file=".env", extra="ignore")

    # LLM. llm_mode "auto" uses llm_provider when its API key is set, MockLLM otherwise;
    # "deepseek" / "anthropic" force that provider; "mock" never calls an API.
    llm_mode: str = "auto"  # auto | mock | deepseek | anthropic
    llm_provider: str = "deepseek"  # provider used by auto mode
    orchestrator_model: str | None = None  # default depends on the provider (see PROVIDER_MODELS)
    classifier_model: str | None = None
    deepseek_api_key: str | None = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = "https://api.deepseek.com"
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

    @model_validator(mode="after")
    def _default_models(self) -> Settings:
        orchestrator, classifier = PROVIDER_MODELS.get(self.provider, PROVIDER_MODELS["deepseek"])
        self.orchestrator_model = self.orchestrator_model or orchestrator
        self.classifier_model = self.classifier_model or classifier
        return self

    @property
    def provider(self) -> str:
        """Provider the current mode points at ("mock" mode still reports the configured provider)."""
        return self.llm_mode if self.llm_mode in PROVIDER_MODELS else self.llm_provider

    @property
    def provider_api_key(self) -> str | None:
        return {"deepseek": self.deepseek_api_key, "anthropic": self.anthropic_api_key}.get(self.provider)

    @property
    def fault_tools(self) -> set[str]:
        return {t.strip() for t in self.fault_inject.split(",") if t.strip()}

    def resolve(self, p: Path) -> Path:
        return p if p.is_absolute() else ROOT / p


@lru_cache
def get_settings() -> Settings:
    return Settings()
