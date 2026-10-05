from __future__ import annotations

import json
from pathlib import Path

import pytest

from iris_bot.config import ROOT, Settings
from iris_bot.repository import DuckDBRepository
from iris_bot.runtime import build_runtime

FIXTURES = json.loads((ROOT / "demo" / "fixtures.json").read_text(encoding="utf-8"))["roles"]
OTP = "135790"


class FakeClock:
    def __init__(self, start: float = 1_800_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(scope="session")
def repo() -> DuckDBRepository:
    return DuckDBRepository(demo_db=ROOT / "demo" / "iris_demo.duckdb")


@pytest.fixture
def settings() -> Settings:
    return Settings(llm_mode="mock", session_secret="test-secret-0123456789abcdef", test_otp=OTP,
                    anthropic_api_key=None)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def make_runtime(settings, repo, clock):
    def _make(mode: str = "proposed", fault_tools: set[str] | None = None):
        return build_runtime(settings, mode=mode, store_path=":memory:", repo=repo, clock=clock,
                             sleep=lambda s: None, fault_tools=fault_tools or set())
    return _make


@pytest.fixture
def fx() -> dict:
    return FIXTURES


def converse(agent, cid: str, messages: list[str]):
    replies = []
    for m in messages:
        replies.append(agent.handle(cid, m))
    return replies


def text_of(replies) -> str:
    return "\n".join(m for r in replies for m in r.messages)


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    return tmp_path
