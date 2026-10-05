"""Composition root: wires settings, policy, data, auth, tools, NLU, LLM and the agent."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from iris_bot.agent import Agent
from iris_bot.auth import IdentityService
from iris_bot.config import Settings
from iris_bot.llm import build_llm
from iris_bot.llm.base import LLMClient
from iris_bot.nlu import KeywordClassifier, load_classifier
from iris_bot.policy import Policy
from iris_bot.repository import Repository, build_repository
from iris_bot.storage import Store
from iris_bot.tools import BankingTools, ToolRuntime


@dataclass
class Runtime:
    settings: Settings
    policy: Policy
    repo: Repository
    store: Store
    identity: IdentityService
    tools: BankingTools
    agent: Agent


def build_runtime(settings: Settings, *, mode: str = "proposed", store_path: Path | str | None = None,
                  repo: Repository | None = None, clock: Callable[[], float] = time.time,
                  sleep: Callable[[float], None] = time.sleep, fault_tools: set[str] | None = None,
                  llm: LLMClient | None = None) -> Runtime:
    """mode="proposed": learned classifier + LLM (Anthropic or Mock). mode="baseline": keyword rules,
    deterministic transaction matching, no LLM at all."""
    policy = Policy.load(settings.resolve(settings.policy_path))
    repo = repo or build_repository(settings)
    store = Store(store_path if store_path is not None else settings.resolve(settings.runtime_db))
    identity = IdentityService(settings.session_secret, settings.test_otp, settings.session_ttl_minutes, clock=clock)
    rt = ToolRuntime(identity, store, max_attempts=policy.tool_max_attempts, backoff_base=policy.backoff_base,
                     backoff_max=policy.backoff_max,
                     fault_tools=fault_tools if fault_tools is not None else settings.fault_tools, sleep=sleep)
    tools = BankingTools(repo, store, policy, rt)
    if mode == "baseline":
        classifier, llm_client = KeywordClassifier(), None
    else:
        classifier = load_classifier(settings.resolve(settings.intent_model_path))
        llm_client = llm or build_llm(settings, policy)
    agent = Agent(policy=policy, identity=identity, tools=tools, store=store, repo=repo, classifier=classifier,
                  llm=llm_client, name=mode)
    return Runtime(settings, policy, repo, store, identity, tools, agent)
