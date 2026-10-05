"""LLM factory: Anthropic when a key is configured (or forced), otherwise the offline MockLLM."""
from __future__ import annotations

from iris_bot.llm.base import LLMClient, LLMUsage, price
from iris_bot.llm.mock import MockLLM

__all__ = ["LLMClient", "LLMUsage", "MockLLM", "build_llm", "price"]


def build_llm(settings, policy=None) -> LLMClient:
    mode = settings.llm_mode
    if mode == "mock" or (mode == "auto" and not settings.anthropic_api_key):
        return MockLLM(settings.orchestrator_model, settings.classifier_model)
    try:
        from iris_bot.llm.anthropic_llm import AnthropicLLM
    except ImportError as exc:  # anthropic extra not installed
        if mode == "anthropic":
            raise RuntimeError("pip install 'iris-bot[llm]' to use IRIS_LLM_MODE=anthropic") from exc
        return MockLLM(settings.orchestrator_model, settings.classifier_model)
    attempts = policy.llm_max_attempts if policy else 3
    return AnthropicLLM(settings.anthropic_api_key, settings.orchestrator_model, settings.classifier_model,
                        max_attempts=attempts)
