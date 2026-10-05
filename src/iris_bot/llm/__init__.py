"""LLM factory: the configured provider (DeepSeek by default, or Anthropic) when its key is set,
otherwise the offline MockLLM."""
from __future__ import annotations

from iris_bot.llm.base import LLMClient, LLMUsage, price
from iris_bot.llm.mock import MockLLM

__all__ = ["LLMClient", "LLMUsage", "MockLLM", "build_llm", "price"]


def build_llm(settings, policy=None) -> LLMClient:
    mode, provider = settings.llm_mode, settings.provider
    mock = MockLLM(settings.orchestrator_model, settings.classifier_model)
    if mode == "mock" or (mode == "auto" and not settings.provider_api_key):
        return mock
    attempts = policy.llm_max_attempts if policy else 3
    try:
        if provider == "deepseek":
            from iris_bot.llm.deepseek_llm import DeepSeekLLM
            return DeepSeekLLM(settings.deepseek_api_key, settings.deepseek_base_url,
                               settings.orchestrator_model, settings.classifier_model, max_attempts=attempts)
        from iris_bot.llm.anthropic_llm import AnthropicLLM
        return AnthropicLLM(settings.anthropic_api_key, settings.orchestrator_model, settings.classifier_model,
                            max_attempts=attempts)
    except ImportError as exc:  # llm extra not installed
        if mode != "auto":
            raise RuntimeError(f"pip install 'iris-bot[llm]' to use IRIS_LLM_MODE={mode}") from exc
        return mock
