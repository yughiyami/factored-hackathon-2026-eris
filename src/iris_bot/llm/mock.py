"""Deterministic offline stand-in for the LLM.

Behaviour mirrors the contracts of DeepSeekLLM / AnthropicLLM so the agent, tests and eval run without network:
- classify_intent: keyword rules (returns None when the rules have no match)
- select_transactions: lists transactions through the same scoped tool executor and ranks them with
  the deterministic matcher
- summarize_handoff: None (the engine uses its deterministic template)
Token usage is ESTIMATED (characters / 4) and priced at list prices so cost metrics have a projection;
`estimated=True` marks it. No API call is ever made.
"""
from __future__ import annotations

from typing import Any

from iris_bot.llm.base import LLMUsage, ToolExecutor, price
from iris_bot.llm.prompts import CLASSIFIER_SYSTEM, ORCHESTRATOR_SYSTEM
from iris_bot.nlu.keywords import KeywordClassifier
from iris_bot.nlu.slots import TXN_ID_RX, match_transactions


def _est(model: str, prompt_chars: int, output_chars: int) -> LLMUsage:
    tin, tout = prompt_chars // 4, max(1, output_chars // 4)
    return LLMUsage(model=model, calls=1, input_tokens=tin, output_tokens=tout,
                    cost_usd=price(model, tin, tout), estimated=True)


class MockLLM:
    name = "mock"

    def __init__(self, orchestrator_model: str = "deepseek-v4-pro",
                 classifier_model: str = "deepseek-flash"):
        self.orchestrator_model = orchestrator_model
        self.classifier_model = classifier_model
        self._rules = KeywordClassifier()

    def classify_intent(self, text: str, lang: str) -> tuple[tuple[str, float] | None, LLMUsage]:
        intent, conf = self._rules.predict(text)
        usage = _est(self.classifier_model, len(CLASSIFIER_SYSTEM) + len(text), 60)
        return ((intent, 0.75) if conf >= 0.6 else None), usage

    def select_transactions(self, text: str, lang: str, execute: ToolExecutor
                            ) -> tuple[list[str], str, LLMUsage]:
        ids = [m.upper() for m in TXN_ID_RX.findall(text or "")]
        if ids:
            got = execute("get_transaction", {"transaction_id": ids[0]})
            chars = len(ORCHESTRATOR_SYSTEM) + len(text) + len(str(got))
            return [ids[0]], "explicit transaction id", _est(self.orchestrator_model, chars, 120)
        txns: list[dict[str, Any]] = execute("list_recent_transactions", {"days": 120})
        ranked = match_transactions(text, txns)
        chars = len(ORCHESTRATOR_SYSTEM) + len(text) + len(str(txns))
        usage = _est(self.orchestrator_model, chars * 2, 160)  # two model turns in the real loop
        if not ranked:
            return [], "no match", usage
        best = ranked[0][1]
        picked = [t["transaction_id"] for t, s in ranked if s >= best - 0.5][:5]
        return picked, "deterministic match", usage

    def summarize_handoff(self, transcript: list[dict[str, str]], facts: dict[str, Any], lang: str
                          ) -> tuple[str | None, LLMUsage]:
        return None, LLMUsage(model=self.classifier_model, estimated=True)
