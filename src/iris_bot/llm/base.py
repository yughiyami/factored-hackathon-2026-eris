"""LLM interface shared by the Anthropic client and the offline MockLLM."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

# USD per million tokens (input, output). Cache reads bill at 0.1x input, cache writes at 1.25x.
PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-opus-5-5": (4.0, 20.0),
}


def price(model: str, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0) -> float:
    pin, pout = PRICES.get(model, (2.0, 10.0))
    return (input_tokens * pin + output_tokens * pout + cache_read * pin * 0.1
            + cache_write * pin * 1.25) / 1e6


@dataclass
class LLMUsage:
    model: str = ""
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    estimated: bool = False  # True for MockLLM (token estimate, no API call)
    errors: list[str] = field(default_factory=list)

    def add(self, other: LLMUsage) -> None:
        self.model = other.model or self.model
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.cost_usd += other.cost_usd
        self.latency_ms += other.latency_ms
        self.estimated = self.estimated or other.estimated
        self.errors += other.errors


ToolExecutor = Callable[[str, dict[str, Any]], Any]


class LLMClient(Protocol):
    name: str

    def classify_intent(self, text: str, lang: str) -> tuple[tuple[str, float] | None, LLMUsage]:
        """Zero-shot intent; returns ((intent, confidence) or None, usage)."""

    def select_transactions(self, text: str, lang: str, execute: ToolExecutor
                            ) -> tuple[list[str], str, LLMUsage]:
        """Find the customer's transactions that match the description using read-only tools."""

    def summarize_handoff(self, transcript: list[dict[str, str]], facts: dict[str, Any], lang: str
                          ) -> tuple[str | None, LLMUsage]:
        """One-paragraph summary for the human agent (None -> use the deterministic template)."""
