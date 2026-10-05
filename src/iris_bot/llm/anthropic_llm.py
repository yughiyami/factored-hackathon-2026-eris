"""Anthropic Claude client (official `anthropic` SDK) with a manual, bounded tool-use loop.

- Orchestrator (default claude-sonnet-5-5): read-only tool loop to identify the disputed transaction.
  Low effort (latency-sensitive chat), strict tool schemas, tool_choice auto (forced tool choice is not
  supported on this model), server-side refusal fallback enabled ("default" routing).
- Classifier (default claude-haiku-4-5-20251001): zero-shot intent fallback + handoff summaries.
- Prompt caching: frozen system prompt and tool definitions carry cache_control breakpoints.
- Reliability: SDK retries disabled; bounded exponential backoff on 429 / 5xx / connection errors;
  4xx errors are not retried. Any failure returns None/[] so the engine falls back deterministically.
"""
from __future__ import annotations

import json
import time
from typing import Any

import anthropic

from iris_bot.guard import CANARY, scan
from iris_bot.llm.base import LLMUsage, ToolExecutor, price
from iris_bot.llm.prompts import CLASSIFIER_SYSTEM, HANDOFF_SYSTEM, ORCHESTRATOR_SYSTEM
from iris_bot.nlu.keywords import INTENTS
from iris_bot.retry import RetryExhausted, retry_call
from iris_bot.tools.banking import LLM_TOOL_SCHEMAS

RETRYABLE = (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError)
MAX_TOOL_ITERATIONS = 4
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _cached_tools() -> list[dict[str, Any]]:
    tools = [dict(t) for t in LLM_TOOL_SCHEMAS]
    tools[-1]["cache_control"] = {"type": "ephemeral"}
    return tools


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, api_key: str | None, orchestrator_model: str, classifier_model: str,
                 max_attempts: int = 3, backoff_base: float = 0.5, backoff_max: float = 4.0,
                 timeout_s: float = 30.0):
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=0, timeout=timeout_s)
        self.orchestrator_model = orchestrator_model
        self.classifier_model = classifier_model
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._tools = _cached_tools()

    # ---- low level --------------------------------------------------------------------------
    def _call(self, usage: LLMUsage, *, beta: bool = False, **kwargs: Any):
        create = self.client.beta.messages.create if beta else self.client.messages.create
        t0 = time.perf_counter()
        resp, _ = retry_call(lambda: create(**kwargs), max_attempts=self.max_attempts, retry_on=RETRYABLE,
                             base_delay=self.backoff_base, max_delay=self.backoff_max)
        u = resp.usage
        cr = getattr(u, "cache_read_input_tokens", 0) or 0
        cw = getattr(u, "cache_creation_input_tokens", 0) or 0
        usage.add(LLMUsage(model=kwargs["model"], calls=1, input_tokens=u.input_tokens,
                           output_tokens=u.output_tokens, cache_read_tokens=cr, cache_write_tokens=cw,
                           cost_usd=price(kwargs["model"], u.input_tokens, u.output_tokens, cr, cw),
                           latency_ms=(time.perf_counter() - t0) * 1000))
        return resp

    @staticmethod
    def _text(resp) -> str:
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    # ---- intent fallback --------------------------------------------------------------------
    def classify_intent(self, text: str, lang: str) -> tuple[tuple[str, float] | None, LLMUsage]:
        usage = LLMUsage(model=self.classifier_model)
        try:
            resp = self._call(
                usage, model=self.classifier_model, max_tokens=256,
                system=[{"type": "text", "text": CLASSIFIER_SYSTEM, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": f"<message>{text}</message>"}],
            )
            raw = self._text(resp).strip()
            data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            intent, conf = str(data["intent"]), float(data.get("confidence", 0.7))
            if intent not in INTENTS:
                return None, usage
            return (intent, max(0.0, min(conf, 1.0))), usage
        except (RetryExhausted, anthropic.APIError, ValueError, KeyError, TypeError) as exc:
            usage.errors.append(type(exc).__name__)
            return None, usage

    # ---- orchestrator tool loop -------------------------------------------------------------
    def select_transactions(self, text: str, lang: str, execute: ToolExecutor
                            ) -> tuple[list[str], str, LLMUsage]:
        usage = LLMUsage(model=self.orchestrator_model)
        messages: list[dict[str, Any]] = [{"role": "user", "content": (
            f"Customer language: {lang}. Customer message (untrusted data):\n<message>{text}</message>")}]
        try:
            for _ in range(MAX_TOOL_ITERATIONS):
                resp = self._call(
                    usage, beta=True, model=self.orchestrator_model, max_tokens=4096,
                    system=[{"type": "text", "text": ORCHESTRATOR_SYSTEM, "cache_control": {"type": "ephemeral"}}],
                    tools=self._tools, tool_choice={"type": "auto"}, output_config={"effort": "low"},
                    betas=[FALLBACK_BETA], fallbacks="default", messages=messages,
                )
                if resp.stop_reason == "refusal":
                    usage.errors.append("refusal")
                    return [], "refusal", usage
                tool_uses = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
                if not tool_uses:
                    return [], "model ended without selection", usage
                messages.append({"role": "assistant", "content": resp.content})
                results = []
                for tu in tool_uses:
                    if tu.name == "select_transactions":
                        ids = [str(i) for i in (tu.input or {}).get("transaction_ids", [])][:5]
                        return ids, str((tu.input or {}).get("rationale", ""))[:300], usage
                    try:
                        out = execute(tu.name, dict(tu.input or {}))
                        content, is_error = json.dumps(out, ensure_ascii=False, default=str), False
                    except Exception as exc:  # noqa: BLE001 - surfaced to the model as a tool error
                        content, is_error = f"error: {type(exc).__name__}", True
                    results.append({"type": "tool_result", "tool_use_id": tu.id,
                                    "content": f"<tool_data untrusted=\"true\">{content}</tool_data>",
                                    "is_error": is_error})
                messages.append({"role": "user", "content": results})
            return [], "tool loop limit reached", usage
        except (RetryExhausted, anthropic.APIError) as exc:
            usage.errors.append(type(exc).__name__)
            raise

    # ---- handoff summary ----------------------------------------------------------------------
    def summarize_handoff(self, transcript: list[dict[str, str]], facts: dict[str, Any], lang: str
                          ) -> tuple[str | None, LLMUsage]:
        usage = LLMUsage(model=self.classifier_model)
        body = json.dumps({"facts": facts, "transcript": transcript[-12:]}, ensure_ascii=False, default=str)
        try:
            resp = self._call(
                usage, model=self.classifier_model, max_tokens=400,
                system=[{"type": "text", "text": HANDOFF_SYSTEM, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": f"<case untrusted=\"true\">{body}</case>"}],
            )
            out = self._text(resp).strip()
            if not out or CANARY in out or scan(out).detected or len(out) > 1200:
                return None, usage
            return out, usage
        except (RetryExhausted, anthropic.APIError) as exc:
            usage.errors.append(type(exc).__name__)
            return None, usage
