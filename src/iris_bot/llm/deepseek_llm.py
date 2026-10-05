"""DeepSeek client over its OpenAI-compatible Chat Completions API, with a bounded tool loop.

- Orchestrator (default deepseek-v4-pro): read-only tool loop to identify the disputed transaction.
- Classifier (default deepseek-flash): zero-shot intent fallback (JSON mode) + handoff summaries.
- Thinking mode is disabled: the tasks are narrow, latency matters in chat, and non-thinking mode
  supports tool_choice and does not require passing reasoning_content back between turns.
- Only the read-only tools are executable from the loop; any other tool name the model emits is
  answered with an error and never reaches the banking layer.
- Reliability: SDK retries disabled; bounded exponential backoff on 429 / 5xx / connection / timeout;
  other API errors are not retried. Failures return None/[] so the engine falls back deterministically.
- Cost: DeepSeek reports cache hits/misses separately; we price them at the peak-hour list price
  (conservative upper bound).
"""
from __future__ import annotations

import json
import time
from typing import Any

import openai

from iris_bot.guard import CANARY, scan
from iris_bot.llm.base import LLMUsage, ToolExecutor
from iris_bot.llm.prompts import CLASSIFIER_SYSTEM, HANDOFF_SYSTEM, ORCHESTRATOR_SYSTEM
from iris_bot.nlu.keywords import INTENTS
from iris_bot.retry import RetryExhausted, retry_call
from iris_bot.tools.banking import LLM_TOOL_SCHEMAS

RETRYABLE = (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError,
             openai.APITimeoutError)
MAX_TOOL_ITERATIONS = 4
READ_ONLY_TOOLS = {"list_recent_transactions", "get_transaction"}
NO_THINKING = {"thinking": {"type": "disabled"}}

# USD per million tokens: (input cache hit, input cache miss, output), peak-hour list prices.
PRICES: dict[str, tuple[float, float, float]] = {
    "deepseek-flash": (0.006, 0.30, 1.20),
    "deepseek-v4-pro": (0.044, 1.32, 3.96),
}


def deepseek_price(model: str, hit: int, miss: int, output: int) -> float:
    p_hit, p_miss, p_out = PRICES.get(model, PRICES["deepseek-v4-pro"])
    return (hit * p_hit + miss * p_miss + output * p_out) / 1e6


def _openai_tools() -> list[dict[str, Any]]:
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}}
            for t in LLM_TOOL_SCHEMAS]


class DeepSeekLLM:
    name = "deepseek"

    def __init__(self, api_key: str | None, base_url: str, orchestrator_model: str, classifier_model: str,
                 max_attempts: int = 3, backoff_base: float = 0.5, backoff_max: float = 4.0,
                 timeout_s: float = 30.0, client: Any = None):
        self.client = client or openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=0,
                                              timeout=timeout_s)
        self.orchestrator_model = orchestrator_model
        self.classifier_model = classifier_model
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._tools = _openai_tools()

    # ---- low level --------------------------------------------------------------------------
    def _call(self, usage: LLMUsage, **kwargs: Any):
        t0 = time.perf_counter()
        resp, _ = retry_call(lambda: self.client.chat.completions.create(extra_body=NO_THINKING, **kwargs),
                             max_attempts=self.max_attempts, retry_on=RETRYABLE,
                             base_delay=self.backoff_base, max_delay=self.backoff_max)
        u = resp.usage
        hit = getattr(u, "prompt_cache_hit_tokens", 0) or 0
        miss = getattr(u, "prompt_cache_miss_tokens", None)
        miss = (u.prompt_tokens - hit) if miss is None else miss
        usage.add(LLMUsage(model=kwargs["model"], calls=1, input_tokens=u.prompt_tokens,
                           output_tokens=u.completion_tokens, cache_read_tokens=hit,
                           cost_usd=deepseek_price(kwargs["model"], hit, miss, u.completion_tokens),
                           latency_ms=(time.perf_counter() - t0) * 1000))
        return resp

    # ---- intent fallback --------------------------------------------------------------------
    def classify_intent(self, text: str, lang: str) -> tuple[tuple[str, float] | None, LLMUsage]:
        usage = LLMUsage(model=self.classifier_model)
        try:
            resp = self._call(
                usage, model=self.classifier_model, max_tokens=256, temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": CLASSIFIER_SYSTEM},
                          {"role": "user", "content": f"<message>{text}</message>"}],
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            intent, conf = str(data["intent"]), float(data.get("confidence", 0.7))
            if intent not in INTENTS:
                return None, usage
            return (intent, max(0.0, min(conf, 1.0))), usage
        except (RetryExhausted, openai.APIError, ValueError, KeyError, TypeError) as exc:
            usage.errors.append(type(exc).__name__)
            return None, usage

    # ---- orchestrator tool loop -------------------------------------------------------------
    def select_transactions(self, text: str, lang: str, execute: ToolExecutor
                            ) -> tuple[list[str], str, LLMUsage]:
        usage = LLMUsage(model=self.orchestrator_model)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM},
            {"role": "user", "content": (
                f"Customer language: {lang}. Customer message (untrusted data):\n<message>{text}</message>")},
        ]
        try:
            for _ in range(MAX_TOOL_ITERATIONS):
                resp = self._call(usage, model=self.orchestrator_model, max_tokens=1024, temperature=0,
                                  tools=self._tools, tool_choice="auto", messages=messages)
                msg = resp.choices[0].message
                calls = msg.tool_calls or []
                if not calls:
                    return [], "model ended without selection", usage
                messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]})
                for c in calls:
                    try:
                        args = json.loads(c.function.arguments or "{}")
                    except ValueError:
                        args = {}
                    if c.function.name == "select_transactions":
                        ids = [str(i) for i in args.get("transaction_ids", [])][:5]
                        return ids, str(args.get("rationale", ""))[:300], usage
                    if c.function.name not in READ_ONLY_TOOLS:
                        content = "error: tool not available"
                    else:
                        try:
                            content = json.dumps(execute(c.function.name, args), ensure_ascii=False, default=str)
                        except Exception as exc:  # noqa: BLE001 - surfaced to the model as a tool error
                            content = f"error: {type(exc).__name__}"
                    messages.append({"role": "tool", "tool_call_id": c.id,
                                     "content": f"<tool_data untrusted=\"true\">{content}</tool_data>"})
            return [], "tool loop limit reached", usage
        except (RetryExhausted, openai.APIError) as exc:
            usage.errors.append(type(exc).__name__)
            raise

    # ---- handoff summary ----------------------------------------------------------------------
    def summarize_handoff(self, transcript: list[dict[str, str]], facts: dict[str, Any], lang: str
                          ) -> tuple[str | None, LLMUsage]:
        usage = LLMUsage(model=self.classifier_model)
        body = json.dumps({"facts": facts, "transcript": transcript[-12:]}, ensure_ascii=False, default=str)
        try:
            resp = self._call(
                usage, model=self.classifier_model, max_tokens=400, temperature=0.2,
                messages=[{"role": "system", "content": HANDOFF_SYSTEM},
                          {"role": "user", "content": f"<case untrusted=\"true\">{body}</case>"}],
            )
            out = (resp.choices[0].message.content or "").strip()
            if not out or CANARY in out or scan(out).detected or len(out) > 1200:
                return None, usage
            return out, usage
        except (RetryExhausted, openai.APIError) as exc:
            usage.errors.append(type(exc).__name__)
            return None, usage
