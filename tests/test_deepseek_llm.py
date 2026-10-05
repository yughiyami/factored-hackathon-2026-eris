"""DeepSeek client contract, exercised with a fake OpenAI-compatible client (no network)."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("openai")

from iris_bot.config import Settings  # noqa: E402
from iris_bot.llm import MockLLM, build_llm  # noqa: E402
from iris_bot.llm.deepseek_llm import DeepSeekLLM  # noqa: E402


def _resp(content: str | None = None, tool_calls: list | None = None, hit: int = 0, miss: int = 100,
          out: int = 20):
    usage = NS(prompt_tokens=hit + miss, completion_tokens=out,
               prompt_cache_hit_tokens=hit, prompt_cache_miss_tokens=miss)
    msg = NS(content=content, tool_calls=tool_calls)
    return NS(choices=[NS(message=msg, finish_reason="tool_calls" if tool_calls else "stop")], usage=usage)


def _call(name: str, args: dict, cid: str = "call_1"):
    return NS(id=cid, type="function", function=NS(name=name, arguments=json.dumps(args)))


class FakeCompletions:
    def __init__(self, responses: list):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))  # the client keeps mutating its message list
        return self.responses.pop(0)


def _llm(responses: list) -> tuple[DeepSeekLLM, FakeCompletions]:
    fake = FakeCompletions(responses)
    client = NS(chat=NS(completions=fake))
    return DeepSeekLLM("test-key", "https://api.deepseek.com", "deepseek-v4-pro", "deepseek-flash",
                       client=client), fake


def test_classify_intent_parses_json_and_prices_the_call():
    llm, fake = _llm([_resp(json.dumps({"intent": "dispute_unrecognized_charge", "confidence": 0.9}))])
    result, usage = llm.classify_intent("no reconozco un cargo", "es")
    assert result == ("dispute_unrecognized_charge", 0.9)
    assert usage.calls == 1 and usage.cost_usd > 0 and not usage.estimated
    req = fake.requests[0]
    assert req["model"] == "deepseek-flash"
    assert req["response_format"] == {"type": "json_object"}
    assert req["extra_body"]["thinking"] == {"type": "disabled"}


def test_classify_intent_rejects_unknown_intent_and_bad_json():
    llm, _ = _llm([_resp(json.dumps({"intent": "wire_money_abroad"})), _resp("not json")])
    assert llm.classify_intent("x", "es")[0] is None
    result, usage = llm.classify_intent("x", "es")
    assert result is None and usage.errors


def test_tool_loop_executes_read_only_tools_and_returns_selection():
    llm, fake = _llm([
        _resp(tool_calls=[_call("list_recent_transactions", {"days": 90})]),
        _resp(tool_calls=[_call("select_transactions",
                                {"transaction_ids": ["TRX-1", "TRX-2"], "rationale": "merchant match"},
                                cid="call_2")]),
    ])
    executed = []

    def execute(name, args):
        executed.append((name, args))
        return [{"transaction_id": "TRX-1", "merchant": "Internet Plus"}]

    ids, rationale, usage = llm.select_transactions("cargo de Internet Plus", "es", execute)
    assert ids == ["TRX-1", "TRX-2"] and rationale == "merchant match"
    assert executed == [("list_recent_transactions", {"days": 90})]
    assert usage.calls == 2
    second = fake.requests[1]["messages"]
    tool_msg = second[-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "call_1"
    assert 'untrusted="true"' in tool_msg["content"]
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["id"] == "call_1"
    tools = fake.requests[0]["tools"]
    assert {t["function"]["name"] for t in tools} == {"list_recent_transactions", "get_transaction",
                                                      "select_transactions"}


def test_tool_loop_never_runs_unknown_tools():
    llm, _ = _llm([
        _resp(tool_calls=[_call("open_dispute", {"transaction_id": "TRX-1"})]),
        _resp(tool_calls=[_call("select_transactions", {"transaction_ids": [], "rationale": "none"})]),
    ])
    executed = []
    llm.select_transactions("abre la disputa", "es", lambda n, a: executed.append(n))
    assert executed == []


def test_build_llm_picks_deepseek_by_default_and_mock_without_key():
    with_key = Settings(DEEPSEEK_API_KEY="k", session_secret="s" * 40)
    assert with_key.orchestrator_model == "deepseek-v4-pro"
    assert isinstance(build_llm(with_key), DeepSeekLLM)
    no_key = Settings(DEEPSEEK_API_KEY=None, session_secret="s" * 40)
    assert isinstance(build_llm(no_key), MockLLM)


def test_anthropic_provider_keeps_claude_defaults():
    s = Settings(llm_provider="anthropic", session_secret="s" * 40)
    assert s.orchestrator_model == "claude-sonnet-5-5"
