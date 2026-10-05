"""Tool runtime: session scoping, bounded retries, fault injection, output sanitization, audit."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from iris_bot.auth import IdentityService, Session
from iris_bot.guard import sanitize_tool_output
from iris_bot.retry import RetryExhausted, retry_call
from iris_bot.storage import Store


class ToolError(Exception):
    """Business error: not retryable (e.g. not found, not eligible)."""


class PermissionDenied(ToolError):
    """The session's customer is not allowed to touch the requested resource."""


class ToolUnavailable(Exception):
    """Transient backend failure: retryable."""


class ToolFailure(Exception):
    """Retries exhausted."""

    def __init__(self, tool: str, attempts: int):
        super().__init__(f"{tool} failed after {attempts} attempts")
        self.tool = tool
        self.attempts = attempts


@dataclass
class CallContext:
    conversation_id: str
    trace_id: str
    token: str | None


class ToolRuntime:
    """Executes a tool function under the session in `token`; every call is audited."""

    def __init__(self, identity: IdentityService, store: Store, *, max_attempts: int = 3,
                 backoff_base: float = 0.2, backoff_max: float = 2.0, fault_tools: set[str] | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.identity = identity
        self.store = store
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.fault_tools = set(fault_tools or ())
        self.sleep = sleep

    def run(self, name: str, fn: Callable[..., Any], ctx: CallContext, *, require_session: bool = True,
            **args: Any) -> Any:
        session: Session | None = None
        if require_session:
            # Raises SessionExpired / InvalidSession: the agent re-authenticates.
            session = self.identity.validate(ctx.token)
        else:
            try:
                session = self.identity.validate(ctx.token)
            except Exception:  # noqa: BLE001 - handoff may happen before authentication
                session = None
        customer_id = session.customer_id if session else None
        t0 = time.perf_counter()
        attempts = 0

        def attempt() -> Any:
            nonlocal attempts
            attempts += 1
            if name in self.fault_tools:
                raise ToolUnavailable(f"injected fault in {name}")
            return fn(session, **args)

        try:
            result, _ = retry_call(attempt, max_attempts=self.max_attempts, retry_on=(ToolUnavailable,),
                                   base_delay=self.backoff_base, max_delay=self.backoff_max, sleep=self.sleep)
        except RetryExhausted as exc:
            self._audit(name, ctx, customer_id, t0, "tool_failure", {"attempts": attempts, "args": args})
            raise ToolFailure(name, exc.attempts) from exc
        except PermissionDenied as exc:
            self._audit(name, ctx, customer_id, t0, "unauthorized_access_attempt",
                        {"args": args, "error": str(exc)})
            raise
        except ToolError as exc:
            self._audit(name, ctx, customer_id, t0, "error", {"args": args, "error": str(exc)})
            raise
        clean, redactions = sanitize_tool_output(result)
        self._audit(name, ctx, customer_id, t0, "ok",
                    {"args": args, "attempts": attempts, "redactions": redactions})
        return clean

    def _audit(self, name: str, ctx: CallContext, customer_id: str | None, t0: float, outcome: str,
               detail: dict[str, Any]) -> None:
        self.store.audit(step="tool", tool=name, trace_id=ctx.trace_id, conversation_id=ctx.conversation_id,
                         customer_id=customer_id, latency_ms=(time.perf_counter() - t0) * 1000,
                         outcome=outcome, detail=detail)
