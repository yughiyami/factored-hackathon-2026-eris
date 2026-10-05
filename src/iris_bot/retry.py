"""Bounded retries with exponential backoff and jitter, shared by the LLM and tool layers."""
from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class RetryExhausted(Exception):
    def __init__(self, attempts: int, last: BaseException):
        super().__init__(f"failed after {attempts} attempts: {last!r}")
        self.attempts = attempts
        self.last = last


def retry_call(
    fn: Callable[[], T],
    *,
    max_attempts: int,
    retry_on: tuple[type[BaseException], ...],
    base_delay: float = 0.2,
    max_delay: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[T, int]:
    """Run fn; retry only on `retry_on` exceptions. Returns (result, attempts_used)."""
    last: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(), attempt
        except retry_on as exc:  # noqa: PERF203
            last = exc
            if attempt == max_attempts:
                break
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            sleep(delay * (0.5 + random.random() / 2))
    raise RetryExhausted(max_attempts, last)  # type: ignore[arg-type]
