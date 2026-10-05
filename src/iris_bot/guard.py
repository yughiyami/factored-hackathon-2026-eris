"""Prompt-injection guard for user text and for tool outputs (tool data is always treated as data).

Deterministic pattern matching in ES / PT-BR / EN. It runs before the classifier and the LLM; a
detection forces a `prompt_injection` handoff and the message is never forwarded to the LLM.
Tool outputs are sanitized: suspicious free-text fields are replaced before they reach the LLM or
the customer.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

_PATTERNS = [
    # override / ignore instructions
    r"\b(ignor[ae]\w*|olvid[ae]\w*|esque[cç]\w*|disregard|forget)\b.{0,40}\b(instruc\w*|instru[cç][oõ]\w*|regras?|reglas?|rules?|prompt|polic\w*|pol[ií]tica\w*)",
    r"\b(previous|anteriores?|prior)\s+(instructions|instrucciones|instru[cç][oõ]es)",
    # role play / persona switch
    r"\b(ahora eres|you are now|voc[eê] agora [eé]|act[uú]a como|act as|finge que|pretend)\b",
    r"\b(modo|mode)\s+(desarrollador|developer|dev|desenvolvedor|admin|administrador|god|dios|deus|debug)\b",
    r"\bjailbreak|\bDAN\b",
    # prompt exfiltration
    r"\b(system prompt|prompt del sistema|prompt do sistema|tus instrucciones|suas instru[cç][oõ]es|your instructions)\b",
    r"\b(revela|muestra|mostra|show|print|imprime)\b.{0,30}\b(prompt|instrucciones|instru[cç][oõ]es|configura[cç][aã]o|secret|secreto|api key|token)",
    # markup / tool spoofing
    r"</?\s*(system|assistant|tool|tool_result|tool_use|instructions?)\s*>",
    r"\[\s*(system|admin|sistema)\s*\]",
    r"\{\{.*\}\}",
    r"<script",
    # direct action coercion
    r"\b(aprueba|aprove|approve|autoriza|autorize|authorize)\b.{0,30}\b(reembolso|refund|reembolso|estorno|devoluci[oó]n|transfer\w*)",
    r"\b(sin verificar|sem verificar|without (verification|checking)|skip (the )?(verification|checks|policy))\b",
]
_RX = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _PATTERNS]

CANARY = "IRIS-CANARY-7f3a"  # appears only in the system prompt; must never be echoed to users


@dataclass
class GuardResult:
    detected: bool
    matches: list[str]


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def scan(text: str) -> GuardResult:
    t = _norm(text or "")
    hits = [rx.pattern[:40] for rx in _RX if rx.search(t)]
    return GuardResult(bool(hits), hits)


REDACTED = "[contenido no confiable omitido]"


def sanitize_tool_output(value: Any) -> tuple[Any, int]:
    """Recursively replace suspicious strings in tool output. Returns (clean_value, n_redactions)."""
    if isinstance(value, str):
        return (REDACTED, 1) if scan(value).detected else (value, 0)
    if isinstance(value, dict):
        out, n = {}, 0
        for k, v in value.items():
            out[k], m = sanitize_tool_output(v)
            n += m
        return out, n
    if isinstance(value, list):
        items, n = [], 0
        for v in value:
            c, m = sanitize_tool_output(v)
            items.append(c)
            n += m
        return items, n
    return value, 0
