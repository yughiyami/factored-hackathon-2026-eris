"""Deterministic slot extraction and transaction matching (amounts, merchants, ids, dates, yes/no)."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from iris_bot.nlu.keywords import normalize

TXN_ID_RX = re.compile(r"\bTRX-[A-Z0-9]{8,}\b", re.IGNORECASE)
CUSTOMER_ID_RX = re.compile(r"\bCLI-[A-Z0-9]{8,}\b", re.IGNORECASE)
DISPUTE_ID_RX = re.compile(r"\bDSP-[A-F0-9]{12}\b", re.IGNORECASE)
OTP_RX = re.compile(r"\b(\d{6})\b")
_AMOUNT_RX = re.compile(r"(?<![\w-])\$?\s?(\d{1,3}(?:[.,\s]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)(?![\w-])")

_MONTHS = {"enero": 1, "janeiro": 1, "febrero": 2, "fevereiro": 2, "marzo": 3, "marco": 3, "abril": 4,
           "mayo": 5, "maio": 5, "junio": 6, "junho": 6, "julio": 7, "julho": 7, "agosto": 8,
           "septiembre": 9, "setiembre": 9, "setembro": 9, "octubre": 10, "outubro": 10,
           "noviembre": 11, "novembro": 11, "diciembre": 12, "dezembro": 12}
_STOP = {"de", "del", "la", "el", "en", "un", "una", "por", "que", "con", "los", "las", "mi", "meu", "minha",
         "compra", "cargo", "cobro", "tienda", "loja", "para", "pra", "uma", "foi", "fue", "no", "nao", "não"}

YES = {"si", "sí", "sip", "claro", "dale", "correcto", "confirmo", "ok", "okay", "de una", "exacto", "afirmativo",
       "sim", "isso", "pode", "certo", "confirmado", "beleza", "va", "sale", "listo", "por favor", "s", "yes",
       "esa", "essa", "eso", "es correcto", "está bien", "esta bien", "tá bom", "ta bom", "perfecto", "perfeito"}
NO = {"no", "não", "nao", "negativo", "cancelar", "cancela", "nop", "nope", "para nada", "nem", "tampoco",
      "n", "todavia no", "todavía no", "ainda não", "ainda nao", "ninguno", "ninguna", "nenhum", "nenhuma"}


def parse_amounts(text: str) -> list[float]:
    out: list[float] = []
    for m in _AMOUNT_RX.finditer(text or ""):
        raw = m.group(1).replace(" ", "")
        if re.fullmatch(r"\d{6}", raw):  # looks like an OTP, not an amount
            continue
        if "," in raw and "." in raw:
            dec = "," if raw.rfind(",") > raw.rfind(".") else "."
            thou = "." if dec == "," else ","
            raw = raw.replace(thou, "").replace(dec, ".")
        elif "," in raw:
            parts = raw.split(",")
            raw = raw.replace(",", ".") if len(parts[-1]) in (1, 2) and len(parts) == 2 else raw.replace(",", "")
        elif raw.count(".") > 1 or (raw.count(".") == 1 and len(raw.split(".")[-1]) == 3):
            raw = raw.replace(".", "")
        try:
            val = float(raw)
        except ValueError:
            continue
        if val > 0:
            out.append(val)
    return out


def parse_day_month(text: str) -> list[tuple[int, int]]:
    t = normalize(text)
    found = [(int(d), _MONTHS[m]) for d, m in re.findall(r"\b(\d{1,2}) de (\w+)", t) if m in _MONTHS]
    found += [(int(d), int(m)) for d, m in re.findall(r"\b(\d{1,2})/(\d{1,2})\b", t) if 1 <= int(m) <= 12]
    return found


def yes_no(text: str) -> bool | None:
    t = normalize(text).strip(" !.¡¿?,")
    first = re.split(r"[\s,.!]+", t)[0] if t else ""
    if t in {normalize(x) for x in NO} or first in {"no", "nao", "negativo", "nop", "ninguno", "ninguna", "nenhum", "nenhuma"}:
        return False
    if t in {normalize(x) for x in YES} or first in {normalize(x) for x in YES}:
        return True
    return None


def parse_choice(text: str, n: int) -> int | None:
    t = normalize(text).strip()
    m = re.fullmatch(r"(?:la |el |a |o |opcion |opcao |numero |n )?(\d)\.?", t)
    if m and 1 <= int(m.group(1)) <= n:
        return int(m.group(1)) - 1
    ordinals = {"primer": 0, "primera": 0, "primero": 0, "segund": 1, "tercer": 2, "cuart": 3, "quart": 3, "quint": 4}
    for k, v in ordinals.items():
        if t.startswith(("la " + k, "el " + k, "a " + k, "o " + k, k)) and v < n:
            return v
    return None


def _merchant_tokens(name: str | None) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", normalize(name or "")) if w not in _STOP}


def match_transactions(text: str, txns: list[dict[str, Any]]) -> list[tuple[dict[str, Any], float]]:
    """Score public transactions against the customer's description. Returns [(txn, score)] desc."""
    amounts = parse_amounts(text)
    dates = parse_day_month(text)
    words = set(re.findall(r"[a-z]{4,}", normalize(text))) - _STOP
    ids = {m.upper() for m in TXN_ID_RX.findall(text or "")}
    scored = []
    for t in txns:
        s = 0.0
        if t["transaction_id"].upper() in ids:
            s += 10
        amt = t.get("amount")
        if amt and any(abs(a - amt) <= max(0.01, 0.005 * amt) for a in amounts):
            s += 3
        mt = _merchant_tokens(t.get("merchant"))
        if mt and mt & words:
            s += 2 * len(mt & words) / len(mt)
        if dates and t.get("date"):
            d = datetime.fromisoformat(str(t["date"]))
            if (d.day, d.month) in dates:
                s += 1.5
        if s > 0:
            scored.append((t, s))
    scored.sort(key=lambda x: -x[1])
    return scored


def has_transaction_hint(text: str) -> bool:
    return bool(parse_amounts(text) or parse_day_month(text) or TXN_ID_RX.search(text or ""))
