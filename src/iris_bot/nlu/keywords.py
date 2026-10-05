"""Keyword-rule intent baseline (ES + PT-BR). Deterministic; used as the baseline and as a fallback."""
from __future__ import annotations

import re
import unicodedata

INTENTS = (
    "dispute_unrecognized_charge", "dispute_wrong_fee", "dispute_status", "card_block_request",
    "balance_inquiry", "credit_eligibility", "talk_to_human", "greeting", "out_of_scope",
)


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", (text or "").lower())
    return "".join(c for c in t if not unicodedata.combining(c))


# Ordered: first match wins. Patterns run on accent-stripped lowercase text.
_RULES: list[tuple[str, list[str]]] = [
    ("talk_to_human", [r"\b(humano|asesor|operador|atendente|ejecutivo|gerente|especialista|funcionari[oa])\b",
                       r"\b(hablar|comunic\w*|pas\w*|transfi?er\w*|falar|atend\w*|contact\w*|llame|ligue)\b"
                       r".{0,25}\b(alguien|alguem|persona|pessoa|agente|ser humano)\b"]),
    ("card_block_request", [r"\bbloque\w*\b", r"\b(robaron|roubaram|perdi|extravi\w*)\b.*\b(tarjeta|cartao)\b"]),
    ("dispute_status", [r"\b(estado|status|andamento|seguimiento|como va)\b.*\b(reclamo|disputa|contestacao|caso|queja|solicitud|pedido)\b",
                        r"\b(reclamo|disputa|contestacao|queja)\b.*\b(estado|status|andamento)\b"]),
    ("dispute_wrong_fee", [r"\b(cobro indebido|cobranca indevida|comision|tarifa|taxa|cuota de manejo|anuidade|"
                           r"cobraron de mas|cobraram a mais|doble cobro|cobrado duas vezes|cobraron dos veces)\b"]),
    ("dispute_unrecognized_charge", [r"\b(no reconozco|nao reconheco|desconozco|desconheco|no hice|nao fiz|"
                                     r"no autorice|nao autorizei)\b",
                                     r"\b(cargo|compra|cobro|cobranca|transaccion|transacao)\b.*"
                                     r"\b(raro|extran\w*|estranh\w*|desconocid\w*|desconhecid\w*)\b"]),
    ("balance_inquiry", [r"\b(saldo|cuanto tengo|quanto tenho|disponible|disponivel)\b"]),
    ("credit_eligibility", [r"\b(prestamo|emprestimo|credito|aumento de cupo|aumento de limite|financiamiento|"
                            r"financiamento)\b"]),
    ("greeting", [r"^\s*(hola|ola|oi|buenas|buenos dias|bom dia|boa tarde|boa noite|buenas tardes|que tal|e ai)"
                  r"\b[\s!.,]*$"]),
]
_COMPILED = [(intent, [re.compile(p) for p in pats]) for intent, pats in _RULES]

_HUMAN_REQUEST = [
    re.compile(r"\b(hablar|comunic\w*|pasa\w*|pase\w*|transfi?er\w*|falar|quiero|quero|necesito|preciso|deseo|"
               r"exijo)\b.{0,30}\b(humano|asesor|agente|persona|pessoa|operador|atendente|alguien|alguem|"
               r"ejecutivo|gerente)\b"),
    re.compile(r"^\s*(un |una |um |uma )?(humano|asesor|agente|operador|atendente)\s*[.!]*\s*$"),
    re.compile(r"\bno quiero (un |hablar con un )?(robot|bot|maquina)\b|\bnao quero (um |falar com )?(robo|bot)\b"),
]


def asks_for_human(text: str) -> bool:
    """Explicit request for a human, checked at every turn (deterministic, high precision)."""
    t = normalize(text)
    if re.search(r"\b(usou|uso|usaron|usaram|utiliz\w*)\b", t):  # "una persona usó mi tarjeta" is not a request
        return False
    return any(rx.search(t) for rx in _HUMAN_REQUEST)


class KeywordClassifier:
    name = "keyword_rules"

    def predict(self, text: str) -> tuple[str, float]:
        t = normalize(text)
        for intent, rxs in _COMPILED:
            if any(rx.search(t) for rx in rxs):
                return intent, 0.9
        return "out_of_scope", 0.3

    def predict_many(self, texts: list[str]) -> list[str]:
        return [self.predict(t)[0] for t in texts]
