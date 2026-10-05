"""Rule-based ES vs PT-BR language detection (marker words + orthography). Default: es."""
from __future__ import annotations

import re

_PT = {
    "não", "nao", "você", "voce", "vocês", "meu", "minha", "cartão", "cartao", "obrigado", "obrigada", "olá",
    "ola", "oi", "estou", "uma", "cobrança", "cobranca", "reconheço", "reconheco", "fatura", "conta", "quero",
    "preciso", "falar", "atendente", "tá", "ta", "também", "tambem", "isso", "está", "esta", "compra", "foi",
    "sim", "bom", "dia", "boa", "tarde", "noite", "valor", "pra", "pro", "com", "nunca", "fiz", "essa", "esse",
    "dinheiro", "saldo", "bloquear", "crédito", "credito", "posso", "ajuda", "sou", "eu", "tudo",
    "bem", "aqui", "agora", "ainda", "contestar", "contestação", "estorno", "taxa", "tarifa", "cobraram",
    "minhas", "meus", "cadê", "pessoa", "humano", "pedido", "andamento", "status", "quanto",
    "tenho", "aprovado", "empréstimo", "emprestimo", "roubado", "perdi", "golpe", "pix", "num",
}
_ES = {
    "no", "mi", "mis", "tarjeta", "cobro", "cobraron", "usted", "hola", "quiero", "necesito", "hablar",
    "agente", "estoy", "una", "reconozco", "cuenta", "gracias", "sí", "si", "buenos", "días", "dias", "buenas",
    "tardes", "noches", "compra", "fue", "nunca", "hice", "esa", "ese", "dinero", "saldo", "bloquear",
    "crédito", "puedo", "ayuda", "soy", "yo", "todo", "bien", "aquí", "aqui", "ahora", "todavía", "todavia",
    "reclamo", "disputa", "comisión", "comision", "cargo", "persona", "humano", "pedido", "estado", "cuánto",
    "cuanto", "tengo", "préstamo", "prestamo", "robaron", "perdí", "perdi", "vos", "che", "plata", "parce",
    "pues", "ahorita", "qué", "que", "cómo", "como", "dónde", "donde", "por", "favor", "del", "los", "las",
    "hay", "este", "esta", "eso", "solicitud", "pana", "chévere", "boludo", "güey", "wey", "ya", "le",
}
_WORD = re.compile(r"[a-záéíóúâêôãõçñü]+", re.IGNORECASE)


def detect_language(text: str, default: str = "es") -> tuple[str, float]:
    """Return (lang, confidence) where lang in {"es", "pt"}."""
    t = (text or "").lower()
    words = _WORD.findall(t)
    pt = sum(1 for w in words if w in _PT)
    es = sum(1 for w in words if w in _ES)
    # Orthographic cues: ã/õ/ç and "lh"/"nh" digraphs are Portuguese; ñ and ¿¡ are Spanish.
    pt += 2 * len(re.findall(r"[ãõç]", t)) + len(re.findall(r"\b\w*(nh|lh)\w*\b", t))
    es += 2 * len(re.findall(r"[ñ¿¡]", t))
    total = pt + es
    if total == 0:
        return default, 0.0
    if pt > es:
        return "pt", pt / total
    if es > pt:
        return "es", es / total
    return default, 0.5
