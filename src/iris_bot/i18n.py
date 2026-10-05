"""Customer-facing reply templates. Spanish (neutral LATAM) and Brazilian Portuguese."""
from __future__ import annotations

from typing import Any

T: dict[str, dict[str, str]] = {
    "welcome": {
        "es": "¡Hola! Soy IRIS, el asistente del banco. Te ayudo con cargos que no reconoces, cobros indebidos y el estado de tus reclamos.",
        "pt": "Olá! Sou a IRIS, assistente do banco. Ajudo com compras que você não reconhece, cobranças indevidas e o andamento das suas contestações.",
    },
    "ask_customer_id": {
        "es": "Para proteger tu cuenta primero necesito verificar tu identidad. ¿Cuál es tu número de cliente? (empieza con CLI-)",
        "pt": "Para proteger sua conta, preciso primeiro verificar sua identidade. Qual é o seu número de cliente? (começa com CLI-)",
    },
    "customer_id_invalid": {
        "es": "No pude leer el número de cliente. Debe tener el formato CLI-XXXXXXXXXXXX.",
        "pt": "Não consegui ler o número de cliente. Ele deve ter o formato CLI-XXXXXXXXXXXX.",
    },
    "otp_sent": {
        "es": "Te enviamos un código de 6 dígitos a tu celular registrado. Escríbelo aquí.",
        "pt": "Enviamos um código de 6 dígitos para o seu celular cadastrado. Digite aqui.",
    },
    "otp_invalid": {
        "es": "El código no es válido. Intenta de nuevo ({left} intento(s) restante(s)).",
        "pt": "O código não é válido. Tente novamente ({left} tentativa(s) restante(s)).",
    },
    "auth_ok": {
        "es": "¡Listo, identidad verificada!",
        "pt": "Pronto, identidade verificada!",
    },
    "session_expired": {
        "es": "Tu sesión expiró por seguridad. Te enviamos un nuevo código de 6 dígitos; escríbelo para continuar donde quedamos.",
        "pt": "Sua sessão expirou por segurança. Enviamos um novo código de 6 dígitos; digite para continuar de onde paramos.",
    },
    "ask_request": {
        "es": "¿En qué te puedo ayudar hoy?",
        "pt": "Como posso ajudar hoje?",
    },
    "clarify": {
        "es": "No estoy seguro de haber entendido. ¿Me cuentas si quieres (1) desconocer un cargo, (2) reclamar un cobro indebido o (3) consultar el estado de un reclamo?",
        "pt": "Não tenho certeza se entendi. Você quer (1) contestar uma compra que não reconhece, (2) reclamar de uma cobrança indevida ou (3) consultar o andamento de uma contestação?",
    },
    "out_of_scope": {
        "es": "Eso está fuera de lo que puedo hacer por este canal. Puedo ayudarte con cargos no reconocidos, cobros indebidos, el estado de tus reclamos y tus saldos. ¿Necesitas algo de eso?",
        "pt": "Isso está fora do que posso fazer por este canal. Posso ajudar com compras não reconhecidas, cobranças indevidas, o andamento das suas contestações e seus saldos. Precisa de algo disso?",
    },
    "candidates": {
        "es": "Encontré estos movimientos recientes. ¿Cuál quieres reclamar? Responde con el número:\n{items}",
        "pt": "Encontrei estas movimentações recentes. Qual você quer contestar? Responda com o número:\n{items}",
    },
    "no_match": {
        "es": "No encontré un movimiento que coincida con lo que describes. Estos son tus movimientos recientes; responde con el número si alguno es:\n{items}",
        "pt": "Não encontrei uma movimentação que corresponda ao que você descreveu. Estas são as recentes; responda com o número se alguma for:\n{items}",
    },
    "no_transactions": {
        "es": "No encontré movimientos recientes que se puedan reclamar en tus productos.",
        "pt": "Não encontrei movimentações recentes que possam ser contestadas nos seus produtos.",
    },
    "selection_invalid": {
        "es": "No entendí la opción. Responde solo con el número del movimiento (1 a {n}).",
        "pt": "Não entendi a opção. Responda só com o número da movimentação (1 a {n}).",
    },
    "confirm_dispute": {
        "es": "Voy a abrir un reclamo por {reason_label}:\n{item}\n¿Confirmas? (sí / no)",
        "pt": "Vou abrir uma contestação por {reason_label}:\n{item}\nVocê confirma? (sim / não)",
    },
    "reason_unrecognized_charge": {"es": "cargo no reconocido", "pt": "compra não reconhecida"},
    "reason_wrong_fee": {"es": "cobro indebido", "pt": "cobrança indevida"},
    "dispute_opened": {
        "es": "Abrí tu reclamo {dispute_id} y verifiqué que quedó registrado. Un analista lo revisará y te avisaremos por este medio. Guarda el número para darle seguimiento.",
        "pt": "Abri sua contestação {dispute_id} e verifiquei que ficou registrada. Um analista vai analisar e avisaremos por aqui. Guarde o número para acompanhar.",
    },
    "dispute_not_verified": {
        "es": "No pude confirmar que el reclamo quedara registrado, así que no lo doy por abierto.",
        "pt": "Não consegui confirmar que a contestação ficou registrada, então não a considero aberta.",
    },
    "dispute_cancelled": {
        "es": "Entendido, no abrí ningún reclamo.",
        "pt": "Entendido, não abri nenhuma contestação.",
    },
    "already_disputed": {
        "es": "Ese movimiento ya tiene un reclamo abierto ({ids}). No hace falta abrir otro.",
        "pt": "Essa movimentação já tem uma contestação aberta ({ids}). Não é preciso abrir outra.",
    },
    "ineligible_declined": {
        "es": "Ese intento de compra fue rechazado, así que no se te cobró nada y no hay monto que reclamar.",
        "pt": "Essa tentativa de compra foi recusada, então nada foi cobrado e não há valor a contestar.",
    },
    "ineligible_reversed": {
        "es": "Ese movimiento ya fue reversado; el dinero ya regresó a tu cuenta.",
        "pt": "Essa movimentação já foi estornada; o dinheiro já voltou para sua conta.",
    },
    "ineligible_too_old": {
        "es": "Ese movimiento tiene {age} días y el plazo para reclamar por este canal es de {max_age} días.",
        "pt": "Essa movimentação tem {age} dias e o prazo para contestar por este canal é de {max_age} dias.",
    },
    "status_none": {
        "es": "No tienes reclamos de cargos o cobros abiertos en los últimos 180 días.",
        "pt": "Você não tem contestações de compras ou cobranças nos últimos 180 dias.",
    },
    "status_list": {
        "es": "Estos son tus reclamos:\n{items}",
        "pt": "Estas são suas contestações:\n{items}",
    },
    "balance": {
        "es": "Tus productos:\n{items}",
        "pt": "Seus produtos:\n{items}",
    },
    "ask_feedback": {
        "es": "¿Esto resuelve tu solicitud? (sí / no)",
        "pt": "Isso resolve sua solicitação? (sim / não)",
    },
    "offer_human": {
        "es": "Lamento no haberlo resuelto. ¿Quieres que te pase con un asesor humano? (sí / no)",
        "pt": "Sinto não ter resolvido. Quer que eu transfira para um atendente humano? (sim / não)",
    },
    "ask_csat": {
        "es": "Antes de terminar: del 1 al 4, ¿qué tan satisfecho quedaste con la atención? (1 = nada, 4 = muy satisfecho)",
        "pt": "Antes de encerrar: de 1 a 4, quanto você ficou satisfeito com o atendimento? (1 = nada, 4 = muito satisfeito)",
    },
    "csat_invalid": {
        "es": "Responde con un número del 1 al 4, por favor.",
        "pt": "Responda com um número de 1 a 4, por favor.",
    },
    "goodbye": {
        "es": "¡Gracias! Que tengas un buen día.",
        "pt": "Obrigado! Tenha um ótimo dia.",
    },
    "handoff": {
        "es": "Te paso con un asesor humano ({reason}). Ya le compartí el resumen de tu caso para que no tengas que repetir nada. Número de atención: {handoff_id}.",
        "pt": "Vou transferir você para um atendente humano ({reason}). Já compartilhei o resumo do seu caso para você não precisar repetir nada. Protocolo: {handoff_id}.",
    },
    "handoff_failed": {
        "es": "Tuvimos un problema técnico al transferirte. Por favor llama a la línea de atención; tu caso quedó registrado en nuestros sistemas.",
        "pt": "Tivemos um problema técnico ao transferir você. Por favor ligue para a central de atendimento; seu caso ficou registrado.",
    },
    "denied": {
        "es": "No encontré esa información entre tus productos. Solo puedo consultar datos de la cuenta verificada en esta conversación.",
        "pt": "Não encontrei essa informação entre os seus produtos. Só posso consultar dados da conta verificada nesta conversa.",
    },
    "fallback_error": {
        "es": "Tuve un problema técnico procesando tu mensaje. No se realizó ninguna acción. ¿Puedes intentar de nuevo?",
        "pt": "Tive um problema técnico ao processar sua mensagem. Nenhuma ação foi realizada. Pode tentar novamente?",
    },
    "closed": {
        "es": "Esta conversación ya terminó. Escribe cualquier mensaje para empezar una nueva.",
        "pt": "Esta conversa já foi encerrada. Envie qualquer mensagem para começar uma nova.",
    },
}

HANDOFF_REASON: dict[str, dict[str, str]] = {
    "fraud_flag": {"es": "posible fraude, lo revisa el equipo de seguridad", "pt": "possível fraude, a equipe de segurança vai analisar"},
    "amount_over_threshold": {"es": "el monto requiere revisión de un asesor", "pt": "o valor exige análise de um atendente"},
    "customer_requested_human": {"es": "lo solicitaste", "pt": "você solicitou"},
    "repeated_unresolved_reason": {"es": "ya tienes un reclamo abierto por el mismo motivo", "pt": "você já tem uma contestação aberta pelo mesmo motivo"},
    "unsupported_intent": {"es": "esa gestión la hace un asesor", "pt": "essa solicitação é feita por um atendente"},
    "prompt_injection": {"es": "tu mensaje necesita revisión manual", "pt": "sua mensagem precisa de revisão manual"},
    "tool_failure": {"es": "nuestros sistemas no responden en este momento", "pt": "nossos sistemas não estão respondendo agora"},
    "not_understood": {"es": "no logré entender tu solicitud", "pt": "não consegui entender sua solicitação"},
    "result_insufficient": {"es": "la respuesta no fue suficiente", "pt": "a resposta não foi suficiente"},
    "auth_failed": {"es": "no fue posible verificar tu identidad", "pt": "não foi possível verificar sua identidade"},
    "no_matching_transaction": {"es": "no encontramos el movimiento", "pt": "não encontramos a movimentação"},
}

STATUS_LABEL = {
    "es": {"Approved": "aprobado", "Pending": "pendiente", "Declined": "rechazado", "Reversed": "reversado",
           "opened": "abierto", "Open": "abierto", "In Process": "en proceso", "Escalated": "escalado",
           "Resolved": "resuelto", "Closed": "cerrado", "Rejected": "rechazado"},
    "pt": {"Approved": "aprovado", "Pending": "pendente", "Declined": "recusado", "Reversed": "estornado",
           "opened": "aberta", "Open": "aberta", "In Process": "em análise", "Escalated": "escalada",
           "Resolved": "resolvida", "Closed": "encerrada", "Rejected": "rejeitada"},
}
TYPE_LABEL = {
    "es": {"Purchase": "Compra", "Payment": "Pago", "Withdrawal": "Retiro", "Adjustment": "Cargo/comisión",
           "Transfer": "Transferencia", "Deposit": "Depósito"},
    "pt": {"Purchase": "Compra", "Payment": "Pagamento", "Withdrawal": "Saque", "Adjustment": "Tarifa/ajuste",
           "Transfer": "Transferência", "Deposit": "Depósito"},
}
PRODUCT_LABEL = {
    "es": {"credit_card": "Tarjeta de crédito", "debit_card": "Tarjeta débito", "savings_account": "Cuenta de ahorros",
           "checking_account": "Cuenta corriente", "personal_loan": "Préstamo personal", "mortgage": "Hipoteca",
           "insurance": "Seguro", "investment": "Inversión"},
    "pt": {"credit_card": "Cartão de crédito", "debit_card": "Cartão de débito", "savings_account": "Poupança",
           "checking_account": "Conta corrente", "personal_loan": "Empréstimo pessoal", "mortgage": "Financiamento imobiliário",
           "insurance": "Seguro", "investment": "Investimento"},
}


def t(key: str, lang: str, **kw: Any) -> str:
    entry = T[key]
    return entry.get(lang, entry["es"]).format(**kw)


def money(amount: float | None, currency: str | None) -> str:
    if amount is None:
        return "-"
    return f"{amount:,.2f} {currency or ''}".strip()


def render_txn(txn: dict[str, Any], lang: str) -> str:
    """Disclosure-safe one-line rendering (fields allowed by policy only)."""
    label = txn.get("merchant") or TYPE_LABEL[lang].get(txn.get("type") or "", txn.get("type") or "")
    date = str(txn.get("date") or "")[:10]
    status = STATUS_LABEL[lang].get(txn.get("status") or "", txn.get("status") or "")
    return f"{date} · {label} · {money(txn.get('amount'), txn.get('currency'))} · {status} · *{txn.get('product_last4')}"
