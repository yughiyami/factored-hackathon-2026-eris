"""Data contracts for the silver layer.

Each column declares: target type, whether it is required (hard contract -> row is
quarantined when violated), an optional numeric range or allowed set (soft contract ->
value is nulled and counted), and an optional value map used to normalize the
Spanish/English mix delivered by the source systems into canonical codes.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Col:
    type: str = "VARCHAR"
    required: bool = False
    range: tuple[float, float] | None = None
    allowed: tuple[str, ...] | None = None
    map: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Contract:
    pk: tuple[str, ...]
    columns: dict[str, Col]
    order_by: str = "_ingested_at"  # newest wins on PK collisions
    fks: dict[str, str] = field(default_factory=dict)  # column -> "table.column"


COUNTRY = {"México": "Mexico", "Mexico": "Mexico", "Colombia": "Colombia", "Argentina": "Argentina"}
SENTIMENT = {
    "Muy Positivo": "very_positive", "Positivo": "positive", "Neutral": "neutral",
    "Negativo": "negative", "Muy Negativo": "very_negative",
    "Very Positive": "very_positive", "Positive": "positive", "Negative": "negative",
    "Very Negative": "very_negative",
}
REASON = {
    "Transaccional": "transactional", "Producto": "product", "Queja": "complaint",
    "Técnico": "technical", "Comercial": "commercial", "Retención": "retention",
}
PRODUCT_TYPE = {
    "Cuenta Ahorro": "savings_account", "Cuenta Corriente": "checking_account",
    "Tarjeta Crédito": "credit_card", "Tarjeta Débito": "debit_card",
    "Préstamo Personal": "personal_loan", "Préstamo Hipotecario": "mortgage",
    "Inversión": "investment", "Seguro": "insurance",
}
TS, D, I, F, B = "TIMESTAMP", "DATE", "INTEGER", "DOUBLE", "BOOLEAN"

CURRENCIES = ("MXN", "COP", "ARS", "USD")
# Literal 'nan' leaked into templated text by the source system.
CAMPAIGN_SUBJECT = {"¡Oferta especial en nan!": "¡Oferta especial!"}

CONTRACTS: dict[str, Contract] = {
    "daily_exchange_rates": Contract(
        pk=("date", "source_currency", "target_currency"),
        columns={
            "date": Col(D, required=True), "source_currency": Col(required=True, allowed=CURRENCIES),
            "target_currency": Col(required=True, allowed=CURRENCIES),
            "exchange_rate": Col(F, required=True, range=(1e-9, 1e9)),
            "buy_rate": Col(F, range=(1e-9, 1e9)), "sell_rate": Col(F, range=(1e-9, 1e9)), "source": Col(),
        },
    ),
    "marketing_campaigns": Contract(
        pk=("campaign_id",),
        columns={
            "campaign_id": Col(required=True), "campaign_name": Col(required=True),
            "campaign_type": Col(required=True), "campaign_objective": Col(required=True),
            "promoted_product": Col(map=PRODUCT_TYPE), "target_segment": Col(),
            "target_country": Col(map=COUNTRY), "start_date": Col(D, required=True),
            "end_date": Col(D, required=True), "budget": Col(F, range=(0, 1e12)),
            "campaign_status": Col(required=True), "expected_conversion_rate": Col(F, range=(0, 100)),
        },
    ),
    "branches": Contract(
        pk=("branch_id",),
        columns={
            "branch_id": Col(required=True), "branch_type": Col(), "city": Col(), "state": Col(),
            "country": Col(map=COUNTRY), "branch_status": Col(),
        },
    ),
    "customers": Contract(
        pk=("customer_id",),
        columns={
            "customer_id": Col(required=True), "document_type": Col(required=True),
            "date_of_birth": Col(D, required=True), "gender": Col(allowed=("M", "F", "O")),
            "city": Col(required=True), "state": Col(required=True),
            "country": Col(required=True, map=COUNTRY, allowed=("Mexico", "Colombia", "Argentina")),
            "detected_accent": Col(allowed=("mexican", "colombian", "argentine", "neutral")),
            "segment": Col(required=True, allowed=("Premium", "Plus", "Basic", "Student")),
            "credit_score": Col(I, range=(300, 850)),
            "estimated_monthly_income": Col(F, range=(0, 1e12)),
            "occupation": Col(), "marital_status": Col(), "education_level": Col(),
            "registration_date": Col(TS, required=True), "registration_branch_id": Col(),
            "customer_status": Col(required=True, allowed=("Active", "Inactive", "Suspended", "Closed")),
            "accepts_marketing": Col(B),
        },
        fks={"registration_branch_id": "branches.branch_id"},
    ),
    "service_agents": Contract(
        pk=("agent_id",),
        columns={
            "agent_id": Col(required=True), "native_accent": Col(), "country_of_origin": Col(map=COUNTRY),
            "assigned_branch_id": Col(), "agent_type": Col(), "experience_level": Col(),
            "languages": Col(), "specialty": Col(), "hire_date": Col(D),
            "avg_csat": Col(F, range=(1, 5)), "total_monthly_interactions": Col(I, range=(0, 1e6)),
            "agent_status": Col(), "work_shift": Col(),
        },
        fks={"assigned_branch_id": "branches.branch_id"},
    ),
    "products": Contract(
        pk=("product_id",),
        columns={
            "product_id": Col(required=True), "customer_id": Col(required=True),
            "product_type": Col(required=True, map=PRODUCT_TYPE, allowed=tuple(PRODUCT_TYPE.values())),
            "currency": Col(allowed=("MXN", "COP", "ARS", "USD")), "current_balance": Col(F),
            "credit_limit": Col(F, range=(0, 1e12)), "interest_rate": Col(F, range=(0, 200)),
            "opening_date": Col(D, required=True), "product_status": Col(), "opening_channel": Col(),
            "has_linked_app": Col(B), "days_past_due": Col(I, range=(0, 10000)),
        },
        fks={"customer_id": "customers.customer_id"},
    ),
    "call_center_interactions": Contract(
        pk=("interaction_id",),
        columns={
            "interaction_id": Col(required=True), "interaction_date": Col(TS, required=True),
            "process_date": Col(D, required=True), "customer_id": Col(required=True), "agent_id": Col(),
            "interaction_type": Col(required=True), "channel": Col(required=True),
            "contact_reason": Col(required=True, map=REASON, allowed=tuple(REASON.values())),
            "duration_seconds": Col(I, range=(0, 6 * 3600)),
            "wait_time_seconds": Col(I, range=(0, 6 * 3600)),
            "was_resolved": Col(B), "requires_followup": Col(B, required=True),
            "detected_sentiment": Col(map=SENTIMENT, allowed=tuple(set(SENTIMENT.values()))),
            "sentiment_score": Col(F, range=(-1, 1)),
            "customer_detected_accent": Col(), "agent_used_accent": Col(),
            "was_escalated": Col(B, required=True), "mentioned_products": Col(),
            "has_transcript": Col(B), "has_recording": Col(B),
        },
        order_by="process_date",
        fks={"customer_id": "customers.customer_id", "agent_id": "service_agents.agent_id"},
    ),
    "call_transcripts": Contract(
        pk=("transcript_id",),
        columns={
            "transcript_id": Col(required=True), "interaction_id": Col(required=True),
            "customer_id": Col(required=True), "agent_id": Col(), "full_text": Col(required=True),
            "customer_text": Col(), "agent_text": Col(), "detected_language": Col(),
            "detected_accent": Col(), "accent_confidence": Col(F, range=(0, 1)),
            "detected_intents": Col(), "main_topics": Col(map=REASON), "audio_quality": Col(),
            "duration_seconds": Col(I, range=(0, 6 * 3600)),
        },
        fks={"interaction_id": "call_center_interactions.interaction_id"},
    ),
    "satisfaction_surveys": Contract(
        pk=("survey_id",),
        columns={
            "survey_id": Col(required=True), "survey_date": Col(TS, required=True),
            "interaction_id": Col(), "customer_id": Col(required=True), "agent_id": Col(),
            "survey_type": Col(required=True, allowed=("CSAT", "NPS", "CES")), "send_channel": Col(),
            "main_score": Col(I, required=True, range=(0, 10)), "nps_category": Col(),
            "question_1_response": Col(I, range=(1, 5)), "question_2_response": Col(I, range=(1, 5)),
            "question_3_response": Col(I, range=(1, 5)), "open_comments": Col(),
            "comment_sentiment": Col(), "response_time_hours": Col(F, range=(0, 24 * 365)),
        },
        fks={"interaction_id": "call_center_interactions.interaction_id"},
    ),
    "complaints": Contract(
        pk=("complaint_id",),
        columns={
            "complaint_id": Col(required=True), "creation_date": Col(TS, required=True),
            "customer_id": Col(required=True), "case_type": Col(required=True),
            "category": Col(required=True), "subcategory": Col(), "reception_channel": Col(required=True),
            "affected_product_id": Col(), "related_branch_id": Col(), "origin_interaction_id": Col(),
            "description": Col(), "claimed_amount": Col(F, range=(0, 1e12)), "currency": Col(),
            "priority": Col(required=True), "status": Col(required=True), "assigned_agent_id": Col(),
            "assignment_date": Col(TS), "first_response_date": Col(TS), "resolution_date": Col(TS),
            "closing_date": Col(TS), "sla_breached": Col(B, required=True),
            "resolution_days": Col(I, range=(0, 3650)), "compensation_granted": Col(F, range=(0, 1e12)),
            "resolution_satisfaction": Col(I, range=(1, 5)), "is_repeat_complainer": Col(B, required=True),
        },
        fks={"customer_id": "customers.customer_id", "affected_product_id": "products.product_id"},
    ),
    "transactions": Contract(
        pk=("transaction_id",),
        columns={
            "transaction_id": Col(required=True), "transaction_date": Col(TS, required=True),
            "process_date": Col(D, required=True), "product_id": Col(required=True),
            "customer_id": Col(required=True), "transaction_type": Col(required=True),
            "transaction_category": Col(), "amount": Col(F, required=True, range=(0, 1e13)),
            "currency": Col(required=True, allowed=CURRENCIES), "amount_usd": Col(F, range=(0, 1e12)),
            "channel": Col(required=True), "branch_id": Col(), "merchant_name": Col(),
            "merchant_category": Col(), "transaction_country": Col(required=True, map=COUNTRY),
            "transaction_city": Col(),
            "transaction_status": Col(required=True, allowed=("Approved", "Declined", "Pending", "Reversed")),
            "response_code": Col(), "is_fraud": Col(B, required=True), "fraud_score": Col(F, range=(0, 100)),
            "latitude": Col(F, range=(-90, 90)), "longitude": Col(F, range=(-180, 180)),
        },
        order_by="process_date",
        fks={"customer_id": "customers.customer_id", "product_id": "products.product_id"},
    ),
    "digital_events": Contract(
        pk=("event_id",),
        columns={
            "event_id": Col(required=True), "event_date": Col(TS, required=True),
            "process_date": Col(D, required=True), "customer_id": Col(), "session_id": Col(required=True),
            "event_type": Col(required=True), "event_category": Col(required=True),
            "channel": Col(required=True), "platform": Col(), "app_version": Col(), "page_url": Col(),
            "action": Col(), "element_id": Col(), "product_id": Col(), "event_value": Col(F),
            "duration_seconds": Col(I, range=(0, 24 * 3600)), "ip_country": Col(map=COUNTRY),
            "is_mobile": Col(B, required=True), "utm_medium": Col(), "utm_campaign": Col(),
        },
        order_by="process_date",
        fks={"customer_id": "customers.customer_id"},
    ),
    "campaign_sends": Contract(
        pk=("send_id",),
        columns={
            "send_id": Col(required=True), "send_date": Col(TS, required=True),
            "process_date": Col(D, required=True), "campaign_id": Col(required=True),
            "customer_id": Col(required=True), "send_channel": Col(required=True),
            "subject": Col(map=CAMPAIGN_SUBJECT), "send_status": Col(required=True),
            "was_delivered": Col(B, required=True), "was_opened": Col(B), "was_clicked": Col(B),
            "click_count": Col(I, range=(0, 1000)), "had_conversion": Col(B, required=True),
            "conversion_value": Col(F, range=(0, 1e12)), "open_device": Col(), "failure_reason": Col(),
            "send_cost": Col(F, range=(0, 1000)),
        },
        order_by="process_date",
        fks={"customer_id": "customers.customer_id", "campaign_id": "marketing_campaigns.campaign_id"},
    ),
}
