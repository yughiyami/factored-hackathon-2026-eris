"""Conversation state and the human-handoff contract."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Stage(StrEnum):
    START = "START"
    AWAIT_CUSTOMER_ID = "AWAIT_CUSTOMER_ID"
    AWAIT_OTP = "AWAIT_OTP"
    AWAIT_REQUEST = "AWAIT_REQUEST"
    AWAIT_CLARIFICATION = "AWAIT_CLARIFICATION"
    AWAIT_TXN_SELECTION = "AWAIT_TXN_SELECTION"
    AWAIT_CONFIRM = "AWAIT_CONFIRM"
    AWAIT_FEEDBACK = "AWAIT_FEEDBACK"
    AWAIT_HANDOFF_OFFER = "AWAIT_HANDOFF_OFFER"
    AWAIT_CSAT = "AWAIT_CSAT"
    CLOSED = "CLOSED"
    HANDED_OFF = "HANDED_OFF"


TERMINAL = {Stage.CLOSED, Stage.HANDED_OFF}
AUTHENTICATED_STAGES = {Stage.AWAIT_REQUEST, Stage.AWAIT_CLARIFICATION, Stage.AWAIT_TXN_SELECTION,
                        Stage.AWAIT_CONFIRM, Stage.AWAIT_FEEDBACK, Stage.AWAIT_HANDOFF_OFFER}


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Fact(BaseModel):
    key: str
    value: Any
    source: str               # tool that produced it
    evidence_id: str | None = None


class ActionRecord(BaseModel):
    action: str               # open_dispute | handoff_to_human | ...
    status: str               # verified | not_verified | failed | denied | cancelled
    reference_id: str | None = None
    detail: str | None = None
    at: str = Field(default_factory=_now)


class ConversationState(BaseModel):
    conversation_id: str
    case_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    channel: str = "web"
    language: str = "es"
    stage: Stage = Stage.START
    resume_stage: Stage | None = None

    # authentication
    claimed_customer_id: str | None = None
    challenge_id: str | None = None
    id_attempts: int = 0
    otp_attempts: int = 0
    token: str | None = None
    customer_id: str | None = None

    # understanding
    pending_request: str | None = None
    request_text: str | None = None
    intent: str | None = None
    intent_confidence: float | None = None
    intent_source: str | None = None
    clarification_turns: int = 0

    # dispute subprocess
    reason: str | None = None
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    selection_attempts: int = 0
    selected_txn: dict[str, Any] | None = None
    confirm_attempts: int = 0
    csat_attempts: int = 0
    offer_trigger: str | None = None

    # evidence trail
    facts: list[Fact] = Field(default_factory=list)
    actions: list[ActionRecord] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    transcript: list[dict[str, str]] = Field(default_factory=list)

    # outcome
    result_delivered: bool = False
    resolved: bool = False
    feedback: str | None = None
    csat: int | None = None
    denied: bool = False
    declined: bool = False
    injection_detected: bool = False
    handoff_id: str | None = None
    handoff_trigger: str | None = None

    # instrumentation
    turns: int = 0
    tool_calls: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    started_at: str = Field(default_factory=_now)
    ended_at: str | None = None

    def add_fact(self, key: str, value: Any, source: str, evidence_id: str | None = None) -> None:
        self.facts = [f for f in self.facts if f.key != key]
        self.facts.append(Fact(key=key, value=value, source=source, evidence_id=evidence_id))
        if evidence_id and evidence_id not in self.evidence_ids:
            self.evidence_ids.append(evidence_id)

    def outcome(self) -> str:
        if self.stage == Stage.HANDED_OFF:
            return "handoff"
        if self.resolved or (self.result_delivered and self.feedback != "no"):
            return "resolved"
        if self.denied:
            return "denied"
        if self.declined:
            return "declined"
        return "unresolved" if self.stage == Stage.CLOSED else "in_progress"


class HandoffPayload(BaseModel):
    """Everything a human agent needs to continue without asking the customer to repeat anything."""
    handoff_id: str
    created_at: str = Field(default_factory=_now)
    conversation_id: str
    case_id: str
    channel: str
    customer_id: str | None
    authenticated: bool
    language: str
    trigger: str
    reason: str
    priority: str
    intent: str | None
    intent_confidence: float | None
    request_summary: str
    verified_facts: list[Fact]
    actions_taken: list[ActionRecord]
    evidence_ids: list[str]
    open_questions: list[str]
    transcript: list[dict[str, str]]
    trace_id: str


class AgentReply(BaseModel):
    conversation_id: str
    case_id: str
    trace_id: str
    messages: list[str]
    stage: Stage
    language: str
    ended: bool
    handoff: HandoffPayload | None = None
    latency_ms: float = 0.0
