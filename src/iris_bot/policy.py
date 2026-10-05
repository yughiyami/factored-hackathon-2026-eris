"""Deterministic policy engine. Rules come from config/policy.yaml; no LLM is involved here."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

HANDOFF_TRIGGERS = (
    "fraud_flag", "amount_over_threshold", "customer_requested_human", "repeated_unresolved_reason",
    "unsupported_intent", "prompt_injection", "tool_failure", "not_understood", "result_insufficient",
    "auth_failed", "no_matching_transaction",
)


@dataclass
class Eligibility:
    eligible: bool                      # can the bot open the dispute automatically?
    reason_code: str                    # ok | too_old | status_declined | status_reversed | ...
    handoff_trigger: str | None = None  # set when a human must take over
    checks: dict[str, Any] = field(default_factory=dict)


class Policy:
    def __init__(self, raw: dict[str, Any]):
        self.raw = raw
        d = raw["dispute"]
        self.max_auto_amount_usd: float = float(d["max_auto_amount_usd"])
        self.max_age_days: int = int(d["max_transaction_age_days"])
        self.allowed_statuses: set[str] = set(d["allowed_statuses"])
        self.disputable_types: set[str] = set(d["disputable_types"])
        self.fraud_score_threshold: float = float(d["fraud_score_threshold"])
        self.unknown_amount_requires_human: bool = bool(d.get("unknown_amount_requires_human", True))
        c = raw["conversation"]
        self.max_clarification_turns: int = int(c["max_clarification_turns"])
        self.min_confidence: float = float(c["min_classifier_confidence"])
        self.max_candidates: int = int(c["max_candidates_shown"])
        self.max_selection_attempts: int = int(c["max_selection_attempts"])
        self.max_otp_attempts: int = int(c["max_otp_attempts"])
        r = raw["repeat_contact"]
        self.repeat_window_days: int = int(r["window_days"])
        self.repeat_min_prior: int = int(r["min_prior_unresolved"])
        self.unresolved_statuses: set[str] = set(r["unresolved_statuses"])
        i = raw["intents"]
        self.automated_intents: set[str] = set(i["automated"])
        self.handoff_intents: set[str] = set(i["handoff"])
        self.conversational_intents: set[str] = set(i["conversational"])
        self.decline_intents: set[str] = set(i["decline"])
        self.triggers: dict[str, dict[str, Any]] = raw["handoff_triggers"]
        rel = raw.get("reliability", {})
        self.tool_max_attempts: int = int(rel.get("tool_max_attempts", 3))
        self.llm_max_attempts: int = int(rel.get("llm_max_attempts", 3))
        self.backoff_base: float = float(rel.get("backoff_base_seconds", 0.2))
        self.backoff_max: float = float(rel.get("backoff_max_seconds", 2.0))

    @classmethod
    def load(cls, path: Path) -> Policy:
        with open(path, encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh))

    # --- handoff triggers -------------------------------------------------------------------
    def trigger_enabled(self, trigger: str) -> bool:
        return bool(self.triggers.get(trigger, {}).get("enabled", False))

    def trigger_priority(self, trigger: str) -> str:
        return str(self.triggers.get(trigger, {}).get("priority", "normal"))

    def intent_route(self, intent: str) -> str:
        """automated | handoff | conversational | decline"""
        if intent in self.automated_intents:
            return "automated"
        if intent in self.handoff_intents:
            return "handoff"
        if intent in self.conversational_intents:
            return "conversational"
        return "decline"

    def needs_clarification(self, confidence: float) -> bool:
        return confidence < self.min_confidence

    def clarification_exhausted(self, turns_used: int) -> bool:
        return turns_used >= self.max_clarification_turns

    def repeated_unresolved(self, prior_open_same_reason: int) -> bool:
        return (self.trigger_enabled("repeated_unresolved_reason")
                and prior_open_same_reason >= self.repeat_min_prior)

    # --- dispute eligibility (pure) ---------------------------------------------------------
    def check_dispute_eligibility(self, txn: dict[str, Any], as_of: datetime) -> Eligibility:
        ts = txn["transaction_date"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        age = (as_of - ts).days
        amount_usd = txn.get("amount_usd")
        fraud = bool(txn.get("is_fraud")) or float(txn.get("fraud_score") or 0) >= self.fraud_score_threshold
        checks = {"age_days": age, "status": txn.get("transaction_status"), "type": txn.get("transaction_type"),
                  "amount_usd": amount_usd, "fraud_flag": fraud}
        # Order matters: risk triggers first (human), then informational ineligibility.
        if fraud and self.trigger_enabled("fraud_flag"):
            return Eligibility(False, "fraud_flag", "fraud_flag", checks)
        status = txn.get("transaction_status")
        if status == "Declined":
            return Eligibility(False, "status_declined", None, checks)
        if status == "Reversed":
            return Eligibility(False, "status_reversed", None, checks)
        if status not in self.allowed_statuses:
            return Eligibility(False, "status_not_allowed", "unsupported_intent", checks)
        if txn.get("transaction_type") not in self.disputable_types:
            return Eligibility(False, "type_not_disputable", "unsupported_intent", checks)
        if age > self.max_age_days:
            return Eligibility(False, "too_old", None, checks)
        if amount_usd is None and self.unknown_amount_requires_human:
            return Eligibility(False, "amount_unknown", "amount_over_threshold", checks)
        if amount_usd is not None and float(amount_usd) > self.max_auto_amount_usd:
            return Eligibility(False, "amount_over_threshold", "amount_over_threshold", checks)
        return Eligibility(True, "ok", None, checks)
