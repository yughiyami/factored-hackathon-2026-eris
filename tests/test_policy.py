from datetime import datetime, timedelta

import pytest

from iris_bot.config import ROOT
from iris_bot.policy import Policy

AS_OF = datetime(2026, 6, 18)


@pytest.fixture(scope="module")
def policy() -> Policy:
    return Policy.load(ROOT / "config" / "policy.yaml")


def txn(**kw):
    base = {"transaction_date": AS_OF - timedelta(days=10), "transaction_status": "Approved",
            "transaction_type": "Purchase", "amount_usd": 50.0, "is_fraud": False, "fraud_score": 5.0}
    return base | kw


def test_eligible_purchase(policy):
    el = policy.check_dispute_eligibility(txn(), AS_OF)
    assert el.eligible and el.reason_code == "ok" and el.handoff_trigger is None


def test_fraud_flag_forces_handoff(policy):
    for t in (txn(is_fraud=True), txn(fraud_score=policy.fraud_score_threshold)):
        el = policy.check_dispute_eligibility(t, AS_OF)
        assert not el.eligible and el.handoff_trigger == "fraud_flag"


def test_amount_threshold(policy):
    el = policy.check_dispute_eligibility(txn(amount_usd=policy.max_auto_amount_usd + 0.01), AS_OF)
    assert el.handoff_trigger == "amount_over_threshold"
    assert policy.check_dispute_eligibility(txn(amount_usd=policy.max_auto_amount_usd), AS_OF).eligible


def test_unknown_amount_requires_human(policy):
    assert policy.check_dispute_eligibility(txn(amount_usd=None), AS_OF).handoff_trigger == "amount_over_threshold"


def test_age_limit(policy):
    ok = txn(transaction_date=AS_OF - timedelta(days=policy.max_age_days))
    old = txn(transaction_date=AS_OF - timedelta(days=policy.max_age_days + 1))
    assert policy.check_dispute_eligibility(ok, AS_OF).eligible
    el = policy.check_dispute_eligibility(old, AS_OF)
    assert not el.eligible and el.reason_code == "too_old" and el.handoff_trigger is None


@pytest.mark.parametrize("status,code", [("Declined", "status_declined"), ("Reversed", "status_reversed")])
def test_status_informational(policy, status, code):
    el = policy.check_dispute_eligibility(txn(transaction_status=status), AS_OF)
    assert not el.eligible and el.reason_code == code


def test_non_disputable_type(policy):
    el = policy.check_dispute_eligibility(txn(transaction_type="Deposit"), AS_OF)
    assert not el.eligible and el.handoff_trigger == "unsupported_intent"


def test_conversation_rules(policy):
    assert policy.max_clarification_turns == 2
    assert policy.min_confidence == 0.6
    assert policy.max_age_days == 120
    assert policy.needs_clarification(0.59) and not policy.needs_clarification(0.6)
    assert policy.clarification_exhausted(2) and not policy.clarification_exhausted(1)
    assert policy.repeated_unresolved(1) and not policy.repeated_unresolved(0)


def test_intent_routes(policy):
    assert policy.intent_route("dispute_unrecognized_charge") == "automated"
    assert policy.intent_route("talk_to_human") == "handoff"
    assert policy.intent_route("card_block_request") == "handoff"
    assert policy.intent_route("greeting") == "conversational"
    assert policy.intent_route("out_of_scope") == "decline"


def test_all_handoff_triggers_configured(policy):
    from iris_bot.policy import HANDOFF_TRIGGERS
    for trig in HANDOFF_TRIGGERS:
        assert policy.trigger_enabled(trig), trig
