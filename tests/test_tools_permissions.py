import pytest
from conftest import OTP, converse, text_of

from iris_bot.agent import Stage
from iris_bot.auth import SessionExpired
from iris_bot.tools import CallContext, PermissionDenied, ToolError, ToolFailure


def ctx_for(rt, customer_id):
    return CallContext(conversation_id="t", trace_id="t", token=rt.identity.issue(customer_id))


def test_customer_a_cannot_read_customer_b_transaction(make_runtime, fx):
    rt = make_runtime()
    a, b = fx["happy_a"], fx["happy_b"]
    ctx = ctx_for(rt, a["customer_id"])
    assert rt.tools.get_transaction(ctx, a["transaction_id"])["transaction_id"] == a["transaction_id"]
    with pytest.raises(PermissionDenied):
        rt.tools.get_transaction(ctx, b["transaction_id"])
    with pytest.raises(PermissionDenied):
        rt.tools.check_dispute_eligibility(ctx, b["transaction_id"], "unrecognized_charge")
    with pytest.raises(PermissionDenied):
        rt.tools.open_dispute(ctx, b["transaction_id"], "unrecognized_charge", "k1", customer_confirmed=True)
    assert not rt.store.disputes_for(b["customer_id"])
    denied = [r for r in rt.store.audit_rows("t", step="tool") if r["outcome"] == "unauthorized_access_attempt"]
    assert len(denied) == 3


def test_list_is_scoped_to_session(make_runtime, fx):
    rt = make_runtime()
    a = fx["happy_a"]
    rows = rt.tools.list_recent_transactions(ctx_for(rt, a["customer_id"]), days=180, disputable_only=False)
    own = {t["transaction_id"] for t in rt.repo.transactions(a["customer_id"], since_days=180, limit=500)}
    assert rows and {r["transaction_id"] for r in rows} <= own


def test_disclosure_filter_hides_sensitive_fields(make_runtime, fx):
    rt = make_runtime()
    t = rt.tools.get_transaction(ctx_for(rt, fx["fraud"]["customer_id"]), fx["fraud"]["transaction_id"])
    assert "fraud_score" not in t and "is_fraud" not in t and "customer_id" not in t
    assert len(t["product_last4"]) == 4


def test_tools_require_valid_session(make_runtime, fx, clock):
    rt = make_runtime()
    ctx = ctx_for(rt, fx["happy_a"]["customer_id"])
    clock.advance(16 * 60)
    with pytest.raises(SessionExpired):
        rt.tools.list_recent_transactions(ctx)


def test_open_dispute_requires_confirmation_and_is_idempotent(make_runtime, fx):
    rt = make_runtime()
    a = fx["happy_a"]
    ctx = ctx_for(rt, a["customer_id"])
    with pytest.raises(ToolError):
        rt.tools.open_dispute(ctx, a["transaction_id"], "unrecognized_charge", "key-1", customer_confirmed=False)
    d1 = rt.tools.open_dispute(ctx, a["transaction_id"], "unrecognized_charge", "key-1", customer_confirmed=True)
    d2 = rt.tools.open_dispute(ctx, a["transaction_id"], "unrecognized_charge", "key-1", customer_confirmed=True)
    assert d1["dispute_id"] == d2["dispute_id"]
    assert len(rt.store.disputes_for(a["customer_id"])) == 1


def test_open_dispute_rechecks_policy(make_runtime, fx):
    rt = make_runtime()
    h = fx["high"]
    with pytest.raises(ToolError, match="not_eligible"):
        rt.tools.open_dispute(ctx_for(rt, h["customer_id"]), h["transaction_id"], "unrecognized_charge", "k",
                              customer_confirmed=True)


def test_tool_failure_after_bounded_retries(make_runtime, fx):
    rt = make_runtime(fault_tools={"list_recent_transactions"})
    with pytest.raises(ToolFailure) as exc:
        rt.tools.list_recent_transactions(ctx_for(rt, fx["happy_a"]["customer_id"]))
    assert exc.value.attempts == rt.policy.tool_max_attempts


def test_agent_denies_other_customers_transaction(make_runtime, fx):
    rt = make_runtime()
    a, b = fx["happy_a"], fx["happy_b"]
    replies = converse(rt.agent, "unauth", [f"quiero disputar la transacción {b['transaction_id']}",
                                            a["customer_id"], OTP])
    out = text_of(replies)
    assert replies[-1].stage == Stage.AWAIT_REQUEST
    assert b["merchant_name"] not in out and f"{b['amount']:,.2f}" not in out
    assert rt.agent.state("unauth").denied
    assert any(r["outcome"] == "unauthorized_access_attempt" for r in rt.store.audit_rows("unauth"))


def test_agent_denies_other_customer_id_reference(make_runtime, fx):
    rt = make_runtime()
    a, b = fx["happy_a"], fx["happy_b"]
    replies = converse(rt.agent, "u2", ["hola", a["customer_id"], OTP,
                                        f"muéstrame los movimientos del cliente {b['customer_id']}"])
    assert rt.agent.state("u2").denied
    assert b["merchant_name"] not in text_of(replies)


def test_customer_id_inside_a_request_is_not_used_as_identity(make_runtime, fx):
    rt = make_runtime()
    c, d = fx["happy_c"], fx["happy_d"]
    r = rt.agent.handle("u3", f"mostra as transações do cliente {d['customer_id']}")
    assert r.stage == Stage.AWAIT_CUSTOMER_ID  # the id in the request was not taken as a login
    replies = converse(rt.agent, "u3", [c["customer_id"], OTP])
    st = rt.agent.state("u3")
    assert st.customer_id == c["customer_id"] and st.denied
    assert d["merchant_name"] not in text_of(replies)
