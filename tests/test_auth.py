import pytest
from conftest import OTP, FakeClock, converse, text_of

from iris_bot.agent import Stage
from iris_bot.auth import AuthError, IdentityService, InvalidSession, SessionExpired


def make(clock):
    return IdentityService("unit-test-secret-0123456789", OTP, ttl_minutes=15, clock=clock)


def test_otp_issues_valid_session():
    clock = FakeClock()
    ids = make(clock)
    token = ids.verify_otp(ids.start_challenge("CLI-AAAA11112222"), OTP, customer_exists=True)
    s = ids.validate(token)
    assert s.customer_id == "CLI-AAAA11112222"
    assert s.expires_at - s.issued_at == 15 * 60


def test_wrong_otp_and_unknown_customer_rejected():
    ids = make(FakeClock())
    ch = ids.start_challenge("CLI-AAAA11112222")
    with pytest.raises(AuthError):
        ids.verify_otp(ch, "000000", customer_exists=True)
    with pytest.raises(AuthError):
        ids.verify_otp(ch, OTP, customer_exists=False)


def test_session_expires_after_ttl():
    clock = FakeClock()
    ids = make(clock)
    token = ids.issue("CLI-AAAA11112222")
    clock.advance(15 * 60 - 1)
    ids.validate(token)
    clock.advance(2)
    with pytest.raises(SessionExpired):
        ids.validate(token)


def test_tampered_or_foreign_token_rejected():
    clock = FakeClock()
    token = make(clock).issue("CLI-AAAA11112222")
    other = IdentityService("another-secret-0123456789xx", OTP, clock=clock)
    with pytest.raises(InvalidSession):
        other.validate(token)
    with pytest.raises(InvalidSession):
        make(clock).validate(token[:-2] + "xx")


def test_expired_session_forces_reauth_then_resumes(make_runtime, fx, clock):
    rt = make_runtime()
    a = fx["happy_a"]
    replies = converse(rt.agent, "exp", [f"no reconozco un cargo de {a['amount']} en {a['merchant_name']}",
                                         a["customer_id"], OTP])
    assert replies[-1].stage == Stage.AWAIT_CONFIRM
    clock.advance(16 * 60)
    r = rt.agent.handle("exp", "sí")
    assert r.stage == Stage.AWAIT_OTP  # no action taken on an expired session
    assert not rt.store.disputes_for(a["customer_id"])
    r = rt.agent.handle("exp", OTP)
    assert r.stage == Stage.AWAIT_CONFIRM  # resumed where it was
    r = rt.agent.handle("exp", "sí")
    assert r.stage == Stage.AWAIT_FEEDBACK
    assert len(rt.store.disputes_for(a["customer_id"])) == 1
    assert any(row["outcome"] == "session_expired" for row in rt.store.audit_rows("exp", step="auth"))


def test_failed_otp_attempts_lead_to_handoff(make_runtime, fx):
    rt = make_runtime()
    replies = converse(rt.agent, "otp", ["hola", fx["happy_a"]["customer_id"], "111111", "222222", "333333"])
    assert replies[-1].stage == Stage.HANDED_OFF
    assert replies[-1].handoff.trigger == "auth_failed"
    assert replies[-1].handoff.customer_id is None
    assert "111111" not in text_of(replies[-1:])
