import json

from conftest import OTP, converse, text_of
from fastapi.testclient import TestClient

from iris_bot.agent import HandoffPayload, Stage
from iris_bot.api.app import create_app


def test_happy_path_es_opens_verified_dispute(make_runtime, fx):
    rt = make_runtime()
    a = fx["happy_a"]
    replies = converse(rt.agent, "h1", [f"Hola, no reconozco un cargo de {a['amount']} en {a['merchant_name']}",
                                        a["customer_id"], OTP, "sí", "sí", "4"])
    assert [r.stage for r in replies] == [Stage.AWAIT_CUSTOMER_ID, Stage.AWAIT_OTP, Stage.AWAIT_CONFIRM,
                                          Stage.AWAIT_FEEDBACK, Stage.AWAIT_CSAT, Stage.CLOSED]
    disputes = rt.store.disputes_for(a["customer_id"])
    assert len(disputes) == 1 and disputes[0]["transaction_id"] == a["transaction_id"]
    assert disputes[0]["dispute_id"] in text_of(replies)
    st = rt.agent.state("h1")
    assert st.outcome() == "resolved" and st.csat == 4
    assert st.actions[-1].action == "open_dispute" and st.actions[-1].status == "verified"


def test_happy_path_pt_with_selection(make_runtime, fx):
    rt = make_runtime()
    c = fx["happy_c"]
    replies = converse(rt.agent, "h2", ["Oi, não reconheço uma compra no meu cartão", c["customer_id"], OTP])
    assert replies[-1].language == "pt"
    assert replies[-1].stage in (Stage.AWAIT_TXN_SELECTION, Stage.AWAIT_CONFIRM)
    if replies[-1].stage == Stage.AWAIT_TXN_SELECTION:
        r = rt.agent.handle("h2", c["merchant_name"])
        assert r.stage == Stage.AWAIT_CONFIRM
    r = rt.agent.handle("h2", "sim")
    assert r.stage == Stage.AWAIT_FEEDBACK
    assert "contestação" in r.messages[0]


def test_baseline_mode_runs_without_llm(make_runtime, fx):
    rt = make_runtime(mode="baseline")
    assert rt.agent.llm is None
    a = fx["happy_a"]
    replies = converse(rt.agent, "b1", [f"no reconozco un cargo de {a['amount']}", a["customer_id"], OTP, "sí"])
    assert replies[-1].stage == Stage.AWAIT_FEEDBACK


def test_fraud_flag_hands_off_with_complete_payload(make_runtime, fx):
    rt = make_runtime()
    f = fx["fraud"]
    replies = converse(rt.agent, "fr", [f"no reconozco una compra de {f['amount']} en {f['merchant_name']}",
                                        f["customer_id"], OTP])
    h = replies[-1].handoff
    assert replies[-1].stage == Stage.HANDED_OFF and h.trigger == "fraud_flag" and h.priority == "high"
    assert not rt.store.disputes_for(f["customer_id"])
    stored = rt.store.handoff(h.handoff_id)["payload"]
    payload = HandoffPayload.model_validate(stored)
    # completeness: every field the human agent needs is present and non-empty
    assert payload.customer_id == f["customer_id"] and payload.authenticated
    assert payload.language == "es" and payload.intent == "dispute_unrecognized_charge"
    assert payload.request_summary and payload.open_questions and payload.transcript
    assert {x.key for x in payload.verified_facts} >= {"disputed_transaction", "eligibility"}
    assert f["transaction_id"] in payload.evidence_ids
    assert payload.reason and payload.trace_id
    assert OTP not in json.dumps(stored)  # OTP redacted from the transcript


def test_amount_over_threshold_hands_off(make_runtime, fx):
    rt = make_runtime()
    h = fx["high"]
    replies = converse(rt.agent, "hi", [f"no reconozco el cargo de {h['amount']} en {h['merchant_name']}",
                                        h["customer_id"], OTP])
    assert replies[-1].handoff and replies[-1].handoff.trigger == "amount_over_threshold"


def test_declined_transaction_is_explained_not_disputed(make_runtime, fx):
    rt = make_runtime()
    d = fx["declined"]
    replies = converse(rt.agent, "dc", [f"no reconozco una compra de {d['amount']} en {d['merchant_name']}",
                                        d["customer_id"], OTP])
    assert replies[-1].stage == Stage.AWAIT_FEEDBACK
    assert not rt.store.disputes_for(d["customer_id"])


def test_repeated_unresolved_reason_hands_off(make_runtime, fx):
    rt = make_runtime()
    r = fx["repeat"]
    replies = converse(rt.agent, "rp", [f"no reconozco una compra de {r['amount']} en {r['merchant_name']}",
                                        r["customer_id"], OTP])
    assert replies[-1].handoff and replies[-1].handoff.trigger == "repeated_unresolved_reason"


def test_clarification_is_bounded(make_runtime, fx):
    rt = make_runtime()
    replies = converse(rt.agent, "cl", ["hola", fx["happy_a"]["customer_id"], OTP, "mmm", "eh no sé", "?"])
    assert [r.stage for r in replies[-3:]] == [Stage.AWAIT_CLARIFICATION, Stage.AWAIT_CLARIFICATION,
                                               Stage.HANDED_OFF]
    assert replies[-1].handoff.trigger == "not_understood"


def test_customer_can_ask_for_human_anytime(make_runtime, fx):
    rt = make_runtime()
    replies = converse(rt.agent, "hu", ["quiero hablar con un asesor"])
    assert replies[-1].stage == Stage.HANDED_OFF
    assert replies[-1].handoff.trigger == "customer_requested_human"
    assert not replies[-1].handoff.authenticated


def test_tool_failure_hands_off_safely(make_runtime, fx):
    rt = make_runtime(fault_tools={"list_recent_transactions"})
    a = fx["happy_a"]
    replies = converse(rt.agent, "tf", ["no reconozco un cargo", a["customer_id"], OTP])
    assert replies[-1].handoff.trigger == "tool_failure"
    assert "DSP-" not in text_of(replies)


def test_api_endpoints(make_runtime, fx):
    rt = make_runtime()
    with TestClient(create_app(runtime=rt)) as client:
        assert client.get("/healthz").json()["status"] == "ok"
        assert "IRIS" in client.get("/").text
        r = client.post("/chat", json={"conversation_id": "api1", "message": "quiero hablar con un asesor"})
        assert r.status_code == 200 and r.json()["stage"] == "HANDED_OFF"
        wa = client.post("/webhook/whatsapp", data={"From": "whatsapp:+5215550000000", "Body": "hola"})
        assert wa.status_code == 200 and "<Response><Message>" in wa.text
        assert len(client.get("/handoffs").json()) == 1
        m = client.get("/metrics").json()
        assert m["handoff_rate"] == 1.0 and m["latency_ms"]["n_turns"] == 2
