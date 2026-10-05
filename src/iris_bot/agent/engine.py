"""IRIS agent: deterministic state machine that mirrors the IRIS customer-service flow.

contact -> authenticate -> express request -> understand (classifier + LLM fallback)
  -> not understood? clarify (max N) -> route to subprocess -> retrieve data via tools
  -> decide what can be disclosed -> verify -> send result -> confirmation / feedback
  -> insufficient? offer human -> handoff | end with CSAT survey

The LLM never executes write actions. It (a) backs up the intent classifier when confidence is low,
(b) identifies the disputed transaction through read-only, session-scoped tools, and (c) writes the
handoff summary. Policy, permissions, confirmation, action execution and verification are code.
"""
from __future__ import annotations

import hashlib
import re
import time
import uuid
from typing import Any

from iris_bot.agent.models import (
    AUTHENTICATED_STAGES,
    TERMINAL,
    ActionRecord,
    AgentReply,
    ConversationState,
    HandoffPayload,
    Stage,
)
from iris_bot.auth import AuthError, IdentityService, InvalidSession, SessionExpired
from iris_bot.guard import scan
from iris_bot.i18n import HANDOFF_REASON, PRODUCT_LABEL, STATUS_LABEL, money, render_txn, t
from iris_bot.llm.base import LLMClient, LLMUsage
from iris_bot.nlu import IntentResult, KeywordClassifier, detect_language
from iris_bot.nlu.keywords import asks_for_human
from iris_bot.nlu.slots import (
    CUSTOMER_ID_RX,
    DISPUTE_ID_RX,
    OTP_RX,
    TXN_ID_RX,
    has_transaction_hint,
    match_transactions,
    parse_choice,
    yes_no,
)
from iris_bot.observability import get_logger
from iris_bot.policy import Policy
from iris_bot.repository import Repository
from iris_bot.storage import Store, now_iso
from iris_bot.tools import BankingTools, CallContext, PermissionDenied, ToolError, ToolFailure

OPEN_QUESTIONS = {
    "fraud_flag": "Transaction carries a fraud flag: confirm card possession, consider blocking the card, open fraud case.",
    "amount_over_threshold": "Amount exceeds the automated limit: decide whether to open the dispute and any provisional credit.",
    "customer_requested_human": "Customer asked for a human: confirm what they need.",
    "repeated_unresolved_reason": "Customer already has an open case for the same reason: review that case before opening another.",
    "unsupported_intent": "Request is outside the bot's automated scope: handle the request.",
    "prompt_injection": "Message matched prompt-injection patterns: review manually; no automated action was taken.",
    "tool_failure": "Backend tools failed after retries: verify system state before acting; nothing unverified was reported.",
    "not_understood": "Bot could not understand the request after clarification: identify the customer's need.",
    "result_insufficient": "Customer said the automated answer was insufficient: find out what is missing.",
    "auth_failed": "Identity could not be verified: authenticate through a stronger channel before sharing data.",
    "no_matching_transaction": "Customer's description did not match any transaction: identify the transaction with them.",
}
INTENT_REASON = {"dispute_unrecognized_charge": "unrecognized_charge", "dispute_wrong_fee": "wrong_fee"}
_HUMAN_RX = KeywordClassifier()


class Agent:
    def __init__(self, *, policy: Policy, identity: IdentityService, tools: BankingTools, store: Store,
                 repo: Repository, classifier: Any, llm: LLMClient | None, name: str = "iris"):
        self.policy = policy
        self.identity = identity
        self.tools = tools
        self.store = store
        self.repo = repo
        self.classifier = classifier
        self.llm = llm
        self.name = name
        self.log = get_logger("iris.agent")

    # ===================================================================== entry point ======
    def handle(self, conversation_id: str, text: str, channel: str = "web") -> AgentReply:
        trace_id = uuid.uuid4().hex
        t0 = time.perf_counter()
        st = self._load(conversation_id, channel)
        text = (text or "").strip()[:1000]
        st.turns += 1
        st.transcript.append({"role": "customer", "text": self._redact(text)})
        out: list[str] = []
        handoff: HandoffPayload | None = None
        usage = LLMUsage()
        ctx = CallContext(conversation_id=conversation_id, trace_id=trace_id, token=st.token)
        tool_calls_before = len(self.store.audit_rows(conversation_id, step="tool"))

        if st.stage in (Stage.START, Stage.AWAIT_REQUEST, Stage.AWAIT_CLARIFICATION, Stage.AWAIT_CUSTOMER_ID):
            lang, conf = detect_language(text, default=st.language)
            if conf >= 0.6 or st.stage == Stage.START:
                st.language = lang
        try:
            if scan(text).detected:
                st.injection_detected = True
                self.store.audit(step="guard", trace_id=trace_id, conversation_id=conversation_id,
                                 customer_id=st.customer_id, outcome="prompt_injection_detected",
                                 detail={"source": "user_text"})
                handoff = self._handoff(st, ctx, "prompt_injection", out, usage)
            elif st.stage != Stage.AWAIT_CSAT and self._asks_for_human(text):
                handoff = self._handoff(st, ctx, "customer_requested_human", out, usage)
            else:
                handoff = self._dispatch(st, ctx, text, out, usage)
        except ToolFailure as exc:
            self.log.warning("tool_failure", tool=exc.tool, trace_id=trace_id)
            handoff = self._handoff(st, ctx, "tool_failure", out, usage)
        except Exception as exc:  # noqa: BLE001 - safe fallback, never leak internals
            self.log.exception("turn_error", trace_id=trace_id, error=type(exc).__name__)
            self.store.audit(step="error", trace_id=trace_id, conversation_id=conversation_id,
                             customer_id=st.customer_id, outcome=type(exc).__name__)
            out.append(t("fallback_error", st.language))

        latency = (time.perf_counter() - t0) * 1000
        st.tool_calls += len(self.store.audit_rows(conversation_id, step="tool")) - tool_calls_before
        st.llm_calls += usage.calls
        st.input_tokens += usage.input_tokens
        st.output_tokens += usage.output_tokens
        st.cost_usd += usage.cost_usd
        for m in out:
            st.transcript.append({"role": "bot", "text": m})
        if st.stage in TERMINAL and not st.ended_at:
            st.ended_at = now_iso()
        self._save(st)
        self.store.audit(step="turn", trace_id=trace_id, conversation_id=conversation_id,
                         customer_id=st.customer_id, latency_ms=latency, input_tokens=usage.input_tokens,
                         output_tokens=usage.output_tokens, cost_usd=usage.cost_usd, outcome=st.stage.value,
                         detail={"agent": self.name, "intent": st.intent, "llm_estimated": usage.estimated,
                                 "case_id": st.case_id})
        self.log.info("turn", trace_id=trace_id, conversation_id=conversation_id, stage=st.stage.value,
                      intent=st.intent, latency_ms=round(latency, 1), cost_usd=round(usage.cost_usd, 6))
        return AgentReply(conversation_id=conversation_id, case_id=st.case_id, trace_id=trace_id,
                          messages=out, stage=st.stage, language=st.language, ended=st.stage in TERMINAL,
                          handoff=handoff, latency_ms=latency)

    # ===================================================================== dispatch =========
    def _dispatch(self, st: ConversationState, ctx: CallContext, text: str, out: list[str],
                  usage: LLMUsage) -> HandoffPayload | None:
        if st.stage in AUTHENTICATED_STAGES:
            try:
                self.identity.validate(st.token)
            except SessionExpired:
                return self._reauth(st, ctx, out)
            except InvalidSession:
                st.token, st.customer_id = None, None
                st.stage = Stage.AWAIT_CUSTOMER_ID
                out.append(t("ask_customer_id", st.language))
                return None
        handler = {
            Stage.START: self._on_start,
            Stage.AWAIT_CUSTOMER_ID: self._on_customer_id,
            Stage.AWAIT_OTP: self._on_otp,
            Stage.AWAIT_REQUEST: self._on_request,
            Stage.AWAIT_CLARIFICATION: self._on_request,
            Stage.AWAIT_TXN_SELECTION: self._on_selection,
            Stage.AWAIT_CONFIRM: self._on_confirm,
            Stage.AWAIT_FEEDBACK: self._on_feedback,
            Stage.AWAIT_HANDOFF_OFFER: self._on_handoff_offer,
            Stage.AWAIT_CSAT: self._on_csat,
        }[st.stage]
        return handler(st, ctx, text, out, usage)

    # ===================================================================== authentication ===
    def _on_start(self, st, ctx, text, out, usage):
        out.append(t("welcome", st.language))
        intent, _ = _HUMAN_RX.predict(text)
        if intent != "greeting" and len(text) > 3 and not CUSTOMER_ID_RX.fullmatch(text.strip()):
            st.pending_request = text
        st.stage = Stage.AWAIT_CUSTOMER_ID
        # Only a message that is just an id counts as identification; an id inside a request is data.
        m = CUSTOMER_ID_RX.fullmatch(text.strip())
        if m:
            return self._on_customer_id(st, ctx, m.group(0), out, usage)
        out.append(t("ask_customer_id", st.language))
        return None

    def _on_customer_id(self, st, ctx, text, out, usage):
        m = CUSTOMER_ID_RX.search(text)
        if not m:
            st.id_attempts += 1
            if st.pending_request is None and len(text) > 8:
                st.pending_request = text
            if st.id_attempts >= self.policy.max_otp_attempts:
                return self._handoff(st, ctx, "auth_failed", out, usage)
            out.append(t("customer_id_invalid", st.language))
            return None
        st.claimed_customer_id = m.group(0).upper()
        st.challenge_id = self.identity.start_challenge(st.claimed_customer_id)
        st.stage = Stage.AWAIT_OTP
        out.append(t("otp_sent", st.language))
        return None

    def _on_otp(self, st, ctx, text, out, usage):
        m = OTP_RX.search(text)
        try:
            if not m or not st.challenge_id:
                raise AuthError("no otp")
            exists = self.repo.customer(st.claimed_customer_id) is not None
            st.token = self.identity.verify_otp(st.challenge_id, m.group(1), customer_exists=exists)
        except AuthError:
            st.otp_attempts += 1
            self.store.audit(step="auth", trace_id=ctx.trace_id, conversation_id=st.conversation_id,
                             outcome="otp_rejected", detail={"attempt": st.otp_attempts})
            left = self.policy.max_otp_attempts - st.otp_attempts
            if left <= 0:
                return self._handoff(st, ctx, "auth_failed", out, usage)
            out.append(t("otp_invalid", st.language, left=left))
            return None
        st.customer_id = st.claimed_customer_id
        st.otp_attempts = 0
        ctx.token = st.token
        self.store.audit(step="auth", trace_id=ctx.trace_id, conversation_id=st.conversation_id,
                         customer_id=st.customer_id, outcome="authenticated")
        out.append(t("auth_ok", st.language))
        if st.resume_stage:
            st.stage, st.resume_stage = st.resume_stage, None
            self._reprompt(st, out)
            return None
        if st.pending_request:
            pending, st.pending_request = st.pending_request, None
            return self._on_request(st, ctx, pending, out, usage)
        st.stage = Stage.AWAIT_REQUEST
        out.append(t("ask_request", st.language))
        return None

    def _reauth(self, st, ctx, out):
        self.store.audit(step="auth", trace_id=ctx.trace_id, conversation_id=st.conversation_id,
                         customer_id=st.customer_id, outcome="session_expired")
        st.resume_stage = st.stage
        st.token = None
        st.claimed_customer_id = st.customer_id
        st.challenge_id = self.identity.start_challenge(st.customer_id or "")
        st.stage = Stage.AWAIT_OTP
        out.append(t("session_expired", st.language))
        return None

    def _reprompt(self, st, out):
        lang = st.language
        if st.stage == Stage.AWAIT_CONFIRM and st.selected_txn:
            out.append(t("confirm_dispute", lang, reason_label=t(f"reason_{st.reason}", lang),
                         item=render_txn(st.selected_txn, lang)))
        elif st.stage == Stage.AWAIT_TXN_SELECTION and st.candidates:
            out.append(t("candidates", lang, items=self._numbered(st.candidates, lang)))
        elif st.stage == Stage.AWAIT_FEEDBACK:
            out.append(t("ask_feedback", lang))
        elif st.stage == Stage.AWAIT_HANDOFF_OFFER:
            out.append(t("offer_human", lang))
        else:
            st.stage = Stage.AWAIT_REQUEST
            out.append(t("ask_request", lang))

    # ===================================================================== understanding ====
    def understand(self, text: str, lang: str, usage: LLMUsage) -> IntentResult:
        intent, conf = self.classifier.predict(text)
        res = IntentResult(intent, conf, getattr(self.classifier, "name", "classifier"))
        if self.llm is not None and self.policy.needs_clarification(conf):
            llm_res, u = self.llm.classify_intent(text, lang)
            usage.add(u)
            if llm_res and not self.policy.needs_clarification(llm_res[1]):
                res = IntentResult(llm_res[0], llm_res[1], "llm")
        return res

    def _on_request(self, st, ctx, text, out, usage):
        other = [c.upper() for c in CUSTOMER_ID_RX.findall(text) if c.upper() != st.customer_id]
        if other:
            return self._deny(st, ctx, out, f"customer id {other[0]} referenced")
        txn_ids = [m.upper() for m in TXN_ID_RX.findall(text)]
        for tid in txn_ids:  # permission check up front: never act on, or describe, another customer's data
            try:
                self.tools.get_transaction(ctx, tid)
            except PermissionDenied:
                return self._deny(st, ctx, out, f"transaction {tid} of another customer referenced")
            except ToolError:
                pass
        res = self.understand(text, st.language, usage)
        if txn_ids and self.policy.needs_clarification(res.confidence):
            res = IntentResult("dispute_unrecognized_charge", 0.6, "rule:transaction_id")
        self.store.audit(step="understand", trace_id=ctx.trace_id, conversation_id=st.conversation_id,
                         customer_id=st.customer_id, outcome=res.intent,
                         detail={"confidence": round(res.confidence, 3), "source": res.source})
        if self.policy.needs_clarification(res.confidence):
            if self.policy.clarification_exhausted(st.clarification_turns):
                return self._handoff(st, ctx, "not_understood", out, usage)
            st.clarification_turns += 1
            st.stage = Stage.AWAIT_CLARIFICATION
            out.append(t("clarify", st.language))
            return None
        st.intent, st.intent_confidence, st.intent_source = res.intent, res.confidence, res.source
        st.request_text = st.request_text or text
        route = self.policy.intent_route(res.intent)
        if route == "conversational":
            st.stage = Stage.AWAIT_REQUEST
            out.append(t("ask_request", st.language))
            return None
        if route == "decline":
            st.declined = True
            st.stage = Stage.AWAIT_REQUEST
            out.append(t("out_of_scope", st.language))
            return None
        if route == "handoff":
            trigger = "customer_requested_human" if res.intent == "talk_to_human" else "unsupported_intent"
            return self._handoff(st, ctx, trigger, out, usage)
        # automated subprocesses
        if res.intent in INTENT_REASON:
            st.reason = INTENT_REASON[res.intent]
            return self._dispute_identify(st, ctx, text, out, usage)
        if res.intent == "dispute_status":
            return self._dispute_status(st, ctx, text, out, usage)
        if res.intent == "balance_inquiry":
            return self._balance(st, ctx, out)
        return self._handoff(st, ctx, "unsupported_intent", out, usage)

    @staticmethod
    def _asks_for_human(text: str) -> bool:
        return asks_for_human(text)

    # ===================================================================== dispute flow =====
    def _executor(self, ctx: CallContext):
        def run(name: str, args: dict[str, Any]) -> Any:
            if name == "list_recent_transactions":
                return self.tools.list_recent_transactions(ctx, days=int(args.get("days", 120)))
            if name == "get_transaction":
                return self.tools.get_transaction(ctx, str(args["transaction_id"]))
            raise ToolError(f"tool {name} not allowed for the model")
        return run

    def _dispute_identify(self, st, ctx, text, out, usage):
        ids_in_text = [m.upper() for m in TXN_ID_RX.findall(text)]
        selected_ids: list[str] = []
        if self.llm is not None:
            try:
                selected_ids, rationale, u = self.llm.select_transactions(text, st.language, self._executor(ctx))
                usage.add(u)
            except PermissionDenied:
                return self._deny(st, ctx, out, "transaction of another customer requested")
            except ToolFailure:
                raise
            except Exception as exc:  # noqa: BLE001 - LLM failure -> deterministic fallback
                self.log.warning("llm_select_failed", error=type(exc).__name__)
                selected_ids = []
        recent = self.tools.list_recent_transactions(ctx, days=self.policy.max_age_days + 60)
        if st.reason == "wrong_fee":
            recent.sort(key=lambda x: x.get("type") != "Adjustment")
        by_id = {r["transaction_id"]: r for r in recent}
        candidates: list[dict[str, Any]] = []
        for tid in dict.fromkeys(ids_in_text + selected_ids):
            try:
                candidates.append(by_id.get(tid) or self.tools.get_transaction(ctx, tid))
            except PermissionDenied:
                return self._deny(st, ctx, out, "transaction of another customer requested")
            except ToolError:
                continue
        if not candidates and self.llm is None:
            ranked = match_transactions(text, recent)
            if ranked:
                best = ranked[0][1]
                candidates = [r for r, s in ranked if s >= best - 0.5]
        if not recent and not candidates:
            out.append(t("no_transactions", st.language))
            st.offer_trigger = "no_matching_transaction"
            st.stage = Stage.AWAIT_HANDOFF_OFFER
            out.append(t("offer_human", st.language))
            return None
        if len(candidates) == 1:
            return self._evaluate(st, ctx, candidates[0], out, usage)
        n = self.policy.max_candidates
        if candidates:
            st.candidates = candidates[:n]
            out.append(t("candidates", st.language, items=self._numbered(st.candidates, st.language)))
        else:
            st.candidates = recent[:n]
            key = "no_match" if has_transaction_hint(text) else "candidates"
            out.append(t(key, st.language, items=self._numbered(st.candidates, st.language)))
        st.stage = Stage.AWAIT_TXN_SELECTION
        return None

    def _on_selection(self, st, ctx, text, out, usage):
        idx = parse_choice(text, len(st.candidates))
        chosen = st.candidates[idx] if idx is not None else None
        if chosen is None:
            ranked = match_transactions(text, st.candidates)
            if ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
                chosen = ranked[0][0]
        if chosen is None and idx is None and yes_no(text) is None:
            # The customer described a transaction that was not in the short list: search all recent ones.
            ranked = match_transactions(text, self.tools.list_recent_transactions(ctx, days=self.policy.max_age_days + 60))
            if ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
                chosen = ranked[0][0]
        if chosen is None and yes_no(text) is False:
            st.offer_trigger = "no_matching_transaction"
            st.stage = Stage.AWAIT_HANDOFF_OFFER
            out.append(t("offer_human", st.language))
            return None
        if chosen is None:
            st.selection_attempts += 1
            if st.selection_attempts >= self.policy.max_selection_attempts:
                return self._handoff(st, ctx, "no_matching_transaction", out, usage)
            out.append(t("selection_invalid", st.language, n=len(st.candidates)))
            return None
        # Re-read through the scoped tool (never trust state alone).
        return self._evaluate(st, ctx, self.tools.get_transaction(ctx, chosen["transaction_id"]), out, usage)

    def _evaluate(self, st, ctx, txn, out, usage):
        st.selected_txn = txn
        st.add_fact("disputed_transaction", {k: txn[k] for k in ("transaction_id", "date", "merchant", "type",
                                                                 "amount", "currency", "status")},
                    "get_transaction", txn["transaction_id"])
        el = self.tools.check_dispute_eligibility(ctx, txn["transaction_id"], st.reason)
        st.add_fact("eligibility", {"eligible": el["eligible"], "reason_code": el["reason_code"],
                                    "age_days": el["checks"].get("age_days"),
                                    "amount_usd": el["checks"].get("amount_usd"),
                                    "prior_open_same_reason": el["prior_open_same_reason"]},
                    "check_dispute_eligibility", txn["transaction_id"])
        lang = st.language
        if el["handoff_trigger"]:
            return self._handoff(st, ctx, el["handoff_trigger"], out, usage)
        if el["reason_code"] == "already_disputed":
            out.append(t("already_disputed", lang, ids=", ".join(el["existing_dispute_ids"])))
            return self._deliver(st, out)
        if el["reason_code"] == "status_declined":
            out.append(t("ineligible_declined", lang))
            return self._deliver(st, out)
        if el["reason_code"] == "status_reversed":
            out.append(t("ineligible_reversed", lang))
            return self._deliver(st, out)
        if el["reason_code"] == "too_old":
            out.append(t("ineligible_too_old", lang, age=el["checks"].get("age_days"),
                         max_age=self.policy.max_age_days))
            st.offer_trigger = "result_insufficient"
            st.stage = Stage.AWAIT_HANDOFF_OFFER
            out.append(t("offer_human", lang))
            return None
        if not el["eligible"]:
            return self._handoff(st, ctx, "unsupported_intent", out, usage)
        st.stage = Stage.AWAIT_CONFIRM
        out.append(t("confirm_dispute", lang, reason_label=t(f"reason_{st.reason}", lang),
                     item=render_txn(txn, lang)))
        return None

    def _on_confirm(self, st, ctx, text, out, usage):
        yn = yes_no(text)
        if yn is None:
            st.confirm_attempts += 1
            if st.confirm_attempts >= self.policy.max_clarification_turns:
                return self._handoff(st, ctx, "not_understood", out, usage)
            self._reprompt(st, out)
            return None
        txn = st.selected_txn or {}
        if yn is False:
            st.actions.append(ActionRecord(action="open_dispute", status="cancelled",
                                           reference_id=txn.get("transaction_id"), detail="customer declined"))
            out.append(t("dispute_cancelled", st.language))
            st.stage = Stage.AWAIT_REQUEST
            out.append(t("ask_request", st.language))
            return None
        key = hashlib.sha256(f"{st.case_id}|{txn['transaction_id']}|{st.reason}".encode()).hexdigest()[:32]
        try:
            rec = self.tools.open_dispute(ctx, txn["transaction_id"], st.reason, key, customer_confirmed=True)
        except ToolError as exc:
            st.actions.append(ActionRecord(action="open_dispute", status="failed",
                                           reference_id=txn["transaction_id"], detail=str(exc)))
            return self._handoff(st, ctx, "result_insufficient", out, usage)
        # Verify: re-read the dispute through the status tool before reporting it.
        status = self.tools.get_dispute_status(ctx, rec["dispute_id"])
        verified = any(d["dispute_id"] == rec["dispute_id"] and d["transaction_id"] == txn["transaction_id"]
                       for d in status["disputes"])
        st.actions.append(ActionRecord(action="open_dispute", status="verified" if verified else "not_verified",
                                       reference_id=rec["dispute_id"], detail=f"transaction {txn['transaction_id']}"))
        if not verified:
            out.append(t("dispute_not_verified", st.language))
            return self._handoff(st, ctx, "tool_failure", out, usage)
        st.add_fact("dispute_opened", {"dispute_id": rec["dispute_id"], "status": rec["status"]},
                    "get_dispute_status", rec["dispute_id"])
        out.append(t("dispute_opened", st.language, dispute_id=rec["dispute_id"]))
        return self._deliver(st, out)

    def _dispute_status(self, st, ctx, text, out, usage):
        m = DISPUTE_ID_RX.search(text)
        try:
            data = self.tools.get_dispute_status(ctx, m.group(0).upper() if m else None)
        except PermissionDenied:
            return self._deny(st, ctx, out, "dispute of another customer requested")
        except ToolError:
            data = {"disputes": [], "complaints": []}
        lang = st.language
        items = [f"• {d['dispute_id']} · {d['created_at'][:10]} · {money(d['amount'], d['currency'])} · "
                 f"{STATUS_LABEL[lang].get(d['status'], d['status'])}" for d in data["disputes"]]
        items += [f"• {c['case_id']} · {str(c['created_at'])[:10]} · {c['subcategory']} · "
                  f"{STATUS_LABEL[lang].get(c['status'], c['status'])}" for c in data["complaints"][:5]]
        for d in data["disputes"]:
            st.evidence_ids.append(d["dispute_id"])
        for c in data["complaints"][:5]:
            st.evidence_ids.append(c["case_id"])
        st.add_fact("dispute_status", {"n_disputes": len(data["disputes"]), "n_cases": len(data["complaints"])},
                    "get_dispute_status")
        out.append(t("status_list", lang, items="\n".join(items)) if items else t("status_none", lang))
        return self._deliver(st, out)

    def _balance(self, st, ctx, out):
        prof = self.tools.get_customer_profile(ctx)
        lang = st.language
        lines = []
        for p in prof["products"]:
            label = PRODUCT_LABEL[lang].get(p["product_type"], p["product_type"])
            line = f"• {label} *{p['product_last4']}: {money(p['balance'], p['currency'])}"
            if p.get("credit_limit"):
                line += f" (límite {money(p['credit_limit'], p['currency'])})" if lang == "es" else \
                        f" (limite {money(p['credit_limit'], p['currency'])})"
            lines.append(line)
        st.add_fact("products", len(prof["products"]), "get_customer_profile")
        out.append(t("balance", lang, items="\n".join(lines) or "-"))
        return self._deliver(st, out)

    def _deliver(self, st, out):
        st.result_delivered = True
        st.stage = Stage.AWAIT_FEEDBACK
        out.append(t("ask_feedback", st.language))
        return None

    # ===================================================================== closing ==========
    def _on_feedback(self, st, ctx, text, out, usage):
        yn = yes_no(text)
        if yn is True:
            st.resolved, st.feedback = True, "yes"
            st.stage = Stage.AWAIT_CSAT
            out.append(t("ask_csat", st.language))
            return None
        if yn is False:
            st.feedback = "no"
            st.offer_trigger = "result_insufficient"
            st.stage = Stage.AWAIT_HANDOFF_OFFER
            out.append(t("offer_human", st.language))
            return None
        return self._on_request(st, ctx, text, out, usage)  # a new request in the same conversation

    def _on_handoff_offer(self, st, ctx, text, out, usage):
        yn = yes_no(text)
        if yn is True:
            return self._handoff(st, ctx, st.offer_trigger or "result_insufficient", out, usage)
        if yn is False:
            st.stage = Stage.AWAIT_CSAT
            out.append(t("ask_csat", st.language))
            return None
        return self._on_request(st, ctx, text, out, usage)

    def _on_csat(self, st, ctx, text, out, usage):
        m = re.search(r"\b([1-4])\b", text)
        if m:
            st.csat = int(m.group(1))
        if m or st.csat_attempts >= 1:  # ask at most twice; never block closing on the survey
            st.stage = Stage.CLOSED
            out.append(t("goodbye", st.language))
            return None
        st.csat_attempts += 1
        out.append(t("csat_invalid", st.language))
        return None

    # ===================================================================== deny / handoff ===
    def _deny(self, st, ctx, out, detail: str):
        st.denied = True
        st.actions.append(ActionRecord(action="data_access", status="denied", detail=detail))
        self.store.audit(step="permission", trace_id=ctx.trace_id, conversation_id=st.conversation_id,
                         customer_id=st.customer_id, outcome="unauthorized_access_attempt",
                         detail={"detail": detail})
        out.append(t("denied", st.language))
        st.stage = Stage.AWAIT_REQUEST
        return None

    def _handoff(self, st: ConversationState, ctx: CallContext, trigger: str, out: list[str],
                 usage: LLMUsage) -> HandoffPayload | None:
        lang = st.language
        handoff_id = "HO-" + uuid.uuid4().hex[:10].upper()
        open_q = [OPEN_QUESTIONS.get(trigger, "Review the case.")]
        if st.reason and not st.selected_txn:
            open_q.append("Which transaction does the customer want to dispute? Not identified yet.")
        if not st.customer_id:
            open_q.append("Customer is not authenticated: verify identity before sharing account data.")
        summary = None
        facts_dict = {f.key: f.value for f in st.facts}
        if self.llm is not None and trigger != "prompt_injection":
            summary, u = self.llm.summarize_handoff(st.transcript, facts_dict, lang)
            usage.add(u)
        if not summary:
            req = (st.request_text or st.pending_request or (st.transcript[0]["text"] if st.transcript else ""))
            summary = (f"Customer ({lang}) contacted via {st.channel}. Intent: {st.intent or 'unknown'}"
                       f"{f' (reason: {st.reason})' if st.reason else ''}. Request: \"{req[:200]}\". "
                       f"Handoff trigger: {trigger}.")
            if st.selected_txn:
                summary += f" Transaction under review: {st.selected_txn['transaction_id']}."
        payload = HandoffPayload(
            handoff_id=handoff_id, conversation_id=st.conversation_id, case_id=st.case_id, channel=st.channel,
            customer_id=st.customer_id, authenticated=st.customer_id is not None, language=lang,
            trigger=trigger, reason=HANDOFF_REASON.get(trigger, {}).get("es", trigger),
            priority=self.policy.trigger_priority(trigger), intent=st.intent,
            intent_confidence=st.intent_confidence, request_summary=summary, verified_facts=list(st.facts),
            actions_taken=list(st.actions), evidence_ids=list(st.evidence_ids), open_questions=open_q,
            transcript=st.transcript[-12:], trace_id=ctx.trace_id,
        )
        try:
            res = self.tools.handoff_to_human(ctx, payload.model_dump(mode="json"))
            verified = bool(res.get("verified"))
        except (ToolFailure, ToolError):
            verified = False
        st.actions.append(ActionRecord(action="handoff_to_human", status="verified" if verified else "failed",
                                       reference_id=handoff_id, detail=trigger))
        st.handoff_trigger = trigger
        st.stage = Stage.HANDED_OFF
        if verified:
            st.handoff_id = handoff_id
            out.append(t("handoff", lang, reason=HANDOFF_REASON.get(trigger, {}).get(lang, trigger),
                         handoff_id=handoff_id))
            return payload
        out.append(t("handoff_failed", lang))
        return payload

    # ===================================================================== helpers ==========
    @staticmethod
    def _numbered(txns: list[dict[str, Any]], lang: str) -> str:
        return "\n".join(f"{i + 1}. {render_txn(x, lang)}" for i, x in enumerate(txns))

    @staticmethod
    def _redact(text: str) -> str:
        return OTP_RX.sub("******", text)

    def _load(self, conversation_id: str, channel: str) -> ConversationState:
        raw = self.store.load_conversation(conversation_id)
        if raw:
            st = ConversationState.model_validate_json(raw)
            if st.stage not in TERMINAL:
                return st
        return ConversationState(conversation_id=conversation_id, channel=channel)

    def _save(self, st: ConversationState) -> None:
        self.store.save_conversation(st.conversation_id, st.model_dump_json())
        self.store.upsert_case({
            "conversation_id": f"{st.conversation_id}:{st.case_id}", "language": st.language, "intent": st.intent,
            "outcome": st.outcome(), "handed_off": int(st.stage == Stage.HANDED_OFF),
            "handoff_trigger": st.handoff_trigger, "resolved": int(st.outcome() == "resolved"),
            "feedback": st.feedback, "csat": st.csat, "turns": st.turns,
            "clarification_turns": st.clarification_turns, "cost_usd": st.cost_usd,
            "started_at": st.started_at, "ended_at": st.ended_at,
        })

    def state(self, conversation_id: str) -> ConversationState | None:
        raw = self.store.load_conversation(conversation_id)
        return ConversationState.model_validate_json(raw) if raw else None
