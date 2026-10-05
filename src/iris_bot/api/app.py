"""FastAPI app: JSON chat, Twilio-compatible WhatsApp webhook, web chat page, agent console, metrics."""
from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from xml.sax.saxutils import escape

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from iris_bot.agent import AgentReply
from iris_bot.config import ROOT, Settings, get_settings
from iris_bot.metrics import compute_metrics
from iris_bot.observability import configure_logging, get_logger
from iris_bot.runtime import Runtime, build_runtime

STATIC = Path(__file__).parent / "static"


class ChatIn(BaseModel):
    conversation_id: str | None = Field(default=None, max_length=100)
    message: str = Field(min_length=1, max_length=1000)


def create_app(settings: Settings | None = None, runtime: Runtime | None = None) -> FastAPI:
    configure_logging()
    log = get_logger("iris.api")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.rt = runtime or build_runtime(settings or get_settings())
        log.info("startup", llm=getattr(app.state.rt.agent.llm, "name", None),
                 data=getattr(app.state.rt.repo, "source", None),
                 classifier=getattr(app.state.rt.agent.classifier, "name", None))
        yield
        app.state.rt.store.close()

    app = FastAPI(title="IRIS dispute-intake bot", version="0.1.0", lifespan=lifespan)

    def rt(request: Request) -> Runtime:
        return request.app.state.rt

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/healthz")
    def healthz(request: Request) -> dict:
        r = rt(request)
        return {"status": "ok", "llm": getattr(r.agent.llm, "name", None),
                "classifier": getattr(r.agent.classifier, "name", None), "data": getattr(r.repo, "source", None),
                "as_of": str(r.repo.as_of)}

    @app.post("/chat", response_model=AgentReply)
    def chat(body: ChatIn, request: Request) -> AgentReply:
        cid = body.conversation_id or ("web:" + uuid.uuid4().hex[:16])
        return rt(request).agent.handle(cid, body.message, channel="web")

    @app.post("/webhook/whatsapp")
    def whatsapp(request: Request, From: str = Form(...), Body: str = Form("")) -> Response:  # noqa: N803
        """Twilio WhatsApp inbound webhook. Replies with TwiML (one <Message> per bot message)."""
        reply = rt(request).agent.handle("wa:" + From[:60], Body[:1000], channel="whatsapp")
        msgs = "".join(f"<Message>{escape(m)}</Message>" for m in reply.messages)
        return Response(content=f'<?xml version="1.0" encoding="UTF-8"?><Response>{msgs}</Response>',
                        media_type="application/xml")

    @app.get("/handoffs")
    def handoffs(request: Request, limit: int = 50) -> list[dict]:
        return rt(request).store.handoffs(limit=min(limit, 200))

    @app.get("/handoffs/{handoff_id}")
    def handoff(handoff_id: str, request: Request) -> dict:
        h = rt(request).store.handoff(handoff_id)
        if h is None:
            raise HTTPException(404, "handoff not found")
        return h

    @app.get("/metrics")
    def metrics(request: Request) -> dict:
        return compute_metrics(rt(request).store)

    @app.get("/demo-info")
    def demo_info(request: Request) -> dict:
        """Synthetic demo identities so visitors can try the bot (mock identity provider only)."""
        r = rt(request)
        fixtures = ROOT / "demo" / "fixtures.json"
        roles = json.loads(fixtures.read_text(encoding="utf-8"))["roles"] if fixtures.exists() else {}
        pick = {k: {"customer_id": v["customer_id"], "try": f"{v.get('merchant_name') or v['transaction_type']} "
                    f"{v['amount']} {v['currency']}"} for k, v in roles.items()
                if k in ("happy_a", "happy_c", "fee", "fraud", "high", "declined")}
        return {"otp": r.settings.test_otp, "customers": pick, "as_of": str(r.repo.as_of),
                "note": "Synthetic data and a mock identity provider; the OTP is a fixed demo code."}

    return app


app = create_app()
