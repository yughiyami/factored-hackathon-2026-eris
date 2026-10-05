# IRIS — AI-first transaction-dispute intake (Factored AI & Data Hackathon 2026, team Eris)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/yughiyami/factored-hackathon-2026-eris)

The deploy uses `render.yaml` (Docker, free plan, committed demo DB, no data lake needed). Set `ANTHROPIC_API_KEY` in Render to use Claude; without it the bot runs on the deterministic MockLLM. On the free plan the service sleeps after 15 minutes idle (first request takes ~1 minute), and the SQLite runtime state (disputes, handoffs, audit log) resets on every restart.

IRIS is a WhatsApp-style assistant for a Latin American bank. It handles one workflow end to end: a customer who **does not recognize a charge** ("no reconozco un cargo") or was **charged a wrong fee** ("cobro indebido"). It works in Spanish and Brazilian Portuguese.

In one conversation, IRIS:

1. authenticates the customer;
2. understands the request (learned classifier, with an LLM fallback);
3. finds the transaction using session-scoped tools;
4. applies a deterministic policy;
5. opens a dispute, and verifies it, only after the customer confirms;
6. hands off to a human with a structured JSON payload whenever policy requires it.

**Why this workflow.** On the bank's 686k historical contacts, CSAT depends almost only on first-contact resolution. Complaints and disputes have the worst FCR (43.6 %) and CSAT (2.43). Historical escalation labels are noise (AUC 0.50), so handoff is a deterministic policy, not a model. Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/KPIS.md](docs/KPIS.md).

## Quickstart (no API key, no data download needed)

```bash
pip install -e ".[dev]"            # add ,llm to use Claude
python -m pytest                   # 54 tests, ~2 s, offline
python -m uvicorn iris_bot.api.app:app --app-dir src --port 8000
# open http://localhost:8000 — the "Demo" panel lists synthetic customer ids and the demo OTP
```

With Docker: `docker compose up --build`. With Claude: copy `.env.example` to `.env`, set `ANTHROPIC_API_KEY` and keep `IRIS_LLM_MODE=auto`. Without a key, the bot uses the deterministic **MockLLM**.

| Endpoint | Purpose |
|---|---|
| `GET /` | WhatsApp-style web chat (static HTML/JS) |
| `POST /chat` | JSON chat: `{"conversation_id": "...", "message": "..."}` |
| `POST /webhook/whatsapp` | Twilio-compatible WhatsApp webhook (form `From`, `Body`; TwiML reply) |
| `GET /handoffs`, `GET /handoffs/{id}` | Agent console view of queued handoff payloads |
| `GET /metrics` | KPI counters: FCR proxy, containment, handoff rate, missed / unnecessary escalation proxies, p50/p95 latency, cost per case |
| `GET /healthz`, `GET /demo-info` | Health check; demo identities |

## Results (offline, synthetic data — see [docs/EVALUATION.md](docs/EVALUATION.md))

- **Intent classifier**, held-out template families, 90 utterances (ES + PT): TF-IDF + LogReg reaches macro-F1 **0.647**, against **0.305** for keyword rules. Above the 0.6 confidence gate it is 87.8 % accurate.
- **Conversational eval**, 50 scripted cases, MockLLM. Proposed vs rules-only baseline:
  - safe automated resolution: **100 % (26/26)** vs 96.2 % in-scope
  - missed transfers: **0/19** vs 1/19
  - unnecessary transfers: 0/31 in both
  - unsafe outcomes: **0/50** in both
  - estimated cost: $0.0026 per successful resolution

## Repository map

```
src/iris_bot/            the bot
  api/app.py             FastAPI: /chat, /webhook/whatsapp, /, /handoffs, /metrics
  agent/engine.py        IRIS state machine (auth → understand → clarify → route → tools → verify → feedback → handoff/CSAT)
  agent/models.py        conversation state + HandoffPayload contract
  policy.py              deterministic policy engine (config/policy.yaml)
  auth.py                mock trusted IdP: OTP → HS256 session (15 min)
  tools/                 session-scoped banking tools (RPA layer), retries, fault injection, audit
  nlu/                   keyword baseline, TF-IDF+LogReg model, ES/PT detection, slot extraction
  llm/                   Anthropic client (manual tool loop, caching, fallback) + offline MockLLM
  guard.py               prompt-injection guard for user text and tool output
  repository.py          DuckDB over demo DB or data/silver parquet
  storage.py, metrics.py SQLite disputes / handoffs / audit_log; KPI counters
src/eris/                data pipeline: ingest → bronze → silver → gold, KPI relevance, scorecard
config/policy.yaml       thresholds and handoff triggers
demo/                    committed demo DuckDB (500 customers, 1.8 MB) + fixtures.json
models/                  intent.joblib (170 KB) + intent_metrics.json
eval/                    cases.jsonl (50 multi-turn cases), intents/ (378 labeled utterances)
scripts/                 build_demo_db, build_intent_splits, train_intent, run_eval
dags/iris_assets.py      Dagster assets wrapping the pipeline and bot artifacts
tests/                   policy, auth/expiry, permission isolation, injection, handoff completeness, flows, API
docs/                    ARCHITECTURE, KPIS, STACK, BOT_DESIGN, EVALUATION, ROADMAP
```

## Make targets

`make setup | data | pipeline | demo | train | eval | test | lint | run | docker | dagster`

## Data pipeline (optional — needs organizer S3 access)

```bash
pip install -e ".[pipeline]"
# .env (git-ignored): AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, DATATHON_BUCKET — see .env.example
make data pipeline demo            # S3 → bronze → silver → gold → reports → demo/iris_demo.duckdb
```

Set `IRIS_DATA_BACKEND=silver` to run the bot on the full lake instead of the demo DB.

## Docs

- [ARCHITECTURE.md](docs/ARCHITECTURE.md): plan, data findings, target architecture
- [KPIS.md](docs/KPIS.md): KPI selection and baselines
- [STACK.md](docs/STACK.md): stack decisions, with rationale and rejected alternatives
- [BOT_DESIGN.md](docs/BOT_DESIGN.md): flow diagram, state machine, policy table, AI vs deterministic, tool contracts
- [EVALUATION.md](docs/EVALUATION.md): intent metrics and baseline-vs-proposed eval (offline / mock)
- [ROADMAP.md](docs/ROADMAP.md): what remains before production

## Hackathon deliverables

- [ ] Public repository named `factored-hackathon-2026-eris` (MIT license, no secrets: `.env` is git-ignored, `.env.example` has placeholders only)
- [ ] Deployed link (Render Blueprint `render.yaml`, uses the committed demo DB): `<add URL>`
- [ ] 4–6 slides: problem and data evidence, KPI selection, architecture and AI-vs-rules split, live demo, evaluation, roadmap
- [ ] Demo video (≤ 5 min): ES happy path, PT happy path, fraud handoff with payload, injection attempt, session expiry
- [x] Data pipeline (bronze/silver/gold, contracts, DQ report) and KPI relevance analysis
- [x] Bot: policy engine, auth, scoped tools, learned intent classifier, LLM integration, handoff payload
- [x] Offline evaluation: intent metrics and 50-case baseline-vs-proposed harness

## License

MIT. See [LICENSE](LICENSE). The dataset is synthetic and provided by the hackathon organizers.
