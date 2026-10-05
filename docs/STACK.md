# IRIS — Technology Stack

Each choice below includes a one-line reason and the alternatives we rejected.
The rule behind most choices: **the data showed that escalation can't be learned and that resolution drives CSAT**. So we use a deterministic core and add AI only where language understanding is needed.

| Layer | Choice | Rationale (one line) | Rejected alternatives |
|---|---|---|---|
| Workflow | Transaction-dispute intake ("no reconozco un cargo" / "cobro indebido") over simulated WhatsApp | Complaints are 17 % of contacts but have the worst CSAT (2.43) and FCR (43.6 %), and the two dispute subcategories are the most frequent | Generic FAQ bot (no measurable FCR lever); card blocking (urgent and irreversible, so a human is better) |
| Language / runtime | Python 3.13 | Same language as the data pipeline and the ML tooling | Node/TypeScript (would split the codebase in two) |
| API | FastAPI + Pydantic v2 + uvicorn | Typed request/response contracts, so the handoff payload is a validated schema | Flask (no typed models); Django (too heavy for one service) |
| LLM provider | **DeepSeek** (default, `IRIS_LLM_PROVIDER=deepseek`) via its OpenAI-compatible API and the `openai` SDK; Anthropic Claude kept as a switchable alternative | Team choice: low cost per case and native tool calling; the provider sits behind one `LLMClient` interface, so switching is a config change | LangChain (adds abstraction we don't need for a single bounded tool loop) |
| Orchestrator model | `deepseek-v4-pro` (env `IRIS_ORCHESTRATOR_MODEL`), thinking mode off | Runs only the read-only "which transaction?" tool loop; non-thinking mode keeps latency low and supports `tool_choice` | Thinking mode (slower, and `reasoning_content` must be passed back between tool turns) |
| Classifier / fallback model | `deepseek-flash` (env `IRIS_CLASSIFIER_MODEL`), JSON output mode | Cheapest model, used for zero-shot intent backup and handoff summaries | Calling the orchestrator model for every message |
| LLM data residency | Synthetic data only reaches the hosted DeepSeek API; a production bank deployment would self-host the open-weight model or use an in-region provider | Customer data must stay under the bank's control; the client already supports a custom `IRIS_DEEPSEEK_BASE_URL` | Sending real customer data to a third-party API |
| Offline mode | `MockLLM`, deterministic and on whenever no API key is set | Tests, CI, the eval harness and the public demo work with no key and no spend | Recorded API responses (they go stale and are hard to maintain) |
| Policy engine | Plain Python + `config/policy.yaml` | Auditable rules that can be versioned; escalation labels in the data are noise (AUC 0.50), so they can't be learned | Rules in prompts (not enforceable); a rules engine such as Drools or OPA (too heavy for ~15 rules) |
| Permissions | Enforced in the tool layer, scoped to `session.customer_id` | The model can't widen scope because no tool accepts a `customer_id` argument | Prompt instructions ("only show the user's data") |
| Auth | Mock trusted IdP: OTP challenge, then an HS256 JWT (PyJWT), 15-min TTL | Mirrors a real bank IdP contract; expiry is testable with an injectable clock | Real SMS OTP (out of scope; needs a telecom provider) |
| Data access | DuckDB over silver Parquet, or a committed 1.8 MB demo DuckDB | Same SQL locally and in the cloud; the deployed demo doesn't need the 5 GB lake | Postgres (an extra service to run); querying pandas in-process (slower, needs more memory) |
| Bot state / audit | SQLite (`disputes`, `handoffs`, `conversations`, `cases`, `audit_log`) | Zero-ops, transactional, gives an idempotency key on disputes | Redis (not durable by default); Postgres (more infrastructure for the demo) |
| Learned component | TF-IDF character n-grams (1–4) + logistic regression, ES + PT-BR | Robust to typos and accents across MX/CO/AR/BR, trains in seconds, and calibrated probabilities feed the confidence gate | Fine-tuned BERT (300 labels are too few, and inference is heavier); LLM-only (cost per message, and a hard dependency on the provider) |
| Language detection | Rule-based ES/PT marker words + orthography | 92 % on the test split, no dependency | `langdetect`/fastText (mis-detects short, mixed ES/PT messages) |
| Injection guard | Deterministic regex (ES/PT/EN) on user text and tool outputs | Runs before any model sees the text; tool data is sanitized and wrapped as untrusted | LLM-as-judge (can itself be injected, and adds latency) |
| Observability | structlog JSON logs + per-turn trace rows in SQLite `audit_log` | Each turn and tool call has a trace_id, latency, tokens, cost and outcome | OpenTelemetry collector (planned; see ROADMAP) |
| Reliability | Bounded exponential backoff (tools and LLM), SDK retries off, safe fallback message | One retry policy that we control; tool failures become a `tool_failure` handoff | Unbounded SDK retries (latency is unpredictable) |
| Channel / UI | `POST /webhook/whatsapp` (Twilio form, TwiML reply), `POST /chat`, static web chat at `/` | The WhatsApp-compatible contract is ready; the web page lets judges try it with no setup | React SPA (needs a build step); Streamlit (not a chat channel) |
| Agent console / KPIs | `GET /handoffs`, `GET /metrics` (JSON) | Enough for the demo and for scraping into a dashboard | A full agent desktop (out of scope) |
| Pipeline DAG | Dagster software-defined assets wrapping the existing scripts (`dags/iris_assets.py`), optional extra | Asset lineage with no rewrite of the pipeline | Airflow (heavier to run for a 10-day build) |
| Packaging | `pyproject.toml` with extras `[llm]`, `[pipeline]`, `[dagster]`, `[dev]` | The bot image installs only what it needs | requirements.txt files (no extras) |
| Container / deploy | Docker (`python:3.13-slim`), docker-compose, Render Blueprint (`render.yaml`) | One-click deploy from a public repo with the demo DB committed | Fly.io/Railway (similar; Render has a free web tier with a Blueprint file); AWS ECS (too much setup) |
| CI | GitHub Actions: ruff + pytest + offline eval | Every push proves the safety tests and the eval still pass without secrets | — |
| Tests | pytest + FastAPI TestClient | Fast (≈2 s) and offline | — |
