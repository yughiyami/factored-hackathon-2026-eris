# IRIS — Remaining Work to Production (honest list)

What exists today is a working prototype. It is offline-evaluated on synthetic data, with a mock identity provider, a mock LLM by default, and a local SQLite store. Below is what would still have to change before it could take real customers.

## Must-have before any real traffic

| Area | Gap today | Needed |
|---|---|---|
| LLM measurement | Conversational eval runs only on MockLLM; the Claude zero-shot intent comparator hasn't run | Run `train_intent.py --llm` and `run_eval.py --llm anthropic`; record real latency, cost, cache hit rate and refusal rate |
| Identity | Mock IdP with a fixed demo OTP | Integrate the bank IdP (OTP to the registered phone, device binding); rate-limit OTP attempts per number; set the session secret from a KMS |
| Channel | Twilio-shaped webhook with no signature check; the web chat is public | Validate `X-Twilio-Signature`; use the WhatsApp Business API with approved templates; add per-sender rate limits and abuse protection |
| Core banking | Read-only DuckDB demo DB; disputes stored in local SQLite | Real core-banking / card-processor APIs for reads and dispute creation (e.g. chargeback reason codes); queue writes with an outbox |
| Data protection | Transcripts stored in SQLite with only the OTP redacted | PII redaction (card numbers, document ids), encryption at rest, retention policy, a data-processing agreement with the LLM provider, regional data residency |
| Human handoff | Payload stored and listed at `GET /handoffs` (JSON) | Integrate the contact-center platform (Genesys/Zendesk/Salesforce); routing by priority; SLA timers; agent feedback loop |
| Policy ownership | YAML edited by engineers | Policy versioning with approvals from the risk and compliance owners; record the policy version in every audit row |
| Evaluation data | 50 scripted cases and 378 utterances, all written by the team | Real, labeled WhatsApp transcripts (ES + PT) with an independent labeling team; red-team set for injection; periodic re-labeling |
| Observability | SQLite audit log + JSON logs | OpenTelemetry traces, a metrics backend, alerts on handoff-rate drift, tool error rate and p95 latency; dashboards |
| Load / resilience | Single process; SQLite | Stateless API replicas, Postgres/Redis for conversation state, circuit breakers on tools, load test at peak volume |

## Should-have

- Calibrate the intent classifier (it currently clears the 0.6 gate on only 46 % of held-out utterances). Retrain on real data, consider a small multilingual encoder, and track per-language drift.
- Ask the reason sub-type (wrong fee vs duplicate charge vs unrecognized) as structured slots, and attach evidence (receipts) through WhatsApp media.
- Proactive status updates through WhatsApp templates when a dispute changes state.
- An A/B or shadow-mode rollout that compares IRIS with the current human FCR / CSAT per contact reason, which is the real test of the KPI hypothesis (CSAT ≈ 2 + FCR).
- Accessibility and tone review of the ES and PT templates by native speakers from MX, CO, AR and BR.
- Full Dagster deployment (schedules, sensors on new S3 partitions, asset checks wired to `contracts.py`).

## Known limitations

- The data is synthetic and its relationships are rule-like. The data shows that escalation is not learnable from history, so handoff stays rule-based by design.
- The "as-of" date is the latest data timestamp (2026-06-18). The transaction age rule is measured against it, not against today.
- The cost figures in `docs/EVALUATION.md` are estimates from MockLLM, not invoices.
