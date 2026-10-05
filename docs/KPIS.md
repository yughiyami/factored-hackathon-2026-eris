# IRIS — KPI Selection

Source: `src/eris/analytics/kpis.py` → `reports/kpi_scorecard.csv`, `reports/kpi_summary.json`,
`data/gold/kpi_flow.parquet` (monthly × country × channel × reason). 686,296 contacts, Jun 2023 – Jun 2026.
All numbers are offline measurements on synthetic data.

## How KPIs were selected

Each of the 14 candidate KPIs, mapped to a stage of the IRIS flow, was scored on:

| Criterion | Weight | Meaning |
|---|---:|---|
| Coverage | 0.20 | Share of contacts where the KPI can be computed |
| Sensitivity | 0.25 | Spread across contact reasons, i.e. it discriminates where to act |
| Stability | 0.15 | 1 − monthly coefficient of variation |
| **CSAT link (adjusted)** | 0.40 | CSAT effect **within** contact-reason × resolution strata |

The adjustment matters: AHT (ρ = −0.17), follow-up (CSAT 2.33 vs 3.00) and negative sentiment
(2.64 vs 2.79) look related to satisfaction, but inside each reason × resolved stratum the effect is
≈ 0 (≤ 0.016). Their raw link comes from FCR. Without the adjustment we would pick the wrong KPIs.

## Selected KPIs

| Tier | KPI | IRIS stage | Baseline | Why |
|---|---|---|---|---|
| **North Star** | CSAT top-2 (score ≥ 3, observed 1–4 scale) | 7. Survey | **68.7 %** (mean 2.77) | Outcome the program answers for |
| **Primary driver** | First-contact resolution (FCR) | 5. Result sufficient | **76.7 %** | Only KPI with a real CSAT effect: resolved 3.00 vs unresolved 2.00 (+1.0 point) |
| **Escalation quality** | Missed escalations: unresolved contacts with no handoff | 6. Human escalation | **90.1 %** of unresolved | The flow says "insufficient → offer a human"; today 9 of 10 unresolved customers are left without one |
| **Escalation quality** | Unnecessary escalations: escalated but resolvable | 6. Human escalation | **76.7 %** of escalations | Human capacity spent on cases that were resolvable |
| **Interaction frequency** | Repeat contact, same reason ≤ 30 days | 3. Understanding loop | **2.7 %** | Measures "really solved"; highest sensitivity across reasons (1.39) |
| **Efficiency** | Cycle time (wait + handling) | 5. Result | **419 s** (AHT 321 s, wait 120 s) | No CSAT effect, but drives cost; 2× spread across reasons |
| **Diagnostic** | Negative sentiment rate | 3. Understanding | 19.2 % | Early-warning signal; not a CSAT driver |

**Rejected as management KPIs:**
- **Raw % redirection (10.0 %):** it is flat across every reason, channel and month, and has no CSAT link. Report it, but split it into the missed and unnecessary rates above.
- **Contacts per customer 90d:** shows no variation.
- **Wait time:** has no CSAT link and 30 % of values are missing.
- **Survey response rate (31 %):** this is the measurement base, not a lever.

## Baseline by contact reason

| Reason | Contacts | FCR | CSAT | Cycle (s) | Redirection | Missed escalation |
|---|---:|---:|---:|---:|---:|---:|
| transactional | 240,056 | 91.5 % | 2.91 | 318 | 9.9 % | 90.1 % |
| product | 150,863 | 89.6 % | 2.90 | 364 | 10.0 % | 90.4 % |
| **complaint** | 117,021 | **43.6 %** | **2.43** | **533** | 10.0 % | 90.1 % |
| technical | 102,899 | 69.9 % | 2.70 | 458 | 10.1 % | 89.9 % |
| commercial | 54,879 | 65.2 % | 2.66 | 638 | 9.8 % | 90.0 % |
| retention | 20,578 | 60.2 % | 2.61 | 578 | 9.8 % | 90.4 % |

**Dispute / complaint operations (67,095 cases)**

| Metric | Value |
|---|---|
| SLA breached | 20.1 % |
| Median resolution | 16 days |
| Median first response | 38 h |
| Unassigned | 34.5 % |
| Open / in process | 70 % |
| Repeat complainers | 15.0 % |

The top dispute subcategories are *Cargo no reconocido* (12,297 cases) and *Cobro indebido* (12,194 cases).

**Simulation (offline, not a production claim):** CSAT ≈ 2 + FCR. Raising complaint FCR from 43.6 % to the
technical-reason level (69.9 %) would move global FCR from 76.7 % to ≈ 81.2 % and mean CSAT from 2.77 to ≈ 2.81.

## KPIs IRIS must instrument (not measurable in the supplied data)

| IRIS stage | KPI | Gap in data |
|---|---|---|
| 1. WhatsApp contact | Cycle time on WhatsApp | AHT is 100 % missing for WhatsApp, Email and Web Chat |
| 2. Authentication | Auth success rate, time-to-auth | No auth events in contact-center data |
| 3. Clarification loop | Clarifying turns per case; % understood on 1st turn | No turn-level data; transcript intents single-valued |
| 4. Routing | Intent/reason classification accuracy (ES/PT) | Needs labeled eval set; no Portuguese data |
| 5. Retrieval/verification | Grounded-answer rate; verified-action rate | System-generated |
| 6. Escalation | Handoff completeness (request, facts, actions, open questions) | System-generated |
| All | Safe automated resolution, containment, unsafe outcomes, p50/p95 latency, cost per resolution | Hackathon evaluation metrics |

## Data caveats

- **Cross-table causality was tested and not found.** The rate of digital errors and declined transactions in the 7 days before a contact (1.0 %) equals the rate in the 7 days after (placebo test). These tables are useful for grounding the agent, not for predicting demand.
- **Agent specialty and channel have no effect** on FCR or CSAT (all within ±1 pp).
- **The CSAT base is small:** CSAT surveys cover 18.6 % of contacts.
