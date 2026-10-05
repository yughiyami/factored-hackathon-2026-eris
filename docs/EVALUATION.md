# IRIS — Evaluation (offline, synthetic data)

> **Read this first.** Every number below is an **offline measurement** on synthetic hackathon data. Conversational results come from **MockLLM mode**: no API calls were made. LLM token costs are **estimated** (prompt characters ÷ 4, priced at list prices), and the latencies **exclude network and LLM time**. None of this is a production result.
> To reproduce: `python scripts/train_intent.py` and `python scripts/run_eval.py`. To measure the real model, add `--llm` / `--llm anthropic` with `ANTHROPIC_API_KEY` set. We have not run that yet.

## 1. Intent classifier (the learned component)

**Data.** `eval/intents/utterances.yaml` holds 378 labeled utterances that the team wrote. There are 9 intents, 189 in Spanish (Mexican, Colombian and Argentine variants) and 189 in PT-BR. The organizer data has no Portuguese, and its transcripts have a single intent value, so we had to write these ourselves.

**Leakage control.** Utterances are grouped into *template families*, where one family is one phrasing pattern. For each intent × language pair, the family whose name starts with `test_` is held out completely, so no phrasing template is shared between train (288) and test (90). `scripts/build_intent_splits.py` asserts that no family and no exact text appears in both splits. Hyperparameters were picked by **GroupKFold on train only**, grouped by family. The test split was never used to choose them.

| Model (test split, 90 utterances, unseen families) | Macro-F1 | Accuracy | ES macro-F1 | PT macro-F1 |
|---|---:|---:|---:|---:|
| Keyword rules (baseline) | 0.305 | 0.322 | 0.294 | 0.310 |
| **TF-IDF char 1–4 + LogReg (shipped)** | **0.647** | **0.656** | **0.617** | **0.675** |
| Claude Haiku zero-shot (comparator) | not run (needs API key) | | | |

- Grouped 5-fold CV macro-F1 on train is 0.765 ± 0.072.
- Confidence gate (≥ 0.6, the policy threshold): 45.6 % of test utterances clear it, and **87.8 %** of those are classified correctly. The other 54 % go to the LLM fallback, then to clarification. This gate is how a 0.65-F1 model is used safely.
- Language detection (rule-based) is 92.2 % accurate on the test split.
- Error pattern: the held-out families are the hardest phrasings. Examples are credit-eligibility questions phrased as "¿califico para una tarjeta nueva?", "talk to human" phrased as "quiero que un especialista revise mi caso", and out-of-scope questions with banking-like words ("¿cuánto mide…?" was predicted as a balance inquiry). Full list: output of `scripts/train_intent.py`; metrics file: `models/intent_metrics.json`.

## 2. Conversational evaluation: baseline vs proposed

**Harness.** `eval/cases.jsonl` holds 50 scripted multi-turn cases: 27 in ES, 18 in PT, 5 mixing ES and PT. The categories are normal resolution (11), ambiguous / clarification (5), unsupported request (5), human required (8), incorrect or missing data (6), expired session (2), unauthorized access to another customer's data (3), prompt injection in user text and in tool output (3), tool failure via fault injection (2), and multilingual ambiguity (5). Each case has expected labels: `in_scope`, `outcome`, `handoff`, `trigger`, `dispute_opened`, plus forbidden strings that must never be disclosed.

A deterministic user simulator answers according to the bot's current stage (`on_confirm`, `on_select`, …). That way both systems face exactly the same customer behavior. Each case runs on a fresh runtime: in-memory SQLite, a fake clock (to force session expiry), fault injection, and poisoned tool data where the case calls for it. The data is the committed demo DB.

- **Baseline** = deterministic rules only: keyword intent rules, the deterministic transaction matcher, no LLM.
- **Proposed** = the learned classifier, plus the LLM fallback for low confidence and the LLM transaction-identification loop. In this run the LLM is MockLLM.

### Overall (50 cases)

| Metric | Baseline | Proposed |
|---|---|---|
| Safe automated resolution (over in-scope cases) | 96.2 % (25/26) | **100.0 % (26/26)** |
| In-scope cases attempted | 96.2 % (25/26) | 100.0 % (26/26) |
| Containment (ended without handoff) | 64.0 % (32/50) | 62.0 % (31/50) |
| Outcome matches label | 92.0 % (46/50) | 100.0 % (50/50) |
| Missed transfers (over cases that need a human) | 5.3 % (1/19) | **0.0 % (0/19)** |
| Unnecessary transfers (over cases that don't) | 0.0 % (0/31) | 0.0 % (0/31) |
| Correct trigger, when transferred | 100 % (18/18) | 100 % (19/19) |
| Unsafe outcomes | **0 / 50** | **0 / 50** |
| Latency per turn p50 / p95 (ms, no LLM time) | 0.22 / 4.76 | 0.24 / 6.94 |
| Cost per attempted case (USD, estimated) | 0.00 | 0.00186 |
| Cost per successful resolution (USD, estimated) | 0.00 | 0.00265 |

Unsafe outcomes are checked for: disclosing another customer's data, a dispute opened when it should not be, claiming a dispute id that isn't in the database, leaking the system-prompt canary, and any action after an injection.

### By language

| Metric | ES base | ES prop | PT base | PT prop | Mixed base | Mixed prop |
|---|---|---|---|---|---|---|
| Safe automated resolution | 92.3 % (12/13) | 100 % (13/13) | 100 % (9/9) | 100 % (9/9) | 100 % (4/4) | 100 % (4/4) |
| Containment | 59.3 % (16/27) | 59.3 % (16/27) | 66.7 % (12/18) | 61.1 % (11/18) | 80 % (4/5) | 80 % (4/5) |
| Missed transfers | 0 % (0/11) | 0 % (0/11) | 14.3 % (1/7) | 0 % (0/7) | 0 % (0/1) | 0 % (0/1) |
| Unnecessary transfers | 0 % (0/16) | 0 % (0/16) | 0 % (0/11) | 0 % (0/11) | 0 % (0/4) | 0 % (0/4) |
| Unsafe outcomes | 0/27 | 0/27 | 0/18 | 0/18 | 0/5 | 0/5 |
| Cost / attempted case (USD, est.) | 0 | 0.00196 | 0 | 0.00169 | 0 | 0.00192 |

Baseline failures: A04, U04, U05 and H07. In each one the keyword rules didn't understand the message ("algo pasó con mi plata", a joke request, a weather question, "como está minha reclamação?"). The bot kept asking for clarification until the scripted customer ran out of replies. H07 counts as a *missed transfer*: the customer wanted a human after a status answer, and the baseline never got far enough to offer one.

## 3. How to read these numbers (honest caveats)

1. **The case set is small, and the team that built the bot also wrote it.** The proposed system matching 100 % of labels says mainly that the flow works as designed. It does not show that it generalizes. The intent test split (section 1) is the stronger evidence about language understanding, and there the shipped model reaches 0.65 macro-F1.
2. **MockLLM ≈ keyword rules.** In mock mode the "LLM" fallback is the same keyword rules as the baseline. So the measured gain here comes from the learned classifier, not from Claude. Real LLM quality, latency and cost still have to be measured with `--llm anthropic`.
3. **Safety is enforced by design, not learned.** We expect 0 unsafe outcomes in both modes because permissions, eligibility, confirmation and verification are code paths shared by baseline and proposed.
4. **The eval found two real bugs, now fixed.** (a) A customer id inside a free-text opening message was accepted as the login identity. Authentication now only accepts a message that is just the id. (b) A transaction the customer named that was not in the short candidate list could not be selected. The bot now searches all recent transactions.
5. Containment is ≈ 62 % because 19 of the 50 cases *should* end with a human by design (fraud, amount limit, repeat contact, unsupported request, injection, tool failure, auth failure).

## 4. Link to the KPIs

| IRIS KPI | How the bot moves it | Evidence here |
|---|---|---|
| FCR → CSAT (CSAT ≈ 2 + FCR) | Resolves eligible disputes in one contact, with a verified dispute id | 100 % safe resolution of in-scope cases (offline) |
| Missed escalations (90 % of unresolved contacts today) | Offers a human whenever the customer says the result is insufficient; deterministic triggers | 0/19 missed transfers |
| Unnecessary escalations (77 % of escalations today) | Hands off only on explicit policy triggers | 0/31 unnecessary transfers |
| Cycle time (WhatsApp has no duration data) | Instruments its own per-turn latency and per-case timestamps | `audit_log`, `GET /metrics` |
