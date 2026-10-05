"""Runtime KPI counters computed from the case table and the audit log."""
from __future__ import annotations

from typing import Any

import numpy as np

from iris_bot.storage import Store

DEFINITIONS = {
    "fcr_proxy": "Closed cases where the customer confirmed the answer resolved the request (feedback=yes), over closed cases.",
    "containment": "Closed cases that ended without a human handoff, over closed cases.",
    "handoff_rate": "Cases handed off to a human, over closed cases.",
    "missed_escalation_proxy": "Closed cases where the customer said the answer was NOT sufficient and no handoff happened.",
    "unnecessary_escalation_proxy": "Handoffs triggered only by low understanding (not_understood) - candidates for better NLU.",
    "latency_ms": "Server-side processing time per customer message (p50/p95).",
    "cost_per_case_usd": "LLM cost per closed case (estimated when running with MockLLM).",
}


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def compute_metrics(store: Store) -> dict[str, Any]:
    cases = store.cases()
    closed = [c for c in cases if c["outcome"] not in ("in_progress",)]
    n = len(closed)
    handed = [c for c in closed if c["handed_off"]]
    lat = [r["latency_ms"] for r in store.audit_rows(step="turn") if r["latency_ms"] is not None]
    cost = sum(c["cost_usd"] or 0 for c in closed)
    csat = [c["csat"] for c in closed if c["csat"]]
    return {
        "cases_total": len(cases),
        "cases_closed": n,
        "fcr_proxy": _rate(sum(1 for c in closed if c["feedback"] == "yes"), n),
        "containment": _rate(n - len(handed), n),
        "handoff_rate": _rate(len(handed), n),
        "missed_escalation_proxy": _rate(sum(1 for c in closed if c["feedback"] == "no" and not c["handed_off"]), n),
        "unnecessary_escalation_proxy": _rate(sum(1 for c in handed if c["handoff_trigger"] == "not_understood"),
                                              len(handed)),
        "handoff_triggers": {t: sum(1 for c in handed if c["handoff_trigger"] == t)
                             for t in sorted({c["handoff_trigger"] for c in handed})},
        "latency_ms": {"p50": round(float(np.percentile(lat, 50)), 1) if lat else None,
                       "p95": round(float(np.percentile(lat, 95)), 1) if lat else None, "n_turns": len(lat)},
        "cost_per_case_usd": round(cost / n, 6) if n else None,
        "csat_mean_1_4": round(float(np.mean(csat)), 2) if csat else None,
        "csat_top2_rate": _rate(sum(1 for c in csat if c >= 3), len(csat)),
        "definitions": DEFINITIONS,
    }
