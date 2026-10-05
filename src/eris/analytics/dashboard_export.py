"""Export gold-layer aggregates for the IRIS KPI Board and render dashboard/iris-kpi-board.html.

Run after gold.py, feature_relevance.py and kpis.py. The page embeds its data, so it can be
published as a static file with no backend.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
F = f"'{ROOT / 'data' / 'gold' / 'interaction_features.parquet'}'"
C = f"'{ROOT / 'data' / 'silver' / 'complaints.parquet'}'"
REPORTS = ROOT / "reports"
DASH = ROOT / "dashboard"

CYCLE = "coalesce(wait_time_seconds, 0) + duration_seconds"
MISSED = "CASE WHEN NOT was_resolved THEN (NOT was_escalated)::INT END"
UNNECESSARY = "CASE WHEN was_escalated THEN was_resolved::INT END"
REPEAT30 = "coalesce((same_reason_as_last AND days_since_last_contact <= 30)::INT, 0)"


def main() -> None:
    con = duckdb.connect()

    def q(sql: str) -> list[dict]:
        return con.execute(sql).df().round(4).to_dict(orient="records")

    out = {
        "monthly": q(f"""SELECT strftime(date_trunc('month', interaction_date), '%Y-%m') m, count(*) n,
            avg(was_resolved::INT) fcr, avg(csat_score) csat, avg((csat_score >= 3)::INT) csat_top2,
            avg(was_escalated::INT) redir, avg({CYCLE}) cycle_s, avg({MISSED}) missed,
            avg({UNNECESSARY}) unnecessary, avg({REPEAT30}) repeat30
            FROM {F} WHERE interaction_date >= '2023-07-01' AND interaction_date < '2026-06-01'
            GROUP BY 1 ORDER BY 1"""),
        "reason": q(f"""SELECT contact_reason r, count(*) n, avg(was_resolved::INT) fcr, avg(csat_score) csat,
            avg({CYCLE}) cycle_s, avg(was_escalated::INT) redir, avg({REPEAT30}) repeat30
            FROM {F} GROUP BY 1 ORDER BY n DESC"""),
        "esc_matrix": q(f"""SELECT was_resolved, was_escalated, count(*) n, avg(csat_score) csat
            FROM {F} GROUP BY ALL ORDER BY ALL"""),
        "disputes": q(f"""SELECT coalesce(subcategory, '(none)') s, count(*) n, avg(sla_breached::INT) sla,
            median(resolution_days) med_days, avg((assigned_agent_id IS NULL)::INT) unassigned
            FROM {C} GROUP BY 1 ORDER BY n DESC"""),
        "dispute_status": q(f"SELECT status s, count(*) n FROM {C} GROUP BY 1 ORDER BY n DESC"),
        "totals": q(f"""SELECT count(*) n, count(DISTINCT customer_id) customers, avg(was_resolved::INT) fcr,
            avg(csat_score) csat, avg((csat_score >= 3)::INT) csat_top2, avg(was_escalated::INT) redir,
            avg({CYCLE}) cycle_s, avg({MISSED}) missed, avg({UNNECESSARY}) unnecessary,
            avg({REPEAT30}) repeat30, count(csat_score) n_csat FROM {F}""")[0],
    }
    cols = ["kpi", "stage", "value", "coverage", "sensitivity", "stability",
            "csat_link_raw", "csat_link_adjusted", "score"]
    out["scorecard"] = pd.read_csv(REPORTS / "kpi_scorecard.csv")[cols].round(4).to_dict(orient="records")
    out["models"] = json.loads((REPORTS / "model_results.json").read_text())

    data = json.dumps(out, ensure_ascii=False, default=str)
    (REPORTS / "dashboard_data.json").write_text(data, encoding="utf-8")
    template = (DASH / "template.html").read_text(encoding="utf-8")
    (DASH / "iris-kpi-board.html").write_text(template.replace("__DATA__", data), encoding="utf-8")
    print("wrote", DASH / "iris-kpi-board.html")


if __name__ == "__main__":
    main()
