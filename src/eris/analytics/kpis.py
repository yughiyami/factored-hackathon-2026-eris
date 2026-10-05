"""IRIS KPI catalog aligned to the customer-service flow, plus a scorecard to pick the best KPIs.

Flow stages (IRIS diagram): contact -> authentication -> request/understanding (clarification
loop) -> routing -> retrieval/verification -> result + confirmation -> (insufficient?) human
escalation -> satisfaction survey.

Every candidate KPI is scored on four criteria so the selection is evidence-based:
  coverage     share of interactions where the KPI can be computed from the data
  sensitivity  spread of the KPI across contact reasons (max - min, relative to the mean)
  stability    1 - coefficient of variation across months (noisy KPIs are hard to manage)
  csat_link    |CSAT difference| between interactions with/without the event (binary KPIs)
               or |Spearman rho| with CSAT (continuous KPIs)
Outputs: data/gold/kpi_flow.parquet, reports/kpi_scorecard.csv, reports/kpi_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold"
REPORTS = ROOT / "reports"
FEATURES = GOLD / "interaction_features.parquet"

# name -> (flow stage, kind, SQL expression evaluated per interaction, direction, definition)
# kind: "rate" = mean of a 0/1 flag; "value" = mean of a continuous measure.
KPIS: dict[str, tuple[str, str, str, str, str]] = {
    "fcr_rate": ("5. Result sufficient", "rate", "was_resolved::INT", "up",
                 "Share of contacts resolved in the first contact"),
    "redirection_rate": ("6. Human escalation", "rate", "was_escalated::INT", "down",
                         "Share of contacts transferred to a supervisor/human"),
    "missed_escalation_rate": ("6. Human escalation", "rate",
                               "CASE WHEN NOT was_resolved THEN (NOT was_escalated)::INT END", "down",
                               "Unresolved contacts that ended without a human handoff"),
    "unnecessary_escalation_rate": ("6. Human escalation", "rate",
                                    "CASE WHEN was_escalated THEN was_resolved::INT END", "down",
                                    "Escalated contacts that were already resolvable"),
    "followup_rate": ("3. Understanding / clarification", "rate", "requires_followup::INT", "down",
                      "Contacts that need a follow-up (proxy of incomplete understanding)"),
    "repeat_contact_7d": ("3. Understanding / clarification", "rate",
                          "CASE WHEN days_since_last_contact IS NOT NULL THEN (days_since_last_contact <= 7)::INT "
                          "ELSE 0 END", "down", "Customer contacted again within 7 days of a previous contact"),
    "repeat_same_reason_30d": ("3. Understanding / clarification", "rate",
                               "coalesce((same_reason_as_last AND days_since_last_contact <= 30)::INT, 0)", "down",
                               "Repeat contact for the same reason within 30 days"),
    "aht_seconds": ("5. Result sufficient", "value", "duration_seconds", "down",
                    "Average handling time (seconds, synchronous channels only)"),
    "wait_seconds": ("1. Contact", "value", "wait_time_seconds", "down",
                     "Queue wait before attention (phone only)"),
    "cycle_time_seconds": ("5. Result sufficient", "value", "coalesce(wait_time_seconds, 0) + duration_seconds",
                           "down", "Wait + handling time"),
    "survey_response_rate": ("7. Satisfaction survey", "rate",
                             "((csat_score IS NOT NULL) OR (nps_score IS NOT NULL) OR (ces_score IS NOT NULL))::INT",
                             "up", "Contacts with any survey answer"),
    "csat_top2": ("7. Satisfaction survey", "rate", "CASE WHEN csat_score IS NOT NULL THEN (csat_score >= 3)::INT END",
                  "up", "CSAT >= 3 on the observed 1-4 scale"),
    "negative_sentiment_rate": ("3. Understanding / clarification", "rate",
                                "(detected_sentiment IN ('negative', 'very_negative'))::INT", "down",
                                "Contacts with negative detected sentiment"),
    "contacts_per_customer_90d": ("1. Contact", "value", "prior_contacts_90d", "down",
                                  "Prior contacts in the last 90 days (interaction frequency)"),
}


# Outcomes are measured, not used to explain CSAT. FCR-derived KPIs are stratified only by
# contact reason (stratifying by was_resolved would erase them by construction).
OUTCOMES = {"csat_top2", "survey_response_rate"}
FCR_DERIVED = {"fcr_rate", "missed_escalation_rate", "unnecessary_escalation_rate"}


def raw_link(con: duckdb.DuckDBPyConnection, base: str, kind: str) -> tuple[float, str]:
    if kind == "rate":
        hi, lo = con.execute(f"""SELECT avg(csat_score) FILTER (k = 1), avg(csat_score) FILTER (k = 0)
                                 FROM ({base}) WHERE csat_score IS NOT NULL""").fetchone()
        if hi is None or lo is None:
            return 0.0, "n/a"
        return abs(hi - lo), f"CSAT {hi:.2f} vs {lo:.2f}"
    d = con.execute(f"SELECT k, csat_score FROM ({base}) WHERE k IS NOT NULL AND csat_score IS NOT NULL").df()
    rho = d["k"].corr(d["csat_score"], method="spearman")
    return abs(rho), f"spearman {rho:+.3f}"


def adjusted_link(con: duckdb.DuckDBPyConnection, expr: str, kind: str, fcr_derived: bool) -> float:
    """Size-weighted CSAT link inside contact_reason (x was_resolved) strata."""
    strata = "contact_reason" if fcr_derived else "contact_reason, was_resolved"
    d = con.execute(f"""SELECT {expr} AS k, csat_score, {strata} FROM '{FEATURES}'
                        WHERE csat_score IS NOT NULL AND ({expr}) IS NOT NULL""").df()
    total, acc = 0, 0.0
    for _, g in d.groupby(strata.split(", ")):
        if kind == "rate":
            if g["k"].nunique() < 2:
                continue
            effect = g.loc[g.k == 1, "csat_score"].mean() - g.loc[g.k == 0, "csat_score"].mean()
        else:
            effect = g["k"].corr(g["csat_score"], method="spearman")
        if pd.notna(effect):
            acc += abs(effect) * len(g)
            total += len(g)
    return acc / total if total else 0.0


def build_flow_mart(con: duckdb.DuckDBPyConnection) -> None:
    rates = ",\n".join(f"avg({expr}) AS {name}" for name, (_, _, expr, _, _) in KPIS.items())
    con.execute(f"""COPY (
        SELECT date_trunc('month', interaction_date)::DATE AS month, country, channel, contact_reason,
               count(*) AS interactions, count(DISTINCT customer_id) AS customers, avg(csat_score) AS csat_avg,
               {rates}
        FROM '{FEATURES}' GROUP BY ALL
    ) TO '{GOLD / 'kpi_flow.parquet'}' (FORMAT parquet)""")


def score_kpis(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows = []
    for name, (stage, kind, expr, direction, definition) in KPIS.items():
        base = f"SELECT {expr} AS k, csat_score, contact_reason, date_trunc('month', interaction_date) m FROM '{FEATURES}'"
        value, coverage = con.execute(f"SELECT avg(k), avg((k IS NOT NULL)::INT) FROM ({base})").fetchone()
        by_reason = con.execute(f"SELECT avg(k) FROM ({base}) GROUP BY contact_reason").df().iloc[:, 0]
        monthly = con.execute(
            f"SELECT avg(k) v FROM ({base}) WHERE m BETWEEN '2023-07-01' AND '2026-05-01' GROUP BY m").df()["v"]
        if name in OUTCOMES:
            link, link_desc, adj = 0.0, "outcome itself", 0.0
        else:
            link, link_desc = raw_link(con, base, kind)
            adj = adjusted_link(con, expr, kind, name in FCR_DERIVED)
        rows.append({
            "kpi": name, "stage": stage, "definition": definition, "direction": direction,
            "value": round(value, 4), "coverage": round(coverage, 3),
            "sensitivity": round((by_reason.max() - by_reason.min()) / abs(value), 3) if value else 0.0,
            "stability": round(max(0.0, 1 - monthly.std() / abs(monthly.mean())), 3),
            "csat_link_raw": round(link, 3), "csat_link_detail": link_desc,
            "csat_link_adjusted": round(adj, 3),
        })
    df = pd.DataFrame(rows)
    # Normalize each criterion to [0, 1] across candidates. The confounder-adjusted CSAT link
    # weighs most: satisfaction is the outcome the program is accountable for, and a raw link
    # that disappears within reason x resolution strata is borrowed from FCR, not real.
    weights = {"coverage": 0.2, "sensitivity": 0.25, "stability": 0.15, "csat_link_adjusted": 0.4}
    score = sum(w * df[c] / (df[c].max() or 1) for c, w in weights.items())
    df["score"] = score.round(3)
    return df.sort_values("score", ascending=False)


def dispute_kpis(con: duckdb.DuckDBPyConnection) -> dict:
    c = SILVER / "complaints.parquet"
    return con.execute(f"""SELECT
        count(*) AS cases,
        avg(sla_breached::INT) AS sla_breach_rate,
        median(resolution_days) AS median_resolution_days,
        median(date_diff('hour', creation_date, first_response_date)) AS median_first_response_hours,
        avg((status IN ('Open', 'In Process'))::INT) AS backlog_share,
        avg(is_repeat_complainer::INT) AS repeat_complainer_rate,
        avg((assigned_agent_id IS NULL)::INT) AS unassigned_rate,
        avg(resolution_satisfaction) AS resolution_satisfaction_avg
        FROM '{c}'""").df().round(4).iloc[0].to_dict()


def main() -> None:
    con = duckdb.connect()
    build_flow_mart(con)
    card = score_kpis(con)
    REPORTS.mkdir(exist_ok=True)
    card.to_csv(REPORTS / "kpi_scorecard.csv", index=False)

    by_reason = con.execute(f"""SELECT contact_reason, count(*) n,
        avg(was_resolved::INT) fcr, avg(csat_score) csat, avg(was_escalated::INT) redirection,
        avg(coalesce(wait_time_seconds, 0) + duration_seconds) cycle_s,
        avg(CASE WHEN NOT was_resolved THEN (NOT was_escalated)::INT END) missed_escalation
        FROM '{FEATURES}' GROUP BY 1 ORDER BY n DESC""").df().round(4)
    channel = con.execute(f"""SELECT channel, count(*) n, avg(was_resolved::INT) fcr, avg(csat_score) csat,
        avg(duration_seconds) aht_s, avg((duration_seconds IS NULL)::INT) aht_missing
        FROM '{FEATURES}' GROUP BY 1 ORDER BY n DESC""").df().round(4)
    summary = {
        "scorecard": card.to_dict(orient="records"),
        "by_reason": by_reason.to_dict(orient="records"),
        "by_channel": channel.to_dict(orient="records"),
        "disputes": dispute_kpis(con),
    }
    (REPORTS / "kpi_summary.json").write_text(json.dumps(summary, indent=2, default=str, ensure_ascii=False))
    pd.set_option("display.width", 220)
    print(card[["kpi", "stage", "value", "coverage", "sensitivity", "stability", "csat_link_detail", "csat_link_adjusted", "score"]]
          .to_string(index=False))
    print("\nDisputes:", summary["disputes"])


if __name__ == "__main__":
    main()
