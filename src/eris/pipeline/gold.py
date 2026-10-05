"""Gold layer: KPI marts and the interaction-level feature table used for modeling.

Leakage policy: every customer-history feature is computed strictly from events that
happened BEFORE the interaction timestamp (window frames end at 1 PRECEDING / range
strictly lower). Agent aggregates such as service_agents.avg_csat are excluded because
they summarize the whole period, including the future.
"""
from __future__ import annotations

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[3]
SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold"


def s(table: str) -> str:
    return f"'{SILVER / (table + '.parquet')}'"


def main() -> None:
    GOLD.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    # Customer product portfolio (snapshot dimension; opening_date used to keep it point-in-time).
    con.execute(f"""CREATE TEMP TABLE prod AS SELECT customer_id, product_type, opening_date,
        coalesce(days_past_due, 0) AS dpd, product_status FROM {s('products')}""")

    con.execute(f"""CREATE TEMP TABLE inter AS
        SELECT i.*, epoch(interaction_date) AS ts FROM {s('call_center_interactions')} i""")

    con.execute(f"""CREATE TEMP TABLE complaints_ts AS
        SELECT customer_id, epoch(creation_date) AS ts FROM {s('complaints')}""")

    features = f"""
    WITH hist AS (
        SELECT interaction_id,
            count(*) OVER w30  AS prior_contacts_30d,
            count(*) OVER w90  AS prior_contacts_90d,
            sum(was_escalated::INT) OVER w90 AS prior_escalations_90d,
            sum((NOT coalesce(was_resolved, FALSE))::INT) OVER w90 AS prior_unresolved_90d,
            (ts - lag(ts) OVER (PARTITION BY customer_id ORDER BY ts)) / 86400.0 AS days_since_last_contact,
            lag(contact_reason) OVER (PARTITION BY customer_id ORDER BY ts) = contact_reason AS same_reason_as_last
        FROM inter
        WINDOW
            w30 AS (PARTITION BY customer_id ORDER BY ts RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING),
            w90 AS (PARTITION BY customer_id ORDER BY ts RANGE BETWEEN 7776000 PRECEDING AND 1 PRECEDING)
    ),
    comp AS (
        SELECT i.interaction_id, count(c.customer_id) AS prior_complaints_90d
        FROM inter i LEFT JOIN complaints_ts c
          ON c.customer_id = i.customer_id AND c.ts < i.ts AND c.ts >= i.ts - 7776000
        GROUP BY 1
    ),
    portfolio AS (
        SELECT i.interaction_id,
            count(p.customer_id) AS n_products,
            count(*) FILTER (p.product_type = 'credit_card') AS n_credit_cards,
            count(*) FILTER (p.product_type IN ('personal_loan', 'mortgage')) AS n_loans,
            max(p.dpd) AS max_days_past_due,
            count(*) FILTER (p.product_status IN ('Blocked', 'Suspended')) AS n_blocked_products
        FROM inter i LEFT JOIN prod p
          ON p.customer_id = i.customer_id AND p.opening_date <= i.interaction_date::DATE
        GROUP BY 1
    ),
    surveys AS (
        SELECT interaction_id,
            max(main_score) FILTER (survey_type = 'CSAT') AS csat_score,
            max(main_score) FILTER (survey_type = 'NPS')  AS nps_score,
            max(main_score) FILTER (survey_type = 'CES')  AS ces_score
        FROM {s('satisfaction_surveys')} GROUP BY 1
    )
    SELECT
        i.interaction_id, i.interaction_date, i.customer_id, i.agent_id,
        -- contact context (known when the contact starts)
        i.interaction_type, i.channel, i.contact_reason,
        hour(i.interaction_date) AS hour_of_day, dayofweek(i.interaction_date) AS day_of_week,
        i.wait_time_seconds,
        -- customer profile
        c.country, c.segment, c.customer_status, c.document_type, c.gender,
        coalesce(c.detected_accent, 'unknown') AS customer_accent,
        c.credit_score, c.estimated_monthly_income,
        date_diff('year', c.date_of_birth, i.interaction_date::DATE) AS customer_age,
        date_diff('day', c.registration_date::DATE, i.interaction_date::DATE) AS tenure_days,
        c.accepts_marketing,
        -- agent profile (static attributes only, no outcome aggregates)
        a.agent_type, a.experience_level, coalesce(a.specialty, 'none') AS agent_specialty,
        a.work_shift, a.native_accent AS agent_accent,
        (coalesce(i.customer_detected_accent, 'x') = coalesce(i.agent_used_accent, 'y')) AS accent_match,
        date_diff('day', a.hire_date, i.interaction_date::DATE) AS agent_tenure_days,
        -- history (strictly before the contact)
        h.prior_contacts_30d, h.prior_contacts_90d, h.prior_escalations_90d, h.prior_unresolved_90d,
        h.days_since_last_contact, h.same_reason_as_last, cp.prior_complaints_90d,
        pf.n_products, pf.n_credit_cards, pf.n_loans, pf.max_days_past_due, pf.n_blocked_products,
        -- in-contact signals (only known after/while the contact happens)
        i.duration_seconds, i.sentiment_score, i.detected_sentiment, i.requires_followup,
        -- outcomes / KPIs
        i.was_resolved, i.was_escalated, sv.csat_score, sv.nps_score, sv.ces_score
    FROM inter i
    JOIN hist h USING (interaction_id)
    JOIN comp cp USING (interaction_id)
    JOIN portfolio pf USING (interaction_id)
    LEFT JOIN surveys sv USING (interaction_id)
    LEFT JOIN {s('customers')} c USING (customer_id)
    LEFT JOIN {s('service_agents')} a ON a.agent_id = i.agent_id
    """
    con.execute(f"COPY ({features}) TO '{GOLD / 'interaction_features.parquet'}' (FORMAT parquet)")

    # KPI mart for IRIS: CSAT, cycle time, interaction frequency, redirection rate.
    con.execute(f"""COPY (
        SELECT date_trunc('month', interaction_date) AS month, country, channel, contact_reason,
            count(*) AS interactions,
            count(DISTINCT customer_id) AS customers,
            count(*) / count(DISTINCT customer_id) AS interactions_per_customer,
            avg(csat_score) AS csat_avg,
            avg((csat_score >= 3)::INT) AS csat_top2_rate,
            -- unknown duration is missing data, not a zero-length contact
            avg(coalesce(wait_time_seconds, 0) + duration_seconds) AS cycle_time_seconds,
            quantile_cont(coalesce(wait_time_seconds, 0) + duration_seconds, 0.9) AS cycle_time_p90,
            avg(was_escalated::INT) AS redirection_rate,
            avg(was_resolved::INT) AS fcr_rate,
            avg((days_since_last_contact <= 7)::INT) AS repeat_contact_7d_rate
        FROM '{GOLD / 'interaction_features.parquet'}' GROUP BY ALL
    ) TO '{GOLD / 'kpi_monthly.parquet'}' (FORMAT parquet)""")

    for t in ("interaction_features", "kpi_monthly"):
        n = con.execute(f"SELECT count(*) FROM '{GOLD / (t + '.parquet')}'").fetchone()[0]
        print(f"{t:24s} rows={n:,}")


if __name__ == "__main__":
    main()
