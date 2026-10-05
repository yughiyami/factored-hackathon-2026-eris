"""Build the small, committed demo database used by the IRIS bot when the 5 GB lake is absent.

Reads data/silver/*.parquet (synthetic hackathon data) and writes:
  demo/iris_demo.duckdb   ~500 customers + their products, last 180 days of transactions, complaints
  demo/fixtures.json      named customer/transaction roles used by tests and the eval harness

"As-of" date: the dataset ends on 2026-06-18, so all relative windows (transaction age, repeat
contact) are computed against the latest transaction timestamp in silver, stored in table `meta`.

Selection is deterministic (md5 ordering), so re-running produces the same database.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
SILVER = ROOT / "data" / "silver"
OUT_DB = ROOT / "demo" / "iris_demo.duckdb"
OUT_FIX = ROOT / "demo" / "fixtures.json"
N_CUSTOMERS = 500
WINDOW_DAYS = 180
DISPUTE_SUBCATS = ("Cargo no reconocido", "Cobro indebido")


def s(table: str) -> str:
    return f"'{(SILVER / (table + '.parquet')).as_posix()}'"


def main() -> None:
    if not (SILVER / "transactions.parquet").exists():
        sys.exit("data/silver not found: run the pipeline first (make pipeline)")
    con = duckdb.connect()
    as_of = con.sql(f"SELECT max(transaction_date) FROM {s('transactions')}").fetchone()[0]
    print("as_of:", as_of)
    con.execute(f"""CREATE TEMP TABLE tx AS
        SELECT transaction_id, transaction_date, product_id, customer_id, transaction_type,
               transaction_category, amount, currency,
               coalesce(amount_usd, CASE WHEN currency = 'USD' THEN amount END) AS amount_usd,
               channel, merchant_name, merchant_category, transaction_country, transaction_city,
               transaction_status, response_code, is_fraud, fraud_score,
               date_diff('day', transaction_date, TIMESTAMP '{as_of}') AS age_days
        FROM {s('transactions')}
        WHERE transaction_date > TIMESTAMP '{as_of}' - INTERVAL {WINDOW_DAYS} DAY
          AND NOT coalesce(_orphan_customer_id, FALSE)""")
    con.execute(f"""CREATE TEMP TABLE cust AS SELECT * FROM {s('customers')}
                    WHERE customer_status = 'Active'""")
    con.execute(f"""CREATE TEMP TABLE open_disputes AS
        SELECT DISTINCT customer_id FROM {s('complaints')}
        WHERE subcategory IN {DISPUTE_SUBCATS} AND status IN ('Open', 'In Process', 'Escalated')
          AND creation_date > TIMESTAMP '{as_of}' - INTERVAL 30 DAY""")
    # Per-customer uniqueness of merchant and amount, so a scripted user can name one transaction.
    con.execute("""CREATE TEMP TABLE txu AS SELECT t.*,
            count(*) OVER (PARTITION BY customer_id, merchant_name) AS merchant_n,
            count(*) OVER (PARTITION BY customer_id, round(amount, 2)) AS amount_n
        FROM tx t""")

    def pick(role: str, where: str, exclude: set[str], extra_customer: str = "TRUE") -> dict:
        row = con.sql(f"""
            SELECT t.customer_id, t.transaction_id, t.transaction_date, t.amount, t.currency,
                   t.amount_usd, t.merchant_name, t.transaction_type, t.transaction_status, t.age_days
            FROM txu t JOIN cust c USING (customer_id)
            WHERE {where} AND {extra_customer}
              AND t.customer_id NOT IN (SELECT unnest(?::VARCHAR[]))
            ORDER BY md5(t.transaction_id) LIMIT 1""", params=[sorted(exclude) or [""]]).fetchone()
        if row is None:
            sys.exit(f"no candidate for role {role}")
        keys = ["customer_id", "transaction_id", "transaction_date", "amount", "currency",
                "amount_usd", "merchant_name", "transaction_type", "transaction_status", "age_days"]
        rec = dict(zip(keys, row))
        rec["transaction_date"] = rec["transaction_date"].isoformat()
        rec["amount"] = round(float(rec["amount"]), 2)
        rec["amount_usd"] = None if rec["amount_usd"] is None else round(float(rec["amount_usd"]), 2)
        exclude.add(rec["customer_id"])
        print(f"  {role:10s} {rec['customer_id']} {rec['transaction_id']} {rec['amount']} "
              f"{rec['currency']} {rec['merchant_name']} age={rec['age_days']}")
        return rec

    clean = ("t.customer_id NOT IN (SELECT customer_id FROM open_disputes) "
             "AND t.customer_id NOT IN (SELECT customer_id FROM tx WHERE is_fraud)")
    purchase_ok = ("t.transaction_type = 'Purchase' AND t.transaction_status = 'Approved' "
                   "AND NOT t.is_fraud AND t.fraud_score < 30 AND t.merchant_name IS NOT NULL "
                   "AND t.merchant_n = 1 AND t.amount_n = 1 AND t.amount_usd IS NOT NULL")
    used: set[str] = set()
    roles = {
        "happy_a": pick("happy_a", f"{purchase_ok} AND t.amount_usd < 300 AND t.age_days <= 60 "
                        f"AND t.currency = 'USD'", used, clean),
        "happy_b": pick("happy_b", f"{purchase_ok} AND t.amount_usd < 300 AND t.age_days <= 60 "
                        f"AND t.currency = 'USD'", used, clean),
        "happy_c": pick("happy_c", f"{purchase_ok} AND t.amount_usd < 300 AND t.age_days <= 60",
                        used, clean),
        "happy_d": pick("happy_d", f"{purchase_ok} AND t.amount_usd < 300 AND t.age_days <= 60",
                        used, clean),
        "fee": pick("fee", "t.transaction_type = 'Adjustment' AND t.transaction_status = 'Approved' "
                    "AND NOT t.is_fraud AND t.amount_usd < 300 AND t.age_days <= 60 AND t.amount_n = 1",
                    used, clean),
        "fraud": pick("fraud", "t.transaction_type = 'Purchase' AND t.is_fraud "
                      "AND t.transaction_status = 'Approved' AND t.amount_usd < 300 "
                      "AND t.merchant_name IS NOT NULL AND t.merchant_n = 1 AND t.age_days <= 90",
                      used, "t.customer_id NOT IN (SELECT customer_id FROM open_disputes)"),
        "high": pick("high", f"{purchase_ok} AND t.amount_usd > 420 AND t.age_days <= 60", used, clean),
        "old": pick("old", f"{purchase_ok} AND t.amount_usd < 300 AND t.age_days BETWEEN 130 AND 170",
                    used, clean),
        "declined": pick("declined", "t.transaction_type = 'Purchase' AND t.transaction_status = 'Declined' "
                         "AND t.merchant_name IS NOT NULL AND t.merchant_n = 1 AND t.amount_n = 1 "
                         "AND t.amount_usd < 300 AND t.age_days <= 60", used, clean),
        "repeat": pick("repeat", f"{purchase_ok} AND t.amount_usd < 300 AND t.age_days <= 60", used,
                       "t.customer_id IN (SELECT customer_id FROM open_disputes)"),
    }

    fixed = sorted(used)
    con.execute("""CREATE TEMP TABLE chosen AS
        SELECT customer_id FROM cust
        WHERE customer_id IN (SELECT DISTINCT customer_id FROM tx WHERE transaction_type = 'Purchase')
          AND customer_id NOT IN (SELECT unnest(?::VARCHAR[]))
        ORDER BY md5(customer_id) LIMIT ?""", [fixed, N_CUSTOMERS - len(fixed)])
    con.execute("INSERT INTO chosen SELECT unnest(?::VARCHAR[])", [fixed])

    OUT_DB.parent.mkdir(parents=True, exist_ok=True)
    if OUT_DB.exists():
        OUT_DB.unlink()
    con.execute(f"ATTACH '{OUT_DB.as_posix()}' AS demo")
    con.execute("""CREATE TABLE demo.customers AS
        SELECT customer_id, country, city, segment, customer_status, registration_date
        FROM cust WHERE customer_id IN (SELECT customer_id FROM chosen)""")
    con.execute(f"""CREATE TABLE demo.products AS
        SELECT product_id, customer_id, product_type, currency, current_balance, credit_limit,
               product_status, opening_date
        FROM {s('products')} WHERE customer_id IN (SELECT customer_id FROM chosen)""")
    con.execute("""CREATE TABLE demo.transactions AS
        SELECT * EXCLUDE (age_days) FROM tx WHERE customer_id IN (SELECT customer_id FROM chosen)
        ORDER BY customer_id, transaction_date""")
    con.execute(f"""CREATE TABLE demo.complaints AS
        SELECT complaint_id, creation_date, customer_id, case_type, category, subcategory,
               affected_product_id, claimed_amount, currency, priority, status, resolution_date
        FROM {s('complaints')} WHERE customer_id IN (SELECT customer_id FROM chosen)""")
    con.execute(f"""CREATE TABLE demo.meta AS SELECT TIMESTAMP '{as_of}' AS as_of_date,
        now() AS built_at, {WINDOW_DAYS} AS window_days, 'data/silver (synthetic)' AS source""")
    for t in ("customers", "products", "transactions", "complaints"):
        print(f"  demo.{t}: {con.sql(f'SELECT count(*) FROM demo.{t}').fetchone()[0]:,} rows")
    con.execute("DETACH demo")
    con.close()

    OUT_FIX.write_text(json.dumps({"as_of_date": str(as_of), "roles": roles}, indent=2,
                                  ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT_DB} ({OUT_DB.stat().st_size / 1e6:.2f} MB) and {OUT_FIX}")


if __name__ == "__main__":
    main()
