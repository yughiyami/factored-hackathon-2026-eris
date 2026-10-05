"""Read-only data access over DuckDB: the committed demo DB or the full silver parquet lake.

The repository is deliberately unscoped (it is the "core banking" system of record). Customer
scoping and permission checks live in the tool layer (iris_bot.tools), which is the only caller.
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import duckdb

TXN_COLS = ("transaction_id, transaction_date, product_id, customer_id, transaction_type, amount, currency, "
            "amount_usd, channel, merchant_name, merchant_category, transaction_status, is_fraud, fraud_score")


class Repository(Protocol):
    as_of: datetime

    def customer(self, customer_id: str) -> dict[str, Any] | None: ...
    def products(self, customer_id: str) -> list[dict[str, Any]]: ...
    def transactions(self, customer_id: str, since_days: int | None = None,
                     limit: int = 50) -> list[dict[str, Any]]: ...
    def transaction(self, transaction_id: str) -> dict[str, Any] | None: ...
    def complaints(self, customer_id: str, since_days: int | None = None) -> list[dict[str, Any]]: ...


class DuckDBRepository:
    def __init__(self, *, demo_db: Path | None = None, silver_dir: Path | None = None,
                 as_of: str | None = None):
        self._lock = threading.Lock()
        if demo_db is not None:
            self._con = duckdb.connect(str(demo_db), read_only=True)
            self.source = f"demo:{demo_db.name}"
            default_as_of = self._con.execute("SELECT as_of_date FROM meta").fetchone()[0]
        elif silver_dir is not None:
            self._con = duckdb.connect()
            p = silver_dir.as_posix()
            self._con.execute(f"CREATE VIEW customers AS SELECT * FROM '{p}/customers.parquet'")
            self._con.execute(f"CREATE VIEW products AS SELECT * FROM '{p}/products.parquet'")
            self._con.execute(f"CREATE VIEW complaints AS SELECT * FROM '{p}/complaints.parquet'")
            self._con.execute(f"""CREATE VIEW transactions AS SELECT * REPLACE (
                coalesce(amount_usd, CASE WHEN currency = 'USD' THEN amount END) AS amount_usd)
                FROM '{p}/transactions.parquet'""")
            self.source = "silver"
            default_as_of = self._con.execute("SELECT max(transaction_date) FROM transactions").fetchone()[0]
        else:
            raise ValueError("demo_db or silver_dir is required")
        self.as_of: datetime = datetime.fromisoformat(as_of) if as_of else default_as_of

    def _rows(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._con.cursor()
            try:
                res = cur.execute(sql, params)
                cols = [d[0] for d in res.description]
                return [dict(zip(cols, r)) for r in res.fetchall()]
            finally:
                cur.close()

    def customer(self, customer_id: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT customer_id, country, city, segment, customer_status FROM customers "
                          "WHERE customer_id = ?", [customer_id])
        return rows[0] if rows else None

    def products(self, customer_id: str) -> list[dict[str, Any]]:
        return self._rows("SELECT product_id, product_type, currency, current_balance, credit_limit, "
                          "product_status FROM products WHERE customer_id = ? ORDER BY product_type",
                          [customer_id])

    def transactions(self, customer_id: str, since_days: int | None = None,
                     limit: int = 50) -> list[dict[str, Any]]:
        since = self.as_of - timedelta(days=since_days) if since_days else datetime(1900, 1, 1)
        return self._rows(f"SELECT {TXN_COLS} FROM transactions WHERE customer_id = ? "
                          "AND transaction_date >= ? AND transaction_date <= ? "
                          "ORDER BY transaction_date DESC LIMIT ?",
                          [customer_id, since, self.as_of, limit])

    def transaction(self, transaction_id: str) -> dict[str, Any] | None:
        rows = self._rows(f"SELECT {TXN_COLS} FROM transactions WHERE transaction_id = ?", [transaction_id])
        return rows[0] if rows else None

    def complaints(self, customer_id: str, since_days: int | None = None) -> list[dict[str, Any]]:
        since = self.as_of - timedelta(days=since_days) if since_days else datetime(1900, 1, 1)
        return self._rows("SELECT complaint_id, creation_date, subcategory, status, claimed_amount, currency "
                          "FROM complaints WHERE customer_id = ? AND creation_date >= ? "
                          "AND creation_date <= ? ORDER BY creation_date DESC",
                          [customer_id, since, self.as_of])


def build_repository(settings) -> DuckDBRepository:
    if settings.data_backend == "silver":
        return DuckDBRepository(silver_dir=settings.resolve(settings.silver_dir), as_of=settings.as_of_date)
    return DuckDBRepository(demo_db=settings.resolve(settings.demo_db), as_of=settings.as_of_date)
