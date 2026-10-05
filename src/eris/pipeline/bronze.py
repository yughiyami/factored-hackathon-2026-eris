"""Bronze layer: land raw CSVs as Parquet, untyped (all VARCHAR), plus lineage columns.

No business logic here: every value is kept as delivered so silver can be rebuilt
deterministically. `_source_file` and `_ingested_at` give row-level lineage.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
BRONZE = ROOT / "data" / "bronze"

TABLES = [
    "customers", "products", "branches", "service_agents", "marketing_campaigns",
    "daily_exchange_rates", "call_center_interactions", "call_transcripts",
    "satisfaction_surveys", "complaints", "transactions", "campaign_sends", "digital_events",
]


def source_glob(table: str) -> str:
    flat = RAW / f"{table}.csv"
    return str(flat) if flat.exists() else str(RAW / table / "**" / "*.csv")


def build(table: str, con: duckdb.DuckDBPyConnection) -> None:
    t0 = time.time()
    out = BRONZE / f"{table}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"""
        COPY (
            SELECT *, filename AS _source_file, current_timestamp AS _ingested_at
            FROM read_csv('{source_glob(table)}', all_varchar=true, union_by_name=true,
                          filename=true, header=true, encoding='utf-8')
        ) TO '{out}' (FORMAT parquet, COMPRESSION zstd)
    """)
    n = con.execute(f"SELECT count(*) FROM '{out}'").fetchone()[0]
    print(f"{table:28s} rows={n:>10,}  {time.time() - t0:5.1f}s", flush=True)


def main(tables: list[str]) -> None:
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    for t in tables or TABLES:
        build(t, con)


if __name__ == "__main__":
    main(sys.argv[1:])
