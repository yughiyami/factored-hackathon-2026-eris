"""Silver layer: typed, normalized, deduplicated tables that satisfy the data contracts.

Steps per table (all deterministic, rebuilt from bronze on every run):
  1. Clean strings: trim, treat '', 'None', 'nan', 'NULL' as NULL.
  2. Normalize categorical values through the contract value maps.
  3. TRY_CAST to the contract type; failed casts become NULL and are counted.
  4. Soft rules (range / allowed set): offending values become NULL and are counted.
  5. Hard rules (required columns): offending rows go to data/quarantine.
  6. Deduplicate on the primary key, newest record wins.
  7. Flag orphan foreign keys with `_orphan_<column>` instead of dropping rows.
A JSON data-quality report is written to reports/dq_silver.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from eris.quality.contracts import CONTRACTS, Col, Contract  # noqa: E402

BRONZE = ROOT / "data" / "bronze"
SILVER = ROOT / "data" / "silver"
QUARANTINE = ROOT / "data" / "quarantine"
REPORT = ROOT / "reports" / "dq_silver.json"
NULL_TOKENS = ("", "None", "none", "nan", "NaN", "NULL", "null", "N/A")


def q(name: str) -> str:
    return f'"{name}"'


def clean_expr(name: str, col: Col) -> str:
    tokens = ", ".join(f"'{t}'" for t in NULL_TOKENS)
    raw = f"CASE WHEN trim({q(name)}) IN ({tokens}) THEN NULL ELSE trim({q(name)}) END"
    if col.map:
        cases = " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in col.map.items())
        raw = f"(CASE ({raw}) {cases} ELSE ({raw}) END)"
    return raw


def cast_expr(expr: str, col: Col) -> str:
    if col.type == "VARCHAR":
        return expr
    if col.type == "INTEGER":
        return f"CAST(round(TRY_CAST({expr} AS DOUBLE)) AS INTEGER)"
    return f"TRY_CAST({expr} AS {col.type})"


def rule_expr(name: str, col: Col) -> str | None:
    if col.range:
        lo, hi = col.range
        return f"{q(name)} BETWEEN {lo} AND {hi}"
    if col.allowed:
        return f"{q(name)} IN ({', '.join(repr(a) for a in col.allowed)})"
    return None


def build(table: str, contract: Contract, con: duckdb.DuckDBPyConnection) -> dict:
    src = BRONZE / f"{table}.parquet"
    cols = contract.columns
    con.execute(f"CREATE OR REPLACE TEMP VIEW raw AS SELECT * FROM '{src}'")
    available = {r[0] for r in con.execute("DESCRIBE raw").fetchall()}
    missing = [c for c in cols if c not in available]
    cols = {c: v for c, v in cols.items() if c in available}

    con.execute(f"""CREATE OR REPLACE TEMP TABLE cleaned AS SELECT
        {", ".join(f"{clean_expr(c, v)} AS {q(c)}" for c, v in cols.items())},
        _source_file, _ingested_at FROM raw""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE typed AS SELECT
        {", ".join(f"{cast_expr(q(c), v)} AS {q(c)}" for c, v in cols.items())},
        _source_file, _ingested_at FROM cleaned""")

    stats: dict = {"rows_bronze": con.execute("SELECT count(*) FROM typed").fetchone()[0],
                   "missing_columns": missing, "cast_failures": {}, "rule_violations": {}, "null_rate": {}}
    for c, v in cols.items():
        if v.type != "VARCHAR":
            n = con.execute(f"""SELECT count(*) FROM cleaned a POSITIONAL JOIN typed b
                WHERE a.{q(c)} IS NOT NULL AND b.{q(c)} IS NULL""").fetchone()[0]
            if n:
                stats["cast_failures"][c] = n

    soft = {c: r for c, v in cols.items() if (r := rule_expr(c, v))}
    for c, rule in soft.items():
        n = con.execute(f"SELECT count(*) FROM typed WHERE {q(c)} IS NOT NULL AND NOT ({rule})").fetchone()[0]
        if n:
            stats["rule_violations"][c] = n
    select_soft = ", ".join(
        f"CASE WHEN {q(c)} IS NOT NULL AND NOT ({soft[c]}) THEN NULL ELSE {q(c)} END AS {q(c)}"
        if c in soft else q(c) for c in cols)
    con.execute(f"CREATE OR REPLACE TEMP TABLE valid AS SELECT {select_soft}, _source_file, _ingested_at FROM typed")

    required = [c for c, v in cols.items() if v.required]
    bad = " OR ".join(f"{q(c)} IS NULL" for c in required) or "FALSE"
    QUARANTINE.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY (SELECT * FROM valid WHERE {bad}) TO '{QUARANTINE / f'{table}.parquet'}' (FORMAT parquet)")
    stats["rows_quarantined"] = con.execute(f"SELECT count(*) FROM valid WHERE {bad}").fetchone()[0]

    pk = ", ".join(q(c) for c in contract.pk)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE dedup AS SELECT * EXCLUDE (_rn) FROM (
        SELECT *, row_number() OVER (PARTITION BY {pk} ORDER BY {q(contract.order_by)} DESC, _source_file DESC) _rn
        FROM valid WHERE NOT ({bad})) WHERE _rn = 1""")
    stats["rows_duplicate_pk"] = stats["rows_bronze"] - stats["rows_quarantined"] - \
        con.execute("SELECT count(*) FROM dedup").fetchone()[0]

    orphan_cols = []
    for c, ref in contract.fks.items():
        if c not in cols:
            continue
        ref_table, ref_col = ref.split(".")
        parent = SILVER / f"{ref_table}.parquet"
        if not parent.exists():
            continue
        orphan_cols.append(f"""({q(c)} IS NOT NULL AND {q(c)} NOT IN
            (SELECT {q(ref_col)} FROM '{parent}')) AS _orphan_{c}""")
    extra = (", " + ", ".join(orphan_cols)) if orphan_cols else ""
    out = SILVER / f"{table}.parquet"
    SILVER.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY (SELECT *{extra} FROM dedup) TO '{out}' (FORMAT parquet, COMPRESSION zstd)")

    stats["rows_silver"] = con.execute(f"SELECT count(*) FROM '{out}'").fetchone()[0]
    stats["orphan_fk"] = {
        c: con.execute(f"SELECT count(*) FILTER (_orphan_{c}) FROM '{out}'").fetchone()[0]
        for c in contract.fks if c in cols and (SILVER / f"{contract.fks[c].split('.')[0]}.parquet").exists()
    }
    for c in cols:
        stats["null_rate"][c] = round(con.execute(
            f"SELECT avg(({q(c)} IS NULL)::INT) FROM '{out}'").fetchone()[0] or 0, 4)
    return stats


def main(tables: list[str]) -> None:
    con = duckdb.connect()
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    for table, contract in CONTRACTS.items():
        if tables and table not in tables:
            continue
        s = build(table, contract, con)
        report[table] = s
        print(f"{table:26s} bronze={s['rows_bronze']:>9,} quarantined={s['rows_quarantined']:>6,} "
              f"dup_pk={s['rows_duplicate_pk']:>6,} silver={s['rows_silver']:>9,} "
              f"cast_fail={sum(s['cast_failures'].values()):>6,} rule_viol={sum(s['rule_violations'].values()):>6,} "
              f"orphans={s['orphan_fk']}", flush=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
