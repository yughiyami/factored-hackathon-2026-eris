"""Dagster software-defined assets for the IRIS data pipeline and bot artifacts.

Wraps the existing, idempotent scripts so the asset graph gives lineage, run history and partial
re-materialization without rewriting the pipeline:

  raw_files -> bronze -> silver -> gold -> feature_relevance
                                \\-> kpis (needs gold)
                         silver -> demo_db (bot demo dataset + eval fixtures)
  intent_splits -> intent_model

Install the extra and launch the UI:  pip install -e ".[dagster]"  &&  dagster dev -f dags/iris_assets.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from dagster import AssetExecutionContext, Definitions, MaterializeResult, MetadataValue, asset

ROOT = Path(__file__).resolve().parents[1]


def _run(context: AssetExecutionContext, script: str) -> MaterializeResult:
    context.log.info(f"running {script}")
    proc = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    tail = "\n".join(proc.stdout.splitlines()[-30:])
    if proc.returncode != 0:
        raise RuntimeError(f"{script} failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    return MaterializeResult(metadata={"script": script, "stdout_tail": MetadataValue.md(f"```\n{tail}\n```")})


@asset(group_name="lake", description="Incremental S3 download of organizer CSVs into data/raw")
def raw_files(context: AssetExecutionContext) -> MaterializeResult:
    return _run(context, "src/eris/ingest/download.py")


@asset(group_name="lake", deps=[raw_files], description="Raw CSV -> bronze parquet with lineage columns")
def bronze(context: AssetExecutionContext) -> MaterializeResult:
    return _run(context, "src/eris/pipeline/bronze.py")


@asset(group_name="lake", deps=[bronze], description="Contracts, quarantine, dedup -> silver + DQ report")
def silver(context: AssetExecutionContext) -> MaterializeResult:
    return _run(context, "src/eris/pipeline/silver.py")


@asset(group_name="lake", deps=[silver], description="Point-in-time features and KPI marts")
def gold(context: AssetExecutionContext) -> MaterializeResult:
    return _run(context, "src/eris/pipeline/gold.py")


@asset(group_name="analytics", deps=[gold], description="KPI variable relevance (LightGBM vs baselines)")
def feature_relevance(context: AssetExecutionContext) -> MaterializeResult:
    return _run(context, "src/eris/model/feature_relevance.py")


@asset(group_name="analytics", deps=[gold], description="IRIS KPI scorecard")
def kpis(context: AssetExecutionContext) -> MaterializeResult:
    return _run(context, "src/eris/analytics/kpis.py")


@asset(group_name="bot", deps=[silver], description="Small committed demo DuckDB + eval fixtures")
def demo_db(context: AssetExecutionContext) -> MaterializeResult:
    return _run(context, "scripts/build_demo_db.py")


@asset(group_name="bot", description="Held-out-by-family train/test splits of labeled utterances")
def intent_splits(context: AssetExecutionContext) -> MaterializeResult:
    return _run(context, "scripts/build_intent_splits.py")


@asset(group_name="bot", deps=[intent_splits], description="TF-IDF + logistic regression intent model")
def intent_model(context: AssetExecutionContext) -> MaterializeResult:
    return _run(context, "scripts/train_intent.py")


@asset(group_name="bot", deps=[demo_db, intent_model], description="Offline baseline-vs-proposed evaluation")
def bot_eval(context: AssetExecutionContext) -> MaterializeResult:
    return _run(context, "scripts/run_eval.py")


defs = Definitions(assets=[raw_files, bronze, silver, gold, feature_relevance, kpis, demo_db, intent_splits,
                           intent_model, bot_eval])
