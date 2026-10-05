"""Bronze ingestion: mirror the organizer S3 bucket into data/raw (idempotent, incremental).

Only objects whose size differs from the local copy are downloaded, so re-running the
script picks up late-arriving partitions without re-downloading everything.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
MAX_RETRIES = 4


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main(tables: list[str]) -> None:
    load_env()
    bucket = os.environ["DATATHON_BUCKET"]
    s3 = boto3.client("s3")
    todo = []
    for prefix in tables or [""]:
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"data/{prefix}"):
            for o in page.get("Contents", []):
                dest = RAW / o["Key"].removeprefix("data/")
                if not dest.exists() or dest.stat().st_size != o["Size"]:
                    todo.append((o["Key"], dest))
    print(f"{len(todo)} objects to download", flush=True)

    def fetch(key: str, dest: Path) -> str:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        for attempt in range(MAX_RETRIES):
            try:
                s3.download_file(bucket, key, str(tmp))
                tmp.replace(dest)
                return key
            except Exception as exc:  # bounded retry with backoff
                if attempt == MAX_RETRIES - 1:
                    print(f"FAILED {key}: {exc}", flush=True)
                    return key
                time.sleep(2**attempt)
        return key

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(fetch, k, d) for k, d in todo]
        for i, f in enumerate(as_completed(futures), 1):
            f.result()
            if i % 500 == 0:
                print(f"{i}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
