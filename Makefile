# IRIS - common tasks. On Windows use Git Bash or WSL (or run the commands directly).
PY ?= python
PORT ?= 8000

.PHONY: setup data pipeline demo train eval test lint run docker dagster

setup:            ## install the bot + dev tools (add [llm] for DeepSeek/Claude, [pipeline] for the data lake)
	$(PY) -m pip install -e ".[llm,dev]"

data:             ## download organizer data from S3 (needs .env with read-only AWS keys)
	$(PY) -m pip install -e ".[pipeline]"
	$(PY) src/eris/ingest/download.py

pipeline:         ## bronze -> silver -> gold -> KPI relevance -> scorecard
	$(PY) src/eris/pipeline/bronze.py
	$(PY) src/eris/pipeline/silver.py
	$(PY) src/eris/pipeline/gold.py
	$(PY) src/eris/model/feature_relevance.py
	$(PY) src/eris/analytics/kpis.py

demo:             ## rebuild the committed demo DuckDB from silver
	$(PY) scripts/build_demo_db.py

train:            ## build intent splits and train the classifier
	$(PY) scripts/build_intent_splits.py
	$(PY) scripts/train_intent.py

eval:             ## offline baseline-vs-proposed evaluation (MockLLM)
	$(PY) scripts/run_eval.py

test:
	$(PY) -m pytest

lint:
	ruff check .

run:              ## start the API + web chat on http://localhost:$(PORT)
	$(PY) -m uvicorn iris_bot.api.app:app --app-dir src --host 0.0.0.0 --port $(PORT) --reload

docker:
	docker compose up --build

dagster:          ## asset graph UI (pip install -e ".[dagster]")
	dagster dev -f dags/iris_assets.py
