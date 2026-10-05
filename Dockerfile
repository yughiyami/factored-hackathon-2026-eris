FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app
RUN useradd --create-home --uid 10001 iris

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install -e ".[llm]"

COPY config ./config
COPY demo ./demo
COPY models ./models

RUN mkdir -p /app/runtime && chown -R iris:iris /app/runtime
USER iris

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT','8000'))"
CMD ["sh", "-c", "uvicorn iris_bot.api.app:app --host 0.0.0.0 --port ${PORT}"]
