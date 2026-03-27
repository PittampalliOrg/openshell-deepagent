FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install --no-cache-dir .

FROM base AS durable-agent
ENV APP_PORT=8000
EXPOSE 8000
CMD ["python", "-m", "src.dapr_durable_agent"]

FROM base AS langgraph-dapr
ENV APP_PORT=8001
EXPOSE 8001
CMD ["python", "-m", "src.langgraph_dapr_app"]

FROM base AS deepagents-test
ENV APP_PORT=8002
EXPOSE 8002
CMD ["python", "-m", "src.deepagents_test_app"]
