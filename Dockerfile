# Self-contained image for the RUL API: the build downloads C-MAPSS (hash-checked), trains the
# four models, and the final stage serves them. Build: `docker build -t rul-api .`
# Run: `docker run --rm -p 8000:8000 rul-api`, then open http://127.0.0.1:8000/docs
FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1
WORKDIR /app
# dependencies first, so this layer is cached while pyproject.toml and uv.lock do not change
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

FROM base AS train
RUN .venv/bin/rul download && .venv/bin/rul train

FROM python:3.12-slim
RUN useradd --create-home --uid 1000 user
WORKDIR /app
COPY --from=base --chown=user /app/.venv /app/.venv
COPY --from=train --chown=user /app/models /app/models
USER user
ENV PYTHONUNBUFFERED=1 \
    RUL_MODELS_DIR=/app/models
EXPOSE 8000
# exec so uvicorn is PID 1 and receives the stop signal
CMD exec /app/.venv/bin/uvicorn rul.api:app --host 0.0.0.0 --port "${PORT:-8000}"
# the slim image has no curl: use Python
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD /app/.venv/bin/python -c "import os, urllib.request as u; u.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=4)"
