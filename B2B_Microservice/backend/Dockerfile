# backend/Dockerfile
### Multi-stage (single here for clarity). Use python:3.11-slim as base.
FROM python:3.11-slim

# Metadata
LABEL maintainer="you@example.com" \
      org.opencontainers.image.source="local/parkwave-demo"

ENV PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system deps (minimal). libpq-dev only if you build psycopg wheels (psycopg2-binary usually wheel)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      libpq-dev \
      curl \
 && rm -rf /var/lib/apt/lists/*

# Create non-root user
ARG APP_USER=appuser
RUN useradd --create-home --shell /bin/bash ${APP_USER}

WORKDIR /usr/src/app

# Copy requirements first to take advantage of Docker layer caching
COPY backend/requirements.txt ./requirements.txt
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r ./requirements.txt

# Copy source code into package path so 'app' module exists
# Expect your python package structure: backend/app/...
COPY backend/app ./app

# Ensure proper ownership
RUN chown -R ${APP_USER}:${APP_USER} /usr/src/app

USER ${APP_USER}

EXPOSE 8000

# Default command — dev-friendly (reload). For production remove --reload and use gunicorn/uvicorn workers
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]