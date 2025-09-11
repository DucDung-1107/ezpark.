FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential gcc libpq-dev curl \
 && rm -rf /var/lib/apt/lists/*

# create non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /usr/src/app

# copy requirements (from backend/requirements.txt)
COPY requirements.txt ./requirements.txt

RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r ./requirements.txt

# copy app source from build context (./backend/app)
COPY app ./app

RUN chown -R appuser:appuser /usr/src/app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]