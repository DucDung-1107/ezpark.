FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /usr/src/app

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r ./requirements.txt

# copy worker code (from cv-worker/worker.py)
COPY worker.py ./worker.py

# optional: copy any helper modules in cv-worker/
COPY . /usr/src/app

RUN chown -R appuser:appuser /usr/src/app
USER appuser

CMD ["python", "worker.py"]