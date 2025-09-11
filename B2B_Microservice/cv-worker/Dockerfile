# cv-worker/Dockerfile (sim mode)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# create non-root user
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /usr/src/app

# install minimal deps
COPY cv-worker/requirements.txt ./requirements.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r ./requirements.txt

# copy worker
COPY cv-worker/worker.py ./worker.py
COPY spots.json /spots.json

RUN chown -R appuser:appuser /usr/src/app /spots.json
USER appuser

CMD ["python", "worker.py"]