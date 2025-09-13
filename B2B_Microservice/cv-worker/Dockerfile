# cv-worker/Dockerfile - YOLO (CPU) friendly image
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# system deps for opencv, pillow, ultralytics, ffmpeg (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ffmpeg libgl1 libglib2.0-0 wget ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# create non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /usr/src/app

# copy requirements first (for layer caching)
COPY requirements.txt ./requirements.txt

# Install PyTorch CPU wheel first (use PyTorch index for CPU wheels) then other requirements
# Note: this will download a CPU-compatible torch wheel; if you have a GPU, change this accordingly.
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r ./requirements.txt

# copy worker files
COPY worker_yolo.py ./worker_yolo.py


# set permissions and switch user
RUN chown -R appuser:appuser /usr/src/app
USER appuser

EXPOSE 5001

CMD ["python", "worker_yolo.py"]