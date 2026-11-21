# ---------- slim, hardened serving image (ONNX Runtime only - no PyTorch) ----------
# Build AFTER `python -m fairml.pipeline` so artifacts/model/ exists.
# Use the same Python minor version you trained with:  docker build --build-arg PYTHON_VERSION=3.12 .
ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_DIR=/app/artifacts/model \
    PORT=8000 \
    USE_QUANTIZED=0 \
    ORT_THREADS=1

WORKDIR /app

# Latest library versions, pinned only where the pickled preprocessor requires it
COPY requirements-serve.txt /tmp/requirements-serve.txt
COPY artifacts/model/constraints.txt /tmp/constraints.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements-serve.txt -c /tmp/constraints.txt \
    && rm -f /tmp/requirements-serve.txt /tmp/constraints.txt

COPY fairml/ fairml/
COPY app/ app/
COPY artifacts/model/ artifacts/model/

# Non-root user. OpenShift runs containers with a random UID in group 0 -> give group 0 the owner's permissions.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin appuser \
    && chgrp -R 0 /app && chmod -R g=u /app
USER 10001

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
