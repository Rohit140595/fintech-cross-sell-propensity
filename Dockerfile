# Multi-stage build — keeps the final image lean.
#
# Usage:
#   docker build -t lending-club-ltv .
#   docker run -p 8000:8000 lending-club-ltv

FROM python:3.9-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

FROM python:3.9-slim AS runtime

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY src/     src/
COPY config.yaml .

ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
