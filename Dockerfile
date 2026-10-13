# DocuMind service image.
# Models (embedding + cross-encoder) download on first run and cache into the
# HF cache volume mounted by docker-compose, so rebuilds stay fast.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models

WORKDIR /srv

# System deps kept minimal; torch wheels are self-contained.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
