# syntax=docker/dockerfile:1

FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Default port 8000 for local, Cloud Run will override with $PORT
ENV PORT=8000

WORKDIR /app

# Install system dependencies (including docling requirements)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    # Docling PDF/image processing dependencies
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (better layer caching)
COPY pyproject.toml ./

# Create virtual environment
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source for editable install
COPY src/ ./src/

# Install dependencies
RUN uv pip install -e .

# Copy remaining application code
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY scripts/ ./scripts/

# Create non-root user for security
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check for local development (Cloud Run ignores this)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start command - uses PORT env var (8000 local, Cloud Run overrides)
CMD uvicorn medical_agent.api.main:app --host 0.0.0.0 --port ${PORT}


