# SiteGiant Pricing Automation - Production Docker Image
# Multi-stage build for smaller image size

# ============================================================
# Stage 1: Builder
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================
# Stage 2: Production
# ============================================================
FROM python:3.11-slim AS production

# Labels
LABEL maintainer="SiteGiant Pricing Team" \
      version="1.0.0" \
      description="SiteGiant Pricing Automation - Pokedata Price Sync"

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    # Production settings
    UVICORN_WORKERS=2 \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=appuser:appgroup . .

# Create data directories with proper permissions
RUN mkdir -p data/input data/output data/cache data/mapping data/sessions logs && \
    chown -R appuser:appgroup data logs

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check using dedicated health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use tini as init system for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Run with production settings (multiple workers)
CMD ["sh", "-c", "uvicorn src.webapp.main:app --host $UVICORN_HOST --port $UVICORN_PORT --workers $UVICORN_WORKERS"]
