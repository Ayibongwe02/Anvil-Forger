# Build stage: compile dependencies
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage: minimal image
FROM python:3.12-slim

WORKDIR /app

# Create non-root user for security
RUN groupadd -r anvil && useradd -r -g anvil anvil

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directories
RUN mkdir -p /app/data/datasets /app/data/models && \
    chown -R anvil:anvil /app

# Copy pre-built Python packages from builder to system Python path
COPY --from=builder /root/.local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /root/.local/bin/* /usr/local/bin/

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000

COPY requirements.txt .
COPY app.py .
COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/

USER anvil

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "180", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
