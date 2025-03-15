FROM nvcr.io/nvidia/tensorflow:24.02-tf2-py3

ENV PYTHONUNBUFFERED=1 \
    TZ=UTC \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    curl \
    git \
    wget \
    build-essential \
    ca-certificates \
    libc6-dev \
    linux-libc-dev \
    libstdc++-11-dev \
    netcat-openbsd \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Verify netcat installation
RUN which nc && nc -h

# Install Python dependencies
RUN pip install --no-cache-dir \
    psycopg2-binary \
    redis \
    pandas \
    numpy \
    websocket-client \
    websockets \
    aiohttp \
    fastapi \
    uvicorn \
    pydantic \
    prometheus-client \
    polygon-api-client \
    pytz \
    pyyaml \
    requests \
    python-dotenv \
    pytest \
    pytest-cov \
    flake8 \
    black \
    isort \
    sqlalchemy

# Create app directories
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/logs /app/data

# Create non-root user
RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser src/utils /app/src/utils
COPY --chown=appuser:appuser src/data_acquisition /app/src/data_acquisition
COPY --chown=appuser:appuser docker/data_acquisition/start.sh /app/start.sh

# Make start script executable
RUN chmod +x /app/start.sh

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose port
EXPOSE 8001

# Start the application
CMD ["/app/start.sh"]