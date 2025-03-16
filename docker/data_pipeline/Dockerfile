FROM nvcr.io/nvidia/tensorflow:24.02-tf2-py3

ENV PYTHONUNBUFFERED=1 \
    TZ=UTC \
    DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib:$LD_LIBRARY_PATH

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
    supervisor \
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
    sqlalchemy \
    pyarrow \
    fastparquet

# Create app directories
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/logs /app/data/raw /app/data/processed /app/features

# Create non-root user
RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser src/utils /app/src/utils
COPY --chown=appuser:appuser src/data_acquisition /app/src/data_acquisition
COPY --chown=appuser:appuser src/data_processing /app/src/data_processing
COPY --chown=appuser:appuser src/config /app/src/config

# Copy common scripts
COPY docker/common /app/docker/common
COPY docker/data_pipeline/init_db.sql /app/docker/data_pipeline/init_db.sql

# Create start script
COPY --chown=appuser:appuser docker/data_pipeline/start.sh /app/start.sh

# Set executable permissions
RUN chmod +x /app/docker/common/*.sh
RUN chmod +x /app/start.sh

# Setup supervisord
COPY docker/data_pipeline/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Add healthcheck script
RUN echo '#!/bin/bash\n\
    if curl -s http://localhost:8001/health | grep -q "healthy" && \
    curl -s http://localhost:8006/health | grep -q "healthy"; then\n\
    exit 0\n\
    else\n\
    exit 1\n\
    fi' > /app/healthcheck.sh && \
    chmod +x /app/healthcheck.sh

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /app/healthcheck.sh

# Expose ports
EXPOSE 8001 8006

# Start supervisord
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]