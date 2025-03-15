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
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with specific version pins for ARM64 compatibility
RUN pip install --no-cache-dir \
    psycopg2-binary \
    redis \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    plotly \
    xgboost \
    lightgbm \
    optuna \
    mlflow \
    tensorboard \
    tensorflow-probability \
    tensorflow-addons \
    tensorflow-model-optimization \
    hmmlearn \
    statsmodels \
    scipy \
    pyarrow \
    fastparquet \
    pyyaml \
    requests \
    python-dotenv \
    pytest \
    pytest-cov \
    flake8 \
    black \
    isort

# Install PyTorch for multi-framework support - use the correct wheels for ARM64
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Create app directories
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/data/backtest /app/results/backtest

# Create non-root user
RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser src/utils /app/src/utils
COPY --chown=appuser:appuser src/continuous_learning /app/src/continuous_learning
COPY --chown=appuser:appuser docker/continuous_learning/start.sh /app/start.sh

# Make start script executable
RUN chmod +x /app/start.sh

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1

# Expose port
EXPOSE 8005

# Start the application
CMD ["/app/start.sh"]