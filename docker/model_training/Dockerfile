FROM nvcr.io/nvidia/tensorflow:24.02-tf2-py3

ENV PYTHONUNBUFFERED=1 \
    TZ=UTC \
    DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    redis-tools \
    netcat-openbsd \
    curl \
    git \
    wget \
    build-essential \
    ca-certificates \
    libc6-dev \
    linux-libc-dev \
    libstdc++-11-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch for multi-framework support with CUDA
RUN pip install --no-cache-dir torch torchvision torchaudio

# Install additional Python dependencies
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
    tensorflow-serving-api \
    tensorflow-probability \
    tensorflow-addons \
    tensorflow-model-optimization \
    tensorflow-hub \
    tensorflow-text \
    transformers \
    pyyaml \
    requests \
    python-dotenv \
    pytest \
    pytest-cov \
    flake8 \
    black \
    isort && \
    echo "Note: tensorflow-data-validation, tensorflow-transform, and tensorflow-metadata packages are not available for ARM64 architecture"

# Create app directories
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/logs /app/models/registry /app/models/checkpoints

# Create non-root user
RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser src/utils /app/src/utils
COPY --chown=appuser:appuser src/model_training /app/src/model_training
COPY --chown=appuser:appuser docker/model_training/start.sh /app/start.sh

# Make start script executable
RUN chmod +x /app/start.sh

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Expose ports
EXPOSE 8003 8500

# Start the application
CMD ["/app/start.sh"]