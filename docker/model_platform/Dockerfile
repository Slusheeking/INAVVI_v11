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
    gnupg \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Set up environment for TensorFlow Serving API
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib:$LD_LIBRARY_PATH
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_GPU_ALLOCATOR=cuda_malloc_async
ENV TF_ENABLE_ONEDNN_OPTS=1
ENV TF_XLA_FLAGS=--tf_xla_enable_xla_devices --tf_xla_auto_jit=2
ENV PYTHONUNBUFFERED=1

# Verify TensorFlow GPU support
RUN python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', tf.config.list_physical_devices('GPU')); print('Built with CUDA:', tf.test.is_built_with_cuda()); print('Built with GPU support:', tf.test.is_built_with_gpu_support())"

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch GPU support
RUN python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count())"

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
    grpcio \
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
    hmmlearn \
    statsmodels \
    scipy \
    pyarrow \
    fastparquet \
    fastapi \
    uvicorn \
    pydantic \
    prometheus-client \
    pyyaml \
    requests \
    python-dotenv \
    pytest \
    pytest-cov \
    flake8 \
    black \
    isort

# Create app directories
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/logs /app/models/registry /app/models/checkpoints /app/data/backtest /app/results/backtest

# Create non-root user
RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser src/utils /app/src/utils
COPY --chown=appuser:appuser src/model_training /app/src/model_training
COPY --chown=appuser:appuser src/model_services /app/src/model_services
COPY --chown=appuser:appuser src/continuous_learning /app/src/continuous_learning

# Copy TensorFlow Serving REST API implementation
COPY docker/model_platform/tf_serving_rest_api.py /app/docker/model_platform/tf_serving_rest_api.py

# Copy TensorFlow Serving API implementation
COPY docker/model_platform/tf_serving_api.py /app/docker/model_platform/tf_serving_api.py

# Copy fixed TensorFlow Serving API implementation
COPY docker/model_platform/tf_serving_api_fixed.py /app/docker/model_platform/tf_serving_api_fixed.py

# Copy matplotlib fix script
COPY docker/model_platform/fix_matplotlib_permissions.py /app/docker/model_platform/fix_matplotlib_permissions.py

# Copy model registry initialization script
COPY docker/model_platform/init_model_registry.py /app/docker/model_platform/init_model_registry.py

# Copy process monitor script
COPY docker/model_platform/process_monitor.py /app/docker/model_platform/process_monitor.py

# Copy common scripts
COPY docker/common /app/docker/common

# Create start script
COPY --chown=appuser:appuser docker/model_platform/start.sh /app/start.sh

# Set executable permissions
RUN chmod +x /app/docker/common/*.sh
RUN chmod +x /app/start.sh
RUN chmod +x /app/docker/model_platform/fix_matplotlib_permissions.py
RUN chmod +x /app/docker/model_platform/init_model_registry.py
RUN chmod +x /app/docker/model_platform/process_monitor.py

# Setup supervisord
COPY docker/model_platform/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy TensorFlow serving supervisord config
COPY docker/model_platform/tf_serving_supervisord.conf /etc/supervisor/conf.d/tf_serving_supervisord.conf

# Add healthcheck script
RUN echo '#!/bin/bash\n\
    if curl -s http://localhost:8003/health | grep -q "healthy" && \
    curl -s http://localhost:8005/health | grep -q "healthy"; then\n\
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
EXPOSE 8003 8005 8500 8501

# Start supervisord
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
