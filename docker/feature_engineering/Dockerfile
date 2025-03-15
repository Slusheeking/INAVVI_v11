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
    netcat-openbsd \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Verify netcat installation
RUN which nc && nc -h

# Install TA-Lib dependencies and required development headers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    unzip \
    libc6-dev \
    linux-libc-dev \
    libstdc++-11-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install TA-Lib with explicit build type for ARM64
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    # Explicitly set build type for ARM64
    ./configure --prefix=/usr --build=aarch64-unknown-linux-gnu && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz && \
    # Create symlink to ensure the library is found
    ln -s /usr/lib/libta_lib.so.0 /usr/lib/libta-lib.so

# Set environment variables to help find the TA-Lib library
ENV LD_LIBRARY_PATH=/usr/lib:/usr/local/lib
ENV LIBRARY_PATH=/usr/lib:/usr/local/lib
ENV C_INCLUDE_PATH=/usr/include
ENV CPLUS_INCLUDE_PATH=/usr/include

# Install Python dependencies (exclude ta-lib for now)
RUN pip install --no-cache-dir \
    psycopg2-binary \
    redis \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    plotly \
    pandas-ta \
    statsmodels \
    scipy \
    pyarrow \
    fastparquet \
    dask \
    distributed \
    pyyaml \
    requests \
    python-dotenv \
    pytest \
    pytest-cov \
    flake8 \
    black \
    isort \
    sqlalchemy

# Install numpy first to avoid dependency issues
RUN pip install --no-cache-dir numpy==1.26.4

# Install ta-lib Python wrapper using a more robust approach for ARM64
RUN LIBRARY_PATH=/usr/lib CFLAGS="-I/usr/include" CPATH="/usr/include" \
    pip install --no-cache-dir --no-build-isolation ta-lib==0.4.28

# Create app directories
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/logs /app/features

# Create non-root user
RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser src/utils /app/src/utils
COPY --chown=appuser:appuser src/feature_engineering /app/src/feature_engineering
COPY --chown=appuser:appuser docker/feature_engineering/start.sh /app/start.sh

# Make start script executable
RUN chmod +x /app/start.sh

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

# Expose port
EXPOSE 8004

# Start the application
CMD ["/app/start.sh"]