# Use NVIDIA's TensorFlow container optimized for ARM64
FROM nvcr.io/nvidia/tensorflow:24.02-tf2-py3

# Install required system packages for Polygon data processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    nvidia-utils-550 build-essential wget python3-dev \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables optimized for GH200 and Polygon data processing
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
ENV TF_CUDA_HOST_MEM_LIMIT_IN_MB=80000
# Optimize thread settings for GH200 with Polygon data
ENV TF_GPU_THREAD_MODE=gpu_private 
ENV TF_GPU_THREAD_COUNT=8
ENV CUDA_DEVICE_MAX_CONNECTIONS=32
ENV TF_GPU_ALLOCATOR=cuda_malloc_async
# Disable CUDA graphs which can cause errors with Polygon data processing
ENV TF_USE_CUDA_GRAPHS=0
ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
ENV TF_CUDNN_USE_AUTOTUNE=1
# Additional settings to improve stability for financial data processing
ENV TF_LAYOUT_OPTIMIZER_DISABLE=1 
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# Install Python packages required for Polygon data processing
RUN pip install --upgrade pip && pip install --no-cache-dir \
    pandas \
    matplotlib \
    scikit-learn==1.3.2 \
    pyarrow==14.0.1 \
    numba>=0.57.0 \
    requests \
    redis \
    urllib3 \
    jupyterlab \
    cupy-cuda12x \
    pycuda \
    polygon-api-client \
    alpaca-trade-api \
    ipywidgets \
    nvidia-ml-py3 \
    packaging \
    pytz \
    aiohttp \
    xgboost \
    schedule \
    slack-sdk

# Create a working directory
WORKDIR /app

# Copy environment file
COPY .env /app/.env

# Copy trading system files
COPY . /app/tests/

# Make sure the environment file is readable
RUN chmod 644 /app/.env

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs

# Verify NVIDIA drivers and TensorFlow GPU support
RUN ldconfig && \
    echo "Testing NVIDIA driver installation and GPU support..." && \
    python -c "import tensorflow as tf; import cupy as cp; \
    print('TensorFlow version:', tf.__version__); \
    print('CuPy version:', cp.__version__); \
    print('GPU available:', tf.config.list_physical_devices('GPU')); \
    # Run a simple test operation to verify GPU functionality \
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'): \
        print('Test matrix multiplication result:', tf.matmul(tf.constant([[1.0, 2.0], [3.0, 4.0]]), tf.constant([[5.0, 6.0], [7.0, 8.0]])).numpy())"

# Set up entry point
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]