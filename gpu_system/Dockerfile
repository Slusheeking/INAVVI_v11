# Pull the latest TensorFlow container with CUDA 12.3 support
FROM nvcr.io/nvidia/tensorflow:23.12-tf2-py3

# Install required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    python3-dev \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for CUDA 12.3 and GH200 optimization
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}

# Enhanced TF32 acceleration
ENV NVIDIA_TF32_OVERRIDE="1"
# Enable tensor cores
ENV CUDA_DEVICE_MAX_CONNECTIONS="32"
# Memory allocator optimization
ENV TF_FORCE_UNIFIED_MEMORY="1"
ENV TF_ENABLE_NUMA_AWARE_ALLOCATORS="1"
# Additional optimizations
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
# XLA compilation optimization
ENV TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
# Memory limits
ENV TF_CUDA_HOST_MEM_LIMIT_IN_MB=80000
# Thread optimization
ENV TF_GPU_THREAD_MODE=gpu_private
ENV TF_GPU_THREAD_COUNT=8
# Async memory allocator
ENV TF_GPU_ALLOCATOR=cuda_malloc_async
# Disable CUDA graphs which can cause errors with financial data processing
ENV TF_USE_CUDA_GRAPHS=0
ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
ENV TF_CUDNN_USE_AUTOTUNE=1
# Additional settings to improve stability for financial data processing
ENV TF_LAYOUT_OPTIMIZER_DISABLE=1
ENV TF_ENABLE_ONEDNN_OPTS=0
# Additional GH200 optimizations
ENV CUDA_AUTO_BOOST="1"
ENV NCCL_IB_DISABLE="0"
ENV NCCL_P2P_LEVEL="NVL"
ENV GOMP_CPU_AFFINITY="0-71"
ENV MALLOC_TRIM_THRESHOLD_="0"
ENV MALLOC_MMAP_THRESHOLD_="131072"

# Install additional Python packages
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
    nvidia-ml-py3 \
    ipywidgets \
    alpaca-trade-api \
    packaging \
    pytz \
    aiohttp \
    xgboost \
    schedule \
    slack-sdk

# Create a working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs

# Copy verification script from the tests directory
COPY verify_tensorflow.py /app/verify_tensorflow.py

# Make verification script executable
RUN chmod +x /app/verify_tensorflow.py

# Add additional scripts for testing
COPY test_tensorflow_gpu.py /app/test_tensorflow_gpu.py
COPY test_tensorflow_gpu_direct.py /app/test_tensorflow_gpu_direct.py

# Set up entry point
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]