#!/bin/bash
# Run the existing Docker container with GPU support

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Building Docker image with GPU support...${NC}"

# Check if nvidia-docker or docker with nvidia runtime is available
if command -v nvidia-docker &> /dev/null; then
    DOCKER_CMD="nvidia-docker"
    echo -e "${GREEN}Using nvidia-docker for GPU support${NC}"
elif docker info | grep -q "Runtimes:.*nvidia"; then
    DOCKER_CMD="docker"
    RUNTIME="--runtime=nvidia"
    echo -e "${GREEN}Using Docker with NVIDIA runtime for GPU support${NC}"
else
    echo -e "${YELLOW}NVIDIA Docker runtime not found. Checking for GPU devices...${NC}"
    if [ -d "/dev/nvidia0" ] || [ -d "/dev/nvidia-caps" ]; then
        DOCKER_CMD="docker"
        GPU_ARGS="--gpus all"
        echo -e "${GREEN}Using Docker with GPU device pass-through${NC}"
    else
        echo -e "${RED}No NVIDIA GPU devices found. Running without GPU support.${NC}"
        DOCKER_CMD="docker"
        echo -e "${YELLOW}Warning: Performance will be limited without GPU acceleration${NC}"
    fi
fi

# Build the Docker image
echo -e "${BLUE}Building Docker image...${NC}"
docker build -t ml-trading-gpu:latest .

# Check if build was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}Docker build failed. Exiting.${NC}"
    exit 1
fi

echo -e "${GREEN}Docker image built successfully.${NC}"

# Run the Docker container with GPU support
echo -e "${BLUE}Running Docker container with GPU support...${NC}"

# Set environment variables for GPU optimization
GPU_ENV="-e NVIDIA_VISIBLE_DEVICES=all \
         -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
         -e TF_FORCE_GPU_ALLOW_GROWTH=true \
         -e TF_XLA_FLAGS='--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit' \
         -e TF_CUDA_HOST_MEM_LIMIT_IN_MB=80000 \
         -e TF_GPU_THREAD_MODE=gpu_private \
         -e TF_GPU_THREAD_COUNT=8 \
         -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
         -e TF_GPU_ALLOCATOR=cuda_malloc_async \
         -e TF_USE_CUDA_GRAPHS=0 \
         -e TF_CUDNN_USE_AUTOTUNE=1 \
         -e TF_LAYOUT_OPTIMIZER_DISABLE=1 \
         -e TF_ENABLE_ONEDNN_OPTS=0"

# Run the container
if [ "$DOCKER_CMD" = "nvidia-docker" ]; then
    nvidia-docker run -it --rm \
        -v $(pwd):/app \
        -v $(pwd)/../..:/home/ubuntu/INAVVI_v11-2 \
        $GPU_ENV \
        ml-trading-gpu:latest \
        /bin/bash -c "cd /app && ./run_gpu_ml_setup.sh"
else
    docker run -it --rm \
        $RUNTIME $GPU_ARGS \
        -v $(pwd):/app \
        -v $(pwd)/../..:/home/ubuntu/INAVVI_v11-2 \
        $GPU_ENV \
        ml-trading-gpu:latest \
        /bin/bash -c "cd /app && ./run_gpu_ml_setup.sh"
fi

# Check if container ran successfully
if [ $? -ne 0 ]; then
    echo -e "${RED}Docker container exited with an error.${NC}"
    exit 1
fi

echo -e "${GREEN}Docker container ran successfully.${NC}"