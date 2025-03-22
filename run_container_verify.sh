#!/bin/bash
# Script to build and run the Docker container and verify GPU stack

# Set colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== INAVVI Trading System Container Verification ===${NC}"
echo "This script will build the Docker container, run it, and verify the GPU stack."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if NVIDIA Docker runtime is installed
if ! docker info | grep -q "Runtimes:.*nvidia"; then
    echo -e "${YELLOW}Warning: NVIDIA Docker runtime might not be installed.${NC}"
    echo "The container may not have access to the GPU."
    echo "Consider installing the NVIDIA Container Toolkit:"
    echo "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

# Check if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPU detected:${NC}"
    nvidia-smi | head -n 10
else
    echo -e "${RED}Warning: nvidia-smi command not found. No NVIDIA GPU detected.${NC}"
    echo "The container may not have access to the GPU."
fi

# Make the verification script executable
chmod +x container_gpu_verify.py

echo -e "\n${YELLOW}Building Docker container...${NC}"
docker-compose -f docker-compose.unified.yml build

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to build Docker container.${NC}"
    exit 1
fi

echo -e "\n${YELLOW}Running Docker container...${NC}"
echo "The container will start and run the verification script."
echo "This will verify TensorFlow, TensorRT, CuPy, and Redis with GPU access."

# Run the container with Redis started and then run the verification script
docker run --rm --gpus all \
    -v $(pwd):/app/project \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/logs:/app/logs \
    trading-system-unified:latest \
    bash -c "redis-server /etc/redis/redis.conf --daemonize yes && sleep 2 && python3 /app/project/container_gpu_verify.py"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to run verification in Docker container.${NC}"
    exit 1
fi

echo -e "\n${GREEN}Verification complete!${NC}"
echo "If all tests passed, your container is properly configured with GPU access."
echo "You can now run the full system with:"
echo -e "${YELLOW}docker-compose -f docker-compose.unified.yml up${NC}"