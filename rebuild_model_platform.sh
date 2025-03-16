#!/bin/bash

# Script to rebuild and restart the model-platform container with GPU support
# for NVIDIA GH200 Grace Hopper Superchip

echo "===== Rebuilding and Restarting Model Platform Container ====="
echo "This script will rebuild the model-platform container with GPU support"
echo "for the NVIDIA GH200 Grace Hopper Superchip."
echo ""

# Check if docker and docker-compose are installed
if ! command -v docker &> /dev/null; then
    echo "Error: docker is not installed. Please install docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi is not available. NVIDIA driver may not be installed."
    echo "Continuing anyway, but GPU support may not work."
else
    echo "NVIDIA GPU information:"
    nvidia-smi
fi

# Stop the model-platform container if it's running
echo ""
echo "Stopping model-platform container if it's running..."
docker-compose stop model-platform

# Rebuild the model-platform container
echo ""
echo "Rebuilding model-platform container..."
docker-compose build model-platform

# Start the model-platform container
echo ""
echo "Starting model-platform container..."
docker-compose up -d model-platform

# Wait for the container to start
echo ""
echo "Waiting for container to start..."
sleep 10

# Check if the container is running
if docker ps | grep -q "ats-model-platform"; then
    echo "Container started successfully!"
else
    echo "Error: Container failed to start."
    echo "Checking container logs:"
    docker-compose logs model-platform
    exit 1
fi

# Check GPU support
echo ""
echo "Checking GPU support in the container..."
docker exec ats-model-platform nvidia-smi || echo "nvidia-smi failed in container"

echo ""
echo "Checking TensorFlow GPU support..."
docker exec ats-model-platform python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU')); print('Built with CUDA:', tf.test.is_built_with_cuda()); print('Built with GPU support:', tf.test.is_built_with_gpu_support())"

echo ""
echo "Checking PyTorch GPU support..."
docker exec ats-model-platform python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count() if torch.cuda.is_available() else 'N/A'); print('CUDA device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'N/A')"

echo ""
echo "Container logs (last 20 lines):"
docker-compose logs --tail=20 model-platform

echo ""
echo "===== Rebuild and Restart Complete ====="
echo "To view the full GPU test results, run:"
echo "docker exec ats-model-platform cat /app/logs/gpu_test_results.log"
echo ""
echo "To run the GPU test manually, run:"
echo "docker exec ats-model-platform python /app/docker/model_platform/test_gpu_support.py"