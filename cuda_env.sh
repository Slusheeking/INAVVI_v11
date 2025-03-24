#!/bin/bash
# CUDA environment variables for GPU detection

# CUDA paths
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# GPU device selection
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# TensorFlow optimizations
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=8
export TF_USE_CUDNN=1
export TF_ENABLE_ONEDNN_OPTS=0

# TensorRT optimizations
export TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT=true
export TF_TRT_ALLOW_CUSTOM_OPS=true

echo "CUDA environment variables set successfully!"
