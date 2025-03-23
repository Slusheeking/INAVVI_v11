# TensorFlow Optimization for NVIDIA GH200 Grace Hopper Superchips

This document outlines the optimizations applied to the INAVVI Trading System to maximize TensorFlow performance on NVIDIA GH200 Grace Hopper Superchips.

## Optimization Summary

The following optimizations have been implemented:

1. **Base Image Selection**
   - Using `nvcr.io/nvidia/tensorflow:24.02-tf2-py3` - NVIDIA's official TensorFlow container with CUDA 12.4 support

2. **CUDA Libraries and Dependencies**
   - Added `libcudnn8-dev`, `libnvinfer-dev`, and `libnvinfer-plugin-dev` for TensorRT support
   - Configured proper library paths and CUDA environment variables

3. **TensorFlow Acceleration Settings**
   - Enabled TF32 precision mode for faster computation with minimal precision loss
   - Configured unified memory for seamless CPU/GPU memory management
   - Enabled NUMA-aware allocators for optimized memory access
   - Set XLA compilation flags for just-in-time compilation
   - Configured optimal thread counts and memory limits

4. **TensorRT Integration**
   - Added TensorRT libraries and Python bindings
   - Configured TensorRT optimization settings for reduced precision operations
   - Enabled dynamic shapes and custom operations
   - Implemented special installation process via nvidia-pyindex (see TENSORRT_INSTALLATION.md)

5. **CuPy Optimization**
   - Configured CuPy to use TF32 precision
   - Set up caching for compiled CUDA kernels
   - Enabled CUB and cuTENSOR accelerators

6. **Package Version Compatibility**
   - Specified compatible versions of NumPy, XGBoost, and other dependencies
   - Ensured proper CUDA version detection and compatibility

## Environment Variables

The following environment variables have been added to optimize performance:

```
NVIDIA_TF32_OVERRIDE=1
CUDA_DEVICE_MAX_CONNECTIONS=32
TF_FORCE_UNIFIED_MEMORY=1
TF_ENABLE_NUMA_AWARE_ALLOCATORS=1
TF_FORCE_GPU_ALLOW_GROWTH=true
TF_XLA_FLAGS=--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_enable_lazy_compilation=false
TF_CUDA_HOST_MEM_LIMIT_IN_MB=32000
TF_GPU_THREAD_MODE=gpu_private
TF_GPU_THREAD_COUNT=8
TF_GPU_ALLOCATOR=cuda_malloc_async
TF_USE_CUDA_GRAPHS=1
XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_gpu_enable_fast_min_max=true
TF_CUDNN_USE_AUTOTUNE=1
CUPY_TF32=1
CUPY_CACHE_DIR=/app/data/cache/cupy
```

## Performance Expectations

With these optimizations, you should expect:

1. **Faster Training Times**: TF32 precision and XLA compilation can provide 2-3x speedup for many models
2. **Reduced Memory Usage**: Unified memory and optimized allocators improve memory efficiency
3. **Higher Throughput**: Optimized thread counts and memory access patterns increase throughput
4. **Better Scaling**: NUMA-aware allocators improve performance on multi-GPU systems

## Troubleshooting

If you encounter issues:

1. **Memory Errors**: Try reducing batch sizes or model complexity
2. **Precision Issues**: If model accuracy suffers, consider disabling TF32 precision
3. **Compatibility Problems**: Check for version conflicts between TensorFlow, CUDA, and other libraries
4. **TensorRT Issues**: See TENSORRT_INSTALLATION.md for detailed troubleshooting of TensorRT installation and configuration

## Rebuild Instructions

To apply these optimizations, run:

```bash
./rebuild_container.sh
```

This script will rebuild the Docker container with all optimizations applied.