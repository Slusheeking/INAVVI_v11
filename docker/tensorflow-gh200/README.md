# TensorFlow GH200 Polygon Data Benchmark

This project provides a benchmark for processing financial market data from Polygon.io using NVIDIA's GH200 Grace Hopper Superchip. The benchmark focuses on optimizing data processing pipelines and financial calculations for high-performance trading systems.

## Overview

The NVIDIA GH200 Grace Hopper Superchip combines an ARM-based CPU (Grace) with an H100 GPU (Hopper) in a unified architecture with coherent memory. This architecture is particularly well-suited for financial data processing, which requires both high computational throughput and efficient memory access patterns.

This benchmark demonstrates:

1. Optimized data loading from Polygon.io API
2. GPU-accelerated financial calculations using custom CUDA kernels
3. Performance comparison between standard and GH200-optimized implementations
4. Visualization of benchmark results

## Key Components

- **polygon_data_source.py**: Standard implementation of Polygon.io data source
- **polygon_data_source_gh200.py**: GH200-optimized implementation with:
  - Unified memory utilization
  - Optimized data transfer between CPU and GPU
  - Enhanced parallelism for data processing
  - GH200-specific TensorFlow optimizations
- **financial_cuda_ops.py**: Custom CUDA kernels for financial calculations:
  - Order book processing
  - Technical indicators (EMA, RSI, Bollinger Bands)
  - Tick data normalization
  - Fused financial operations
- **polygon_benchmark.py**: Comprehensive benchmark script that compares performance

## Performance Optimizations

The GH200-optimized implementation includes several key optimizations:

1. **Unified Memory**: Leverages GH200's unified memory architecture to reduce data transfer overhead
2. **Batch Processing**: Processes data in optimized batches to maximize GPU utilization
3. **XLA Compilation**: Uses TensorFlow's XLA (Accelerated Linear Algebra) for just-in-time compilation
4. **Custom CUDA Kernels**: Implements financial calculations directly in CUDA for maximum performance
5. **TensorFlow Data Pipeline Optimizations**: Enhances TensorFlow's data pipeline with GH200-specific settings
6. **Memory Pre-allocation**: Pre-allocates GPU memory to avoid fragmentation
7. **Parallel Data Fetching**: Uses ThreadPoolExecutor for efficient parallel data fetching

## Usage

### Building the Container

```bash
./run.sh build
```

### Starting the Container

```bash
./run.sh start
```

### Running the Polygon Benchmark

Standard benchmark:
```bash
./run.sh polygon
```

GH200-optimized benchmark:
```bash
./run.sh polygon-gh200
```

### Accessing JupyterLab

JupyterLab is available at http://localhost:8888 after starting the container.

### Stopping the Container

```bash
./run.sh stop
```

## Benchmark Results

The benchmark compares:

1. **Data Processing Throughput**: Records processed per second
2. **CUDA Processing Time**: Time to perform financial calculations
3. **Operation Speedups**: Speedup for individual operations (technical indicators, MACD, order book processing)

Results are saved as:
- PNG charts: `polygon_benchmark_aggregates.png`
- CSV data: `polygon_benchmark_aggregates_results.csv`

## Environment Variables

The container includes environment variables optimized for GH200:

- `TF_FORCE_GPU_ALLOW_GROWTH=true`: Enables dynamic GPU memory allocation
- `TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"`: Enables XLA JIT compilation
- `TF_CUDA_HOST_MEM_LIMIT_IN_MB=80000`: Sets host memory limit for CUDA operations
- `TF_GPU_THREAD_MODE=gpu_private`: Optimizes GPU thread mode
- `TF_GPU_THREAD_COUNT=8`: Sets optimal thread count for GH200
- `TF_GPU_ALLOCATOR=cuda_malloc_async`: Uses asynchronous CUDA memory allocation
- `TF_USE_CUDA_GRAPHS=0`: Disables CUDA graphs which can cause errors with financial data

## Requirements

- NVIDIA GH200 Grace Hopper Superchip
- NVIDIA Container Toolkit
- Docker
- Polygon.io API key (set in polygon_data_source.py)

## License

This project is licensed under the MIT License - see the LICENSE file for details.