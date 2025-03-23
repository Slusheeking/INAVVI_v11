# TensorRT Installation Guide for INAVVI Trading System

This document outlines the approach for TensorRT integration in the INAVVI trading system, specifically optimized for NVIDIA GH200 Grace Hopper Superchips.

## Overview

TensorRT is a high-performance deep learning inference optimizer and runtime that delivers low latency and high throughput for deep learning inference applications. It's particularly valuable for our trading system where inference speed can directly impact trading decisions.

## Installation Approach

Our system uses the NVIDIA TensorFlow container (`nvcr.io/nvidia/tensorflow:24.02-tf2-py3`) which comes with TensorRT pre-installed. This provides several advantages:

1. **Pre-optimized Configuration**: The NVIDIA container includes TensorRT that's already configured to work optimally with the included TensorFlow version.
2. **Compatibility Assurance**: Eliminates version compatibility issues between TensorFlow and TensorRT.
3. **System Library Integration**: Proper integration with system-level CUDA libraries.

## Verification

We verify TensorRT availability in our Dockerfile with:

```python
python3 -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"
```

This command is executed during container build to confirm TensorRT is properly installed and accessible.

## Environment Variables

We've configured several environment variables to optimize TensorRT performance:

```dockerfile
# TensorRT optimization settings
ENV TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT=true
ENV TF_TRT_ALLOW_CUSTOM_OPS=true
ENV TF_TRT_USE_IMPLICIT_BATCH=false
ENV TF_TRT_ALLOW_DYNAMIC_SHAPES=true
ENV TF_TRT_ALLOW_REDUCED_PRECISION=true
ENV TF_TRT_ALLOW_REDUCED_PRECISION_REDUCTION=true
ENV TF_TRT_ALLOW_REDUCED_PRECISION_CONVERSION=true
ENV TF_TRT_ALLOW_REDUCED_PRECISION_WEIGHTS=true
ENV TF_TRT_ALLOW_REDUCED_PRECISION_VARIABLES=true
ENV TF_TRT_ALLOW_REDUCED_PRECISION_CONSTANTS=true
ENV TF_TRT_ALLOW_REDUCED_PRECISION_ACTIVATION=true
ENV TF_TRT_ALLOW_REDUCED_PRECISION_ACCUMULATION=true
ENV TF_TRT_ALLOW_REDUCED_PRECISION_REDUCTION_AGGREGATION=true
ENV TF_TRT_ALLOW_REDUCED_PRECISION_REDUCTION_INITIALIZATION=true
ENV TF_TRT_ALLOW_REDUCED_PRECISION_REDUCTION_ACCUMULATION=true
```

These settings enable various TensorRT optimizations including reduced precision operations, which can significantly accelerate inference on GH200 hardware.

## Fallback Mechanism

Our `rebuild_container.sh` script includes a fallback mechanism if the build fails. It creates a simplified Dockerfile with minimal dependencies to ensure the system can still be built and run, even if some advanced features like TensorRT are not available.

## Usage in the Trading System

To use TensorRT for model optimization in the trading system:

1. **Model Conversion**: Convert TensorFlow models to TensorRT format:

```python
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

# Load your saved model
saved_model_dir = "path/to/saved_model"
conversion_params = trt.TrtConversionParams(
    precision_mode=trt.TrtPrecisionMode.FP16,  # Use FP16 for better performance
    max_workspace_size_bytes=8000000000,  # 8GB
    maximum_cached_engines=100
)

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=saved_model_dir,
    conversion_params=conversion_params
)

# Convert the model
converter.convert()

# Save the converted model
converter.save("path/to/tensorrt_model")
```

2. **Inference with TensorRT Model**:

```python
import tensorflow as tf

# Load the TensorRT model
trt_model = tf.saved_model.load("path/to/tensorrt_model")
infer = trt_model.signatures["serving_default"]

# Perform inference
input_tensor = tf.constant([[...]])  # Your input data
result = infer(input_tensor)
```

## Troubleshooting

If you encounter issues with TensorRT:

1. **Verify TensorRT Installation**:
   ```bash
   python3 -c "import tensorrt; print(tensorrt.__version__)"
   ```

2. **Check TensorFlow-TensorRT Integration**:
   ```bash
   python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU')); from tensorflow.python.compiler.tensorrt import trt_convert as trt; print(trt.is_tensorrt_enabled())"
   ```

3. **Inspect CUDA Configuration**:
   ```bash
   python3 -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"
   ```

4. **Common Issues**:
   - Incompatible TensorRT and TensorFlow versions
   - Insufficient GPU memory
   - Unsupported layer types in your model
   - CUDA version mismatches

## References

- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [TensorFlow-TensorRT Integration](https://www.tensorflow.org/tfrt)
- [NVIDIA NGC Containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow)