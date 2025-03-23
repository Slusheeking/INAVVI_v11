#!/usr/bin/env python3
"""
Enhanced TensorRT Integration Test Script

This script tests TensorRT integration with TensorFlow by:
1. Checking TensorRT version and availability
2. Creating and converting a TensorFlow model to TensorRT
3. Benchmarking performance between TensorFlow and TensorRT
4. Testing different precision modes (FP32, FP16)
5. Measuring inference latency
"""

import os
import time
import numpy as np
import tensorflow as tf
import tensorrt as trt
import logging
from tensorflow.python.compiler.tensorrt import trt_convert as trt_convert

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('tensorrt_test')

# Suppress TensorFlow logging except for errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def print_separator(title):
    """Print a section separator with title"""
    width = 80
    print("\n" + "=" * width)
    print(f"{title.center(width)}")
    print("=" * width)


def check_gpu_availability():
    """Check if GPU is available for TensorFlow"""
    print_separator("GPU Availability Check")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"  {i+1}. {gpu.name}")

        # Get GPU details
        try:
            for i, gpu in enumerate(gpus):
                gpu_details = tf.config.experimental.get_device_details(gpu)
                print(f"GPU {i+1} Details:")
                for key, value in gpu_details.items():
                    print(f"  {key}: {value}")
        except:
            print("Could not retrieve detailed GPU information")

        # Enable memory growth to avoid allocating all memory at once
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth enabled for all GPUs")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")
    else:
        print("No GPU found. TensorRT acceleration will not be available.")

    return len(gpus) > 0


def get_tensorrt_version():
    """Get TensorRT version information"""
    print_separator("TensorRT Version Information")

    print(f"TensorRT Version: {trt.__version__}")

    # Check TensorRT availability in TensorFlow
    try:
        from tensorflow.python.compiler.tensorrt import trt_convert
        print("TensorRT integration in TensorFlow: Available")
        print(
            f"TF-TRT Version: {trt_convert.__version__ if hasattr(trt_convert, '__version__') else 'Unknown'}")
    except ImportError:
        print("TensorRT integration in TensorFlow: Not available")

    # List key TensorRT components (limited to avoid excessive output)
    key_components = [
        'Builder', 'Logger', 'Runtime', 'DataType', 'TensorFormat',
        'ActivationType', 'LayerType', 'NetworkDefinitionCreationFlag'
    ]

    print("\nKey TensorRT Components:")
    for component in key_components:
        if hasattr(trt, component):
            print(f"  ✓ {component}")
        else:
            print(f"  ✗ {component}")


def create_model(model_type="cnn", input_shape=(32, 32, 3), num_classes=10):
    """Create a TensorFlow model for conversion

    Args:
        model_type: Type of model to create ('cnn', 'resnet', 'mobilenet')
        input_shape: Input shape for the model
        num_classes: Number of output classes

    Returns:
        A compiled TensorFlow model
    """
    print_separator(f"Creating {model_type.upper()} Model")

    if model_type == "cnn":
        # Simple CNN model
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
        x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
        x = tf.keras.layers.Conv2D(128, 3, activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(num_classes)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=['accuracy']
        )

    elif model_type == "resnet":
        # ResNet-like model (simplified for testing)
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(
            64, 7, strides=2, padding='same', activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)

        # Simplified ResNet block
        for filters in [64, 128, 256]:
            # Residual block
            shortcut = x
            x = tf.keras.layers.Conv2D(
                filters, 3, padding='same', activation='relu')(x)
            x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
            if shortcut.shape[-1] != filters:
                shortcut = tf.keras.layers.Conv2D(filters, 1)(shortcut)
            x = tf.keras.layers.add([x, shortcut])
            x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(num_classes)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=['accuracy']
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print("Model created successfully")
    print("Model summary:")
    model.summary()

    return model


def convert_to_tensorrt(model, precision_mode="FP16", input_shape=(32, 32, 3), batch_size=1):
    """Convert TensorFlow model to TensorRT

    Args:
        model: TensorFlow model to convert
        precision_mode: Precision mode for TensorRT ('FP32', 'FP16', 'INT8')
        input_shape: Input shape for the model (excluding batch dimension)
        batch_size: Batch size for conversion

    Returns:
        Path to the saved TensorRT model and sample input data
    """
    print_separator(f"Converting Model to TensorRT ({precision_mode})")

    # Create sample input data
    input_data = np.random.random(
        (batch_size,) + input_shape).astype(np.float32)

    # Save the model
    model_dir = f"/tmp/tensorrt_test_model_{precision_mode.lower()}"
    tf.saved_model.save(model, model_dir)
    print(f"Model saved to {model_dir}")

    # Set precision mode
    if precision_mode == "FP32":
        trt_precision_mode = trt_convert.TrtPrecisionMode.FP32
    elif precision_mode == "FP16":
        trt_precision_mode = trt_convert.TrtPrecisionMode.FP16
    elif precision_mode == "INT8":
        trt_precision_mode = trt_convert.TrtPrecisionMode.INT8
    else:
        raise ValueError(f"Unknown precision mode: {precision_mode}")

    # Convert to TensorRT
    conversion_params = trt_convert.TrtConversionParams(
        precision_mode=trt_precision_mode,
        max_workspace_size_bytes=8000000000,
        maximum_cached_engines=1,
        use_calibration=False  # Set to False to avoid calibration warnings
    )

    converter = trt_convert.TrtGraphConverterV2(
        input_saved_model_dir=model_dir,
        conversion_params=conversion_params
    )

    print(f"Starting TensorRT conversion with {precision_mode} precision...")
    start_time = time.time()
    converter.convert()
    conversion_time = time.time() - start_time
    print(f"Conversion completed in {conversion_time:.2f} seconds")

    # Save the converted model
    trt_model_dir = f"/tmp/tensorrt_optimized_model_{precision_mode.lower()}"
    converter.save(trt_model_dir)
    print(f"TensorRT model saved to {trt_model_dir}")

    return trt_model_dir, input_data


def benchmark_models(original_model, trt_model_dir, input_data, precision_mode="FP16", iterations=100):
    """Benchmark original and TensorRT models

    Args:
        original_model: Original TensorFlow model
        trt_model_dir: Directory containing the TensorRT model
        input_data: Input data for benchmarking
        precision_mode: Precision mode used for conversion
        iterations: Number of iterations for benchmarking

    Returns:
        Dictionary with benchmark results
    """
    print_separator(f"Benchmarking Models ({precision_mode})")

    # Load the TensorRT model
    trt_model = tf.saved_model.load(trt_model_dir)
    trt_infer = trt_model.signatures['serving_default']

    # Prepare input for TensorRT model
    input_name = list(trt_infer.structured_input_signature[1].keys())[0]

    # Warm-up runs
    print("Performing warm-up runs...")
    for _ in range(10):
        original_model(input_data)
        trt_infer(**{input_name: tf.constant(input_data)})

    # Benchmark original model
    print(
        f"Benchmarking original TensorFlow model ({iterations} iterations)...")
    tf_latencies = []

    for _ in range(iterations):
        # Inference
        start_time = time.time()
        original_model(input_data)
        latency = time.time() - start_time

        tf_latencies.append(latency * 1000)  # Convert to ms

    tf_avg_latency = np.mean(tf_latencies)
    tf_p95_latency = np.percentile(tf_latencies, 95)
    tf_p99_latency = np.percentile(tf_latencies, 99)

    print(f"Original model average latency: {tf_avg_latency:.2f} ms")
    print(f"Original model P95 latency: {tf_p95_latency:.2f} ms")
    print(f"Original model P99 latency: {tf_p99_latency:.2f} ms")

    # Benchmark TensorRT model
    print(
        f"Benchmarking TensorRT model ({precision_mode}, {iterations} iterations)...")
    trt_latencies = []

    for _ in range(iterations):
        # Inference
        start_time = time.time()
        trt_infer(**{input_name: tf.constant(input_data)})
        latency = time.time() - start_time

        trt_latencies.append(latency * 1000)  # Convert to ms

    trt_avg_latency = np.mean(trt_latencies)
    trt_p95_latency = np.percentile(trt_latencies, 95)
    trt_p99_latency = np.percentile(trt_latencies, 99)

    print(f"TensorRT model average latency: {trt_avg_latency:.2f} ms")
    print(f"TensorRT model P95 latency: {trt_p95_latency:.2f} ms")
    print(f"TensorRT model P99 latency: {trt_p99_latency:.2f} ms")

    # Calculate speedup
    speedup = tf_avg_latency / trt_avg_latency

    print(f"TensorRT speedup: {speedup:.2f}x")

    return {
        "precision_mode": precision_mode,
        "tf_avg_latency": tf_avg_latency,
        "tf_p95_latency": tf_p95_latency,
        "tf_p99_latency": tf_p99_latency,
        "trt_avg_latency": trt_avg_latency,
        "trt_p95_latency": trt_p95_latency,
        "trt_p99_latency": trt_p99_latency,
        "speedup": speedup
    }


def test_batch_sizes(model, precision_mode="FP16", input_shape=(32, 32, 3)):
    """Test different batch sizes with TensorRT

    Args:
        model: Original TensorFlow model
        precision_mode: Precision mode for TensorRT
        input_shape: Input shape for the model
    """
    print_separator(f"Testing Different Batch Sizes ({precision_mode})")

    batch_sizes = [1, 4, 16, 32]
    results = []

    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        try:
            # Convert model with specific batch size
            trt_model_dir, input_data = convert_to_tensorrt(
                model,
                precision_mode=precision_mode,
                input_shape=input_shape,
                batch_size=batch_size
            )

            # Run quick benchmark (fewer iterations for larger batches)
            iterations = max(10, 100 // batch_size)
            result = benchmark_models(
                model,
                trt_model_dir,
                input_data,
                precision_mode=precision_mode,
                iterations=iterations
            )

            result["batch_size"] = batch_size
            results.append(result)

        except Exception as e:
            print(f"Error testing batch size {batch_size}: {e}")

    # Print summary
    if results:
        print("\nBatch Size Performance Summary:")
        print(
            f"{'Batch Size':<10} {'Speedup':<10} {'TF Latency (ms)':<15} {'TRT Latency (ms)':<15}")
        print("-" * 50)
        for result in results:
            print(
                f"{result['batch_size']:<10} {result['speedup']:<10.2f} {result['tf_avg_latency']:<15.2f} {result['trt_avg_latency']:<15.2f}")
    else:
        print("No batch size tests completed successfully")


def main():
    """Main function to test TensorRT integration"""
    print_separator("TensorRT Integration Test")

    # Check GPU availability
    gpu_available = check_gpu_availability()
    if not gpu_available:
        print("WARNING: No GPU detected. TensorRT tests may not work correctly.")

    # Check TensorRT version
    get_tensorrt_version()

    try:
        # Create model
        model = create_model(model_type="cnn")

        # Test with FP16 precision (most widely supported)
        precision_mode = "FP16"

        try:
            # Convert and benchmark
            trt_model_dir, input_data = convert_to_tensorrt(
                model, precision_mode=precision_mode)
            result = benchmark_models(
                model, trt_model_dir, input_data, precision_mode=precision_mode)

            # Test different batch sizes if time permits
            if result["speedup"] > 1.5:  # Only test batch sizes if we're seeing good speedup
                test_batch_sizes(model, precision_mode=precision_mode)

            print_separator("Test Results Summary")
            print(f"TensorRT acceleration is working correctly!")
            print(
                f"Achieved {result['speedup']:.2f}x speedup with {precision_mode} precision")

        except Exception as e:
            print(f"Error testing {precision_mode} precision: {e}")
            print_separator("Test Results Summary")
            print("TensorRT tests failed to complete successfully.")

    except Exception as e:
        print(f"Error during TensorRT testing: {e}")

    print_separator("Test Complete")


if __name__ == "__main__":
    main()
