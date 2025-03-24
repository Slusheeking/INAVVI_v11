#!/usr/bin/env python3
"""
Comprehensive GPU Acceleration Test Suite
-----------------------------------------
Tests GPU capabilities across multiple frameworks:
1. TensorFlow - Basic and advanced GPU operations
2. TensorRT - Integration and model optimization
3. XGBoost - GPU-accelerated training and inference
4. CuPy - CUDA-accelerated numerical operations

This script provides a comprehensive benchmark of all GPU-accelerated 
frameworks in the system for ML/AI workloads.
"""

import os
import sys
import time
import json
import logging
import numpy as np
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger('unified_gpu_test')

# Set environment variables to optimize GPU usage for NVIDIA GH200 Grace Hopper Superchips
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allow memory growth
# Match CUDA device IDs with PCI bus IDs
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = 'all'  # Use all available GPUs
# Use private thread pool for each GPU
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '8'  # Number of threads
os.environ['TF_USE_CUDNN'] = '1'  # Enable cuDNN
# Disable OneDNN optimizations which can cause issues
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Enable XLA JIT compilation
os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2"
# Allow TensorRT to use native segments
os.environ['TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT'] = 'true'
# Additional settings for GH200 Grace Hopper Superchips
os.environ['NVIDIA_VISIBLE_DEVICES'] = 'all'
os.environ['NVIDIA_DRIVER_CAPABILITIES'] = 'compute,utility,video,graphics'
os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
os.environ['TF_ENABLE_NUMA_AWARE_ALLOCATORS'] = '1'

# Define utility functions


def print_section(title, level=1):
    """Print a section separator with title"""
    if level == 1:
        width = 80
        print("\n" + "=" * width)
        print(f"{title.center(width)}")
        print("=" * width)
    elif level == 2:
        width = 70
        print("\n" + "-" * width)
        print(f"{title.center(width)}")
        print("-" * width)
    else:
        print(f"\n--- {title} ---")


def print_result(name, result, success_threshold=None):
    """Print test result with optional success indicator"""
    if success_threshold is not None:
        success = result >= success_threshold
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name}: {result:.4f} {status}")
    else:
        print(f"{name}: {result}")


def run_command(command):
    """Run shell command and return output"""
    try:
        result = subprocess.check_output(
            command, shell=True, universal_newlines=True)
        return result.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        return None


def format_memory_size(size_bytes):
    """Format memory size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0 or unit == 'GB':
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0


def check_env_vars():
    """Print important environment variables for GPU detection"""
    env_vars = [
        'CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'CUDA_PATH', 'LD_LIBRARY_PATH',
        'TF_FORCE_GPU_ALLOW_GROWTH', 'TF_GPU_THREAD_MODE', 'TF_GPU_THREAD_COUNT',
        'TF_XLA_FLAGS', 'TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT'
    ]

    print("GPU-related environment variables:")
    for var in env_vars:
        print(f"  {var}={os.environ.get(var, 'not set')}")


# Dictionary to store all test results
test_results = {
    "timestamp": datetime.now().isoformat(),
    "system_info": {},
    "tensorflow": {"status": "not_tested"},
    "tensorrt": {"status": "not_tested"},
    "xgboost": {"status": "not_tested"},
    "cupy": {"status": "not_tested"},
    "summary": {}
}

# =============================================================================
# SYSTEM INFORMATION
# =============================================================================


def get_system_info():
    """Get system and GPU information"""
    print_section("System Information")

    system_info = {}

    # Python version
    system_info["python_version"] = sys.version
    print(f"Python version: {sys.version}")

    # CUDA version
    cuda_version = run_command("nvcc --version | grep release")
    if cuda_version:
        cuda_version = cuda_version.split("release ")[-1].split(",")[0]
        system_info["cuda_version"] = cuda_version
        print(f"CUDA version: {cuda_version}")
    else:
        system_info["cuda_version"] = "Not found"
        print("CUDA not found or nvcc not in PATH")

    # GPU information using nvidia-smi
    try:
        gpu_count = int(run_command(
            "nvidia-smi --query-gpu=count --format=csv,noheader"))
        system_info["gpu_count"] = gpu_count
        print(f"GPU count: {gpu_count}")

        if gpu_count > 0:
            # Get GPU details
            gpu_name = run_command(
                "nvidia-smi --query-gpu=name --format=csv,noheader")
            system_info["gpu_name"] = gpu_name
            print(f"GPU: {gpu_name}")

            gpu_memory = run_command(
                "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits")
            system_info["gpu_memory_mb"] = int(gpu_memory.strip())
            print(f"GPU memory: {gpu_memory} MiB")

            gpu_driver = run_command(
                "nvidia-smi --query-gpu=driver_version --format=csv,noheader")
            system_info["gpu_driver"] = gpu_driver
            print(f"GPU driver: {gpu_driver}")
    except Exception as e:
        logger.error(f"Error getting GPU information: {e}")
        system_info["gpu_error"] = str(e)
        print(f"Error getting GPU information: {e}")

    # Check important environment variables
    check_env_vars()

    test_results["system_info"] = system_info
    return system_info


# =============================================================================
# TENSORFLOW TESTS
# =============================================================================

def check_gpu_availability():
    """Check if GPU is available for TensorFlow with enhanced GH200 detection"""
    print_section("GPU Availability Check", level=2)

    try:
        import tensorflow as tf

        # Print TensorFlow build information
        print(f"TensorFlow version: {tf.__version__}")
        print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
        print(
            f"TensorFlow built with GPU support: {tf.test.is_built_with_gpu_support()}")

        # Try multiple approaches to detect GPUs
        print("\nAttempting GPU detection using multiple methods:")

        # Method 1: Standard TF physical devices
        gpus = tf.config.list_physical_devices('GPU')
        print(
            f"Method 1 - tf.config.list_physical_devices: Found {len(gpus)} GPU(s)")

        # Method 2: TF device lib
        from tensorflow.python.client import device_lib
        devices = device_lib.list_local_devices()
        gpu_devices = [d for d in devices if d.device_type == 'GPU']
        print(f"Method 2 - device_lib: Found {len(gpu_devices)} GPU(s)")

        # Method 3: Direct CUDA check via nvidia-smi
        nvidia_smi_output = run_command("nvidia-smi -L")
        print(
            f"Method 3 - nvidia-smi: {'GPU detected' if nvidia_smi_output else 'No GPU detected'}")
        if nvidia_smi_output:
            print(f"  nvidia-smi output: {nvidia_smi_output}")

        # Combine results
        gpu_detected = len(gpus) > 0 or len(
            gpu_devices) > 0 or bool(nvidia_smi_output)

        if gpus:
            print(f"\nFound {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"  {i+1}. {gpu.name}")

            # Get GPU details
            try:
                for i, gpu in enumerate(gpus):
                    gpu_details = tf.config.experimental.get_device_details(
                        gpu)
                    print(f"GPU {i+1} Details:")
                    for key, value in gpu_details.items():
                        print(f"  {key}: {value}")
            except Exception as e:
                print(f"Could not retrieve detailed GPU information: {e}")

            # Enable memory growth to avoid allocating all memory at once
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("Memory growth enabled for all GPUs")
            except RuntimeError as e:
                print(f"Error setting memory growth: {e}")
        else:
            print("\nNo GPU found through TensorFlow's standard detection.")

            # Diagnostic information
            print("\nEnvironment variables:")
            for var in ['CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES', 'CUDA_HOME',
                        'TF_FORCE_GPU_ALLOW_GROWTH', 'TF_FORCE_UNIFIED_MEMORY']:
                print(f"  {var}={os.environ.get(var, 'not set')}")

            # Check if GPU is visible to system but not to TensorFlow
            nvidia_smi = run_command("nvidia-smi")
            if nvidia_smi and "GPU " in nvidia_smi:
                print("\nGPU is visible to system but not to TensorFlow!")
                print("This might be due to driver/CUDA/TensorFlow version mismatch.")
                print("Attempting to force GPU detection...")

                # Try to force GPU detection for GH200
                os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'

                # Try again with forced settings
                try:
                    gpus = tf.config.list_physical_devices('GPU')
                    if gpus:
                        print(f"Forced detection found {len(gpus)} GPU(s)")
                        gpu_detected = True
                except Exception as e:
                    print(f"Forced detection failed: {e}")

        return gpu_detected
    except ImportError as e:
        print(f"TensorFlow not available: {e}")
        return False


def test_tensorflow():
    """Test TensorFlow GPU capabilities"""
    print_section("TensorFlow GPU Test")

    tf_results = {"status": "not_available"}

    try:
        import tensorflow as tf
        tf_results["version"] = tf.__version__
        print(f"TensorFlow version: {tf.__version__}")

        # Check if TensorFlow was built with CUDA support
        cuda_build = tf.test.is_built_with_cuda()
        tf_results["cuda_build"] = cuda_build
        print(f"TensorFlow built with CUDA: {cuda_build}")

        # Check for GPU availability
        gpu_available = check_gpu_availability()

        # Check if TensorFlow can see GPUs
        gpus = tf.config.list_physical_devices('GPU')
        tf_results["gpu_count"] = len(gpus)

        if gpus:
            print(f"TensorFlow GPU count: {len(gpus)}")
            tf_results["status"] = "available"

            # Basic operations test
            print_section("Basic Operations Test", level=2)
            try:
                with tf.device('/GPU:0'):
                    # Create random tensors
                    a = tf.random.normal([5000, 5000])
                    b = tf.random.normal([5000, 5000])

                    # Warmup
                    c = tf.matmul(a, b)
                    _ = c.numpy()

                    # Timed run
                    start_time = time.time()
                    c = tf.matmul(a, b)
                    _ = c.numpy()  # Force execution
                    basic_op_time = time.time() - start_time

                    # Calculate performance
                    ops = 2 * 5000**3  # Approximate FLOPs for matrix multiplication
                    gflops = ops / (basic_op_time * 1e9)

                    tf_results["basic_op_time"] = basic_op_time
                    tf_results["basic_op_gflops"] = gflops
                    # Record which device was actually used
                    tf_results["device_used"] = c.device

                    print(
                        f"Matrix multiplication time: {basic_op_time:.4f} seconds")
                    print(f"Estimated performance: {gflops:.2f} GFLOPS")
                    print(f"Executed on device: {c.device}")

                    del a, b, c
                    if hasattr(tf, 'keras'):
                        tf.keras.backend.clear_session()
            except Exception as e:
                print(f"Error in basic operations test: {e}")
                tf_results["basic_op_error"] = str(e)

            # CNN inference test
            print_section("CNN Inference Test", level=2)
            try:
                # Create a simple CNN model
                model = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(
                        32, 3, activation='relu', input_shape=(224, 224, 3)),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(64, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(128, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(10)
                ])

                # Compile model
                model.compile(optimizer='adam',
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                                  from_logits=True),
                              metrics=['accuracy'])

                # Create sample data
                batch_size = 16
                test_images = tf.random.normal([batch_size, 224, 224, 3])

                # Warmup
                _ = model(test_images, training=False)

                # Timed inference
                iterations = 50
                start_time = time.time()
                for _ in range(iterations):
                    _ = model(test_images, training=False)
                inference_time = (time.time() - start_time) / iterations

                inference_throughput = batch_size / inference_time

                tf_results["inference_time"] = inference_time
                tf_results["inference_throughput"] = inference_throughput

                print(f"Average inference time: {inference_time*1000:.2f} ms")
                print(
                    f"Inference throughput: {inference_throughput:.2f} images/sec")

                # Training test
                print_section("Training Test", level=2)

                # Generate random training data
                train_images = tf.random.normal([batch_size * 10, 224, 224, 3])
                train_labels = tf.random.uniform(
                    [batch_size * 10], minval=0, maxval=10, dtype=tf.int32)

                # Training setup
                train_dataset = tf.data.Dataset.from_tensor_slices(
                    (train_images, train_labels))
                train_dataset = train_dataset.shuffle(
                    buffer_size=1024).batch(batch_size)

                # Timed training
                start_time = time.time()
                history = model.fit(train_dataset, epochs=3, verbose=1)
                training_time = time.time() - start_time

                tf_results["training_time"] = training_time
                tf_results["training_loss"] = float(
                    history.history['loss'][-1])

                print(f"Training time: {training_time:.2f} seconds")
                print(f"Final loss: {history.history['loss'][-1]:.4f}")

                # Clean up
                del model, test_images, train_images, train_labels
                if hasattr(tf, 'keras'):
                    tf.keras.backend.clear_session()

            except Exception as e:
                print(f"Error in CNN test: {e}")
                tf_results["cnn_error"] = str(e)
        else:
            print("No GPU detected by TensorFlow")
            tf_results["status"] = "no_gpu"

    except ImportError as e:
        print(f"TensorFlow not available: {e}")
        tf_results["error"] = str(e)

    test_results["tensorflow"] = tf_results
    return tf_results

# =============================================================================
# TENSORRT TESTS
# =============================================================================


def get_tensorrt_version():
    """Get TensorRT version information using same approach as test_tensorrt.py"""
    print_section("TensorRT Version Information", level=2)

    try:
        # Direct import like in test_tensorrt.py
        import tensorrt as trt
        version_info = {"version": trt.__version__}
        print(f"TensorRT Version: {trt.__version__}")

        # Check TensorRT availability in TensorFlow
        try:
            from tensorflow.python.compiler.tensorrt import trt_convert
            version_info["tf_integration"] = True
            print("TensorRT integration in TensorFlow: Available")
            print(
                f"TF-TRT Version: {trt_convert.__version__ if hasattr(trt_convert, '__version__') else 'Unknown'}")
        except ImportError:
            version_info["tf_integration"] = False
            print("TensorRT integration in TensorFlow: Not available")

        # List key TensorRT components
        key_components = [
            'Builder', 'Logger', 'Runtime', 'DataType', 'TensorFormat',
            'ActivationType', 'LayerType', 'NetworkDefinitionCreationFlag'
        ]

        print("\nKey TensorRT Components:")
        version_info["components"] = {}
        for component in key_components:
            has_component = hasattr(trt, component)
            version_info["components"][component] = has_component
            print(f"  {'✓' if has_component else '✗'} {component}")

        return version_info

    except ImportError as e:
        print(f"TensorRT not available: {e}")
        return {"error": str(e)}


def test_tensorrt():
    """Test TensorRT integration with TensorFlow using the same approach as test_tensorrt.py"""
    print_section("TensorRT Integration Test")

    trt_results = {"status": "not_available"}

    try:
        # Direct imports like in test_tensorrt.py
        import tensorflow as tf
        import tensorrt as trt
        from tensorflow.python.compiler.tensorrt import trt_convert as trt_convert

        # Get TensorRT version information
        version_info = get_tensorrt_version()
        trt_results.update(version_info)
        trt_results["status"] = "available"

        # Check for GPU
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("No GPU detected, skipping TensorRT tests")
            trt_results["status"] = "no_gpu"
            test_results["tensorrt"] = trt_results
            return trt_results

        # Create model for TensorRT conversion
        print_section("Creating Model", level=2)

        # Create a CNN model
        input_shape = (32, 32, 3)
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                32, 3, activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=['accuracy'])

        print("Model created successfully")

        # Convert to TensorRT
        print_section("TensorRT Conversion", level=2)

        try:
            # Create sample input data
            batch_size = 1
            input_data = np.random.random(
                (batch_size,) + input_shape).astype(np.float32)

            # Save the model
            model_dir = "/tmp/tensorrt_test_model"
            tf.saved_model.save(model, model_dir)
            print(f"Model saved to {model_dir}")

            # Set precision mode (FP16)
            precision_mode = "FP16"
            trt_precision_mode = trt_convert.TrtPrecisionMode.FP16

            # Configure TensorRT parameters
            conversion_params = trt_convert.TrtConversionParams(
                precision_mode=trt_precision_mode,
                max_workspace_size_bytes=8000000000,
                maximum_cached_engines=1,
                use_calibration=False
            )

            # Convert to TensorRT
            print(
                f"Starting TensorRT conversion with {precision_mode} precision...")
            start_time = time.time()
            converter = trt_convert.TrtGraphConverterV2(
                input_saved_model_dir=model_dir,
                conversion_params=conversion_params
            )

            converter.convert()
            conversion_time = time.time() - start_time
            trt_results["conversion_time"] = conversion_time
            print(f"Conversion completed in {conversion_time:.2f} seconds")

            # Save the converted model
            trt_model_dir = "/tmp/tensorrt_optimized_model"
            converter.save(trt_model_dir)
            print(f"TensorRT model saved to {trt_model_dir}")

            # Benchmark comparison
            print_section("Performance Benchmark", level=2)

            # Load the TensorRT model
            trt_model = tf.saved_model.load(trt_model_dir)
            trt_infer = trt_model.signatures['serving_default']

            # Get input tensor name
            input_name = list(
                trt_infer.structured_input_signature[1].keys())[0]

            # Warmup runs
            print("Performing warm-up runs...")
            for _ in range(10):
                model(input_data)
                trt_infer(**{input_name: tf.constant(input_data)})

            # Benchmark original model
            iterations = 100
            print(
                f"Benchmarking original TensorFlow model ({iterations} iterations)...")

            start_time = time.time()
            for _ in range(iterations):
                _ = model(input_data)
            tf_time = (time.time() - start_time) / iterations

            # Benchmark TensorRT model
            print(f"Benchmarking TensorRT model ({iterations} iterations)...")

            start_time = time.time()
            for _ in range(iterations):
                _ = trt_infer(**{input_name: tf.constant(input_data)})
            trt_time = (time.time() - start_time) / iterations

            # Calculate speedup
            speedup = tf_time / trt_time

            trt_results["tf_inference_time"] = tf_time
            trt_results["trt_inference_time"] = trt_time
            trt_results["speedup"] = speedup

            print(f"Original model average time: {tf_time*1000:.2f} ms")
            print(f"TensorRT model average time: {trt_time*1000:.2f} ms")
            print(f"TensorRT speedup: {speedup:.2f}x")

            if speedup > 1.0:
                print("TensorRT acceleration is working correctly!")
            else:
                print("TensorRT did not provide speed improvement. Check configuration.")

        except Exception as e:
            print(f"Error during TensorRT conversion or benchmarking: {e}")
            trt_results["error"] = str(e)

    except ImportError as e:
        print(
            f"TensorRT or TensorFlow-TensorRT integration not available: {e}")
        trt_results["error"] = str(e)

    test_results["tensorrt"] = trt_results
    return trt_results

# =============================================================================
# XGBOOST TESTS
# =============================================================================


def test_xgboost():
    """Test XGBoost GPU acceleration"""
    print_section("XGBoost GPU Test")

    xgb_results = {"status": "not_available"}

    try:
        import xgboost as xgb
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        xgb_results["version"] = xgb.__version__
        print(f"XGBoost version: {xgb.__version__}")

        # Check if GPU is available for XGBoost
        try:
            # Create a small test DMatrix
            X = np.random.rand(10, 5)
            y = np.random.randint(0, 2, 10)
            dtrain = xgb.DMatrix(X, label=y)

            # Try to train with GPU
            param = {"tree_method": "gpu_hist", "gpu_id": 0}
            bst = xgb.train(param, dtrain, num_boost_round=1)
            xgb_results["status"] = "available"
            print("GPU support: Available")

            # Generate a larger dataset for testing
            print_section("Generating Dataset", level=2)
            n_samples = 50000
            n_features = 20

            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=int(n_features * 0.7),
                n_redundant=int(n_features * 0.2),
                n_classes=2,
                random_state=42
            )

            # Split dataset
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            print(
                f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

            # Convert to DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)

            # Benchmark CPU vs GPU
            print_section("CPU vs GPU Benchmark", level=2)

            # Common parameters
            common_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'eta': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1
            }

            # Test CPU
            try:
                cpu_params = common_params.copy()
                cpu_params['tree_method'] = 'hist'

                print("Testing CPU training...")
                start_time = time.time()
                cpu_model = xgb.train(
                    cpu_params,
                    dtrain,
                    num_boost_round=100,
                    evals=[(dtest, 'test')],
                    verbose_eval=False
                )
                cpu_train_time = time.time() - start_time

                # Prediction time
                start_time = time.time()
                cpu_preds = cpu_model.predict(dtest)
                cpu_pred_time = time.time() - start_time

                # Metrics
                cpu_preds_binary = np.round(cpu_preds)
                cpu_accuracy = accuracy_score(y_test, cpu_preds_binary)

                xgb_results["cpu_train_time"] = cpu_train_time
                xgb_results["cpu_pred_time"] = cpu_pred_time
                xgb_results["cpu_accuracy"] = cpu_accuracy

                print(f"CPU training time: {cpu_train_time:.4f} seconds")
                print(f"CPU prediction time: {cpu_pred_time:.4f} seconds")
                print(f"CPU accuracy: {cpu_accuracy:.4f}")

            except Exception as e:
                print(f"Error testing CPU: {e}")
                xgb_results["cpu_error"] = str(e)

            # Test GPU
            try:
                gpu_params = common_params.copy()
                gpu_params['tree_method'] = 'gpu_hist'
                gpu_params['gpu_id'] = 0

                print("Testing GPU training...")
                start_time = time.time()
                gpu_model = xgb.train(
                    gpu_params,
                    dtrain,
                    num_boost_round=100,
                    evals=[(dtest, 'test')],
                    verbose_eval=False
                )
                gpu_train_time = time.time() - start_time

                # Prediction time
                start_time = time.time()
                gpu_preds = gpu_model.predict(dtest)
                gpu_pred_time = time.time() - start_time

                # Metrics
                gpu_preds_binary = np.round(gpu_preds)
                gpu_accuracy = accuracy_score(y_test, gpu_preds_binary)

                xgb_results["gpu_train_time"] = gpu_train_time
                xgb_results["gpu_pred_time"] = gpu_pred_time
                xgb_results["gpu_accuracy"] = gpu_accuracy

                print(f"GPU training time: {gpu_train_time:.4f} seconds")
                print(f"GPU prediction time: {gpu_pred_time:.4f} seconds")
                print(f"GPU accuracy: {gpu_accuracy:.4f}")

                # Compare speed
                if xgb_results.get("cpu_train_time"):
                    train_speedup = xgb_results["cpu_train_time"] / \
                        gpu_train_time
                    pred_speedup = xgb_results["cpu_pred_time"] / gpu_pred_time

                    xgb_results["train_speedup"] = train_speedup
                    xgb_results["pred_speedup"] = pred_speedup

                    print(f"Training speedup: {train_speedup:.2f}x")
                    print(f"Prediction speedup: {pred_speedup:.2f}x")

            except Exception as e:
                print(f"Error testing GPU: {e}")
                xgb_results["gpu_error"] = str(e)

        except Exception as e:
            print(f"GPU support: Not available")
            print(f"Error: {e}")
            xgb_results["status"] = "no_gpu"
            xgb_results["error"] = str(e)

    except ImportError as e:
        print(f"XGBoost not available: {e}")
        xgb_results["error"] = str(e)

    test_results["xgboost"] = xgb_results
    return xgb_results

# =============================================================================
# CUPY TESTS
# =============================================================================


def test_cupy():
    """Test CuPy GPU acceleration"""
    print_section("CuPy GPU Test")

    cupy_results = {"status": "not_available"}

    try:
        import cupy as cp

        cupy_results["version"] = cp.__version__
        print(f"CuPy version: {cp.__version__}")

        # Check CUDA availability
        cuda_available = cp.cuda.is_available()
        cupy_results["cuda_available"] = cuda_available
        print(f"CUDA available: {cuda_available}")

        if not cuda_available:
            print("CUDA not available, skipping CuPy tests")
            cupy_results["status"] = "no_gpu"
            test_results["cupy"] = cupy_results
            return cupy_results

        # Get CUDA details
        cuda_version = cp.cuda.runtime.runtimeGetVersion()
        cupy_results["cuda_version"] = cuda_version
        print(f"CUDA version: {cuda_version}")

        gpu_count = cp.cuda.runtime.getDeviceCount()
        cupy_results["gpu_count"] = gpu_count
        print(f"GPU count: {gpu_count}")

        if gpu_count > 0:
            # Get device properties
            props = cp.cuda.runtime.getDeviceProperties(0)
            cupy_results["gpu_name"] = props['name'].decode('utf-8')
            cupy_results["gpu_compute_capability"] = f"{props['major']}.{props['minor']}"
            cupy_results["gpu_memory"] = props['totalGlobalMem']

            print(f"GPU name: {props['name'].decode('utf-8')}")
            print(f"Compute capability: {props['major']}.{props['minor']}")
            print(
                f"Total memory: {props['totalGlobalMem'] / (1024**3):.2f} GB")

            cupy_results["status"] = "available"

            # Matrix multiplication benchmark
            print_section("Matrix Multiplication Benchmark", level=2)

            # Perform multiple sizes for benchmark
            sizes = [1000, 2000, 5000]

            for size in sizes:
                print(f"\nMatrix size: {size}x{size}")

                # Create matrices on GPU
                try:
                    a = cp.random.random((size, size), dtype=cp.float32)
                    b = cp.random.random((size, size), dtype=cp.float32)

                    # Warm up
                    c = cp.matmul(a, b)
                    cp.cuda.Stream.null.synchronize()

                    # Benchmark
                    start_time = time.time()
                    c = cp.matmul(a, b)
                    cp.cuda.Stream.null.synchronize()
                    elapsed = time.time() - start_time

                    # Calculate performance
                    gflops = 2 * size**3 / elapsed / 1e9

                    # Store results
                    cupy_results[f"matmul_{size}_time"] = elapsed
                    cupy_results[f"matmul_{size}_gflops"] = gflops

                    print(
                        f"Matrix multiplication completed in {elapsed:.4f} seconds")
                    print(f"Performance: {gflops:.2f} GFLOPS")

                    # Clean up to avoid memory issues with large matrices
                    del a, b, c
                    cp.get_default_memory_pool().free_all_blocks()

                except Exception as e:
                    print(f"Error in matrix multiplication (size {size}): {e}")
                    cupy_results[f"matmul_{size}_error"] = str(e)

            # Element-wise operations benchmark
            print_section("Element-wise Operations Benchmark", level=2)

            try:
                size = 50000000  # 50M elements
                print(f"Array size: {size} elements")

                # Create arrays on GPU
                a = cp.random.random(size, dtype=cp.float32)
                b = cp.random.random(size, dtype=cp.float32)

                # Warm up
                c = cp.sin(a) * cp.sqrt(b) + cp.cos(a)
                cp.cuda.Stream.null.synchronize()

                # Benchmark
                start_time = time.time()
                c = cp.sin(a) * cp.sqrt(b) + cp.cos(a)
                cp.cuda.Stream.null.synchronize()
                elapsed = time.time() - start_time

                # Calculate throughput
                throughput = size / elapsed / 1e9

                cupy_results["elementwise_time"] = elapsed
                cupy_results["elementwise_throughput"] = throughput

                print(
                    f"Element-wise operations completed in {elapsed:.4f} seconds")
                print(f"Throughput: {throughput:.2f} billion elements/second")

                # Clean up
                del a, b, c
                cp.get_default_memory_pool().free_all_blocks()

            except Exception as e:
                print(f"Error in element-wise operations: {e}")
                cupy_results["elementwise_error"] = str(e)

            # Memory transfer benchmark
            print_section("Memory Transfer Benchmark", level=2)

            try:
                size = 100000000  # 100M elements (400MB for float32)
                print(f"Array size: {size} elements ({size*4/1e6:.1f} MB)")

                # Create array on CPU
                a_cpu = np.random.random(size).astype(np.float32)

                # GPU to CPU transfer (Device to Host)
                start_time = time.time()
                a_gpu = cp.array(a_cpu)
                cp.cuda.Stream.null.synchronize()
                h2d_time = time.time() - start_time
                h2d_bandwidth = size * 4 / h2d_time / 1e9  # GB/s

                # CPU to GPU transfer (Host to Device)
                start_time = time.time()
                a_cpu_back = cp.asnumpy(a_gpu)
                d2h_time = time.time() - start_time
                d2h_bandwidth = size * 4 / d2h_time / 1e9  # GB/s

                cupy_results["h2d_time"] = h2d_time
                cupy_results["h2d_bandwidth"] = h2d_bandwidth
                cupy_results["d2h_time"] = d2h_time
                cupy_results["d2h_bandwidth"] = d2h_bandwidth

                print(
                    f"Host to Device transfer: {h2d_time:.4f} seconds ({h2d_bandwidth:.2f} GB/s)")
                print(
                    f"Device to Host transfer: {d2h_time:.4f} seconds ({d2h_bandwidth:.2f} GB/s)")

                # Clean up
                del a_cpu, a_gpu, a_cpu_back
                cp.get_default_memory_pool().free_all_blocks()

            except Exception as e:
                print(f"Error in memory transfer benchmark: {e}")
                cupy_results["memory_transfer_error"] = str(e)

        else:
            print("No CUDA devices found")
            cupy_results["status"] = "no_gpu"

    except ImportError as e:
        print(f"CuPy not available: {e}")
        cupy_results["error"] = str(e)

    test_results["cupy"] = cupy_results
    return cupy_results

# =============================================================================
# MAIN FUNCTION
# =============================================================================


def main():
    """Run all GPU acceleration tests"""
    print_section("Comprehensive GPU Acceleration Test Suite")
    print(f"Test started at: {datetime.now().isoformat()}")

    # Get system information
    system_info = get_system_info()

    # Initialize results
    all_tests_success = True
    all_tests_run = True

    # Run tests with proper error handling to ensure all tests run
    print_section("Running GPU Acceleration Tests", level=2)

    try:
        # 1. TensorFlow Test
        tf_results = test_tensorflow()
        all_tests_success = all_tests_success and (
            tf_results.get("status") == "available")
    except Exception as e:
        logger.error(f"Error in TensorFlow tests: {e}")
        test_results["tensorflow"]["error"] = str(e)
        all_tests_run = False

    try:
        # 2. TensorRT Test
        trt_results = test_tensorrt()
        all_tests_success = all_tests_success and (
            trt_results.get("status") == "available")
    except Exception as e:
        logger.error(f"Error in TensorRT tests: {e}")
        test_results["tensorrt"]["error"] = str(e)
        all_tests_run = False

    try:
        # 3. XGBoost Test
        xgb_results = test_xgboost()
        all_tests_success = all_tests_success and (
            xgb_results.get("status") == "available")
    except Exception as e:
        logger.error(f"Error in XGBoost tests: {e}")
        test_results["xgboost"]["error"] = str(e)
        all_tests_run = False

    try:
        # 4. CuPy Test
        cupy_results = test_cupy()
        all_tests_success = all_tests_success and (
            cupy_results.get("status") == "available")
    except Exception as e:
        logger.error(f"Error in CuPy tests: {e}")
        test_results["cupy"]["error"] = str(e)
        all_tests_run = False

    # Generate summary
    test_results["summary"]["all_tests_success"] = all_tests_success
    test_results["summary"]["all_tests_run"] = all_tests_run
    test_results["summary"]["timestamp"] = datetime.now().isoformat()

    # Print final results
    print_section("GPU Acceleration Test Results")

    print("Component Status Summary:")
    components = ['tensorflow', 'tensorrt', 'xgboost', 'cupy']

    for component in components:
        status = test_results[component].get("status", "unknown")
        status_display = "✅ Available" if status == "available" else "❌ Not Available"
        print(f"{component.ljust(10)}: {status_display}")

    # Print performance summary if available
    print("\nPerformance Highlights:")

    if test_results["tensorflow"].get("status") == "available":
        if "inference_throughput" in test_results["tensorflow"]:
            print(
                f"TensorFlow CNN Inference: {test_results['tensorflow']['inference_throughput']:.2f} images/sec")

    if test_results["tensorrt"].get("status") == "available":
        if "speedup" in test_results["tensorrt"]:
            print(
                f"TensorRT Acceleration: {test_results['tensorrt']['speedup']:.2f}x speedup")

    if test_results["xgboost"].get("status") == "available":
        if "train_speedup" in test_results["xgboost"]:
            print(
                f"XGBoost GPU Speedup: {test_results['xgboost']['train_speedup']:.2f}x training, {test_results['xgboost']['pred_speedup']:.2f}x prediction")

    if test_results["cupy"].get("status") == "available":
        if "matmul_5000_gflops" in test_results["cupy"]:
            print(
                f"CuPy Matrix Multiply: {test_results['cupy']['matmul_5000_gflops']:.2f} GFLOPS")
        if "h2d_bandwidth" in test_results["cupy"] and "d2h_bandwidth" in test_results["cupy"]:
            print(
                f"CuPy Memory Transfer: {test_results['cupy']['h2d_bandwidth']:.2f} GB/s (H→D), {test_results['cupy']['d2h_bandwidth']:.2f} GB/s (D→H)")

    # Write results to file
    results_file = "gpu_acceleration_test_results.json"
    try:
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"\nDetailed results saved to {results_file}")
    except Exception as e:
        print(f"Error saving results to file: {e}")

    print("\nTest completed successfully!" if all_tests_run else "\nSome tests failed to run completely. Check logs for details.")
    print_section("END OF TEST")


if __name__ == "__main__":
    main()
