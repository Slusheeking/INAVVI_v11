#!/usr/bin/env python3
"""
ML Model Trainer Runner Script
This script runs the ML model trainer with proper module imports
"""

import os
import sys
import time
import redis
import logging
import traceback
import argparse
import platform
import dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ml_trainer_runner')

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    dotenv.load_dotenv(env_path)
    logger.info(f"Loaded environment variables from {env_path}")

# Try to import the Slack reporting module
try:
    from slack_reporter import SlackReporter, GPUStatsTracker
    SLACK_AVAILABLE = True
    logger.info("Slack reporting module available")
except ImportError:
    SLACK_AVAILABLE = False
    logger.warning("Slack reporting module not available. Continuing without Slack notifications.")

def set_gpu_environment_variables(use_gh200=True):
    """Set environment variables for optimal GPU performance"""
    # TensorFlow GPU optimization
    # Settings optimized for NVIDIA GH200 Grace Hopper Superchip
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
    os.environ['TF_CUDA_HOST_MEM_LIMIT_IN_MB'] = '80000'  # Optimized for GH200's 80GB memory
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = '8'
    os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '32'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    os.environ['TF_USE_CUDA_GRAPHS'] = '0'  # Disable CUDA graphs which can cause issues with data processing
    os.environ['TF_CUDNN_USE_AUTOTUNE'] = '1'
    os.environ['TF_LAYOUT_OPTIMIZER_DISABLE'] = '1'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Add LD_LIBRARY_PATH for NVIDIA libraries
    if 'LD_LIBRARY_PATH' in os.environ:
        os.environ['LD_LIBRARY_PATH'] = f"/usr/local/nvidia/lib:/usr/local/nvidia/lib64:{os.environ['LD_LIBRARY_PATH']}"
    else:
        os.environ['LD_LIBRARY_PATH'] = "/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
        
    # Set verbose logging for debugging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    # Set CUDA device order
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    # NVIDIA visibility
    os.environ['NVIDIA_VISIBLE_DEVICES'] = 'all'
    os.environ['NVIDIA_DRIVER_CAPABILITIES'] = 'compute,utility'
    
    # XLA flags for better performance
    if platform.system() == 'Linux':
        os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
    
    # GH200-specific optimizations
    if use_gh200:
        # Enable TF32 computation for GH200
        os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
        
        # For ARM CPU side of GH200
        os.environ["GOMP_CPU_AFFINITY"] = "0-15"
        
        # Optimize memory transfer
        os.environ["CUDA_AUTO_BOOST"] = "1"
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "8"
        
        # NVLink optimizations
        os.environ["NCCL_IB_DISABLE"] = "0"
        os.environ["NCCL_P2P_LEVEL"] = "NVL"
        
        # GH200 compute capability
        os.environ["TF_CUDA_COMPUTE_CAPABILITIES"] = "9.0"
        
        logger.info("Applied GH200-specific optimizations")
    
    logger.info("Environment variables set for optimal GPU performance")

# Parse command line arguments
parser = argparse.ArgumentParser(description='ML Model Trainer Runner')
parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
parser.add_argument('--no-slack', action='store_true', help='Disable Slack reporting')
parser.add_argument('--diagnostics-only', action='store_true', help='Run diagnostics only, no training')
args = parser.parse_args()

# Override environment variables based on command line args
if args.no_gpu:
    os.environ['USE_GPU'] = 'false'
if args.no_slack:
    os.environ['USE_SLACK_REPORTING'] = 'false'

# Set GPU environment variables
use_gh200 = os.environ.get('USE_GH200', 'true').lower() == 'true'
if use_gh200:
    logger.info("Enabling GH200-specific optimizations")
set_gpu_environment_variables(use_gh200=use_gh200)

try:
    # Import required modules
    # Use direct imports instead of package imports
    sys.path.insert(0, os.path.join(project_root, 'tests'))
    
    # Import the modules directly
    import ml_model_trainer
    import data_pipeline_integration
    
    MLModelTrainer = ml_model_trainer.MLModelTrainer
    DataPipelineIntegration = data_pipeline_integration.DataPipelineIntegration
    
    # Initialize Slack reporting if enabled
    slack_reporter = None
    gpu_tracker = None
    if SLACK_AVAILABLE and os.environ.get('USE_SLACK_REPORTING', 'true').lower() == 'true' and not args.no_slack:
        webhook_url = os.environ.get('SLACK_WEBHOOK_URL', '')
        bot_token = os.environ.get('SLACK_BOT_TOKEN', '')
        if webhook_url or bot_token:
            slack_reporter = SlackReporter(webhook_url=webhook_url, bot_token=bot_token)
            gpu_tracker = GPUStatsTracker(polling_interval=10.0)
            logger.info("Initialized Slack reporting")
    
    # Check for NVIDIA GPU using nvidia-smi
    nvidia_gpu_available = False
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            # Check if GH200 is detected
            if "GH200" in result.stdout:
                logger.info("NVIDIA GH200 GPU detected!")
                if not use_gh200:
                    logger.warning("GH200 detected but GH200-specific optimizations are disabled")
                    logger.warning("Consider enabling GH200 optimizations for better performance")
            else:
                logger.info("NVIDIA GPU detected (not GH200)")
                
            logger.info(result.stdout)
            nvidia_gpu_available = True
            
            # Report to Slack if available
            if slack_reporter:
                gpu_name = "Unknown"
                for line in result.stdout.splitlines():
                    if "NVIDIA" in line and "GB" in line:
                        gpu_name = line.strip()
                slack_reporter.send_message(f"🔍 *GPU Detected*\nGPU: {gpu_name}")
        else:
            logger.warning("nvidia-smi command failed with error:")
            logger.warning(result.stderr)
    except FileNotFoundError:
        logger.warning("nvidia-smi command not found. NVIDIA drivers may not be installed.")
    except Exception as e:
        logger.error(f"Error running nvidia-smi: {e}")
    
    # Run diagnostics only if requested
    if args.diagnostics_only:
        logger.info("Running diagnostics only mode")
        ml_model_trainer.run_diagnostics()
        sys.exit(0)
    
    # Configure TensorFlow for GPU
    import tensorflow as tf
    
    # Set TensorFlow logging level to reduce noise
    tf.get_logger().setLevel('ERROR')
    
    # Get all physical devices
    physical_devices = tf.config.list_physical_devices()
    logger.info(f"All physical devices: {physical_devices}")
    
    # Get GPU devices
    gpus = tf.config.list_physical_devices('GPU')

    # Check if CuPy can access the GPU
    if not gpus:
        logger.warning(f"No GPU detected by TensorFlow initially. Attempting manual configuration for NVIDIA {'GH200' if use_gh200 else 'GPU'}...")
        
        # Additional environment variables specifically for GH200
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['TF_CUDA_COMPUTE_CAPABILITIES'] = '9.0'  # For GH200 Grace Hopper
        os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
        
        # Try to manually initialize TensorFlow with GPU
        try:
            # Create a TensorFlow session with explicit GPU configuration
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.visible_device_list = "0"
            config.log_device_placement = True
            tf.compat.v1.Session(config=config)
            
            # Check again for GPUs
            # Try to import CuPy as an alternative way to verify CUDA access
            try:
                import cupy as cp
                logger.info(f"CuPy version: {cp.__version__}")
                if cp.cuda.is_available():
                    logger.info("CUDA is available through CuPy")
            except ImportError:
                logger.warning("CuPy not installed, skipping CuPy GPU check")
                
            # Check if CuPy can access the GPU
            cupy_gpu_available = False
            try:
                import cupy as cp
                if cp.cuda.is_available():
                    cupy_gpu_available = True
                    logger.info(f"CUDA is available through CuPy version {cp.__version__}")
                    logger.info(f"CuPy CUDA device: {cp.cuda.Device().id}")
                    
                    # Get device properties to check for GH200
                    device_props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
                    device_name = device_props["name"].decode()
                    
                    if "GH200" in device_name:
                        logger.info(f"CuPy detected GH200 device: {device_name}")
                        # Get memory info
                        free, total = cp.cuda.runtime.memGetInfo()
                        logger.info(f"CuPy CUDA memory: {free/(1024**3):.2f}GB free / {total/(1024**3):.2f}GB total")
                    else:
                        logger.info(f"CuPy detected GPU device: {device_name}")
                        logger.info(f"CuPy CUDA memory: {cp.cuda.Device().mem_info}")
            except ImportError:
                logger.warning("CuPy not installed, cannot use GPU through CuPy")
                
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"Successfully detected GPU after manual configuration: {gpus}")
            elif not cupy_gpu_available:
                logger.error("No GPU detected by TensorFlow after manual configuration.")
                logger.error("Please ensure NVIDIA drivers and CUDA are properly installed.")
                if nvidia_gpu_available:
                    logger.error("NVIDIA GPU was detected by nvidia-smi but not by TensorFlow.")
                    logger.error("This indicates a TensorFlow-CUDA compatibility issue. Exiting.")
                logger.error("Exiting as GPU is mandatory for this application.")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Error during manual GPU configuration: {e}")
            sys.exit(1)
    
    if gpus and os.environ.get('USE_GPU', 'true').lower() == 'true':
        # TensorFlow can access the GPU
        os.environ['TF_GPU_AVAILABLE'] = 'true'
        os.environ['USE_TENSORFLOW_GPU'] = 'true'
        
        # Log GPU information
        logger.info(f"TensorFlow can access {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            logger.info(f"  GPU {i}: {gpu}")
        
        # Configure memory growth for all GPUs
        try:
            for gpu in gpus:
                try:
                    # First try to set memory growth
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"Memory growth enabled for {gpu}")
                except Exception as e:
                    logger.warning(f"Unable to set memory growth for {gpu}: {e}")
                    
                # Then try to set virtual device configuration
                try:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)]
                    )
                    logger.info(f"Virtual device configuration set for {gpu}")
                except Exception as e:
                    logger.warning(f"Unable to set virtual device configuration for {gpu}: {e}")
        except RuntimeError as e:
            logger.error(f"Error configuring GPU: {e}")
        
        # Run a simple test operation
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
                c = tf.matmul(a, b)
                result = c.numpy()
            logger.info(f"Test matrix multiplication on GPU successful: {result}")
        except Exception as e:
            logger.warning(f"Test operation on GPU failed: {e}")
    else:
        # TensorFlow cannot access the GPU
        os.environ['TF_GPU_AVAILABLE'] = 'false'
        os.environ['USE_TENSORFLOW_GPU'] = 'false'
        
        # Check if CuPy can access the GPU
        if 'cupy_gpu_available' in locals() and cupy_gpu_available:
            logger.warning("No GPU detected by TensorFlow, but CuPy can access the GPU")
            logger.info("Will use CuPy for GPU operations and TensorFlow on CPU")
        else:
            logger.warning("No GPU detected by TensorFlow or CuPy")
            logger.info("All computations will use CPU")
    
    # Create Redis client with retry logic
    redis_client = None
    try:
        redis_client = redis.Redis(
            host=os.environ.get('REDIS_HOST', 'localhost'),
            port=int(os.environ.get('REDIS_PORT', 6380)),
            db=int(os.environ.get('REDIS_DB', 0)),
            socket_timeout=5,
            socket_connect_timeout=5
        )
        redis_client.ping()
        logger.info(f'Connected to Redis at {os.environ.get("REDIS_HOST", "localhost")}:{os.environ.get("REDIS_PORT", 6380)}')
    except Exception as e:
        logger.warning(f'Redis connection failed: {e}')
        logger.info('Creating a mock Redis client for testing')
        
        # Create a mock Redis client for testing
        class MockRedis:
            def __init__(self):
                self.data = {}
                logger.info('Using MockRedis for testing')
            
            def set(self, key, value):
                self.data[key] = value
                return True
            
            def get(self, key):
                return self.data.get(key)
            
            def hset(self, name, key, value):
                if name not in self.data:
                    self.data[name] = {}
                self.data[name][key] = value
                return True
            
            def hget(self, name, key):
                if name in self.data and key in self.data[name]:
                    return self.data[name][key]
                return None
            
            def hgetall(self, name):
                return self.data.get(name, {})
            
            def sadd(self, name, *values):
                if name not in self.data:
                    self.data[name] = set()
                for value in values:
                    self.data[name].add(value)
                return len(values)
            
            def smembers(self, name):
                return self.data.get(name, set())
            
            def zrange(self, name, start, end, withscores=False):
                return []
            
            def publish(self, channel, message):
                return 0
            
            def ping(self):
                return True
        
        redis_client = MockRedis()
    
    # Create data loader
    try:
        data_loader = DataPipelineIntegration(
            redis_host=os.environ.get('REDIS_HOST', 'redis'),
            redis_port=int(os.environ.get('REDIS_PORT', 6380)),
            redis_db=int(os.environ.get('REDIS_DB', 0)),
            polygon_api_key=os.environ.get('POLYGON_API_KEY', ''),
            unusual_whales_api_key=os.environ.get('UNUSUAL_WHALES_API_KEY', ''),
            use_gpu=os.environ.get('USE_GPU', 'true').lower() == 'true',
            use_gh200=use_gh200
        )
        logger.info('Successfully created DataPipelineIntegration')
    except Exception as e:
        logger.error(f'Error creating DataPipelineIntegration: {e}')
        logger.info('Attempting to import GPU-optimized data clients directly')
        
        try:
            import gpu_optimized_polygon_api_client
            import gpu_optimized_unusual_whales_client
            
            # Create a direct data loader using the GPU-optimized clients
            data_loader = DataPipelineIntegration(
                redis_host=os.environ.get('REDIS_HOST', 'redis'),
                redis_port=int(os.environ.get('REDIS_PORT', 6380)),
                redis_db=int(os.environ.get('REDIS_DB', 0)),
                polygon_api_key=os.environ.get('POLYGON_API_KEY', ''),
                unusual_whales_api_key=os.environ.get('UNUSUAL_WHALES_API_KEY', ''),
                use_gpu=os.environ.get('USE_GPU', 'true').lower() == 'true',
                use_gh200=use_gh200
            )
            logger.info('Successfully created DataPipelineIntegration with direct clients')
        except Exception as e:
            logger.error(f'Fatal error: Could not create data loader: {e}')
            sys.exit(1)
    
    # Start GPU tracking if available
    if gpu_tracker:
        gpu_tracker.start()
    
    # Create model trainer with custom directories
    try:
        trainer = MLModelTrainer(redis_client, data_loader)
        trainer.config['models_dir'] = os.environ.get('MODELS_DIR', './models')
        trainer.config['data_dir'] = os.environ.get('DATA_DIR', './data')
        logger.info(f'Successfully created MLModelTrainer with models_dir={trainer.config["models_dir"]} and data_dir={trainer.config["data_dir"]}')
        
        # Set Slack reporter if available
        if slack_reporter:
            trainer.slack_reporter = slack_reporter
            trainer.gpu_tracker = gpu_tracker
        
        # Train all models
        training_start_time = time.time()
        logger.info('Starting model training...')
        
        # Report training start to Slack if available
        if slack_reporter:
            slack_reporter.send_message("🚀 ML Trainer starting model training...")
        
        success = trainer.train_all_models()
        training_duration = time.time() - training_start_time
        
        # Calculate hours, minutes, seconds
        hours, remainder = divmod(training_duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f'Model training completed with success={success} in {int(hours)}h {int(minutes)}m {int(seconds)}s')
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f'Error in model training: {e}')
        traceback.print_exc()
        
        # Log the error but exit with error code
        logger.error('Exiting with error code 1')
        
        # Report error to Slack if available
        if slack_reporter:
            slack_reporter.report_error(f"Error in model training: {str(e)}", traceback.format_exc(), "training")
        
        # Stop GPU tracking if available
        if gpu_tracker: gpu_tracker.stop()
        sys.exit(1)

except Exception as e:
    logger.error(f'Unhandled exception: {e}')
    # Stop GPU tracking if available
    if 'gpu_tracker' in locals() and gpu_tracker:
        gpu_tracker.stop()
    traceback.print_exc()
    sys.exit(1)