#!/bin/bash
# Run ML Model Trainer
# This script starts the ML model trainer with NVIDIA GH200 GPU support

# Set default values
REDIS_HOST=${REDIS_HOST:-"localhost"}
REDIS_PORT=${REDIS_PORT:-6380}  # Using port 6380 as specified in .env file
REDIS_DB=${REDIS_DB:-0}
USE_GPU=${USE_GPU:-"true"}
MODELS_DIR=${MODELS_DIR:-"./models"}
DATA_DIR=${DATA_DIR:-"./data"}
LOG_LEVEL=${LOG_LEVEL:-"INFO"}
USE_GH200=${USE_GH200:-"true"}
USE_SLACK_REPORTING=${USE_SLACK_REPORTING:-"true"}
SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL:-""}
SLACK_BOT_TOKEN=${SLACK_BOT_TOKEN:-""}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --redis-host)
      REDIS_HOST="$2"
      shift 2
      ;;
    --redis-port)
      REDIS_PORT="$2"
      shift 2
      ;;
    --redis-db)
      REDIS_DB="$2"
      shift 2
      ;;
    --models-dir)
      MODELS_DIR="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --no-gpu)
      USE_GPU="false"
      shift
      ;;
    --no-gh200)
      USE_GH200="false"
      shift
      ;;
    --no-slack)
      USE_SLACK_REPORTING="false"
      shift
      ;;
    --diagnostics-only)
      DIAGNOSTICS_ONLY="true"
      shift
      ;;
    --slack-webhook)
      SLACK_WEBHOOK_URL="$2"
      shift 2
      ;;
    --log-level)
      LOG_LEVEL="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --redis-host HOST     Redis host (default: localhost)"
      echo "  --redis-port PORT     Redis port (default: 6380)"
      echo "  --redis-db DB         Redis database (default: 0)"
      echo "  --models-dir DIR      Directory to store models (default: ./models)"
      echo "  --data-dir DIR        Directory to store data (default: ./data)"
      echo "  --no-gpu              Disable GPU acceleration"
      echo "  --no-gh200            Disable GH200-specific optimizations"
      echo "  --no-slack            Disable Slack reporting"
      echo "  --diagnostics-only    Run diagnostics only, no training"
      echo "  --slack-webhook URL   Set Slack webhook URL for notifications"
      echo "  --log-level LEVEL     Set logging level (default: INFO)"
      echo "  --help                Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Load environment variables from .env file if it exists
if [ -f .env ]; then
  echo "Loading environment variables from .env file"
  export $(grep -v '^#' .env | xargs)
fi

# Check for required API keys
if [ -z "$POLYGON_API_KEY" ]; then
  echo "Error: POLYGON_API_KEY is not set"
  echo "Please set it in the .env file or as an environment variable"
  exit 1
fi

# Check for GPU support if enabled
if [ "$USE_GPU" = "true" ]; then
  # Check if CUDA is available
  if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected"
    nvidia_output=$(nvidia-smi)
    echo "$nvidia_output"
    
    # Check if GH200 is detected
    if echo "$nvidia_output" | grep -q "GH200"; then
      echo "NVIDIA GH200 GPU detected!"
      USE_GH200="true"
    fi
  else
    echo "Error: NVIDIA GPU not detected but USE_GPU=true"
    echo "GPU is required for this application. Exiting."
    exit 1
  fi
  
  # Check if TensorFlow can see the GPU
  echo "Checking TensorFlow GPU support..."
  python3 -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU')); print('TensorFlow version:', tf.__version__)"
fi

# Create directories if they don't exist
mkdir -p "$MODELS_DIR"
mkdir -p "$DATA_DIR"

# Set environment variables
export REDIS_HOST=$REDIS_HOST
export REDIS_PORT=$REDIS_PORT
export REDIS_DB=$REDIS_DB
export USE_GPU=$USE_GPU
export MODELS_DIR=$MODELS_DIR
export DATA_DIR=$DATA_DIR
export LOG_LEVEL=$LOG_LEVEL
export USE_GH200=$USE_GH200
export USE_SLACK_REPORTING=$USE_SLACK_REPORTING
export SLACK_WEBHOOK_URL=$SLACK_WEBHOOK_URL
export SLACK_BOT_TOKEN=$SLACK_BOT_TOKEN

# Set additional GPU optimization environment variables
export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
export TF_CUDA_HOST_MEM_LIMIT_IN_MB=80000
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=8
export CUDA_DEVICE_MAX_CONNECTIONS=32
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_USE_CUDA_GRAPHS=0
export TF_CUDNN_USE_AUTOTUNE=1
export TF_LAYOUT_OPTIMIZER_DISABLE=1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Set NVIDIA environment variables
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility

# GH200-specific optimizations
if [ "$USE_GH200" = "true" ]; then
  echo "Applying GH200-specific optimizations..."
  
  # Enable TF32 computation for GH200
  export NVIDIA_TF32_OVERRIDE=1
  
  # For ARM CPU side of GH200
  export GOMP_CPU_AFFINITY="0-15"
  
  # Optimize memory transfer
  export CUDA_AUTO_BOOST=1
  export CUDA_DEVICE_MAX_CONNECTIONS=8
  
  # NVLink optimizations
  export NCCL_IB_DISABLE=0
  export NCCL_P2P_LEVEL=NVL
fi

echo "Starting ML Model Trainer..."
echo "Redis: $REDIS_HOST:$REDIS_PORT/$REDIS_DB"
echo "GPU Acceleration: $USE_GPU (GH200 optimizations: $USE_GH200)"
echo "Slack Reporting: $USE_SLACK_REPORTING"
echo "Models Directory: $MODELS_DIR"
echo "Data Directory: $DATA_DIR"
echo "Log Level: $LOG_LEVEL"

# Build command with appropriate flags
CMD="python3 run_ml_trainer.py"

if [ "$USE_GPU" = "false" ]; then
  CMD="$CMD --no-gpu"
fi

if [ "$USE_SLACK_REPORTING" = "false" ]; then
  CMD="$CMD --no-slack"
fi

if [ "$DIAGNOSTICS_ONLY" = "true" ]; then
  CMD="$CMD --diagnostics-only"
fi

# Run the ML trainer using the Python script
echo "Running ML trainer: $CMD"
$CMD