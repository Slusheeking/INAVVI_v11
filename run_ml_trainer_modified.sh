#!/bin/bash
# Modified ML Model Trainer script
# This script starts the ML model trainer with GPU support
# Modified to use Redis on port 6380 or run without Redis

# Set default values
REDIS_HOST=${REDIS_HOST:-"localhost"}
REDIS_PORT=${REDIS_PORT:-6380}  # Changed to 6380
REDIS_DB=${REDIS_DB:-0}
USE_GPU=${USE_GPU:-"true"}
MODELS_DIR=${MODELS_DIR:-"./models"}
DATA_DIR=${DATA_DIR:-"./data"}
LOG_LEVEL=${LOG_LEVEL:-"INFO"}

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
if [ -f tests/.env ]; then
  echo "Loading environment variables from tests/.env file"
  export $(grep -v '^#' tests/.env | xargs)
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
    nvidia-smi
  else
    echo "Warning: NVIDIA GPU not detected but USE_GPU=true"
    echo "Continuing with CPU-only mode"
    USE_GPU="false"
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

echo "Starting ML Model Trainer..."
echo "Redis: $REDIS_HOST:$REDIS_PORT/$REDIS_DB"
echo "GPU Acceleration: $USE_GPU"
echo "Models Directory: $MODELS_DIR"
echo "Data Directory: $DATA_DIR"
echo "Log Level: $LOG_LEVEL"

# Run the ML trainer directly from the tests directory
cd $(dirname "$0")
python3 -c "
import os
import redis
import sys
import importlib.util

# Import ml_model_trainer directly
spec = importlib.util.spec_from_file_location('ml_model_trainer', 'tests/ml_model_trainer.py')
ml_model_trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml_model_trainer)

# Import data_pipeline_integration directly
spec = importlib.util.spec_from_file_location('data_pipeline_integration', 'tests/data_pipeline_integration.py')
data_pipeline_integration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_pipeline_integration)

# Configure GPU memory growth to avoid OOM errors
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus and os.environ.get('USE_GPU', 'true').lower() == 'true':
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'GPU memory growth enabled for {len(gpus)} GPUs')
    except RuntimeError as e:
        print(f'Error configuring GPU memory growth: {e}')

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
    print(f'Connected to Redis at {os.environ.get(\"REDIS_HOST\", \"localhost\")}:{os.environ.get(\"REDIS_PORT\", 6380)}')
except Exception as e:
    print(f'Warning: Redis connection failed: {e}')
    print('Continuing without Redis - some functionality will be limited')
    
    # Create a mock Redis client for testing
    class MockRedis:
        def __init__(self):
            self.data = {}
            
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
    
    redis_client = MockRedis()

# Create data loader
DataPipelineIntegration = data_pipeline_integration.DataPipelineIntegration
data_loader = DataPipelineIntegration(
    redis_host=os.environ.get('REDIS_HOST', 'localhost'),
    redis_port=int(os.environ.get('REDIS_PORT', 6380)),
    redis_db=int(os.environ.get('REDIS_DB', 0)),
    polygon_api_key=os.environ.get('POLYGON_API_KEY', ''),
    unusual_whales_api_key=os.environ.get('UNUSUAL_WHALES_API_KEY', ''),
    use_gpu=os.environ.get('USE_GPU', 'true').lower() == 'true'
)

# Create model trainer with custom directories
MLModelTrainer = ml_model_trainer.MLModelTrainer
trainer = MLModelTrainer(redis_client, data_loader)
trainer.config['models_dir'] = os.environ.get('MODELS_DIR', './models')
trainer.config['data_dir'] = os.environ.get('DATA_DIR', './data')

# Train all models
success = trainer.train_all_models()
sys.exit(0 if success else 1)
"

# Exit with the same status as the Python script
exit $?