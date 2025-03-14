#!/bin/bash
# Run the full system test for the autonomous trading system

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "Project root: $PROJECT_ROOT"
echo "Current directory: $(pwd)"
echo "Date: $(date)"
echo "-------------------------------------"

# Ensure environment variables are set
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
else
    echo "Warning: .env file not found. Make sure environment variables are set manually."
fi

# Check for required environment variables
required_vars=("POLYGON_API_KEY" "UNUSUAL_WHALES_API_KEY" "ALPACA_API_KEY_ID" "ALPACA_API_SECRET_KEY")
missing_vars=0

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Error: Required environment variable $var is not set."
        missing_vars=$((missing_vars+1))
    fi
done

if [ $missing_vars -gt 0 ]; then
    echo "Please set all required environment variables before running the test."
    exit 1
fi

# Check for TensorFlow and GPU availability
echo "Checking for TensorFlow and GPU availability..."
python -c "
import sys
try:
    import tensorflow as tf
    print(f'TensorFlow version: {tf.__version__}')
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f'GPU available: {len(gpus)} GPU(s) found')
        for gpu in gpus:
            print(f'  {gpu}')
    else:
        print('No GPU available, using CPU')
except ImportError:
    print('TensorFlow not installed, some features may not work')
except Exception as e:
    print(f'Error checking GPU: {e}')
"

# Check for XGBoost
echo "Checking for XGBoost..."
python -c "
import sys
try:
    import xgboost as xgb
    print(f'XGBoost version: {xgb.__version__}')
except ImportError:
    print('XGBoost not installed, some features may not work')
except Exception as e:
    print(f'Error checking XGBoost: {e}')
"

# Check database connection
echo "Checking database connection..."
python -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
try:
    from autonomous_trading_system.src.config.database_config import get_db_connection_string
    from sqlalchemy import create_engine, text
    
    connection_string = get_db_connection_string()
    engine = create_engine(connection_string)
    
    with engine.connect() as conn:
        result = conn.execute(text('SELECT version()'))
        version = result.scalar()
        print(f'Connected to database: {version}')
        
        # Check if TimescaleDB extension is installed
        result = conn.execute(text(\"\"\"
            SELECT extname FROM pg_extension WHERE extname = 'timescaledb'
        \"\"\"))
        if result.rowcount == 0:
            print('Warning: TimescaleDB extension not installed')
        else:
            print('TimescaleDB extension installed')
except Exception as e:
    print(f'Error connecting to database: {e}')
"

# Create output directory for reports
REPORT_DIR="$PROJECT_ROOT/results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPORT_DIR"
echo "Reports will be saved to: $REPORT_DIR"

# Run the full system test
echo "Running full system test..."
cd "$PROJECT_ROOT"

# Set Python path to include project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Run the test with output logging
python -m autonomous_trading_system.src.tests.full_system_test 2>&1 | tee "$REPORT_DIR/full_system_test.log"

# Check the exit code (use the exit code of the first command in the pipe)
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "Full system test completed successfully."
    echo "Test results saved to: $REPORT_DIR/full_system_test.log"
    exit 0
else
    echo "Full system test failed."
    echo "Check logs at: $REPORT_DIR/full_system_test.log"
    exit 1
fi