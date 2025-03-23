#!/bin/bash
# Production Readiness Verification Script
# This script checks all critical files to ensure they are ready for production

set -e

# Display banner
echo "========================================================"
echo "  INAVVI Trading System Production Readiness Check"
echo "========================================================"
echo "This script will verify that all files are production-ready."
echo ""

# Check critical files
echo "Checking critical files..."

# Check Docker files
if [ ! -f "Dockerfile.unified" ]; then
    echo "Error: Dockerfile.unified not found."
    exit 1
fi

if [ ! -f "docker-compose.unified.yml" ]; then
    echo "Error: docker-compose.unified.yml not found."
    exit 1
fi

# Check configuration files
if [ ! -f ".env" ]; then
    echo "Error: .env file not found."
    exit 1
fi

if [ ! -f "redis/redis.conf" ]; then
    echo "Error: redis/redis.conf not found."
    exit 1
fi

if [ ! -f "prometheus/prometheus.yml" ]; then
    echo "Error: prometheus/prometheus.yml not found."
    exit 1
fi

# Check core Python modules
CORE_MODULES=(
    "api_clients.py"
    "data_pipeline.py"
    "stock_selection_engine.py"
    "ml_engine.py"
    "trading_engine.py"
    "unified_system.py"
    "gpu_utils.py"
    "config.py"
)

for module in "${CORE_MODULES[@]}"; do
    if [ ! -f "$module" ]; then
        echo "Error: Core module $module not found."
        exit 1
    fi
done

# Check monitoring system
if [ ! -f "monitoring_system/monitoring_system.py" ]; then
    echo "Error: monitoring_system.py not found."
    exit 1
fi

# Check scripts
SCRIPTS=(
    "scripts/start_system.py"
    "scripts/stop_system.py"
    "start.sh"
    "stop.sh"
)

for script in "${SCRIPTS[@]}"; do
    if [ ! -f "$script" ]; then
        echo "Error: Script $script not found."
        exit 1
    fi
done

# Check test files
if [ ! -f "tests/integration/test_system_integration.py" ]; then
    echo "Error: Integration test file not found."
    exit 1
fi

# Check for required directories
REQUIRED_DIRS=(
    "data"
    "models"
    "logs"
    "redis"
    "prometheus"
    "monitoring_system"
    "tests"
    "scripts"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "Error: Required directory $dir not found."
        exit 1
    fi
done

# Validate .env file
echo "Validating .env file..."
source .env

if [ -z "$POLYGON_API_KEY" ] || [ "$POLYGON_API_KEY" == "your_polygon_api_key" ]; then
    echo "Error: POLYGON_API_KEY is not set in .env file."
    exit 1
fi

if [ -z "$REDIS_HOST" ] || [ -z "$REDIS_PORT" ]; then
    echo "Error: Redis configuration is incomplete in .env file."
    exit 1
fi

# Check Python syntax for core modules
echo "Checking Python syntax..."
for module in "${CORE_MODULES[@]}"; do
    if ! python -m py_compile "$module" 2>/dev/null; then
        echo "Error: Syntax error in $module."
        python -m py_compile "$module"
        exit 1
    fi
done

# Check Docker configuration
echo "Checking Docker configuration..."
if ! docker-compose -f docker-compose.unified.yml config >/dev/null; then
    echo "Error: Invalid docker-compose.unified.yml configuration."
    exit 1
fi

# Check NVIDIA GPU availability
echo "Checking NVIDIA GPU availability..."
if ! nvidia-smi &>/dev/null; then
    echo "Warning: NVIDIA GPU not detected or drivers not properly installed."
    echo "The system requires GPU acceleration for optimal performance."
    echo "You can still proceed, but performance will be limited."
fi

# Check Redis configuration
echo "Checking Redis configuration..."
if grep -q "bind 127.0.0.1" redis/redis.conf; then
    echo "Warning: Redis is configured to bind only to localhost."
    echo "For Docker deployment, Redis should bind to 0.0.0.0."
    echo "Consider updating redis/redis.conf."
fi

# Check for required Python packages
echo "Checking for required Python packages..."
REQUIRED_PACKAGES=(
    "tensorflow"
    "pandas"
    "numpy"
    "redis"
    "prometheus_client"
    "polygon-api-client"
    "scikit-learn"
    "xgboost"
)

MISSING_PACKAGES=()
for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! pip list | grep -q "$package"; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo "Warning: The following required Python packages are missing:"
    for package in "${MISSING_PACKAGES[@]}"; do
        echo "  - $package"
    done
    echo "These packages will be installed in the Docker container, but are missing in the local environment."
fi

# All checks passed
echo ""
echo "========================================================"
echo "  Production Readiness Check: PASSED"
echo "========================================================"
echo "All critical files are present and properly configured."
echo "The system is ready for production deployment."
echo ""
echo "Next steps:"
echo "1. Run setup_production.sh to deploy the system"
echo "2. Monitor system performance and logs"
echo "3. Set up backup and recovery procedures"
echo "========================================================"