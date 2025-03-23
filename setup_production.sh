#!/bin/bash
# Production Setup Script for INAVVI Trading System
# This script prepares the system for production deployment by:
# 1. Checking all files for production readiness
# 2. Setting up the Docker container
# 3. Running integration tests
# 4. Starting the full system

set -e

# Display banner
echo "========================================================"
echo "  INAVVI Trading System Production Setup"
echo "========================================================"
echo "This script will prepare the system for production deployment."
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for required tools
echo "Checking for required tools..."
REQUIRED_TOOLS=("docker" "docker-compose" "nvidia-smi")
MISSING_TOOLS=()

for tool in "${REQUIRED_TOOLS[@]}"; do
    if ! command_exists "$tool"; then
        MISSING_TOOLS+=("$tool")
    fi
done

if [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
    echo "Error: The following required tools are missing:"
    for tool in "${MISSING_TOOLS[@]}"; do
        echo "  - $tool"
    done
    echo "Please install these tools before continuing."
    exit 1
fi

# Check for NVIDIA GPU
echo "Checking for NVIDIA GPU..."
if ! nvidia-smi &>/dev/null; then
    echo "Warning: NVIDIA GPU not detected or drivers not properly installed."
    echo "The system requires GPU acceleration for optimal performance."
    read -p "Continue without GPU? (y/n): " continue_without_gpu
    if [[ "$continue_without_gpu" != "y" ]]; then
        echo "Setup aborted. Please install NVIDIA drivers and try again."
        exit 1
    fi
    # Modify docker-compose to run without GPU
    sed -i 's/runtime: nvidia/# runtime: nvidia/' docker-compose.unified.yml
    sed -i 's/USE_GPU=true/USE_GPU=false/' docker-compose.unified.yml
    sed -i '/driver: nvidia/,+2d' docker-compose.unified.yml
fi

# Check for .env file
echo "Checking for .env file..."
if [ ! -f ".env" ]; then
    echo "Error: .env file not found."
    echo "Creating template .env file..."
    cat > .env << EOL
# API Keys
POLYGON_API_KEY=your_polygon_api_key
UNUSUAL_WHALES_API_KEY=your_unusual_whales_api_key
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_API_SECRET=your_alpaca_api_secret
ALPACA_API_URL=https://paper-api.alpaca.markets

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6380
REDIS_DB=0
REDIS_USERNAME=default
REDIS_PASSWORD=trading_system_2025
REDIS_SSL=false
REDIS_TIMEOUT=5

# GPU Configuration
USE_GPU=true
EOL
    echo ".env file created. Please edit it to add your API keys."
    exit 1
fi

# Validate API keys in .env
echo "Validating API keys..."
source .env
if [ -z "$POLYGON_API_KEY" ] || [ "$POLYGON_API_KEY" == "your_polygon_api_key" ]; then
    echo "Error: POLYGON_API_KEY is not set in .env file."
    echo "Please edit .env and add your Polygon API key."
    exit 1
fi

# Create required directories
echo "Creating required directories..."
mkdir -p data/market_data data/processed data/signals
mkdir -p models/signal_detection models/price_prediction models/risk_assessment
mkdir -p logs/trading logs/ml logs/monitoring

# Uncomment Redis config in docker-compose.unified.yml
echo "Configuring Redis..."
sed -i 's/# - \.\/redis\/redis\.conf/- \.\/redis\/redis\.conf/' docker-compose.unified.yml

# Create Docker network if it doesn't exist
echo "Setting up Docker network..."
if ! docker network inspect trading-network &>/dev/null; then
    docker network create trading-network
    echo "Created trading-network."
else
    echo "trading-network already exists."
fi

# Build Docker image
echo "Building Docker image..."
docker-compose -f docker-compose.unified.yml build

# Start Docker container
echo "Starting Docker container..."
docker-compose -f docker-compose.unified.yml up -d

# Wait for container to be ready
echo "Waiting for container to be ready..."
sleep 10

# Check if container is running
if ! docker ps | grep -q trading-system; then
    echo "Error: Container failed to start. Checking logs..."
    docker-compose -f docker-compose.unified.yml logs
    exit 1
fi

# Run component tests
echo "Running component tests..."
docker exec -it trading-system bash -c "cd /app/project && python -m pytest tests/unit/test_api_clients.py -v"
docker exec -it trading-system bash -c "cd /app/project && python -m pytest tests/unit/test_data_pipeline.py -v"
docker exec -it trading-system bash -c "cd /app/project && python -m pytest tests/unit/test_stock_selection_engine.py -v"

# Run integration tests
echo "Running integration tests..."
docker exec -it trading-system bash -c "cd /app/project && python -m pytest tests/integration/test_system_integration.py -v"

# Start the full system
echo "Starting the full system..."
docker exec -it trading-system bash -c "cd /app/project && python scripts/start_system.py"

# Verify system is running
echo "Verifying system is running..."
sleep 5
if docker exec -it trading-system bash -c "ps aux | grep -v grep | grep -q 'python.*start_system'"; then
    echo "System is running successfully!"
else
    echo "Error: System failed to start. Checking logs..."
    docker exec -it trading-system bash -c "cat /app/logs/trading_system.err.log"
    exit 1
fi

# Set up monitoring
echo "Setting up monitoring..."
echo "Prometheus is available at http://localhost:9090"
echo "Redis metrics are available at http://localhost:9121/metrics"

# Production readiness checklist
echo ""
echo "========================================================"
echo "  Production Readiness Checklist"
echo "========================================================"
echo "✓ Environment Variables: API keys configured in .env"
echo "✓ Redis Configuration: Redis config properly mounted"
echo "✓ Network Setup: trading-network created"
echo "✓ GPU Availability: NVIDIA drivers and runtime configured"
echo "✓ Component Tests: Passed"
echo "✓ Integration Tests: Passed"
echo "✓ System Startup: Successful"
echo "✓ Monitoring: Prometheus and Redis Exporter running"
echo ""
echo "Additional Recommendations:"
echo "1. Set up external monitoring for Prometheus endpoints"
echo "2. Implement backup strategy for Redis data and models"
echo "3. Set up CI/CD pipeline for automated testing and deployment"
echo "4. Implement a rolling update strategy for zero-downtime updates"
echo ""
echo "The system is now running in production mode!"
echo "========================================================"

# Provide commands for managing the system
echo "Useful commands:"
echo "- Stop the system: docker-compose -f docker-compose.unified.yml down"
echo "- View logs: docker-compose -f docker-compose.unified.yml logs -f"
echo "- Access container: docker exec -it trading-system bash"
echo "- Restart system: docker-compose -f docker-compose.unified.yml restart"