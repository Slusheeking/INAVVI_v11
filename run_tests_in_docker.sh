#!/bin/bash
# Docker Test Runner for Trading System
# This script runs all tests inside the Docker container

set -e

# Display banner
echo "========================================================"
echo "  INAVVI Trading System Docker Test Runner"
echo "========================================================"
echo "This script runs all tests inside the Docker container."
echo ""

# Check if container is running
if ! docker ps | grep -q trading-system; then
    echo "Error: trading-system container is not running."
    echo "Please start the container first with:"
    echo "  docker-compose -f docker-compose.unified.yml up -d"
    exit 1
fi

# Function to run tests and check result
run_test() {
    local test_path=$1
    local test_name=$2
    
    echo "Running $test_name..."
    if ! docker exec -it trading-system bash -c "cd /app/project && python -m pytest $test_path -v"; then
        echo "Error: $test_name failed."
        exit 1
    fi
    echo "$test_name passed successfully."
    echo ""
}

# Run unit tests
echo "Running unit tests..."
run_test "tests/unit/test_api_clients.py" "API Clients Tests"
run_test "tests/unit/test_data_pipeline.py" "Data Pipeline Tests"
run_test "tests/unit/test_stock_selection_engine.py" "Stock Selection Engine Tests"
run_test "tests/unit/test_ml_engine.py" "ML Engine Tests"
run_test "tests/unit/test_trading_engine.py" "Trading Engine Tests"
run_test "tests/unit/test_gpu_utils.py" "GPU Utils Tests"
run_test "tests/unit/test_monitoring_system.py" "Monitoring System Tests"
run_test "tests/unit/test_unified_system.py" "Unified System Tests"

# Run integration tests
echo "Running integration tests..."
run_test "tests/integration/test_system_integration.py" "System Integration Tests"

# Run GPU verification tests
echo "Running GPU verification tests..."
docker exec -it trading-system bash -c "cd /app && python verify_tensorflow.py"
docker exec -it trading-system bash -c "cd /app && python test_tensorflow_gpu.py"

# All tests passed
echo ""
echo "========================================================"
echo "  All tests passed successfully!"
echo "========================================================"
echo "The system is verified and ready for production use."
echo ""
echo "Next steps:"
echo "1. Start the full system with: docker exec -it trading-system bash -c \"cd /app/project && python scripts/start_system.py\""
echo "2. Monitor system performance with Prometheus: http://localhost:9090"
echo "3. Check system logs: docker exec -it trading-system bash -c \"cat /app/logs/trading_system.out.log\""
echo "========================================================"