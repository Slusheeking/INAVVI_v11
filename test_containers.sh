#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
  if [ "$1" == "PASS" ]; then
    echo -e "${GREEN}[PASS]${NC} $2"
  elif [ "$1" == "FAIL" ]; then
    echo -e "${RED}[FAIL]${NC} $2"
  else
    echo -e "${YELLOW}[INFO]${NC} $2"
  fi
}

# Function to check if a container is running
check_container_running() {
  local container_name=$1
  if docker ps | grep -q "$container_name"; then
    print_status "PASS" "Container $container_name is running"
    return 0
  else
    print_status "FAIL" "Container $container_name is not running"
    return 1
  fi
}

# Function to check if a container's logs contain a specific string
check_container_logs() {
  local container_name=$1
  local search_string=$2
  if docker logs "$container_name" 2>&1 | grep -q "$search_string"; then
    print_status "PASS" "Container $container_name logs contain '$search_string'"
    return 0
  else
    print_status "FAIL" "Container $container_name logs do not contain '$search_string'"
    return 1
  fi
}

# Function to check if a container's health endpoint is responding
check_container_health() {
  local container_name=$1
  local port=$2
  local endpoint=${3:-"/health"}
  local max_retries=${4:-5}
  local retry_interval=${5:-2}
  
  for ((i=1; i<=max_retries; i++)); do
    if curl -s "http://localhost:$port$endpoint" > /dev/null; then
      print_status "PASS" "Container $container_name health endpoint is responding"
      return 0
    else
      print_status "INFO" "Waiting for $container_name health endpoint (attempt $i/$max_retries)..."
      sleep $retry_interval
    fi
  done
  
  print_status "FAIL" "Container $container_name health endpoint is not responding after $max_retries attempts"
  return 1
}

# Function to check if TimescaleDB is accessible
check_timescaledb() {
  local host=${1:-"localhost"}
  local port=${2:-"5435"}  # Updated default port to 5435
  local db=${3:-"ats_db"}  # Using the correct DB name from config
  local user=${4:-"ats_user"}  # Using the correct user from config
  local password=${5:-"ats_password"}  # Using the correct password from config
  
  if PGPASSWORD=$password psql -h $host -p $port -U $user -d $db -c '\dt' > /dev/null 2>&1; then
    print_status "PASS" "TimescaleDB is accessible at $host:$port"
    return 0
  else
    print_status "FAIL" "TimescaleDB is not accessible at $host:$port"
    return 1
  fi
}

# Function to check if GPU is available in a container
check_gpu_in_container() {
  local container_name=$1
  
  if docker exec $container_name nvidia-smi > /dev/null 2>&1; then
    print_status "PASS" "GPU is available in container $container_name"
    return 0
  else
    print_status "FAIL" "GPU is not available in container $container_name"
    return 1
  fi
}

# Function to check if TA-Lib is installed in a container
check_talib_in_container() {
  local container_name=$1
  
  if docker exec $container_name python -c "import talib; print('TA-Lib version:', talib.__version__)" > /dev/null 2>&1; then
    print_status "PASS" "TA-Lib is installed in container $container_name"
    return 0
  else
    print_status "FAIL" "TA-Lib is not installed in container $container_name"
    return 1
  fi
}

# Function to check if PyTorch is installed in a container
check_pytorch_in_container() {
  local container_name=$1
  
  if docker exec $container_name python -c "import torch; print('PyTorch version:', torch.__version__)" > /dev/null 2>&1; then
    print_status "PASS" "PyTorch is installed in container $container_name"
    return 0
  else
    print_status "FAIL" "PyTorch is not installed in container $container_name"
    return 1
  fi
}

# Main testing function
run_tests() {
  local test_mode=$1
  local failures=0
  
  print_status "INFO" "Starting container tests in $test_mode mode..."
  
  # Create .env file if it doesn't exist
  if [ ! -f .env ]; then
    print_status "INFO" "Creating .env file from .env.sample..."
    cp .env.sample .env
  fi
  
  # Start containers based on test mode
  if [ "$test_mode" == "individual" ]; then
    print_status "INFO" "Starting individual container tests..."
    
    # Test TimescaleDB
    print_status "INFO" "Testing TimescaleDB container..."
    docker-compose up -d timescaledb
    sleep 10
    check_timescaledb || ((failures++))
    
    # Test System Controller
    print_status "INFO" "Testing System Controller container..."
    docker-compose up -d system-controller
    sleep 5
    check_container_running "autonomous_trading_system-system-controller-1" || ((failures++))
    check_container_health "autonomous_trading_system-system-controller-1" 8000 || ((failures++))
    
    # Test Data Acquisition
    print_status "INFO" "Testing Data Acquisition container..."
    docker-compose up -d data-acquisition
    sleep 5
    check_container_running "autonomous_trading_system-data-acquisition-1" || ((failures++))
    check_container_health "autonomous_trading_system-data-acquisition-1" 8001 || ((failures++))
    
    # Test Trading Strategy
    print_status "INFO" "Testing Trading Strategy container..."
    docker-compose up -d trading-strategy
    sleep 5
    check_container_running "autonomous_trading_system-trading-strategy-1" || ((failures++))
    check_container_health "autonomous_trading_system-trading-strategy-1" 8002 || ((failures++))
    
    # Test Model Training
    print_status "INFO" "Testing Model Training container..."
    docker-compose up -d model-training
    sleep 5
    check_container_running "autonomous_trading_system-model-training-1" || ((failures++))
    check_container_health "autonomous_trading_system-model-training-1" 8003 || ((failures++))
    check_pytorch_in_container "autonomous_trading_system-model-training-1" || ((failures++))
    if [ "${GPU_ENABLED:-false}" == "true" ]; then
      check_gpu_in_container "autonomous_trading_system-model-training-1" || ((failures++))
    fi
    
    # Test Feature Engineering
    print_status "INFO" "Testing Feature Engineering container..."
    docker-compose up -d feature-engineering
    sleep 5
    check_container_running "autonomous_trading_system-feature-engineering-1" || ((failures++))
    check_container_health "autonomous_trading_system-feature-engineering-1" 8004 || ((failures++))
    check_talib_in_container "autonomous_trading_system-feature-engineering-1" || ((failures++))
    
    # Test Continuous Learning
    print_status "INFO" "Testing Continuous Learning container..."
    docker-compose up -d continuous-learning
    sleep 5
    check_container_running "autonomous_trading_system-continuous-learning-1" || ((failures++))
    check_container_health "autonomous_trading_system-continuous-learning-1" 8005 || ((failures++))
    
  elif [ "$test_mode" == "full" ]; then
    print_status "INFO" "Starting full system test..."
    
    # Start all containers
    docker-compose up -d
    sleep 30
    
    # Check if all containers are running
    for container in system-controller data-acquisition trading-strategy model-training feature-engineering continuous-learning; do
      check_container_running "autonomous_trading_system-$container-1" || ((failures++))
    done
    
    # Check health endpoints
    check_container_health "autonomous_trading_system-system-controller-1" 8000 || ((failures++))
    check_container_health "autonomous_trading_system-data-acquisition-1" 8001 || ((failures++))
    check_container_health "autonomous_trading_system-trading-strategy-1" 8002 || ((failures++))
    check_container_health "autonomous_trading_system-model-training-1" 8003 || ((failures++))
    check_container_health "autonomous_trading_system-feature-engineering-1" 8004 || ((failures++))
    check_container_health "autonomous_trading_system-continuous-learning-1" 8005 || ((failures++))
    
    # Check special features
    check_talib_in_container "autonomous_trading_system-feature-engineering-1" || ((failures++))
    check_pytorch_in_container "autonomous_trading_system-model-training-1" || ((failures++))
    
    if [ "${GPU_ENABLED:-false}" == "true" ]; then
      check_gpu_in_container "autonomous_trading_system-model-training-1" || ((failures++))
    fi
    
    # Check for expected log messages
    check_container_logs "autonomous_trading_system-system-controller-1" "Starting System Controller" || ((failures++))
    check_container_logs "autonomous_trading_system-data-acquisition-1" "Starting Data Acquisition" || ((failures++))
    check_container_logs "autonomous_trading_system-trading-strategy-1" "Starting Trading Strategy" || ((failures++))
    check_container_logs "autonomous_trading_system-model-training-1" "Starting Model Training" || ((failures++))
    check_container_logs "autonomous_trading_system-feature-engineering-1" "Starting Feature Engineering" || ((failures++))
    check_container_logs "autonomous_trading_system-continuous-learning-1" "Starting Continuous Learning" || ((failures++))
    
    # Check monitoring containers
    print_status "INFO" "Testing monitoring containers..."
    for container in prometheus alertmanager grafana node-exporter cadvisor timescaledb-exporter redis-exporter loki promtail; do
      check_container_running "$container" || ((failures++))
    done
    
    # Check Prometheus endpoint
    check_container_health "prometheus" 9090 "/graph" || ((failures++))
    
    # Check AlertManager endpoint
    check_container_health "alertmanager" 9093 "/#/alerts" || ((failures++))
    
    # Check Grafana endpoint
    check_container_health "grafana" 3000 "/login" || ((failures++))
    
    # Check Loki endpoint
    check_container_health "loki" 3100 "/ready" || ((failures++))
    
    # Verify Prometheus targets are being scraped
    print_status "INFO" "Checking Prometheus targets..."
    if curl -s "http://localhost:9090/api/v1/targets" | grep -q "\"health\":\"up\""; then
      print_status "PASS" "Prometheus targets are being scraped"
    else
      print_status "FAIL" "Some Prometheus targets are not being scraped"
      ((failures++))
    fi
  else
    print_status "FAIL" "Invalid test mode: $test_mode"
    return 1
  fi
  
  # Print test summary
  if [ $failures -eq 0 ]; then
    print_status "PASS" "All tests passed successfully!"
    return 0
  else
    print_status "FAIL" "$failures tests failed."
    return 1
  fi
}

# Clean up function
cleanup() {
  print_status "INFO" "Cleaning up containers..."
  docker-compose down
}

# Parse command line arguments
test_mode="full"
cleanup_after=true

while [[ $# -gt 0 ]]; do
  case $1 in
    --mode)
      test_mode="$2"
      shift 2
      ;;
    --no-cleanup)
      cleanup_after=false
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--mode individual|full] [--no-cleanup]"
      exit 1
      ;;
  esac
done

# Run tests
run_tests "$test_mode"
test_result=$?

# Clean up if requested
if [ "$cleanup_after" = true ]; then
  cleanup
fi

exit $test_result