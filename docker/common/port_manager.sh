#!/bin/bash

# Port Manager Script
# This script checks and frees up ports needed by the trading system

# Define colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Define the ports used by our system
SYSTEM_PORTS=(
  5433  # TimescaleDB
  6379  # Redis
  8000  # System Controller API
  8001  # Data Pipeline API
  8002  # Trading Strategy API
  8003  # Model Training API
  8004  # Feature Engineering API
  8005  # Continuous Learning API
  8500  # TensorFlow Serving gRPC
  8501  # TensorFlow Serving REST
  9090  # Prometheus
  3001  # Grafana
)

# Function to check if a port is in use
check_port() {
  local port=$1
  if lsof -i:${port} > /dev/null 2>&1; then
    return 0 # Port is in use
  else
    return 1 # Port is free
  fi
}

# Function to free up a port by stopping the process using it
free_port() {
  local port=$1
  local force=$2
  
  echo -e "${YELLOW}Attempting to free port ${port}...${NC}"
  
  # Get the PID of the process using the port
  local pid=$(lsof -t -i:${port} 2>/dev/null)
  
  if [ -z "$pid" ]; then
    echo -e "${GREEN}Port ${port} is already free.${NC}"
    return 0
  fi
  
  echo -e "${YELLOW}Port ${port} is used by process ${pid}${NC}"
  
  # Try to stop Docker container using this port
  local container=$(docker ps --format '{{.Names}}' | xargs -I{} docker port {} 2>/dev/null | grep ":${port}" | cut -d: -f1)
  if [ ! -z "$container" ]; then
    echo -e "${YELLOW}Stopping Docker container ${container} using port ${port}...${NC}"
    docker stop ${container} > /dev/null 2>&1
    sleep 2
    
    # Check if port is now free
    if ! check_port ${port}; then
      echo -e "${GREEN}Successfully freed port ${port} by stopping Docker container ${container}.${NC}"
      return 0
    fi
  fi
  
  # If force is enabled or it's a known system port, try to kill the process
  if [ "$force" = "true" ] || [[ " ${SYSTEM_PORTS[@]} " =~ " ${port} " ]]; then
    echo -e "${YELLOW}Stopping process ${pid} using port ${port}...${NC}"
    kill -15 ${pid} > /dev/null 2>&1
    sleep 2
    
    # If process is still running, force kill it
    if ps -p ${pid} > /dev/null 2>&1; then
      echo -e "${YELLOW}Process ${pid} still running. Force killing...${NC}"
      kill -9 ${pid} > /dev/null 2>&1
      sleep 1
    fi
    
    # Check if port is now free
    if ! check_port ${port}; then
      echo -e "${GREEN}Successfully freed port ${port}.${NC}"
      return 0
    else
      echo -e "${RED}Failed to free port ${port}.${NC}"
      return 1
    fi
  else
    echo -e "${RED}Port ${port} is in use by a non-system process. Use --force to kill it.${NC}"
    return 1
  fi
}

# Function to stop all Docker containers related to our system
stop_system_containers() {
  echo -e "${YELLOW}Stopping all trading system containers...${NC}"
  
  # Get all containers with our system prefix
  local containers=$(docker ps -a --format '{{.Names}}' | grep "^ats-")
  
  if [ -z "$containers" ]; then
    echo -e "${GREEN}No trading system containers running.${NC}"
    return 0
  fi
  
  # Stop each container
  for container in ${containers}; do
    echo -e "${YELLOW}Stopping container ${container}...${NC}"
    docker stop ${container} > /dev/null 2>&1
  done
  
  echo -e "${GREEN}All trading system containers stopped.${NC}"
}

# Function to ensure all system ports are free
ensure_ports_free() {
  local force=$1
  local all_free=true
  
  echo -e "${YELLOW}Checking system ports...${NC}"
  
  for port in "${SYSTEM_PORTS[@]}"; do
    if check_port ${port}; then
      echo -e "${YELLOW}Port ${port} is in use.${NC}"
      if ! free_port ${port} ${force}; then
        all_free=false
      fi
    else
      echo -e "${GREEN}Port ${port} is free.${NC}"
    fi
  done
  
  if [ "$all_free" = "true" ]; then
    echo -e "${GREEN}All system ports are free and ready to use.${NC}"
    return 0
  else
    echo -e "${RED}Some system ports could not be freed. Use --force to attempt to free all ports.${NC}"
    return 1
  fi
}

# Main function
main() {
  local force=false
  
  # Parse command line arguments
  while [[ $# -gt 0 ]]; do
    case $1 in
      --force)
        force=true
        shift
        ;;
      --stop-containers)
        stop_system_containers
        shift
        ;;
      --check-only)
        for port in "${SYSTEM_PORTS[@]}"; do
          if check_port ${port}; then
            echo -e "${RED}Port ${port} is in use.${NC}"
          else
            echo -e "${GREEN}Port ${port} is free.${NC}"
          fi
        done
        exit 0
        ;;
      --help)
        echo "Usage: $0 [OPTIONS]"
        echo "Options:"
        echo "  --force            Force kill processes using system ports"
        echo "  --stop-containers  Stop all trading system containers"
        echo "  --check-only       Only check port status without freeing them"
        echo "  --help             Display this help message"
        exit 0
        ;;
      *)
        echo "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
    esac
  done
  
  # First stop all system containers
  stop_system_containers
  
  # Then ensure all ports are free
  ensure_ports_free ${force}
  return $?
}

# Run the main function with all arguments
main "$@"