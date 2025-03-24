#!/bin/bash
# rebuild_container.sh
# Script to rebuild and restart the trading system container with all fixes

set -e

echo "=== Trading System Container Rebuild Script ==="
echo "This script will rebuild and restart the trading system container with all fixes"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "Error: Docker is not running or you don't have permission to use it."
  exit 1
fi

# Aggressively stop and remove any existing trading-system container
echo "Checking for existing trading-system container..."
if docker ps -a | grep -q trading-system; then
  echo "Found existing trading-system container. Stopping it aggressively..."
  # Try graceful stop first with short timeout
  docker stop --time=10 trading-system || true
  # Force kill if still running
  docker kill trading-system 2>/dev/null || true
  echo "Forcefully removing trading-system container..."
  docker rm -f trading-system || true
  # Double check if removed
  if docker ps -a | grep -q trading-system; then
    echo "Container still exists, using stronger force..."
    docker rm -f trading-system || true
    sleep 2
    # Last resort - use low-level docker commands
    docker container rm -f trading-system 2>/dev/null || true
  fi
  echo "Waiting to ensure cleanup is complete..."
  sleep 3
fi

# Remove old fix files that are now consolidated into the Dockerfile
echo "Removing old fix files that are now incorporated in the Dockerfile..."
OLD_FILES=(
  "fix_all_container_issues.sh"
  "fix_container_issues.sh"
  "fix_frontend_access.sh"
  "fix_tf_gpu.py"
  "fixed_startup.sh"
  "Dockerfile.unified.fixed"
  "Dockerfile.unified.original"
  "rebuild_with_fixed_dockerfile.sh"
  "CONTAINER_FIX_README.md"
  "verify_tensorflow.py"
  "test_tensorflow_gpu.py"
  "test_tensorflow_gpu_direct.py"
  "test_cupy.py"
  "test_cupy_gpu.py"
  "test_xgboost.py"
  "rebuild_with_portfolio_updater.sh"
)

for file in "${OLD_FILES[@]}"; do
  if [ -f "$file" ]; then
    rm -f "$file" && echo "Removed: $file"
  fi
done

# Remove any dangling images
echo "Cleaning up dangling images..."
docker image prune -f

# Remove existing network and recreate it
echo "Removing existing trading-network if it exists..."
docker network rm trading-network || true
echo "Creating trading-network..."
docker network create trading-network

# Forcefully rebuild the image first
echo "Forcefully rebuilding the trading system container image..."
docker-compose -f docker-compose.unified.yml build --no-cache trading-system

# Start the container with the updated configuration
echo "Starting the trading system container with all fixes..."
docker-compose -f docker-compose.unified.yml up -d --force-recreate --remove-orphans

# Wait for the container to start
echo "Waiting for container to start..."
sleep 10

# Check container status
CONTAINER_STATUS=$(docker inspect --format='{{.State.Status}}' trading-system 2>/dev/null || echo "not running")
echo "Container status: $CONTAINER_STATUS"

if [ "$CONTAINER_STATUS" != "running" ]; then
  echo "Error: Container failed to start properly."
  echo "Container logs:"
  docker logs --tail 50 trading-system
  exit 1
fi

# Check container health
HEALTH_STATUS=$(docker inspect --format='{{.State.Health.Status}}' trading-system 2>/dev/null || echo "no health check")
echo "Container health status: $HEALTH_STATUS"

# Display container logs
echo "Recent container logs:"
docker logs --tail 20 trading-system

echo "=== Rebuild Complete ==="
echo "The container should now be running with all fixes applied."

# Ask if the user wants to open the frontend in a browser
echo "Would you like to open the frontend in a browser? (y/n)"
read -r open_browser

if [[ "$open_browser" == "y" || "$open_browser" == "Y" ]]; then
  # Wait a moment for the frontend to be fully ready
  echo "Launching frontend in browser..."
  sleep 5
  
  # Detect operating system and open the browser accordingly
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    xdg-open http://127.0.0.1:5000
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open http://127.0.0.1:5000
  elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    start http://127.0.0.1:5000
  else
    echo "Unsupported operating system detected."
    echo "Please manually open http://127.0.0.1:5000 in your browser."
  fi
  
  echo "Frontend should now be opening in your browser."
else
  echo "You can access the frontend at http://127.0.0.1:5000"
  fi
  
  echo "You can access Jupyter Lab at http://localhost:8888"
  echo "Monitor logs with: docker logs -f trading-system"
  echo "Monitor portfolio updater logs with: docker logs -f trading-system | grep 'update_portfolio'"
  echo "Portfolio data is being updated from Alpaca every minute and stored in Redis"