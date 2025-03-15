#!/bin/bash

# Script to remove unused containers and rename current containers
# Created for INAVVI_v11-1 autonomous trading system

echo "Starting container cleanup process..."

# Stop and remove old containers
echo "Stopping and removing old containers..."

# List of containers to remove
CONTAINERS_TO_REMOVE=(
  "tensorrt-optimizer-v1-1"
  "trading-strategy-v1-1"
  "model-training-tensorflow-v1-1"
  "model-training-pytorch-v1-1"
  "feature-engineering-tensorflow-v1-1"
  "feature-engineering-pytorch-v1-1"
)

# Stop and remove each container
for container in "${CONTAINERS_TO_REMOVE[@]}"; do
  echo "Processing container: $container"
  
  # Check if container exists
  if docker ps -a --format '{{.Names}}' | grep -q "^$container$"; then
    echo "  Container $container exists, stopping it..."
    docker stop $container || echo "  Warning: Failed to stop $container, it may already be stopped"
    
    echo "  Removing container $container..."
    docker rm $container || echo "  Warning: Failed to remove $container"
  else
    echo "  Container $container not found, skipping"
  fi
done

echo "Old containers have been removed"

# Stop current containers to apply the new names
echo "Stopping current containers to apply new names..."
docker-compose down

# Start containers with new names
echo "Starting containers with new names..."
docker-compose up -d

echo "Container cleanup and renaming process completed!"
echo "New container names are now using the 'ats-' prefix for better readability"
echo ""
echo "To verify the changes, run: docker ps"