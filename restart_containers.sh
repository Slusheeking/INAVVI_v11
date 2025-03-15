#!/bin/bash

echo "Restarting containers with updated start scripts..."

# Stop all containers
echo "Stopping all containers..."
docker-compose down

# Start containers with updated start scripts
echo "Starting containers with updated start scripts..."
docker-compose up -d

echo "Container restart process completed!"
echo "To verify the changes, run: docker ps"