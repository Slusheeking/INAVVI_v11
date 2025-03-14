#!/bin/bash

# Stop the Autonomous Trading System
# This script stops the system using Docker Compose

echo "Stopping Autonomous Trading System..."

# Stop the containers using Docker Compose
docker-compose down

echo "System stopped successfully!"
echo "Note: The existing containers (TimescaleDB, TensorFlow) are still running."
echo "To view the running containers, run: docker ps"