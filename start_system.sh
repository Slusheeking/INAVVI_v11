#!/bin/bash

# Start the Autonomous Trading System
# This script starts the system using Docker Compose

# Set the script to exit on error
set -e

echo "Starting Autonomous Trading System..."

# Check if the required existing containers are running
echo "Checking if required existing containers are running..."

# Check TimescaleDB container
if ! docker ps | grep -q "timescaledb-v1-1"; then
    echo "Error: TimescaleDB container (timescaledb-v1-1) is not running."
    echo "Please start the TimescaleDB container before running this script."
    exit 1
fi

# Check TensorFlow model training container
if ! docker ps | grep -q "model-training-tensorflow-v1-1"; then
    echo "Error: TensorFlow model training container (model-training-tensorflow-v1-1) is not running."
    echo "Please start the TensorFlow model training container before running this script."
    exit 1
fi

# Check TensorFlow feature engineering container
if ! docker ps | grep -q "feature-engineering-tensorflow-v1-1"; then
    echo "Error: TensorFlow feature engineering container (feature-engineering-tensorflow-v1-1) is not running."
    echo "Please start the TensorFlow feature engineering container before running this script."
    exit 1
fi

echo "All required existing containers are running."

# Build and start the system using Docker Compose
echo "Building and starting the system..."
docker-compose build
docker-compose up -d

# Check if the containers started successfully
echo "Checking if the containers started successfully..."
if ! docker-compose ps | grep -q "Up"; then
    echo "Error: Failed to start the containers."
    echo "Please check the logs for more information:"
    echo "docker-compose logs"
    exit 1
fi

echo "System started successfully!"
echo "To view the logs, run: docker-compose logs -f"
echo "To stop the system, run: ./stop_system.sh"