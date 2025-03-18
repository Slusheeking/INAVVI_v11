#!/bin/bash
# Run ML and Data Ingestion Tests in Docker Container
# This script builds and runs the Docker container to test the ML setup and data ingestion

# Set the current directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "Error: Docker is not running or not installed"
  exit 1
fi

# Build the Docker image
echo "Building Docker image..."
docker-compose build

# Run the container with the test script
echo "Running ML setup test in container..."
docker-compose run --rm tensorflow-gpu python /app/test_ml_setup.py

# Run the data ingestion test
echo "Running data ingestion test in container..."
docker-compose run --rm tensorflow-gpu python /app/test_data_ingestion_system.py

echo "Test completed."