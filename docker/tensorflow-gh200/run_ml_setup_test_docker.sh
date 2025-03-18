#!/bin/bash
# Run ML Setup Test in Docker Container
# This script uses the existing Docker container to test the ML setup with proper error handling

# Set the current directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo -e "${RED}Error: Docker is not running or not installed${NC}"
  exit 1
fi

# Check if container is running
if ! docker ps | grep -q tensorflow-gh200; then
  echo -e "${YELLOW}Container not running. Starting container...${NC}"
  docker-compose up -d
fi

# Copy the necessary files to the container
echo -e "${BLUE}Copying test files to container...${NC}"
docker cp test_ml_setup.py tensorflow-gh200:/app/test_ml_setup.py
docker cp polygon_data_source_ultra.py tensorflow-gh200:/app/polygon_data_source_ultra.py
docker cp load_env.py tensorflow-gh200:/app/load_env.py
docker cp .env tensorflow-gh200:/app/.env

# Run the container with the test script
echo -e "${GREEN}Running ML setup test in container...${NC}"
docker exec -it tensorflow-gh200 python /app/test_ml_setup.py || {
    echo -e "${RED}ML setup test failed. Check the logs for details.${NC}"
    exit 1
}

echo -e "${GREEN}Test completed.${NC}"