#!/bin/bash
# Run Data Ingestion Test in Docker Container
# This script uses the existing Docker container to test the data ingestion system with market hours handling

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
docker cp test_data_ingestion_system.py tensorflow-gh200:/app/test_data_ingestion_system.py
docker cp data_ingestion_api_clients.py tensorflow-gh200:/app/data_ingestion_api_clients.py
docker cp polygon_data_source_ultra.py tensorflow-gh200:/app/polygon_data_source_ultra.py
docker cp market_hours_handler.py tensorflow-gh200:/app/market_hours_handler.py
docker cp load_env.py tensorflow-gh200:/app/load_env.py
docker cp .env tensorflow-gh200:/app/.env

# Install required packages
echo -e "${BLUE}Installing required packages...${NC}"
docker exec -it tensorflow-gh200 pip install pytz

# Run the container with the test script
echo -e "${GREEN}Running data ingestion test in container...${NC}"

# Run with market hours handling
echo -e "${YELLOW}Testing with market hours handling...${NC}"
docker exec -it tensorflow-gh200 python /app/test_data_ingestion_system.py --non-market-hours || {
    echo -e "${RED}Test failed with non-market hours flag, trying without flag...${NC}"
    docker exec -it tensorflow-gh200 python /app/test_data_ingestion_system.py
}

echo -e "${GREEN}Test completed.${NC}"