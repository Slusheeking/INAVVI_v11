#!/bin/bash

# Setup script for enhanced Polygon.io implementation

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up enhanced Polygon.io implementation...${NC}"

# Stop any running containers
echo -e "${BLUE}Stopping existing containers...${NC}"
docker-compose down

# Update Redis port in docker-compose.yml to avoid conflicts
echo -e "${BLUE}Updating Redis port to avoid conflicts...${NC}"
sed -i 's/- "6379:6379"/- "6380:6379"/' docker-compose.yml

# Build the container with enhanced implementation
echo -e "${BLUE}Building container with enhanced implementation...${NC}"
docker-compose build

# Start the containers
echo -e "${BLUE}Starting containers...${NC}"
docker-compose up -d

# Check if containers are running
echo -e "${BLUE}Checking container status...${NC}"
REDIS_RUNNING=$(docker ps | grep redis-cache | wc -l)
TF_RUNNING=$(docker ps | grep tensorflow-gh200 | wc -l)

if [ "$REDIS_RUNNING" -eq 0 ]; then
    echo -e "${RED}Redis container is not running. Checking logs...${NC}"
    docker-compose logs redis
    echo -e "${YELLOW}Continuing without Redis...${NC}"
fi

if [ "$TF_RUNNING" -eq 0 ]; then
    echo -e "${RED}TensorFlow container is not running. Checking logs...${NC}"
    docker-compose logs tensorflow-gpu
    echo -e "${RED}Cannot continue without TensorFlow container. Exiting.${NC}"
    exit 1
fi

# Wait for containers to be ready
echo -e "${BLUE}Waiting for containers to be ready...${NC}"
sleep 5

# Copy files to the container
echo -e "${BLUE}Copying files to the container...${NC}"
docker cp polygon_data_source_enhanced.py tensorflow-gh200:/app/ || echo -e "${YELLOW}Warning: Failed to copy polygon_data_source_enhanced.py${NC}"
docker cp enhanced_benchmark.py tensorflow-gh200:/app/ || echo -e "${YELLOW}Warning: Failed to copy enhanced_benchmark.py${NC}"
docker cp ENHANCED_README.md tensorflow-gh200:/app/ || echo -e "${YELLOW}Warning: Failed to copy ENHANCED_README.md${NC}"

# Run a small test to verify the setup
echo -e "${BLUE}Running a small test...${NC}"
docker exec -it tensorflow-gh200 python -c "
import sys
try:
    import redis
    print('Redis module imported successfully')
    
    # Try to connect to Redis
    try:
        r = redis.Redis(host='redis', port=6379, db=0)
        r.ping()
        print('Successfully connected to Redis')
    except Exception as e:
        print(f'Warning: Could not connect to Redis: {e}')
        print('Continuing with local cache only')
    
    print('Enhanced implementation is ready to use')
except ImportError as e:
    print(f'Error: {e}')
    print('Enhanced implementation may not work correctly')
    sys.exit(1)
" || echo -e "${RED}Test failed. Container might not be running properly.${NC}"

echo -e "${GREEN}Setup complete!${NC}"
echo -e "${YELLOW}You can now run the enhanced implementation with:${NC}"
echo -e "  ./run.sh enhanced"
echo -e "${YELLOW}Or run a scaling test with:${NC}"
echo -e "  ./run.sh scaling-test"
echo -e "${YELLOW}Or run a large-scale benchmark with:${NC}"
echo -e "  ./run.sh large-scale"