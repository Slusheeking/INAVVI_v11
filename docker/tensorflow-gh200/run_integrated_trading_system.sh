#!/bin/bash
# Run Integrated Trading System
# This script runs the integrated stock selection and data ingestion system

# Set the current directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Python 3 is installed
echo -e "${BLUE}Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3 and try again.${NC}"
    exit 1
fi

# Check if required packages are installed
echo -e "${BLUE}Checking required packages...${NC}"
REQUIRED_PACKAGES=("redis" "numpy" "pandas" "requests" "asyncio")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! python3 -c "import $package" &> /dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo -e "${YELLOW}The following packages are missing: ${MISSING_PACKAGES[*]}${NC}"
    echo -e "${YELLOW}Installing missing packages...${NC}"
    pip3 install ${MISSING_PACKAGES[*]}
fi

# Check if Redis is running (if not using mock Redis)
if [ "$1" != "--mock-redis" ]; then
    echo -e "${BLUE}Checking if Redis is running...${NC}"
    if redis-cli ping &> /dev/null; then
        echo -e "${GREEN}Redis is running.${NC}"
    else
        echo -e "${RED}Redis is not running. Please start Redis and try again.${NC}"
        echo -e "${YELLOW}You can start Redis with: docker run --name redis -p 6379:6379 -d redis${NC}"
        exit 1
    fi
fi

# Check if API keys are set
echo -e "${BLUE}Checking API keys...${NC}"
if [ -f ".env" ]; then
    source .env
    echo -e "${GREEN}Environment variables loaded from .env file.${NC}"
else
    echo -e "${YELLOW}No .env file found. Using default API keys.${NC}"
fi

# Verify API keys are valid
echo -e "${BLUE}Verifying API keys...${NC}"
if [ -z "$POLYGON_API_KEY" ] || [ "$POLYGON_API_KEY" == "wFvpCGZq4glxZU_LlRc2Qpw6tQGB5Fmf" ]; then
    echo -e "${RED}Invalid or default Polygon API key. Please set a valid API key in .env file.${NC}"
    echo -e "${YELLOW}You can get a Polygon API key from: https://polygon.io/${NC}"
    exit 1
fi

# Parse command line arguments
DATA_ONLY=""
SELECTION_ONLY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --data-only)
            DATA_ONLY="--data-only"
            shift
            ;;
        --selection-only)
            SELECTION_ONLY="--selection-only"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Run the integrated trading system
echo -e "${BLUE}Starting integrated trading system...${NC}"
echo -e "${BLUE}Press Ctrl+C to stop the system.${NC}"

# Build the command with proper arguments
CMD="python3 run_stock_selection_with_data_ingestion.py"

if [ -n "$DATA_ONLY" ]; then
    CMD="$CMD --data-only"
fi

if [ -n "$SELECTION_ONLY" ]; then
    CMD="$CMD --selection-only" 
fi

# Execute the command
$CMD

# Check if the script exited with an error
if [ $? -ne 0 ]; then
    echo -e "${RED}The integrated trading system exited with an error.${NC}"
    exit 1
fi

echo -e "${GREEN}Integrated trading system stopped.${NC}"