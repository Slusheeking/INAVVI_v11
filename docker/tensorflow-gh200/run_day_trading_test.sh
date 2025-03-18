#!/bin/bash
# Script to run the Day Trading System test

# Set API keys
export POLYGON_API_KEY="wFvpCGZq4glxZU_LlRc2Qpw6tQGB5Fmf"
export UNUSUAL_WHALES_API_KEY="4ad71b9e-7ace-4f24-bdfc-532ace219a18"

# Check if API keys are set
if [ -z "$POLYGON_API_KEY" ]; then
    echo "Error: POLYGON_API_KEY environment variable not set properly"
    echo "Please check the script to ensure the API key is correctly defined"
    exit 1
fi

if [ -z "$UNUSUAL_WHALES_API_KEY" ]; then
    echo "Error: UNUSUAL_WHALES_API_KEY environment variable not set properly"
    echo "Please check the script to ensure the API key is correctly defined"
    exit 1
fi

# Set the working directory to the project root
cd "$(dirname "$0")/../.." || exit 1

echo "Using Polygon API key: $POLYGON_API_KEY"
echo "Using Unusual Whales API key: $UNUSUAL_WHALES_API_KEY"

# Parse command line arguments
SYMBOLS=""
NUM_SYMBOLS=5
USE_MOCK=true

# Process command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --symbols)
            SYMBOLS="$2"
            shift
            shift
            ;;
        --num-symbols)
            NUM_SYMBOLS="$2"
            shift
            shift
            ;;
        --use-mock)
            USE_MOCK="$2"
            shift
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --symbols SYMBOLS      Comma-separated list of symbols to test"
            echo "  --num-symbols N        Number of symbols to test if symbols not provided"
            echo "  --use-mock true|false  Use mock data for testing (default: true)"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build command
CMD="python docker/tensorflow-gh200/test_day_trading_system.py"

if [ -n "$SYMBOLS" ]; then
    CMD="$CMD --symbols $SYMBOLS"
fi

CMD="$CMD --num-symbols $NUM_SYMBOLS"

if [ "$USE_MOCK" = "false" ]; then
    CMD="$CMD --no-use-mock"
fi

# Run the test
echo "Running Day Trading System test with command:"
echo "$CMD"
echo ""

eval "$CMD"

# Check exit status
if [ $? -ne 0 ]; then
    echo "Test failed with exit code $?"
    exit 1
fi

echo ""
echo "Test completed successfully"