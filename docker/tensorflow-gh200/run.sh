#!/bin/bash

# TensorFlow GH200 Container Management Script

set -e
trap 'echo -e "${RED}An error occurred. Exiting...${NC}"; exit 1' ERR

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
SYMBOLS="AAPL,MSFT,GOOGL,AMZN,TSLA"
DATA_TYPE="aggregates"
NUM_RUNS=1
MAX_SYMBOLS=50

# Function to display usage
function show_usage {
    echo -e "${YELLOW}Usage:${NC}"
    echo -e "  $0 [command]"
    echo -e "\n${YELLOW}Commands:${NC}"
    echo -e "  build       Build the TensorFlow container"
    echo -e "  start       Start the container"
    echo -e "  stop        Stop the container"
    echo -e "  polygon     Run standard Polygon.io data benchmark"
    echo -e "  polygon-gh200  Run optimized Polygon.io data benchmark for GH200"
    echo -e "  polygon-fixed  Run fixed Polygon.io data benchmark for GH200"
    echo -e "  benchmark-fixed  Run fixed benchmark implementation"
    echo -e "  compare-impl  Compare original and fixed implementations"
    echo -e "  enhanced      Run enhanced Polygon.io implementation with Redis caching"
    echo -e "  scaling-test  Run scaling test with increasing number of symbols"
    echo -e "  large-scale   Run large-scale benchmark with many symbols"
    echo -e "  turbo        Run turbo-charged Polygon.io implementation"
    echo -e "  ultra        Run ultra-optimized Polygon.io implementation"
    echo -e "  compare     Run both benchmarks and compare results"
    echo -e "  view-results  View the latest benchmark results"
    echo -e "  clean       Clean up benchmark results"
    echo -e "  jupyter     Get the Jupyter URL"
    echo -e "  shell       Open a shell in the container"
    echo -e "  logs        Show container logs"
    echo -e "  help        Show this help message"
}

# Function to build the container
function build_container {
    echo -e "${GREEN}Building TensorFlow GH200 container...${NC}"
    echo -e "${BLUE}This may take several minutes...${NC}"
    docker-compose build
    echo -e "${GREEN}Build complete!${NC}"
}

# Function to start the container
function start_container {
    echo -e "${GREEN}Starting TensorFlow GH200 container...${NC}"
    docker-compose up -d --force-recreate
    echo -e "${GREEN}Container started!${NC}"
    echo -e "${YELLOW}JupyterLab is available at:${NC} http://localhost:8888"
}

# Function to stop the container
function stop_container {
    echo -e "${GREEN}Stopping TensorFlow GH200 container...${NC}"
    docker-compose down
    echo -e "${GREEN}Container stopped!${NC}"
}

# Function to run Polygon data benchmark
function run_polygon_benchmark {
    echo -e "${GREEN}Running Polygon.io data benchmark...${NC}"
    echo -e "${BLUE}Testing standard implementation with symbols: ${SYMBOLS}${NC}"
    docker exec -it tensorflow-gh200 python /app/polygon_benchmark.py --symbols ${SYMBOLS} --data-type ${DATA_TYPE} --num-runs ${NUM_RUNS}
    echo -e "${GREEN}Standard benchmark complete!${NC}"
}

# Function to run optimized Polygon data benchmark for GH200
function run_polygon_gh200_benchmark {
    echo -e "${GREEN}Running optimized Polygon.io data benchmark for GH200...${NC}"
    echo -e "${BLUE}Testing GH200-optimized implementation with symbols: ${SYMBOLS}${NC}"
    docker exec -it tensorflow-gh200 python /app/polygon_benchmark.py --symbols ${SYMBOLS} --data-type ${DATA_TYPE} --num-runs ${NUM_RUNS}
    echo -e "${GREEN}GH200-optimized benchmark complete!${NC}"
}

# Function to run fixed Polygon data benchmark for GH200
function run_polygon_fixed_benchmark {
    echo -e "${GREEN}Running fixed Polygon.io data benchmark for GH200...${NC}"
    echo -e "${BLUE}Testing fixed GH200 implementation with symbols: ${SYMBOLS}${NC}"
    docker exec -it tensorflow-gh200 python -c "import polygon_data_source_gh200_fixed as pg; pg.process_polygon_data_with_cuda_gh200(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])"
    echo -e "${GREEN}Fixed GH200 benchmark complete!${NC}"
}

# Function to run fixed benchmark
function run_fixed_benchmark {
    echo -e "${GREEN}Running fixed benchmark implementation...${NC}"
    echo -e "${BLUE}This will run both standard and GH200-optimized implementations${NC}"
    docker exec -it tensorflow-gh200 python /app/polygon_benchmark_fixed.py --symbols ${SYMBOLS} --data-type ${DATA_TYPE} --num-runs ${NUM_RUNS}
    echo -e "${GREEN}Fixed benchmark complete!${NC}"
}

# Function to run enhanced implementation
function run_enhanced_implementation {
    echo -e "${GREEN}Running enhanced Polygon.io implementation...${NC}"
    echo -e "${BLUE}This implementation includes Redis caching, connection pooling, and parallel processing${NC}"
    docker exec -it tensorflow-gh200 python /app/enhanced_benchmark.py --symbols ${SYMBOLS} --data-type ${DATA_TYPE} --num-runs ${NUM_RUNS} --compare
    echo -e "${GREEN}Enhanced implementation complete!${NC}"
}

# Function to run turbo implementation
function run_turbo_implementation {
    echo -e "${GREEN}Running turbo-charged Polygon.io implementation...${NC}"
    echo -e "${BLUE}This implementation includes Redis caching, optimized HTTP requests, and GPU acceleration${NC}"
    docker exec -it tensorflow-gh200 python /app/turbo_benchmark_fixed.py --symbols 5
    echo -e "${GREEN}Turbo implementation complete!${NC}"
}

# Function to run ultra implementation
function run_ultra_implementation {
    echo -e "${GREEN}Running ultra-optimized Polygon.io implementation...${NC}"
    echo -e "${BLUE}This implementation includes custom CUDA kernels, shared memory parallelism, zero-copy memory, and async processing${NC}"
    docker exec -it tensorflow-gh200 python /app/ultra_benchmark.py --symbols 5
    echo -e "${GREEN}Ultra implementation complete!${NC}"
}

# Function to run scaling test
function run_scaling_test {
    echo -e "${GREEN}Running scaling test...${NC}"
    echo -e "${BLUE}This will test performance with increasing number of symbols${NC}"
    docker exec -it tensorflow-gh200 python /app/enhanced_benchmark.py --scaling-test --max-symbols ${MAX_SYMBOLS} --data-type ${DATA_TYPE}
    echo -e "${GREEN}Scaling test complete!${NC}"
}

# Function to run large-scale benchmark
function run_large_scale_benchmark {
    echo -e "${GREEN}Running large-scale benchmark...${NC}"
    echo -e "${BLUE}This will test performance with many symbols${NC}"
    docker exec -it tensorflow-gh200 python /app/enhanced_benchmark.py --symbols ${SYMBOLS} --data-type ${DATA_TYPE} --large-scale
    echo -e "${GREEN}Large-scale benchmark complete!${NC}"
}

# Function to compare implementations
function run_implementation_comparison {
    echo -e "${GREEN}Comparing original and fixed implementations...${NC}"
    echo -e "${BLUE}This will run both original and fixed implementations${NC}"
    docker exec -it tensorflow-gh200 python /app/compare_implementations.py ${SYMBOLS}
    echo -e "${GREEN}Comparison complete!${NC}"
}

# Function to run both benchmarks and compare results
function run_comparison {
    echo -e "${GREEN}Running comprehensive benchmark comparison...${NC}"
    echo -e "${BLUE}This will run both standard and GH200-optimized implementations${NC}"
    echo -e "${BLUE}Using symbols: ${SYMBOLS}, data type: ${DATA_TYPE}, runs: ${NUM_RUNS}${NC}"
    
    # Run with comparison flag
    docker exec -it tensorflow-gh200 python /app/polygon_benchmark.py --symbols ${SYMBOLS} --data-type ${DATA_TYPE} --num-runs ${NUM_RUNS} --compare
    
    echo -e "${GREEN}Benchmark comparison complete!${NC}"
    echo -e "${YELLOW}Results saved to polygon_benchmark_${DATA_TYPE}.png and polygon_benchmark_${DATA_TYPE}_results.csv${NC}"
}

# Function to view benchmark results
function view_results {
    echo -e "${GREEN}Checking for benchmark results...${NC}"
    
    # Check if results exist in the container
    if docker exec tensorflow-gh200 test -f /app/polygon_benchmark_${DATA_TYPE}_results.csv; then
        echo -e "${BLUE}Found benchmark results for ${DATA_TYPE} data${NC}"
        echo -e "${YELLOW}CSV Results:${NC}"
        docker exec -it tensorflow-gh200 cat /app/polygon_benchmark_${DATA_TYPE}_results.csv
        
        echo -e "\n${YELLOW}The benchmark chart has been saved inside the container.${NC}"
        echo -e "${YELLOW}You can view it by accessing JupyterLab at http://localhost:8888${NC}"
    else
        echo -e "${RED}No benchmark results found for ${DATA_TYPE} data${NC}"
        echo -e "${YELLOW}Run a benchmark first with:${NC}"
        echo -e "  $0 polygon"
        echo -e "  $0 polygon-gh200"
        echo -e "  $0 compare"
    fi
}

# Function to clean up benchmark results
function clean_results {
    echo -e "${GREEN}Cleaning up benchmark results...${NC}"
    docker exec tensorflow-gh200 rm -f /app/polygon_benchmark_*.png /app/polygon_benchmark_*.csv
    echo -e "${GREEN}Cleanup complete!${NC}"
}

# Function to get Jupyter URL
function get_jupyter_url {
    echo -e "${GREEN}JupyterLab is available at:${NC} http://localhost:8888"
}

# Function to open a shell in the container
function open_shell {
    echo -e "${GREEN}Opening shell in the container...${NC}"
    docker exec -it tensorflow-gh200 bash
}

# Function to show container logs
function show_logs {
    echo -e "${GREEN}Showing container logs...${NC}"
    docker-compose logs -f
}

# Main script logic
case "$1" in
    build)
        build_container
        ;;
    start)
        start_container
        ;;
    stop)
        stop_container
        ;;
    polygon)
        run_polygon_benchmark
        ;;
    polygon-gh200)
        run_polygon_gh200_benchmark
        ;;
    polygon-fixed)
        run_polygon_fixed_benchmark
        ;;
    benchmark-fixed)
        run_fixed_benchmark
        ;;
    enhanced)
        run_enhanced_implementation
        ;;
    turbo)
        run_turbo_implementation
        ;;
    ultra)
        run_ultra_implementation
        ;;
    scaling-test)
        run_scaling_test
        ;;
    large-scale)
        run_large_scale_benchmark
        ;;
    compare-impl)
        run_implementation_comparison
        ;;
    compare)
        run_comparison
        ;;
    view-results)
        view_results
        ;;
    clean)
        clean_results
        ;;
    jupyter)
        get_jupyter_url
        ;;
    shell)
        open_shell
        ;;
    logs)
        show_logs
        ;;
    help)
        show_usage
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        show_usage
        exit 1
        ;;
esac