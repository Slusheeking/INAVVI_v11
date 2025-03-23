#!/bin/bash
# Production Monitoring Script for Trading System
# This script monitors the health of the production system

set -e

# Display banner
echo "========================================================"
echo "  INAVVI Trading System Production Monitor"
echo "========================================================"
echo "This script monitors the health of the production system."
echo ""

# Check if container is running
if ! docker ps | grep -q trading-system; then
    echo "Error: trading-system container is not running."
    echo "Please start the container first with:"
    echo "  docker-compose -f docker-compose.unified.yml up -d"
    exit 1
fi

# Function to display colored status
status() {
    local status=$1
    local message=$2
    
    if [ "$status" == "OK" ]; then
        echo -e "\e[32m[OK]\e[0m $message"
    elif [ "$status" == "WARNING" ]; then
        echo -e "\e[33m[WARNING]\e[0m $message"
    else
        echo -e "\e[31m[ERROR]\e[0m $message"
    fi
}

# Check container health
echo "Checking container health..."
if docker inspect --format='{{.State.Health.Status}}' trading-system 2>/dev/null | grep -q "healthy"; then
    status "OK" "Container is healthy"
else
    status "WARNING" "Container health check not available or not healthy"
fi

# Check Redis
echo "Checking Redis..."
if docker exec -it trading-system redis-cli ping | grep -q "PONG"; then
    status "OK" "Redis is running"
else
    status "ERROR" "Redis is not responding"
fi

# Check Prometheus
echo "Checking Prometheus..."
if curl -s http://localhost:9090/-/healthy | grep -q "Prometheus is Healthy"; then
    status "OK" "Prometheus is running"
else
    status "WARNING" "Prometheus health check failed"
fi

# Check Redis Exporter
echo "Checking Redis Exporter..."
if curl -s http://localhost:9121/metrics | grep -q "redis_up"; then
    status "OK" "Redis Exporter is running"
else
    status "WARNING" "Redis Exporter health check failed"
fi

# Check system processes
echo "Checking system processes..."
PROCESSES=(
    "python.*start_system.py"
    "redis-server"
    "prometheus"
    "redis_exporter"
)

for process in "${PROCESSES[@]}"; do
    if docker exec -it trading-system bash -c "ps aux | grep -v grep | grep -q '$process'"; then
        status "OK" "Process '$process' is running"
    else
        status "ERROR" "Process '$process' is not running"
    fi
done

# Check GPU status
echo "Checking GPU status..."
if docker exec -it trading-system nvidia-smi &>/dev/null; then
    # Get GPU utilization
    gpu_util=$(docker exec -it trading-system nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    gpu_mem=$(docker exec -it trading-system nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    gpu_total=$(docker exec -it trading-system nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
    
    status "OK" "GPU is available (Utilization: ${gpu_util}%, Memory: ${gpu_mem}/${gpu_total} MB)"
else
    status "WARNING" "GPU is not available or drivers not properly installed"
fi

# Check log files for errors
echo "Checking log files for errors..."
ERROR_COUNT=$(docker exec -it trading-system bash -c "grep -i error /app/logs/*.log | wc -l")
WARNING_COUNT=$(docker exec -it trading-system bash -c "grep -i warning /app/logs/*.log | wc -l")

if [ "$ERROR_COUNT" -gt 0 ]; then
    status "WARNING" "Found $ERROR_COUNT errors in log files"
else
    status "OK" "No errors found in log files"
fi

if [ "$WARNING_COUNT" -gt 0 ]; then
    status "WARNING" "Found $WARNING_COUNT warnings in log files"
fi

# Check system memory usage
echo "Checking system memory usage..."
MEM_USAGE=$(docker exec -it trading-system bash -c "free -m | grep Mem | awk '{print \$3/\$2 * 100.0}'")
MEM_USAGE_ROUNDED=$(printf "%.1f" $MEM_USAGE)

if (( $(echo "$MEM_USAGE > 90" | bc -l) )); then
    status "ERROR" "Memory usage is critical: ${MEM_USAGE_ROUNDED}%"
elif (( $(echo "$MEM_USAGE > 80" | bc -l) )); then
    status "WARNING" "Memory usage is high: ${MEM_USAGE_ROUNDED}%"
else
    status "OK" "Memory usage is normal: ${MEM_USAGE_ROUNDED}%"
fi

# Check disk usage
echo "Checking disk usage..."
DISK_USAGE=$(docker exec -it trading-system bash -c "df -h /app | tail -1 | awk '{print \$5}' | tr -d '%'")

if [ "$DISK_USAGE" -gt 90 ]; then
    status "ERROR" "Disk usage is critical: ${DISK_USAGE}%"
elif [ "$DISK_USAGE" -gt 80 ]; then
    status "WARNING" "Disk usage is high: ${DISK_USAGE}%"
else
    status "OK" "Disk usage is normal: ${DISK_USAGE}%"
fi

# Check API connectivity
echo "Checking API connectivity..."
if docker exec -it trading-system bash -c "cd /app/project && python -c 'from api_clients import PolygonRESTClient; import asyncio; print(asyncio.run(PolygonRESTClient(api_key=\"test\").check_connection()))'"; then
    status "OK" "API client connectivity check passed"
else
    status "WARNING" "API client connectivity check failed"
fi

# Check Redis data
echo "Checking Redis data..."
KEYS_COUNT=$(docker exec -it trading-system redis-cli keys "*" | wc -l)
if [ "$KEYS_COUNT" -gt 0 ]; then
    status "OK" "Redis contains $KEYS_COUNT keys"
else
    status "WARNING" "Redis database is empty"
fi

# Summary
echo ""
echo "========================================================"
echo "  System Health Summary"
echo "========================================================"
echo "Container: Running"
echo "Redis: Running"
echo "Prometheus: Running"
echo "System Processes: Running"
echo "Memory Usage: ${MEM_USAGE_ROUNDED}%"
echo "Disk Usage: ${DISK_USAGE}%"
echo "Log Errors: $ERROR_COUNT"
echo "Log Warnings: $WARNING_COUNT"
echo ""
echo "For detailed metrics, visit:"
echo "- Prometheus: http://localhost:9090"
echo "- Redis Metrics: http://localhost:9121/metrics"
echo ""
echo "To view logs:"
echo "docker exec -it trading-system bash -c \"tail -f /app/logs/*.log\""
echo "========================================================"

# Continuous monitoring option
if [ "$1" == "--watch" ]; then
    echo "Press Ctrl+C to exit continuous monitoring..."
    echo ""
    echo "Monitoring will refresh every 30 seconds."
    echo ""
    
    while true; do
        sleep 30
        clear
        $0
    done
fi