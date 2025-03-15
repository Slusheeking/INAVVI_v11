#!/bin/bash

# Script to check the status of all ATS containers
# For use with the new ats-* naming convention

echo "Checking status of Autonomous Trading System containers..."
echo "--------------------------------------------------------"
echo

# Check all containers with ats- prefix
echo "=== Container Status ==="
docker ps -a --filter "name=ats-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo

# Check running containers only
echo "=== Running Containers ==="
docker ps --filter "name=ats-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo

# Check resource usage
echo "=== Resource Usage ==="
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}" $(docker ps --filter "name=ats-" -q)
echo

# Check for containers in error state
echo "=== Containers in Error State ==="
docker ps -a --filter "name=ats-" --filter "status=exited" --format "table {{.Names}}\t{{.Status}}\t{{.Command}}"
echo

# Check logs for containers in error state (last 10 lines)
ERROR_CONTAINERS=$(docker ps -a --filter "name=ats-" --filter "status=exited" --format "{{.Names}}")
if [ -n "$ERROR_CONTAINERS" ]; then
    echo "=== Error Container Logs ==="
    for container in $ERROR_CONTAINERS; do
        echo "--- $container logs ---"
        docker logs --tail 10 $container
        echo
    done
fi

echo "Container status check completed."
echo "For detailed logs of a specific container, run: docker logs [container-name]"