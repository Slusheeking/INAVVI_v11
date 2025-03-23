#!/bin/bash
# Master Production Management Script for Trading System
# This script provides a unified interface for managing the production system

set -e

# Display banner
echo "========================================================"
echo "  INAVVI Trading System Production Manager"
echo "========================================================"
echo "This script provides a unified interface for managing the production system."
echo ""

# Function to display help
show_help() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  setup       - Set up the production environment"
    echo "  verify      - Verify production readiness"
    echo "  start       - Start the production system"
    echo "  stop        - Stop the production system"
    echo "  restart     - Restart the production system"
    echo "  status      - Check the status of the production system"
    echo "  logs        - View system logs"
    echo "  test        - Run tests in the Docker container"
    echo "  monitor     - Monitor the production system"
    echo "  backup      - Create a backup of the production data"
    echo "  restore     - Restore from a backup"
    echo "  update      - Update the production system"
    echo "  help        - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup    - Set up the production environment"
    echo "  $0 start    - Start the production system"
    echo "  $0 monitor  - Monitor the production system"
    echo ""
}

# Check if a command was provided
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

# Process command
case "$1" in
    setup)
        echo "Setting up the production environment..."
        bash ./verify_production_readiness.sh
        if [ $? -ne 0 ]; then
            echo "Production readiness verification failed. Please fix the issues and try again."
            exit 1
        fi
        bash ./setup_production.sh
        ;;
    verify)
        echo "Verifying production readiness..."
        bash ./verify_production_readiness.sh
        ;;
    start)
        echo "Starting the production system..."
        docker-compose -f docker-compose.unified.yml up -d
        echo "Waiting for container to initialize..."
        sleep 10
        docker exec -it trading-system bash -c "cd /app/project && python scripts/start_system.py"
        echo "Production system started."
        ;;
    stop)
        echo "Stopping the production system..."
        docker exec -it trading-system bash -c "cd /app/project && python scripts/stop_system.py"
        docker-compose -f docker-compose.unified.yml down
        echo "Production system stopped."
        ;;
    restart)
        echo "Restarting the production system..."
        docker exec -it trading-system bash -c "cd /app/project && python scripts/stop_system.py"
        docker-compose -f docker-compose.unified.yml restart
        sleep 10
        docker exec -it trading-system bash -c "cd /app/project && python scripts/start_system.py"
        echo "Production system restarted."
        ;;
    status)
        echo "Checking the status of the production system..."
        if docker ps | grep -q trading-system; then
            echo "Container status: Running"
            if docker exec -it trading-system bash -c "ps aux | grep -v grep | grep -q 'python.*start_system.py'"; then
                echo "Trading system status: Running"
            else
                echo "Trading system status: Stopped"
            fi
            if docker exec -it trading-system redis-cli ping | grep -q "PONG"; then
                echo "Redis status: Running"
            else
                echo "Redis status: Stopped"
            fi
            if curl -s http://localhost:9090/-/healthy | grep -q "Prometheus is Healthy"; then
                echo "Prometheus status: Running"
            else
                echo "Prometheus status: Stopped"
            fi
        else
            echo "Container status: Stopped"
        fi
        ;;
    logs)
        if [ $# -eq 2 ]; then
            echo "Viewing logs for $2..."
            case "$2" in
                trading)
                    docker exec -it trading-system bash -c "tail -f /app/logs/trading_system.out.log"
                    ;;
                redis)
                    docker exec -it trading-system bash -c "tail -f /var/log/redis/redis-server.out.log"
                    ;;
                prometheus)
                    docker exec -it trading-system bash -c "tail -f /var/log/prometheus.out.log"
                    ;;
                all)
                    docker-compose -f docker-compose.unified.yml logs -f
                    ;;
                *)
                    echo "Unknown log type: $2"
                    echo "Available log types: trading, redis, prometheus, all"
                    exit 1
                    ;;
            esac
        else
            echo "Viewing all logs..."
            docker-compose -f docker-compose.unified.yml logs -f
        fi
        ;;
    test)
        echo "Running tests in the Docker container..."
        bash ./run_tests_in_docker.sh
        ;;
    monitor)
        echo "Monitoring the production system..."
        if [ $# -eq 2 ] && [ "$2" == "--watch" ]; then
            bash ./monitor_production.sh --watch
        else
            bash ./monitor_production.sh
        fi
        ;;
    backup)
        echo "Creating a backup of the production data..."
        if [ $# -eq 2 ] && [ "$2" == "--remote" ]; then
            bash ./backup_production_data.sh --remote
        elif [ $# -eq 2 ] && [ "$2" == "--schedule" ]; then
            bash ./backup_production_data.sh --schedule
        else
            bash ./backup_production_data.sh
        fi
        ;;
    restore)
        if [ $# -ne 2 ]; then
            echo "Error: Backup file not specified."
            echo "Usage: $0 restore <backup_file>"
            exit 1
        fi
        echo "Restoring from backup $2..."
        if [ -f "$2" ]; then
            bash "$2"
        else
            echo "Error: Backup file $2 not found."
            exit 1
        fi
        ;;
    update)
        echo "Updating the production system..."
        echo "Pulling latest code..."
        git pull
        
        echo "Stopping the system..."
        docker exec -it trading-system bash -c "cd /app/project && python scripts/stop_system.py"
        
        echo "Rebuilding the Docker image..."
        docker-compose -f docker-compose.unified.yml build
        
        echo "Restarting the system..."
        docker-compose -f docker-compose.unified.yml up -d
        
        echo "Waiting for container to initialize..."
        sleep 10
        
        echo "Starting the trading system..."
        docker exec -it trading-system bash -c "cd /app/project && python scripts/start_system.py"
        
        echo "Production system updated and restarted."
        ;;
    shell)
        echo "Opening shell in the Docker container..."
        docker exec -it trading-system bash
        ;;
    help)
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        show_help
        exit 1
        ;;
esac

# Make all scripts executable
chmod +x setup_production.sh
chmod +x verify_production_readiness.sh
chmod +x run_tests_in_docker.sh
chmod +x monitor_production.sh
chmod +x backup_production_data.sh