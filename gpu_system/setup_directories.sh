#!/bin/bash

# Create necessary directories
echo "Creating required directories..."

# Project root directories
mkdir -p ../data
mkdir -p ../data/cache
mkdir -p ../models
mkdir -p ../logs

# Prometheus and Redis directories
mkdir -p ./prometheus/data
mkdir -p ./redis/data

# Set permissions
echo "Setting permissions..."
chmod -R 777 ../data
chmod -R 777 ../models
chmod -R 777 ../logs
chmod -R 777 ./prometheus/data
chmod -R 777 ./redis/data

echo "Creating .gitignore entries..."
cat >> ../.gitignore << EOL
# Environment variables
.env

# Data and cache directories
data/
data/cache/

# Model files
models/

# Log files
logs/

# Redis data
redis/data/

# Prometheus data
prometheus/data/

# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd
EOL

echo "Setup complete. Next steps:"
echo "1. Edit .env file with your API keys:"
echo "   nano ../.env"
echo ""
echo "2. Build and start the container:"
echo "   docker-compose -f docker-compose.unified.yml up --build"
echo ""
echo "3. Access services at:"
echo "   - Jupyter Lab: http://localhost:8888"
echo "   - Redis: localhost:6379"
echo "   - Prometheus: http://localhost:9090"
echo "   - Redis Exporter: http://localhost:9121"
