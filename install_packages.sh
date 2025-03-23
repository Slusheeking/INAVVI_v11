#!/bin/bash
# Script to install required Python packages in the container

echo "=== Installing Required Python Packages ==="

# Install python-dotenv
echo "Installing python-dotenv..."
pip install python-dotenv

# Install pandas
echo "Installing pandas..."
pip install pandas

# Install aiohttp
echo "Installing aiohttp..."
pip install aiohttp

# Install any other required packages
echo "Installing other required packages..."
pip install requests websockets prometheus_client

echo "=== All packages installed successfully ==="