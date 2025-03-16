#!/bin/bash

# Source dependency check library
source /app/docker/common/dependency_check.sh

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Validate common required environment variables
validate_env_var "ENVIRONMENT" true "development"
validate_env_var "LOG_LEVEL" false "INFO"
validate_env_var "TZ" false "UTC"
validate_env_var "VERSION" false "latest"

# Database environment variables
validate_env_var "TIMESCALEDB_HOST" true
validate_env_var "TIMESCALEDB_PORT" false "5432"
validate_env_var "TIMESCALEDB_DATABASE" true
validate_env_var "TIMESCALEDB_USER" true
validate_env_var "TIMESCALEDB_PASSWORD" true

# Redis environment variables
validate_env_var "REDIS_HOST" true
validate_env_var "REDIS_PORT" false "6379"
validate_env_var "REDIS_PASSWORD" true

# API Keys (critical for trading)
validate_env_var "POLYGON_API_KEY" true
validate_env_var "ALPACA_API_KEY" true
validate_env_var "ALPACA_API_SECRET" true
validate_env_var "ALPACA_API_BASE_URL" false "https://paper-api.alpaca.markets"

# API Keys (non-critical)
validate_env_var "UNUSUAL_WHALES_API_KEY" false

# Slack Integration (non-critical)
validate_env_var "SLACK_BOT_TOKEN" false
validate_env_var "SLACK_WEBHOOK_URL" false
validate_env_var "SLACK_CHANNEL" false "system-notifications"

# Trading Configuration
validate_env_var "MAX_POSITION_SIZE" false "0.1"
validate_env_var "RISK_PERCENTAGE" false "0.02"
validate_env_var "MAX_POSITIONS" false "10"
validate_env_var "MAX_POSITION_VALUE" false "10000"
validate_env_var "MAX_DAILY_VALUE" false "50000"

# GPU settings (non-critical)
validate_env_var "GPU_ENABLED" false "false"
validate_env_var "NVIDIA_VISIBLE_DEVICES" false "all"

# Feature Engineering Settings
validate_env_var "FEATURE_STORE_PATH" false "/app/features"
validate_env_var "FEATURE_CACHE_ENABLED" false "true"
validate_env_var "FEATURE_CACHE_TTL" false "3600"

# Model Settings
validate_env_var "MODEL_REGISTRY_PATH" false "/app/models/registry"
validate_env_var "MODEL_CHECKPOINT_PATH" false "/app/models/checkpoints"
validate_env_var "MODEL_SERVING_PORT" false "8500"

# Backtesting Settings
validate_env_var "BACKTEST_DATA_PATH" false "/app/data/backtest"
validate_env_var "BACKTEST_RESULTS_PATH" false "/app/results/backtest"

# Monitoring Settings
validate_env_var "PROMETHEUS_ENABLED" false "true"
validate_env_var "GRAFANA_ENABLED" false "true"

# Check for placeholder API keys
if [ "$POLYGON_API_KEY" = "your_polygon_api_key" ]; then
  echo -e "${RED}[ERROR] POLYGON_API_KEY is set to the placeholder value. Please update with a real API key.${NC}"
  exit 1
fi

if [ "$ALPACA_API_KEY" = "your_alpaca_api_key" ] || [ "$ALPACA_API_SECRET" = "your_alpaca_api_secret" ]; then
  echo -e "${RED}[ERROR] ALPACA_API_KEY or ALPACA_API_SECRET is set to the placeholder value. Please update with real API keys.${NC}"
  exit 1
fi

if [ "$UNUSUAL_WHALES_API_KEY" = "your_unusual_whales_api_key" ]; then
  echo -e "${YELLOW}[WARN] UNUSUAL_WHALES_API_KEY is set to the placeholder value. This is non-critical but some features will be disabled.${NC}"
fi

# Validate API keys if requested
if [ "${VALIDATE_APIS:-false}" = "true" ]; then
  echo "[INFO] Validating API keys..."
  source /app/docker/common/api_validator.sh
fi

echo -e "${GREEN}[SUCCESS] Environment validation completed successfully.${NC}"