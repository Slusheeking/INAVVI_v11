#!/bin/bash

# API Validator Script
# This script validates all external API dependencies to ensure they're working properly
# before starting the services.

# Source common scripts
source /app/docker/common/dependency_check.sh

# Set service name for logging
export SERVICE_NAME="API Validator"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to validate Polygon API
validate_polygon_api() {
    echo "[INFO] Validating Polygon API..."
    
    # Check if API key is set
    if [ -z "$POLYGON_API_KEY" ] || [ "$POLYGON_API_KEY" = "your_polygon_api_key" ]; then
        echo -e "${RED}[ERROR] Polygon API key is not set or is using the default value.${NC}"
        echo "[ERROR] Please set a valid POLYGON_API_KEY in your .env file."
        return 1
    fi
    
    # Test API with a simple request
    echo "[INFO] Testing Polygon API with a simple request..."
    RESPONSE=$(curl -s -X GET "https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-01/2023-01-07?apiKey=$POLYGON_API_KEY")
    
    # Check if response contains an error
    if echo "$RESPONSE" | grep -q "\"status\": \"ERROR\""; then
        ERROR_MSG=$(echo "$RESPONSE" | grep -o "\"message\": \"[^\"]*\"")
        echo -e "${RED}[ERROR] Polygon API test failed: $ERROR_MSG${NC}"
        return 1
    elif echo "$RESPONSE" | grep -q "\"results\""; then
        echo -e "${GREEN}[SUCCESS] Polygon API is working correctly.${NC}"
        set_feature_flag "polygon_api" 1 "/app/feature_flags.json" "API validated successfully"
        return 0
    else
        echo -e "${RED}[ERROR] Unexpected response from Polygon API.${NC}"
        echo "$RESPONSE"
        return 1
    fi
}

# Function to validate Unusual Whales API
validate_unusual_whales_api() {
    echo "[INFO] Validating Unusual Whales API..."
    
    # Check if API key is set
    if [ -z "$UNUSUAL_WHALES_API_KEY" ] || [ "$UNUSUAL_WHALES_API_KEY" = "your_unusual_whales_api_key" ]; then
        echo -e "${YELLOW}[WARN] Unusual Whales API key is not set or is using the default value.${NC}"
        echo "[WARN] This is a non-critical API. The system will continue without Unusual Whales data."
        set_feature_flag "unusual_whales_api" 0 "/app/feature_flags.json" "API key not set"
        return 0
    fi
    
    # Test API with a simple request
    echo "[INFO] Testing Unusual Whales API with a simple request..."
    RESPONSE=$(curl -s -X GET "https://api.unusualwhales.com/api/flow/unusual_options?api_key=$UNUSUAL_WHALES_API_KEY&limit=1")
    
    # Check if response contains an error
    if echo "$RESPONSE" | grep -q "\"error\""; then
        ERROR_MSG=$(echo "$RESPONSE" | grep -o "\"error\": \"[^\"]*\"")
        echo -e "${YELLOW}[WARN] Unusual Whales API test failed: $ERROR_MSG${NC}"
        echo "[WARN] This is a non-critical API. The system will continue without Unusual Whales data."
        set_feature_flag "unusual_whales_api" 0 "/app/feature_flags.json" "API test failed"
        return 0
    elif echo "$RESPONSE" | grep -q "\"data\""; then
        echo -e "${GREEN}[SUCCESS] Unusual Whales API is working correctly.${NC}"
        set_feature_flag "unusual_whales_api" 1 "/app/feature_flags.json" "API validated successfully"
        return 0
    else
        echo -e "${YELLOW}[WARN] Unexpected response from Unusual Whales API.${NC}"
        echo "$RESPONSE"
        echo "[WARN] This is a non-critical API. The system will continue without Unusual Whales data."
        set_feature_flag "unusual_whales_api" 0 "/app/feature_flags.json" "Unexpected API response"
        return 0
    fi
}

# Function to validate Alpaca API
validate_alpaca_api() {
    echo "[INFO] Validating Alpaca API..."
    
    # Check if API keys are set
    if [ -z "$ALPACA_API_KEY" ] || [ "$ALPACA_API_KEY" = "your_alpaca_api_key" ] || 
       [ -z "$ALPACA_API_SECRET" ] || [ "$ALPACA_API_SECRET" = "your_alpaca_api_secret" ]; then
        echo -e "${RED}[ERROR] Alpaca API keys are not set or are using the default values.${NC}"
        echo "[ERROR] Please set valid ALPACA_API_KEY and ALPACA_API_SECRET in your .env file."
        return 1
    fi
    
    # Test API with a simple request
    echo "[INFO] Testing Alpaca API with a simple request..."
    RESPONSE=$(curl -s -X GET \
        -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
        -H "APCA-API-SECRET-KEY: $ALPACA_API_SECRET" \
        "$ALPACA_API_BASE_URL/v2/account")
    
    # Check if response contains an error
    if echo "$RESPONSE" | grep -q "\"code\""; then
        ERROR_MSG=$(echo "$RESPONSE" | grep -o "\"message\": \"[^\"]*\"")
        echo -e "${RED}[ERROR] Alpaca API test failed: $ERROR_MSG${NC}"
        return 1
    elif echo "$RESPONSE" | grep -q "\"account_number\""; then
        echo -e "${GREEN}[SUCCESS] Alpaca API is working correctly.${NC}"
        
        # Check if using paper trading
        if echo "$ALPACA_API_BASE_URL" | grep -q "paper-api"; then
            echo -e "${YELLOW}[WARN] Using Alpaca Paper Trading API. No real money will be used.${NC}"
            set_feature_flag "alpaca_api" 1 "/app/feature_flags.json" "Paper trading API validated"
        else
            echo -e "${GREEN}[INFO] Using Alpaca Live Trading API. REAL MONEY WILL BE USED!${NC}"
            set_feature_flag "alpaca_api" 1 "/app/feature_flags.json" "Live trading API validated"
        fi
        
        # Get account information
        BUYING_POWER=$(echo "$RESPONSE" | grep -o "\"buying_power\":\"[^\"]*\"" | cut -d'"' -f4)
        CASH=$(echo "$RESPONSE" | grep -o "\"cash\":\"[^\"]*\"" | cut -d'"' -f4)
        EQUITY=$(echo "$RESPONSE" | grep -o "\"equity\":\"[^\"]*\"" | cut -d'"' -f4)
        
        echo "[INFO] Account Information:"
        echo "[INFO] Buying Power: $BUYING_POWER"
        echo "[INFO] Cash: $CASH"
        echo "[INFO] Equity: $EQUITY"
        
        return 0
    else
        echo -e "${RED}[ERROR] Unexpected response from Alpaca API.${NC}"
        echo "$RESPONSE"
        return 1
    fi
}

# Function to validate Slack integration
validate_slack_integration() {
    echo "[INFO] Validating Slack integration..."
    
    # Check if Slack tokens are set
    if [ -z "$SLACK_BOT_TOKEN" ] || [ "$SLACK_BOT_TOKEN" = "xoxb-your-token" ] || 
       [ -z "$SLACK_WEBHOOK_URL" ] || [ "$SLACK_WEBHOOK_URL" = "https://hooks.slack.com/services/your/webhook/url" ]; then
        echo -e "${YELLOW}[WARN] Slack tokens are not set or are using the default values.${NC}"
        echo "[WARN] This is a non-critical integration. The system will continue without Slack notifications."
        set_feature_flag "slack_integration" 0 "/app/feature_flags.json" "Tokens not set"
        return 0
    fi
    
    # Test Slack webhook
    echo "[INFO] Testing Slack webhook..."
    RESPONSE=$(curl -s -X POST -H "Content-type: application/json" \
        --data "{\"text\":\"API Validator: Testing Slack integration\"}" \
        "$SLACK_WEBHOOK_URL")
    
    if [ "$RESPONSE" = "ok" ]; then
        echo -e "${GREEN}[SUCCESS] Slack webhook is working correctly.${NC}"
        set_feature_flag "slack_webhook" 1 "/app/feature_flags.json" "Webhook validated"
    else
        echo -e "${YELLOW}[WARN] Slack webhook test failed: $RESPONSE${NC}"
        echo "[WARN] This is a non-critical integration. The system will continue without Slack webhook notifications."
        set_feature_flag "slack_webhook" 0 "/app/feature_flags.json" "Webhook test failed"
    fi
    
    # Test Slack bot token
    echo "[INFO] Testing Slack bot token..."
    RESPONSE=$(curl -s -X POST -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
        -H "Content-type: application/json" \
        --data "{\"channel\":\"$SLACK_CHANNEL\",\"text\":\"API Validator: Testing Slack bot integration\"}" \
        "https://slack.com/api/chat.postMessage")
    
    if echo "$RESPONSE" | grep -q "\"ok\":true"; then
        echo -e "${GREEN}[SUCCESS] Slack bot token is working correctly.${NC}"
        set_feature_flag "slack_bot" 1 "/app/feature_flags.json" "Bot token validated"
        return 0
    else
        ERROR_MSG=$(echo "$RESPONSE" | grep -o "\"error\":\"[^\"]*\"")
        echo -e "${YELLOW}[WARN] Slack bot token test failed: $ERROR_MSG${NC}"
        echo "[WARN] This is a non-critical integration. The system will continue without Slack bot notifications."
        set_feature_flag "slack_bot" 0 "/app/feature_flags.json" "Bot token test failed"
        return 0
    fi
}

# Main validation function
validate_all_apis() {
    echo "[INFO] Starting API validation..."
    
    # Create feature flags file if it doesn't exist
    if [ ! -f "/app/feature_flags.json" ]; then
        echo "{\"service\": \"$SERVICE_NAME\", \"features\": {}, \"updated_at\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\"}" > "/app/feature_flags.json"
    fi
    
    # Validate each API
    POLYGON_RESULT=0
    UNUSUAL_WHALES_RESULT=0
    ALPACA_RESULT=0
    SLACK_RESULT=0
    
    validate_polygon_api
    POLYGON_RESULT=$?
    
    validate_unusual_whales_api
    UNUSUAL_WHALES_RESULT=$?
    
    validate_alpaca_api
    ALPACA_RESULT=$?
    
    validate_slack_integration
    SLACK_RESULT=$?
    
    # Summarize results
    echo ""
    echo "API Validation Summary:"
    echo "----------------------"
    
    if [ $POLYGON_RESULT -eq 0 ]; then
        echo -e "${GREEN}✓ Polygon API: VALID${NC}"
    else
        echo -e "${RED}✗ Polygon API: INVALID${NC}"
    fi
    
    if [ $UNUSUAL_WHALES_RESULT -eq 0 ]; then
        if grep -q "\"unusual_whales_api\":{\"available\":1" "/app/feature_flags.json"; then
            echo -e "${GREEN}✓ Unusual Whales API: VALID${NC}"
        else
            echo -e "${YELLOW}⚠ Unusual Whales API: NOT CONFIGURED (non-critical)${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ Unusual Whales API: INVALID (non-critical)${NC}"
    fi
    
    if [ $ALPACA_RESULT -eq 0 ]; then
        if echo "$ALPACA_API_BASE_URL" | grep -q "paper-api"; then
            echo -e "${YELLOW}⚠ Alpaca API: VALID (PAPER TRADING)${NC}"
        else
            echo -e "${GREEN}✓ Alpaca API: VALID (LIVE TRADING)${NC}"
        fi
    else
        echo -e "${RED}✗ Alpaca API: INVALID${NC}"
    fi
    
    if [ $SLACK_RESULT -eq 0 ]; then
        if grep -q "\"slack_webhook\":{\"available\":1" "/app/feature_flags.json" || grep -q "\"slack_bot\":{\"available\":1" "/app/feature_flags.json"; then
            echo -e "${GREEN}✓ Slack Integration: VALID${NC}"
        else
            echo -e "${YELLOW}⚠ Slack Integration: NOT CONFIGURED (non-critical)${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ Slack Integration: INVALID (non-critical)${NC}"
    fi
    
    echo ""
    
    # Determine if validation passed
    if [ $POLYGON_RESULT -eq 0 ] && [ $ALPACA_RESULT -eq 0 ]; then
        echo -e "${GREEN}API Validation PASSED. All critical APIs are working.${NC}"
        return 0
    else
        echo -e "${RED}API Validation FAILED. One or more critical APIs are not working.${NC}"
        return 1
    fi
}

# Execute validation
validate_all_apis
exit $?