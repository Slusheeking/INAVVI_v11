#!/bin/bash

# dependency_check.sh - Common functions for dependency checking

# Check if a dependency is available
# Arguments:
#   $1: Dependency name (for logs)
#   $2: Check command (will be executed to test if dependency is available)
#   $3: Is critical (true/false) - if true, exit on failure
#   $4: Max retries (default: 30)
#   $5: Initial retry interval in seconds (default: 2)
function check_dependency() {
  local dependency_name=$1
  local check_command=$2
  local is_critical=${3:-false}
  local max_retries=${4:-30}
  local retry_interval=${5:-2}
  local retry_count=0
  
  echo "[DEBUG] Checking dependency: $dependency_name (critical: $is_critical, max_retries: $max_retries)"
  
  until eval "$check_command"; do
    retry_count=$((retry_count+1))
    if [ $retry_count -ge $max_retries ]; then
      echo "[ERROR] Failed to connect to $dependency_name after $max_retries attempts."
      if [ "$is_critical" = true ]; then
        echo "[FATAL] Critical dependency $dependency_name unavailable. Exiting."
        exit 1
      else
        echo "[WARN] Non-critical dependency $dependency_name unavailable. Continuing with reduced functionality."
        return 1
      fi
    fi
    
    echo "[INFO] Waiting for $dependency_name... (Attempt $retry_count/$max_retries)"
    sleep $retry_interval
    
    # Exponential backoff with maximum cap
    retry_interval=$((retry_interval*2))
    if [ $retry_interval -gt 30 ]; then
      retry_interval=30
    fi
  done
  
  echo "[SUCCESS] $dependency_name connection successful"
  return 0
}

# Validate required environment variables
# Arguments:
#   $1: Variable name
#   $2: Is critical (true/false) - if true, exit on missing variable
#   $3: Default value (optional) - if provided, use this when variable is missing
function validate_env_var() {
  local var_name=$1
  local is_critical=${2:-true}
  local default_value=$3
  echo "[DEBUG] Validating environment variable: $var_name (critical: $is_critical, default: ${default_value:-none})"
  
  # Use indirect reference to check if variable is set
  local var_value="${!var_name}"
  
  if [ -z "$var_value" ]; then
    if [ -n "$default_value" ]; then
      echo "[WARN] $var_name not set, using default value: $default_value"
      # Export the variable with default value
      export $var_name="$default_value"
      return 0
    elif [ "$is_critical" = true ]; then
      echo "[ERROR] Required environment variable $var_name is not set. Exiting."
      exit 1
    else
      echo "[WARN] Environment variable $var_name is not set. Some features may be unavailable."
      return 1
    fi
  else
    echo "[DEBUG] Environment variable $var_name is set"
  fi
  
  return 0
}

# Check TimescaleDB connection
# Arguments:
#   $1: Is critical (true/false) - if true, exit on failure
function check_timescaledb() {
  local is_critical=${1:-true}
  echo "[DEBUG] Starting TimescaleDB check (critical: $is_critical)"
  echo "[DEBUG] TimescaleDB host: $TIMESCALEDB_HOST, port: $TIMESCALEDB_PORT, database: $TIMESCALEDB_DATABASE, user: $TIMESCALEDB_USER"
  
  local check_cmd="PGPASSWORD=\$TIMESCALEDB_PASSWORD psql -h \$TIMESCALEDB_HOST -U \$TIMESCALEDB_USER -d \$TIMESCALEDB_DATABASE -c '\q'"
  
  check_dependency "TimescaleDB" "$check_cmd" "$is_critical"
  local result=$?
  
  if [ $result -eq 0 ]; then
    # Check if TimescaleDB extension is installed
    echo "[DEBUG] Checking TimescaleDB extension..."
    if PGPASSWORD=$TIMESCALEDB_PASSWORD psql -h $TIMESCALEDB_HOST -U $TIMESCALEDB_USER -d $TIMESCALEDB_DATABASE -c "SELECT extname FROM pg_extension WHERE extname = 'timescaledb';" | grep -q "timescaledb"; then
      echo "[SUCCESS] TimescaleDB extension is installed"
    else
      echo "[WARN] TimescaleDB extension is not installed. Hypertables will not be created."
    fi
  fi
  
  return $result
}

# Check Redis connection
# Arguments:
#   $1: Is critical (true/false) - if true, exit on failure
function check_redis() {
  local is_critical=${1:-false}
  echo "[DEBUG] Starting Redis check (critical: $is_critical)"
  
  # First check basic connectivity
  local port_check="nc -z \$REDIS_HOST \$REDIS_PORT"
  echo "[DEBUG] Redis host: $REDIS_HOST, port: $REDIS_PORT"
  check_dependency "Redis port" "$port_check" "$is_critical"
  local port_result=$?
  
  if [ $port_result -eq 0 ]; then
    # Only check authentication if port check succeeds
    echo "[DEBUG] Redis port check successful, checking authentication..."
    redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD ping | grep -q "PONG"
    if [ $? -ne 0 ]; then
      echo "[WARN] Redis authentication failed. Some features requiring Redis may be unavailable."
      return 1
    fi
    echo "[SUCCESS] Redis authentication successful"
  else
    echo "[DEBUG] Redis port check failed, skipping authentication check"
  fi
  
  return $port_result
}

# Check HTTP service health
# Arguments:
#   $1: Service name
#   $2: Service URL (including protocol, host, port, and path)
#   $3: Is critical (true/false) - if true, exit on failure
function check_http_service() {
  local service_name=$1
  local service_url=$2
  local is_critical=${3:-false}
  local max_retries=${4:-30}
  local retry_interval=${5:-2}
  
  echo "[DEBUG] Checking HTTP service: $service_name at $service_url (critical: $is_critical)"
  local check_cmd="curl -s -o /dev/null -w '%{http_code}' $service_url | grep -q '2[0-9][0-9]'"
  check_dependency "$service_name" "$check_cmd" "$is_critical"
  local result=$?
  
  if [ $result -eq 0 ]; then
    echo "[SUCCESS] HTTP service $service_name is available and returning 2xx status"
  else
    echo "[WARN] HTTP service $service_name is not available or not returning 2xx status"
  fi
  
  return $result
}

# Create health status file
# Arguments:
#   $1: Status (starting, degraded, healthy)
#   $2: Status file path (default: /app/health_status.json)
function set_health_status() {
  local status=$1
  local status_file=${2:-/app/health_status.json}
  local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  local service_name=${SERVICE_NAME:-"Unknown Service"}
  local previous_status="unknown"
  
  # Read previous status if file exists
  if [ -f "$status_file" ]; then
    previous_status=$(grep -o '"status":"[^"]*"' "$status_file" | cut -d'"' -f4)
  fi
  
  echo "[INFO] Service $service_name changing status from $previous_status to $status"
  
  # Create a more detailed health status file
  cat > $status_file << EOF
{
  "status": "$status",
  "timestamp": "$timestamp",
  "service": "$service_name",
  "previous_status": "$previous_status",
  "dependencies": {
    "timescaledb": "$([ -z "$TIMESCALEDB_HOST" ] && echo "not_configured" || echo "configured")",
    "redis": "$([ -z "$REDIS_HOST" ] && echo "not_configured" || echo "configured")"
  }
}
EOF
  
  echo "[INFO] Health status set to: $status"
}

# Initialize feature flags based on dependency availability
# Arguments:
#   $1: Feature name
#   $2: Is available (0/1)
#   $3: Feature flags file (default: /app/feature_flags.json)
function set_feature_flag() {
  local feature=$1
  local available=$2
  local flags_file=${3:-/app/feature_flags.json}
  local service_name=${SERVICE_NAME:-"Unknown Service"}
  local reason=${4:-"Not specified"}
  
  echo "[INFO] Setting feature flag '$feature' to: $available (reason: $reason)"
  
  # Create file with empty JSON object if it doesn't exist
  if [ ! -f "$flags_file" ]; then
    echo "{\"service\": \"$service_name\", \"features\": {}, \"updated_at\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\"}" > "$flags_file"
  fi
  
  # Read current flags
  local current_flags=$(cat "$flags_file")
  
  # Create a more detailed feature flags file
  cat > "$flags_file" << EOF
{
  "service": "$service_name",
  "features": $(echo "$current_flags" | grep -q "\"features\":" && echo "$current_flags" | sed -n 's/.*"features": *\({[^}]*}\).*/\1/p' | sed "s/\(.*\)\(}\)/\1, \"$feature\": {\"available\": $available, \"reason\": \"$reason\"}\2/" || echo "{\"$feature\": {\"available\": $available, \"reason\": \"$reason\"}}"),
  "updated_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF
  
  echo "[INFO] Feature flag '$feature' set to: $available"
}