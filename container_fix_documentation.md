# Container Naming Convention Update

## Overview

This document outlines the changes made to implement a consistent naming convention for all containers in the Autonomous Trading System (ATS). The new naming convention uses the `ats-` prefix for all containers to improve readability, organization, and maintainability.

## Changes Implemented

The following files were updated to reflect the new container naming convention:

### 1. Docker Compose Configuration

- Updated `docker-compose.yml` to use the `ats-` prefix for all container names
- Example: `system-controller` → `ats-system-controller`

### 2. Service Start Scripts

All service start scripts were updated to reference the new container names when checking for dependencies:

- `docker/data_acquisition/start.sh`
- `docker/feature_engineering/start.sh`
- `docker/model_training/start.sh`
- `docker/trading_strategy/start.sh`
- `docker/continuous_learning/start.sh`

Example change:
```bash
# Before
until curl -s http://system-controller:8000/health > /dev/null; do
    # ...
done

# After
until curl -s http://ats-system-controller:8000/health > /dev/null; do
    # ...
done
```

### 3. Monitoring Configuration

The following monitoring configuration files were updated:

- `monitoring/prometheus/prometheus.yml` - Updated all target references
- `monitoring/promtail/config.yml` - Updated client URL
- `monitoring/grafana/provisioning/datasources/grafana_prometheus_datasource.yaml` - Updated datasource URLs
- `monitoring/grafana/provisioning/datasources/grafana_loki_datasource.yaml` - Updated datasource URL
- `monitoring/loki/local-config.yaml` - Updated alertmanager URL

## Benefits of the New Naming Convention

1. **Improved Readability**: The `ats-` prefix clearly identifies containers as part of the Autonomous Trading System.
2. **Better Organization**: When viewing containers in Docker management tools, all ATS containers are grouped together alphabetically.
3. **Reduced Naming Conflicts**: The prefix reduces the risk of naming conflicts with other containers running on the same host.
4. **Consistent Identification**: Makes it easier to identify system components in logs, monitoring, and debugging.

## Implementation Process

The implementation was performed using the `cleanup_containers.sh` script, which:

1. Stopped and removed old containers with previous naming
2. Recreated containers with the new naming convention
3. Ensured all dependencies and connections were properly updated

## Verification

After implementing these changes, the system was tested to ensure:

1. All containers start correctly with the new names
2. Inter-container communication works properly
3. Monitoring systems correctly track and display container metrics
4. Logging systems properly collect and display logs from all containers

## Future Considerations

When adding new containers to the system, ensure they follow the `ats-` prefix naming convention to maintain consistency across the entire platform.