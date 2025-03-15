# Container Naming Convention

This document outlines the container naming convention for the Autonomous Trading System (ATS).

## Naming Convention

All containers now follow a consistent naming pattern:

```
ats-[component]-[optional-qualifier]
```

For example:
- `ats-timescaledb-main` - Main TimescaleDB database
- `ats-redis-cache` - Redis cache service
- `ats-model-training-tf` - TensorFlow-based model training service

## Container Name Changes

The following container names have been updated in the docker-compose.yml file:

| Old Name | New Name |
|----------|----------|
| timescaledb | ats-timescaledb-main |
| redis | ats-redis-cache |
| system-controller | ats-system-controller |
| data-acquisition | ats-data-acquisition |
| trading-strategy | ats-trading-strategy |
| model-training | ats-model-training-tf |
| feature-engineering | ats-feature-engineering |
| continuous-learning | ats-continuous-learning |
| prometheus | ats-prometheus |
| node-exporter | ats-node-exporter |
| cadvisor | ats-cadvisor |
| timescaledb-exporter | ats-timescaledb-main-exporter |
| redis-exporter | ats-redis-cache-exporter |
| loki | ats-loki-logs |
| promtail | ats-promtail-collector |
| alertmanager | ats-alertmanager |
| grafana | ats-grafana-dashboard |

## Important Note on Container References

When changing container names, it's critical to update all references to these containers in start scripts and configuration files. In Docker networking, containers can reference each other using their container names as hostnames. If you change a container name, you must update all scripts that reference that container by name.

For example, if a start script contains:
```bash
curl -s http://system-controller:8000/health
```

And you rename the container to `ats-system-controller`, you must update the script to:
```bash
curl -s http://ats-system-controller:8000/health
```

Failure to update these references will result in containers being unable to communicate with each other, leading to startup failures and unhealthy containers.

## Cleanup Script

A cleanup script has been created to remove unused containers and apply the new naming convention:

```bash
./cleanup_containers.sh
```

This script will:
1. Stop and remove old/unused containers
2. Stop current containers
3. Start containers with the new naming convention

## Restart Script

If you need to restart the containers after updating the start scripts, you can use:

```bash
./restart_containers.sh
```

## Unused Containers Removed

The following containers have been identified as unused and will be removed by the cleanup script:

- tensorrt-optimizer-v1-1
- trading-strategy-v1-1
- model-training-tensorflow-v1-1
- model-training-pytorch-v1-1
- feature-engineering-tensorflow-v1-1
- feature-engineering-pytorch-v1-1

## Maintaining the Convention

When adding new services to the docker-compose.yml file, please follow these guidelines:

1. Use the `ats-` prefix for all container names
2. Include the main component name (e.g., `redis`, `model-training`)
3. Add a qualifier if needed (e.g., `main`, `backup`, `tf`, `pytorch`)
4. Use hyphens to separate words
5. Update all scripts and configuration files that reference container names

Example for a new service:
```yaml
new-service:
  image: example/new-service:latest
  container_name: ats-new-service-purpose
```

## Benefits

This naming convention provides several benefits:
- Clear identification of containers belonging to the ATS system
- Easy filtering in Docker commands (e.g., `docker ps | grep ats-`)
- Consistent naming across development and production environments
- Better organization of related containers

## Troubleshooting

If containers fail to start after renaming, check:
1. Start scripts for references to old container names
2. Health check endpoints that might be using old container names
3. Docker Compose dependency conditions that might need updating
4. Network configurations that might reference container names

For more details on fixing container startup issues after renaming, see `container_fix_documentation.md`.