"""
Monitoring System Package

This package provides monitoring and observability for the trading system:
1. Prometheus metrics collection and exposure
2. System health monitoring (CPU, memory, GPU)
3. Trading metrics collection
4. ML model performance tracking
5. Frontend notifications for alerts and status updates
6. Redis integration for real-time metrics
"""

from .monitoring_system import MonitoringSystem, FrontendNotifier

__all__ = ["MonitoringSystem", "FrontendNotifier"]
