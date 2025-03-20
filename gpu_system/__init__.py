"""
GPU System Package

This package contains modules for GPU acceleration and system configuration.
It includes components for:
- GH200 Grace Hopper Superchip acceleration
- GPU utilities and optimizations
- System verification tools
- Docker and container configuration
"""

__version__ = "1.0.0"

from .gh200_accelerator import GH200Accelerator, optimize_for_gh200
from .gpu_utils import log_memory_usage, configure_gpu
