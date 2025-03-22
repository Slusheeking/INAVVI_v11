# Product Context

This file provides a high-level overview of the project and the expected product that will be created. It is based on the files in the working directory and will be updated as the project evolves. This file is intended to inform all other modes of the project's goals and context.

2025-03-21 01:55:43 - Initial creation of Memory Bank for INAVVI_v11-3 project.
2025-03-21 01:59:56 - Updated hardware specifications and added TensorFlow, TensorRT, and CuPy frameworks.
2025-03-21 02:06:48 - Testing Memory Bank update functionality.
2025-03-21 02:43:12 - Updated with comprehensive system architecture details based on code review.
2025-03-21 02:56:32 - Added code quality standards and dynamic configuration approach.
2025-03-21 05:30:38 - Added Monitoring System with Prometheus metrics and Slack notifications.
2025-03-21 17:15:45 - Standardized API client usage and fixed Redis authentication issues.
2025-03-21 17:29:45 - Standardized Redis port configuration across all system components.
2025-03-22 02:45:00 - Enhanced TensorFlow, TensorRT, and CuPy integration in Docker container.

## Project Goal

* Develop an autonomous AI trading system leveraging NVIDIA GH200 Grace Hopper Superchips for high-performance computing
* Create a unified system that integrates multiple components for market prediction, trade execution, and continuous learning
* Build a system capable of adapting to changing market conditions through real-time data analysis and model updates

## Key Features

### Core Components
* **Trading Engine**: Handles order execution, position management, risk control, and pattern recognition
* **Stock Selection Engine**: Identifies trading opportunities based on various signals and criteria
* **Learning Engine**: Continuously updates models based on market feedback and performance
* **ML Engine**: Core machine learning functionality for prediction and analysis
* **Data Pipeline**: Processes and transforms market data for model consumption
* **GPU Utilities**: Optimizes GPU usage for maximum performance
* **API Clients**: Interfaces with external data sources (Polygon.io, Unusual Whales) and trading platforms (Alpaca)
* **Monitoring System**: Comprehensive monitoring with Prometheus metrics and Slack notifications

### AI/ML Capabilities
* **Trading Models**: Implementation of XGBoost, CNNs, LSTM networks, DQN/PPO reinforcement learning, FinBERT NLP, and Monte Carlo simulations
* **Deep Learning Frameworks**: TensorFlow for model development, TensorRT for inference optimization, and CuPy for GPU-accelerated numerical computing
* **Model Training**: Continuous learning with automated model updates and deployment
* **Signal Detection**: Real-time pattern recognition for trading opportunities

### Infrastructure
* **High-Performance Computing**: Optimized for NVIDIA GH200 480GB Grace Hopper Superchips with 97871MiB memory and CUDA 12.4
* **Time-Series Data Architecture**: PostgreSQL+TimescaleDB for historical data and Redis for low-latency caching
* **Containerization**: Docker-based deployment for scalability and reproducibility

### Code Quality and Configuration
* **Dynamic Configuration**: Configuration-driven approach for all system parameters including ticker universe
* **Strict Linting**: Comprehensive linting with ruff to ensure code quality
* **Separation of Concerns**: Clear separation between production and test code
* **Error Handling**: Standardized error handling and logging patterns across all components
* **Feature Flags**: Configuration-driven feature flags for enabling/disabling functionality

## Overall Architecture

### Hardware Optimization
* **GPU Acceleration**: Leverages NVIDIA GH200 480GB Grace Hopper Superchips with 97871MiB memory and CUDA 12.4
* **Hardware-Optimized Libraries**: TensorRT for inference optimization, CuPy for GPU-accelerated numerical computing, and custom CUDA kernels
* **Memory Management**: Specialized memory allocation strategies for GH200 architecture

### ML Infrastructure
* **Model Quantization**: Mixed-precision quantization for optimal performance-accuracy tradeoff
* **Distributed Training**: Multi-node training capabilities for large-scale model development
* **Model Serving**: Optimized inference pipelines with TensorRT for low-latency predictions

### System Design
* **Unified System**: Central orchestrator (unified_system.py) manages all components
* **Docker Containerization**: Uses Docker for deployment and scaling (Dockerfile.unified, docker-compose.unified.yml)
* **Configuration Management**: Centralized configuration through config.py and config.env
* **API Integration**: Connects to external data sources and trading platforms through api_clients.py
* **Modular Design**: Separate components with clear interfaces for maintainability and testability
* **Monitoring**: Prometheus and Redis Exporter for system metrics and performance monitoring