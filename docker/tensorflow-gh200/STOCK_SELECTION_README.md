# Stock Selection System

A production-ready dynamic stock selection system for algorithmic trading, integrated with real-time market data ingestion.

## Overview

The Stock Selection System is a sophisticated algorithmic trading component that dynamically selects the most promising stocks for trading based on multiple data sources and criteria. It integrates with the Data Ingestion System to provide a complete market data pipeline for algorithmic trading.

## Features

- **Dynamic Universe Building**: Automatically builds and maintains a universe of tradable stocks based on configurable criteria.
- **Multi-factor Ranking**: Ranks stocks using a weighted combination of volume, volatility, momentum, and options activity factors.
- **Time-of-day Adaptation**: Adjusts focus list size and composition based on market hours and volatility conditions.
- **Real-time Data Integration**: Integrates with Polygon.io for market data and Unusual Whales for options flow data.
- **Redis-backed Storage**: Uses Redis for efficient data storage, retrieval, and inter-process communication.
- **Asynchronous Processing**: Leverages asyncio for non-blocking I/O and concurrent processing.
- **Robust Error Handling**: Implements comprehensive error handling and graceful degradation.

## Requirements

- Python 3.10+
- Redis server
- Polygon.io API key
- Unusual Whales API key (optional but recommended)
- Required Python packages:
  - redis
  - numpy
  - pandas
  - asyncio
  - pytz
  - requests

## Setup

1. Ensure Redis is running:
   ```bash
   # Start Redis using Docker
   docker run --name redis -p 6379:6379 -d redis
   ```

2. Set up API keys in `.env` file:
   ```
   POLYGON_API_KEY=your_polygon_api_key
   UNUSUAL_WHALES_API_KEY=your_unusual_whales_api_key
   REDIS_HOST=localhost
   REDIS_PORT=6379
   REDIS_DB=0
   ```

3. Make the run script executable:
   ```bash
   chmod +x run_integrated_trading_system.sh
   ```

## Usage

Run the integrated trading system:

```bash
./run_integrated_trading_system.sh
```

Run only the data ingestion component:

```bash
./run_integrated_trading_system.sh --data-only
```

Run only the stock selection component:

```bash
./run_integrated_trading_system.sh --selection-only
```

## System Architecture

The Stock Selection System consists of the following components:

1. **Universe Building**: Creates and maintains a universe of tradable stocks based on configurable criteria.
2. **Watchlist Generation**: Identifies stocks with the highest potential based on multiple factors.
3. **Focus List Management**: Maintains a smaller, more focused list of stocks for active trading.
4. **Factor Calculation**: Computes various factors for ranking stocks:
   - Volume Factor: Based on relative volume compared to historical averages.
   - Volatility Factor: Based on Average True Range (ATR) as a percentage of price.
   - Momentum Factor: Based on short-term and mid-term price momentum.
   - Options Factor: Based on options flow data from Unusual Whales.

## Data Flow

1. The Data Ingestion System collects and processes market data from various sources.
2. The Stock Selection System uses this data to build and maintain the universe of tradable stocks.
3. Stocks are ranked based on multiple factors and added to the watchlist.
4. The focus list is updated based on time of day and market conditions.
5. Trading signals are generated for stocks in the focus list.

## Performance Considerations

- The system is designed to handle a universe of 2000+ stocks.
- The watchlist is typically limited to 100 stocks for efficient processing.
- The focus list is dynamically sized based on market conditions, typically ranging from 5 to 50 stocks.
- Redis is used for efficient data storage and retrieval, with appropriate TTL values to manage memory usage.
- Asynchronous processing is used to maximize throughput and minimize latency.

## Error Handling

The system implements comprehensive error handling:

- API errors are caught and logged, with appropriate fallback mechanisms.
- Redis connection errors are treated as critical and will cause the system to exit.
- Process errors are caught and logged, with the system attempting to continue operation when possible.
- Graceful shutdown is implemented to ensure clean termination of all components.

## Monitoring

The system logs detailed information about its operation, including:

- System startup and shutdown events
- Universe building and maintenance activities
- Watchlist and focus list updates
- Factor calculations and ranking results
- API calls and responses
- Error conditions and recovery attempts

## Extending the System

The Stock Selection System is designed to be extensible:

- Additional data sources can be added by implementing appropriate client classes.
- New factors can be added by implementing additional factor calculation methods.
- The ranking algorithm can be customized by adjusting the weights in the configuration.
- Additional filtering criteria can be added to the universe building and watchlist generation components.