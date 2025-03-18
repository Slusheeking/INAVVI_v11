# Day Trading System Test

This directory contains a comprehensive test framework for an AI day trading system that integrates multiple data sources:

1. **Polygon REST API** - For historical and reference data
2. **Polygon WebSocket** - For real-time market data
3. **Unusual Whales API** - For options flow, dark pool, and insider trading data

## Overview

The day trading system test simulates a complete trading day by:

1. Fetching historical data for a set of symbols
2. Retrieving options contracts data
3. Getting unusual options flow data
4. Fetching dark pool trading data
5. Retrieving insider trading information
6. Collecting real-time market data (or using mock data for testing)
7. Analyzing all data sources to generate trading signals
8. Simulating trade execution with position limits

## Files

- `test_day_trading_system.py` - Main test script that implements the day trading system
- `run_day_trading_test.sh` - Shell script to run the test with different parameters
- `polygon_data_source_ultra.py` - Enhanced Polygon data source with GPU acceleration
- `../src/data_acquisition/api/unusual_whales_client.py` - Enhanced Unusual Whales API client

## Requirements

- Python 3.8+
- Polygon API key
- Unusual Whales API key
- Required Python packages:
  - pandas
  - numpy
  - requests
  - websockets
  - asyncio

## Usage

### Running the Test

To run the day trading system test with default parameters:

```bash
./docker/tensorflow-gh200/run_day_trading_test.sh
```

### Command Line Options

The test script supports the following command line options:

- `--symbols` - Comma-separated list of symbols to test (default: AAPL,MSFT,GOOGL,AMZN,TSLA)
- `--num-symbols` - Number of symbols to test if symbols not provided (default: 5)
- `--use-mock` - Whether to use mock data for testing (default: true)

Example:

```bash
./docker/tensorflow-gh200/run_day_trading_test.sh --symbols AAPL,MSFT,NVDA --use-mock false
```

### API Keys

The test uses the following API keys:

- Polygon API Key: `wFvpCGZq4glxZU_LlRc2Qpw6tQGB5Fmf`
- Unusual Whales API Key: `4ad71b9e-7ace-4f24-bdfc-532ace219a18`

These keys are set in the `run_day_trading_test.sh` script. You can override them by setting the environment variables:

```bash
export POLYGON_API_KEY="your-polygon-api-key"
export UNUSUAL_WHALES_API_KEY="your-unusual-whales-api-key"
```

## Data Sources and Endpoints

### Polygon REST API

Key endpoints used:

- `/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from}/{to}` - Historical aggregates
- `/v3/reference/tickers` - Stock universe information
- `/v2/snapshot/locale/us/markets/stocks/tickers` - Current market snapshots
- `/v1/open-close/{symbol}/{date}` - Daily open/close data
- `/v3/reference/options/contracts` - Options contracts data

### Polygon WebSocket

Key channels used:

- `T.{symbol}` - Real-time trades
- `Q.{symbol}` - Real-time quotes
- `A.{symbol}` - Second-by-second aggregates

### Unusual Whales API

Key endpoints used:

- `/options-flow` - Options flow data
- `/darkpool/recent` - Recent dark pool trades
- `/darkpool/{ticker}` - Dark pool trades for a specific ticker
- `/insider/transactions` - Insider trading data
- `/insider/{ticker}` - Insider data for a specific ticker

## Position Limits

The day trading system implements position limits:

- Maximum position value per stock: $2,500
- Maximum total position value per day: $5,000

These limits are enforced during signal execution to simulate realistic trading constraints.

## Performance Metrics

The test script reports the following performance metrics:

- Timing for each data source
- Total execution time
- Number of records processed from each data source
- Number of signals generated
- Number of trades executed

## Mock Data

For testing during non-market hours, the script can generate mock data:

- Mock WebSocket messages for real-time data
- Mock options flow data
- Mock dark pool trades
- Mock insider trading data

This allows for testing the system's functionality without requiring live market data.

## Signal Generation

The system generates trading signals based on a combination of data sources:

1. Historical price patterns
2. Options flow activity
3. Dark pool transactions
4. Insider trading patterns
5. Real-time market data

A signal is generated when multiple data sources indicate a potential trading opportunity.

## Trade Execution

The system simulates trade execution with:

- Position sizing based on signal confidence
- Position limits enforcement
- Tracking of daily position values
- Rejection of trades that would exceed limits

## Extending the System

To extend the system with new data sources or strategies:

1. Add new data fetching methods to the `DayTradingSystem` class
2. Modify the `analyze_data` and `_generate_signal` methods to incorporate the new data
3. Update the `run_simulation` method to fetch the new data

## Troubleshooting

If you encounter issues:

1. Check that the API keys are valid
2. Ensure all required Python packages are installed
3. Verify network connectivity to the API endpoints
4. Check the log output for specific error messages

For WebSocket connection issues, try using mock data with `--use-mock true`.