# Machine Learning Trading Models

This package implements advanced machine learning models for trading strategies that combine data from Polygon.io and Unusual Whales APIs.

## Overview

The ML trading system integrates multiple data sources and machine learning models to generate trading signals:

1. **Data Sources**:
   - Polygon.io API for historical price data, options contracts, and technical indicators
   - Unusual Whales API for options flow, dark pool activity, and insider trading data

2. **Machine Learning Models**:
   - **Classification Models**: XGBoost, Random Forest, Gradient Boosting
   - **Time Series Models**: LSTM, GRU, TCN (Temporal Convolutional Networks)
   - **HFT Models**: Kalman filters, linear regression slope models, statistical arbitrage

3. **Trading Strategies**:
   - Combined Options Flow Analysis
   - Dark Pool + Stock Aggregates Integration
   - Market Structure Monitoring
   - Earnings Trade Setup
   - Technical Pattern + Options Flow Triggers
   - Real-time Trade Execution Framework
   - Historical Backtesting Environment

## Model Integration Framework

The system uses a multi-model approach with a meta-model decision framework:

1. **Pre-Market Analysis**:
   - Ensemble models combining previous day's options flow with pre-market price action
   - Sentiment analysis models for news + options flow correlation

2. **Market Open**:
   - Real-time anomaly detection models for unusual volume/price/options relationships
   - Fast-processing Random Forest models for quick decision support

3. **Regular Trading Hours**:
   - LSTM/GRU networks for continuous pattern recognition
   - XGBoost for feature importance ranking of incoming options flow data
   - Bayesian updating models to continuously refine probability estimates

## Requirements

- Python 3.8+
- TensorFlow 2.5+
- XGBoost
- scikit-learn
- pandas
- numpy
- Polygon.io API key
- Unusual Whales API key (optional, mock data will be used if not provided)

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   ```

2. Install required packages:
   ```
   pip install tensorflow xgboost scikit-learn pandas numpy
   ```

3. Set up API keys:
   ```
   export POLYGON_API_KEY="your-polygon-api-key"
   export UNUSUAL_WHALES_API_KEY="your-unusual-whales-api-key"
   ```

## Usage

### Using the Shell Script

The easiest way to run the ML trading models is to use the provided shell script:

```bash
./run_ml_trading.sh --symbols AAPL,MSFT,GOOGL --mode backtest
```

Options:
- `-s, --symbols`: Comma-separated list of symbols (default: AAPL,MSFT,GOOGL,AMZN,TSLA)
- `-m, --mode`: Mode to run (premarket, trading, backtest) (default: backtest)
- `--start-date`: Start date for backtest (YYYY-MM-DD) (default: 30 days ago)
- `--end-date`: End date for backtest (YYYY-MM-DD) (default: today)
- `--polygon-key`: Polygon API key
- `--unusual-whales-key`: Unusual Whales API key
- `-h, --help`: Show help message

### Using the Python Script Directly

You can also run the Python script directly:

```bash
python run_ml_trading_models.py --symbols AAPL,MSFT,GOOGL --mode backtest
```

The options are the same as for the shell script.

## Modes

### Pre-Market Analysis

Analyzes historical data, options flow, and market conditions to generate pre-market trading signals:

```bash
./run_ml_trading.sh --mode premarket
```

### Trading Hours Analysis

Combines real-time data with historical patterns to generate intraday trading signals:

```bash
./run_ml_trading.sh --mode trading
```

### Backtesting

Evaluates trading strategies on historical data:

```bash
./run_ml_trading.sh --mode backtest --start-date 2023-01-01 --end-date 2023-12-31
```

## Model Details

### Classification Models

- **XGBoost**: Excellent for identifying patterns in options flow combined with price action
- **Random Forest**: Good for handling the mixed data types from both APIs without overfitting
- **Gradient Boosting**: Particularly effective for ranking unusual options activity by probability of success

### Time Series Models

- **LSTM Networks**: Superior for capturing sequential patterns in options flow and corresponding price movements
- **GRU Networks**: Slightly faster than LSTM with similar effectiveness for shorter time horizons
- **Temporal Convolutional Networks (TCN)**: Effective for capturing multi-timeframe patterns

### Technical Indicator Processing

- **Candlestick Pattern Recognition**: Important for entry/exit timing
- **Momentum Oscillators (RSI, Stochastic)**: Critical for confirming options flow direction
- **Trend Indicators (MACD, Moving Averages)**: Essential for filtering out noise in options data

### HFT-Specific Models

- **Kalman Filters**: For real-time noise reduction
- **Linear Regression Slope Models**: For micro-trend identification
- **Statistical Arbitrage Models**: For options-equity price divergence

## Feature Engineering

The system creates a rich set of features from multiple data sources:

1. **Price-based Features**:
   - Moving averages (SMA, EMA)
   - Momentum indicators (RSI, Stochastic)
   - Volatility measures (Bollinger Bands)
   - Candlestick patterns

2. **Options Flow Features**:
   - Put/call ratio
   - Volume/open interest ratio
   - Unusual activity scores
   - Premium levels

3. **Dark Pool Features**:
   - Dark pool volume ratio
   - Block trade analysis
   - Price impact assessment

## Example Output

The system generates trading signals with direction and confidence scores:

```
Results Summary:
AAPL: Direction: BUY, Confidence: 0.85
MSFT: Direction: NEUTRAL, Confidence: 0.52
GOOGL: Direction: SELL, Confidence: 0.78
```

For backtests, it provides performance metrics:

```
Results Summary:
AAPL: Return: 12.45%
MSFT: Return: 8.72%
GOOGL: Return: -3.21%
```

## Extending the System

To add new models or data sources:

1. Add new feature engineering methods to `FeatureEngineering` class
2. Implement new model classes or extend existing ones
3. Update the `ModelIntegrationFramework` to incorporate the new models

## License

This project is licensed under the MIT License - see the LICENSE file for details.