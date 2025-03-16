-- Create tables for market data
CREATE TABLE IF NOT EXISTS market_prices (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,
    PRIMARY KEY (symbol, timestamp, timeframe)
);

CREATE TABLE IF NOT EXISTS market_quotes (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    bid_price NUMERIC NOT NULL,
    ask_price NUMERIC NOT NULL,
    bid_size NUMERIC NOT NULL,
    ask_size NUMERIC NOT NULL,
    PRIMARY KEY (symbol, timestamp)
);

CREATE TABLE IF NOT EXISTS market_trades (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    price NUMERIC NOT NULL,
    size NUMERIC NOT NULL,
    conditions VARCHAR(100),
    PRIMARY KEY (symbol, timestamp)
);

-- Create feature tables
CREATE TABLE IF NOT EXISTS market_features (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    feature_name VARCHAR(50) NOT NULL,
    feature_value NUMERIC NOT NULL,
    feature_metadata JSONB,
    PRIMARY KEY (symbol, timestamp, timeframe, feature_name)
);

CREATE TABLE IF NOT EXISTS feature_metadata (
    feature_name VARCHAR(50) NOT NULL PRIMARY KEY,
    description TEXT,
    category VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    parameters JSONB
);

-- Create hypertables if TimescaleDB extension is available
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable('market_prices', 'timestamp', if_not_exists => TRUE);
        PERFORM create_hypertable('market_quotes', 'timestamp', if_not_exists => TRUE);
        PERFORM create_hypertable('market_trades', 'timestamp', if_not_exists => TRUE);
        PERFORM create_hypertable('market_features', 'timestamp', if_not_exists => TRUE);
    END IF;
END
$$;

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_market_prices_symbol ON market_prices (symbol);
CREATE INDEX IF NOT EXISTS idx_market_quotes_symbol ON market_quotes (symbol);
CREATE INDEX IF NOT EXISTS idx_market_trades_symbol ON market_trades (symbol);
CREATE INDEX IF NOT EXISTS idx_market_features_symbol ON market_features (symbol);
CREATE INDEX IF NOT EXISTS idx_market_features_name ON market_features (feature_name);