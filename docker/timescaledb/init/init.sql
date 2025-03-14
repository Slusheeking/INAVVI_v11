-- TimescaleDB initialization script for Autonomous Trading System
-- This script creates all tables and indexes needed for the system

-- Create extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- 1. crypto_aggs table
CREATE TABLE IF NOT EXISTS crypto_aggs (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol CHARACTER VARYING NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,
    vwap NUMERIC,
    transactions INTEGER,
    timeframe CHARACTER VARYING NOT NULL,
    source CHARACTER VARYING,
    multiplier INTEGER,
    timespan_unit CHARACTER VARYING,
    exchange CHARACTER VARYING
);

-- Create hypertable for time partitioning
SELECT create_hypertable('crypto_aggs', 'timestamp', if_not_exists => TRUE);

-- Create index for symbol-based queries
CREATE INDEX IF NOT EXISTS idx_crypto_aggs_symbol ON crypto_aggs (symbol, timeframe, timestamp DESC);

-- 2. feature_metadata table
CREATE TABLE IF NOT EXISTS feature_metadata (
    feature_name CHARACTER VARYING NOT NULL PRIMARY KEY,
    description TEXT,
    formula TEXT,
    parameters JSONB,
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ,
    version CHARACTER VARYING,
    is_active BOOLEAN
);

-- 3. features table
CREATE TABLE IF NOT EXISTS features (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol CHARACTER VARYING NOT NULL,
    feature_name CHARACTER VARYING NOT NULL,
    feature_value NUMERIC NOT NULL,
    timeframe CHARACTER VARYING NOT NULL,
    feature_group CHARACTER VARYING,
    CONSTRAINT features_pkey PRIMARY KEY (symbol, timestamp, feature_name, timeframe)
);

-- Create hypertable for time partitioning
SELECT create_hypertable('features', 'timestamp', if_not_exists => TRUE);

-- Create index for symbol-based queries
CREATE INDEX IF NOT EXISTS idx_features_symbol ON features (symbol, feature_group, timeframe, timestamp DESC);

-- 4. market_status table
CREATE TABLE IF NOT EXISTS market_status (
    timestamp TIMESTAMPTZ NOT NULL,
    market CHARACTER VARYING NOT NULL,
    status CHARACTER VARYING NOT NULL,
    next_open TIMESTAMPTZ,
    next_close TIMESTAMPTZ,
    early_close BOOLEAN,
    late_open BOOLEAN,
    CONSTRAINT market_status_pkey PRIMARY KEY (timestamp, market)
);

-- Create hypertable for time partitioning
SELECT create_hypertable('market_status', 'timestamp', if_not_exists => TRUE);

-- 5. model_performance table
CREATE TABLE IF NOT EXISTS model_performance (
    id INTEGER PRIMARY KEY,
    model_name CHARACTER VARYING NOT NULL,
    timeframe CHARACTER VARYING NOT NULL,
    accuracy DOUBLE PRECISION,
    precision DOUBLE PRECISION,
    recall DOUBLE PRECISION,
    f1_score DOUBLE PRECISION,
    timestamp TIMESTAMPTZ NOT NULL
);

-- Create index for model_name-based queries
CREATE INDEX IF NOT EXISTS idx_model_performance_model_name ON model_performance (model_name, timeframe);

-- 6. model_training_runs table
CREATE TABLE IF NOT EXISTS model_training_runs (
    run_id UUID PRIMARY KEY,
    model_id UUID NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    status CHARACTER VARYING NOT NULL,
    parameters JSONB,
    metrics JSONB,
    logs TEXT
);

-- Create index for model_id-based queries
CREATE INDEX IF NOT EXISTS idx_model_training_runs_model_id ON model_training_runs (model_id);

-- 7. models table
CREATE TABLE IF NOT EXISTS models (
    model_id UUID PRIMARY KEY,
    model_name CHARACTER VARYING NOT NULL,
    model_type CHARACTER VARYING NOT NULL,
    target CHARACTER VARYING NOT NULL,
    features TEXT[],
    parameters JSONB,
    metrics JSONB,
    created_at TIMESTAMPTZ NOT NULL,
    trained_at TIMESTAMPTZ,
    version CHARACTER VARYING NOT NULL,
    status CHARACTER VARYING NOT NULL,
    file_path CHARACTER VARYING
);

-- Create index for model_type-based queries
CREATE INDEX IF NOT EXISTS idx_models_model_type ON models (model_type, target);

-- 8. options_aggs table
CREATE TABLE IF NOT EXISTS options_aggs (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol CHARACTER VARYING NOT NULL,
    underlying CHARACTER VARYING NOT NULL,
    expiration DATE NOT NULL,
    strike NUMERIC NOT NULL,
    option_type CHARACTER(1) NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume INTEGER,
    open_interest INTEGER,
    timeframe CHARACTER VARYING NOT NULL,
    multiplier INTEGER,
    timespan_unit CHARACTER VARYING,
    source CHARACTER VARYING,
    CONSTRAINT options_aggs_pkey PRIMARY KEY (symbol, timestamp, timeframe)
);

-- Create hypertable for time partitioning
SELECT create_hypertable('options_aggs', 'timestamp', if_not_exists => TRUE);

-- Create index for underlying-based queries
CREATE INDEX IF NOT EXISTS idx_options_aggs_underlying ON options_aggs (underlying, expiration, strike, option_type);

-- 9. options_flow table
CREATE TABLE IF NOT EXISTS options_flow (
    alert_rule TEXT,
    all_opening_trades BOOLEAN,
    ask TEXT,
    bid TEXT,
    created_at TEXT,
    end_time BIGINT,
    er_time TEXT,
    expiry TEXT,
    expiry_count BIGINT,
    has_floor BOOLEAN,
    has_multileg BOOLEAN,
    has_singleleg BOOLEAN,
    has_sweep BOOLEAN,
    id TEXT PRIMARY KEY,
    iv_end TEXT,
    iv_start TEXT,
    marketcap TEXT,
    next_earnings_date TEXT,
    open_interest BIGINT,
    option_chain TEXT,
    price_data_hidden BOOLEAN,
    rule_id TEXT,
    sector TEXT,
    start_time BIGINT,
    strike DOUBLE PRECISION,
    ticker TEXT,
    total_ask_side_prem TEXT,
    total_bid_side_prem TEXT,
    trade_count BIGINT,
    type TEXT,
    volume BIGINT,
    volume_oi_ratio TEXT
);

-- Create index for ticker-based queries
CREATE INDEX IF NOT EXISTS idx_options_flow_ticker ON options_flow (ticker);

-- 10. orders table
CREATE TABLE IF NOT EXISTS orders (
    order_id UUID PRIMARY KEY,
    external_order_id CHARACTER VARYING,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol CHARACTER VARYING NOT NULL,
    order_type CHARACTER VARYING NOT NULL,
    side CHARACTER VARYING NOT NULL,
    quantity NUMERIC NOT NULL,
    price NUMERIC,
    status CHARACTER VARYING NOT NULL,
    signal_id UUID,
    strategy_id UUID,
    filled_quantity NUMERIC,
    filled_price NUMERIC,
    commission NUMERIC,
    updated_at TIMESTAMPTZ
);

-- Create index for symbol-based queries
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_orders_strategy_id ON orders (strategy_id);

-- 11. positions table
CREATE TABLE IF NOT EXISTS positions (
    position_id UUID PRIMARY KEY,
    symbol CHARACTER VARYING NOT NULL,
    quantity NUMERIC NOT NULL,
    entry_price NUMERIC NOT NULL,
    current_price NUMERIC,
    entry_time TIMESTAMPTZ NOT NULL,
    last_update TIMESTAMPTZ,
    strategy_id UUID,
    status CHARACTER VARYING NOT NULL,
    pnl NUMERIC,
    pnl_percentage NUMERIC,
    metadata JSONB
);

-- Create index for symbol-based queries
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions (symbol);
CREATE INDEX IF NOT EXISTS idx_positions_strategy_id ON positions (strategy_id);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions (status);

-- 12. price_1d table
CREATE TABLE IF NOT EXISTS price_1d (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    CONSTRAINT price_1d_pkey PRIMARY KEY (symbol, timestamp)
);

-- Create hypertable for time partitioning
SELECT create_hypertable('price_1d', 'timestamp', if_not_exists => TRUE);

-- 13. quotes table
CREATE TABLE IF NOT EXISTS quotes (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol CHARACTER VARYING NOT NULL,
    bid_price NUMERIC,
    ask_price NUMERIC,
    bid_size INTEGER,
    ask_size INTEGER,
    exchange CHARACTER VARYING,
    conditions TEXT[],
    sequence_number BIGINT,
    tape CHARACTER(1),
    source CHARACTER VARYING,
    CONSTRAINT quotes_pkey PRIMARY KEY (symbol, timestamp, sequence_number)
);

-- Create hypertable for time partitioning
SELECT create_hypertable('quotes', 'timestamp', if_not_exists => TRUE);

-- Create index for symbol-based queries
CREATE INDEX IF NOT EXISTS idx_quotes_symbol ON quotes (symbol, timestamp DESC);

-- 14. schema_migrations table
CREATE TABLE IF NOT EXISTS schema_migrations (
    id INTEGER PRIMARY KEY,
    version CHARACTER VARYING NOT NULL,
    name CHARACTER VARYING NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL,
    duration_ms INTEGER,
    status CHARACTER VARYING NOT NULL
);

-- 15. stock_aggs table
CREATE TABLE IF NOT EXISTS stock_aggs (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol CHARACTER VARYING NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume BIGINT NOT NULL,
    vwap NUMERIC,
    transactions INTEGER,
    timeframe CHARACTER VARYING NOT NULL,
    source CHARACTER VARYING,
    multiplier INTEGER,
    timespan_unit CHARACTER VARYING,
    adjusted BOOLEAN,
    otc BOOLEAN,
    CONSTRAINT stock_aggs_pkey PRIMARY KEY (symbol, timestamp, timeframe)
);

-- Create hypertable for time partitioning
SELECT create_hypertable('stock_aggs', 'timestamp', if_not_exists => TRUE);

-- Create index for symbol-based queries
CREATE INDEX IF NOT EXISTS idx_stock_aggs_symbol ON stock_aggs (symbol, timeframe, timestamp DESC);

-- 16. strategies table
CREATE TABLE IF NOT EXISTS strategies (
    id INTEGER PRIMARY KEY,
    name CHARACTER VARYING NOT NULL,
    description TEXT,
    parameters JSONB,
    is_active BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

-- 17. strategy_performance table
CREATE TABLE IF NOT EXISTS strategy_performance (
    id INTEGER PRIMARY KEY,
    strategy_id INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    risk_reward_ratio DOUBLE PRECISION,
    drawdown_pct DOUBLE PRECISION,
    win_rate DOUBLE PRECISION,
    sharpe_ratio DOUBLE PRECISION,
    sortino_ratio DOUBLE PRECISION,
    profit_factor DOUBLE PRECISION,
    max_consecutive_wins INTEGER,
    max_consecutive_losses INTEGER,
    avg_profit_per_trade DOUBLE PRECISION,
    avg_loss_per_trade DOUBLE PRECISION,
    total_trades INTEGER,
    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
);

-- Create index for strategy_id-based queries
CREATE INDEX IF NOT EXISTS idx_strategy_performance_strategy_id ON strategy_performance (strategy_id);

-- 18. system_metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    timestamp TIMESTAMPTZ NOT NULL,
    metric_name CHARACTER VARYING NOT NULL,
    metric_value NUMERIC NOT NULL,
    component CHARACTER VARYING,
    host CHARACTER VARYING,
    tags JSONB,
    CONSTRAINT system_metrics_pkey PRIMARY KEY (timestamp, metric_name, component)
);

-- Create hypertable for time partitioning
SELECT create_hypertable('system_metrics', 'timestamp', if_not_exists => TRUE);

-- 19. temp_ticker_details table
CREATE TABLE IF NOT EXISTS temp_ticker_details (
    ticker TEXT PRIMARY KEY,
    name TEXT,
    market TEXT,
    locale TEXT,
    type TEXT,
    currency TEXT,
    last_updated TIMESTAMP WITHOUT TIME ZONE,
    active BOOLEAN,
    primary_exchange TEXT,
    description TEXT,
    sic_code TEXT,
    sic_description TEXT,
    ticker_root TEXT,
    homepage_url TEXT,
    total_employees TEXT,
    list_date TEXT,
    share_class_shares_outstanding TEXT,
    weighted_shares_outstanding TEXT,
    market_cap TEXT,
    phone_number TEXT,
    address TEXT,
    metadata TEXT
);

-- 20. ticker_details table
CREATE TABLE IF NOT EXISTS ticker_details (
    ticker TEXT PRIMARY KEY,
    name TEXT,
    market TEXT,
    locale TEXT,
    type TEXT,
    currency TEXT,
    last_updated TIMESTAMP WITHOUT TIME ZONE,
    active BOOLEAN,
    primary_exchange TEXT,
    description TEXT,
    sic_code TEXT,
    sic_description TEXT,
    ticker_root TEXT,
    homepage_url TEXT,
    total_employees BIGINT,
    list_date TIMESTAMP WITHOUT TIME ZONE,
    share_class_shares_outstanding BIGINT,
    weighted_shares_outstanding BIGINT,
    market_cap BIGINT,
    phone_number TEXT,
    address TEXT,
    metadata TEXT
);

-- 21. trade_executions table
CREATE TABLE IF NOT EXISTS trade_executions (
    id INTEGER PRIMARY KEY,
    order_id UUID NOT NULL,
    strategy_id INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    slippage_pct DOUBLE PRECISION,
    execution_time_ms DOUBLE PRECISION,
    market_impact_pct DOUBLE PRECISION,
    venue CHARACTER VARYING,
    liquidity_type CHARACTER VARYING,
    execution_price DOUBLE PRECISION NOT NULL,
    target_price DOUBLE PRECISION,
    quantity INTEGER NOT NULL,
    commission DOUBLE PRECISION,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
);

-- Create index for order_id-based queries
CREATE INDEX IF NOT EXISTS idx_trade_executions_order_id ON trade_executions (order_id);
CREATE INDEX IF NOT EXISTS idx_trade_executions_strategy_id ON trade_executions (strategy_id);

-- 22. trades table
CREATE TABLE IF NOT EXISTS trades (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol CHARACTER VARYING NOT NULL,
    price NUMERIC NOT NULL,
    size INTEGER NOT NULL,
    exchange CHARACTER VARYING,
    conditions TEXT[],
    tape CHARACTER(1),
    sequence_number BIGINT,
    trade_id CHARACTER VARYING,
    source CHARACTER VARYING,
    CONSTRAINT trades_pkey PRIMARY KEY (symbol, timestamp, trade_id)
);

-- Create hypertable for time partitioning
SELECT create_hypertable('trades', 'timestamp', if_not_exists => TRUE);

-- Create index for symbol-based queries
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol, timestamp DESC);

-- 23. trading_metrics table
CREATE TABLE IF NOT EXISTS trading_metrics (
    timestamp TIMESTAMPTZ NOT NULL,
    metric_name CHARACTER VARYING NOT NULL,
    metric_value NUMERIC NOT NULL,
    symbol CHARACTER VARYING,
    strategy_id UUID,
    timeframe CHARACTER VARYING,
    tags JSONB,
    CONSTRAINT trading_metrics_pkey PRIMARY KEY (timestamp, metric_name, symbol, strategy_id)
);

-- Create hypertable for time partitioning
SELECT create_hypertable('trading_metrics', 'timestamp', if_not_exists => TRUE);

-- Create index for symbol-based queries
CREATE INDEX IF NOT EXISTS idx_trading_metrics_symbol ON trading_metrics (symbol, metric_name);
CREATE INDEX IF NOT EXISTS idx_trading_metrics_strategy_id ON trading_metrics (strategy_id);

-- 24. trading_signals table
CREATE TABLE IF NOT EXISTS trading_signals (
    signal_id UUID PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol CHARACTER VARYING NOT NULL,
    signal_type CHARACTER VARYING NOT NULL,
    confidence NUMERIC NOT NULL,
    model_id UUID,
    timeframe CHARACTER VARYING NOT NULL,
    parameters JSONB,
    features_snapshot JSONB
);

-- Create index for symbol-based queries
CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol ON trading_signals (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trading_signals_model_id ON trading_signals (model_id);