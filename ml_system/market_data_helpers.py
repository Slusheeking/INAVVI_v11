#!/usr/bin/env python3
"""
Market Data Helpers for Stock Selection System
This module provides helper functions for retrieving and processing market data.
"""

import logging
import json
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('market_data_helpers')


async def get_all_active_tickers(polygon_api):
    """Get all active tickers from Polygon API dynamically"""
    try:
        # Use the v3 reference/tickers endpoint
        endpoint = "v3/reference/tickers"
        params = {
            "market": "stocks",
            "active": "true",
            "limit": 1000
        }

        response = await polygon_api._make_request(endpoint, params)

        # Process the response based on the actual API structure
        if isinstance(response, dict) and "results" in response:
            return response["results"]
        else:
            logger.warning(
                f"Unexpected response format from Polygon API: {type(response)}")
            return []
    except Exception as e:
        logger.error(f"Error getting active tickers: {str(e)}")
        return []


async def get_previous_day_data(polygon_api, ticker):
    """Get previous day's trading data"""
    try:
        # Use the proper Polygon API client method
        response = await polygon_api.get_aggregates(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_date=(datetime.datetime.now() -
                       datetime.timedelta(days=5)).strftime("%Y-%m-%d"),
            to_date=datetime.datetime.now().strftime("%Y-%m-%d"),
            limit=5
        )

        # Process the response based on the actual API structure
        if isinstance(response, pd.DataFrame) and not response.empty:
            # Extract the most recent day's data
            latest_data = response.iloc[0]
            return {
                "close": float(latest_data.get("close", 0)),
                "volume": int(latest_data.get("volume", 0)),
                "open": float(latest_data.get("open", 0)),
                "high": float(latest_data.get("high", 0)),
                "low": float(latest_data.get("low", 0))
            }
        else:
            logger.warning(f"No data returned for {ticker}")
            return None
    except Exception as e:
        logger.error(
            f"Error getting previous day data for {ticker}: {str(e)}")
        return None


async def check_options_availability(unusual_whales, ticker):
    """Check if options are available for a ticker"""
    try:
        # Use Unusual Whales client to check for options data
        options_data = await unusual_whales.get_flow_alerts(ticker, limit=1)

        # If we get any data back, options are available
        return len(options_data) > 0
    except Exception as e:
        logger.error(
            f"Error checking options availability for {ticker}: {str(e)}")
        # Return False if there's an error
        return False


async def get_pre_market_movers(redis_client):
    """Get pre-market movers"""
    try:
        # Use Polygon websocket client to get pre-market data
        pre_market_movers = set()

        # Check Redis for pre-market data that the websocket client would have stored
        pre_market_keys = redis_client.keys("stock:*:pre_market")

        for key in pre_market_keys:
            ticker = key.decode('utf-8').split(':')[1]
            data = redis_client.hgetall(key)

            # Check for significant movement (e.g., > 2%)
            if b'percent_change' in data:
                percent_change = float(data[b'percent_change'])
                if abs(percent_change) > 2.0:
                    pre_market_movers.add(ticker)

        return pre_market_movers
    except Exception as e:
        logger.error(f"Error getting pre-market movers: {str(e)}")
        return set()


async def get_unusual_options_activity(unusual_whales):
    """Get stocks with unusual options activity"""
    try:
        # Get data from Unusual Whales
        data = await unusual_whales.get_alerts(limit=100)

        if isinstance(data, list) and data:
            # Extract tickers
            return set(item.get('ticker') for item in data if 'ticker' in item)
        elif isinstance(data, pd.DataFrame) and not data.empty:
            # Handle DataFrame format if that's what the client returns
            return set(data['ticker'].unique())

        return set()

    except Exception as e:
        logger.error(f"Error getting unusual options activity: {str(e)}")
        return set()


async def get_technical_setups(redis_client):
    """Get stocks with technical setups"""
    try:
        # Use Polygon API to get technical pattern data
        technical_setups = set()

        # Get stocks from Redis that have been processed by the Polygon API client
        # and have technical indicators stored
        technical_keys = redis_client.keys("stock:*:technical")

        for key in technical_keys:
            ticker = key.decode('utf-8').split(':')[1]
            data = redis_client.hgetall(key)

            # Check for technical setups based on indicators
            # For example, golden cross (SMA 5 crossing above SMA 20)
            if b'sma_5' in data and b'sma_20' in data:
                sma_5 = float(data[b'sma_5'])
                sma_20 = float(data[b'sma_20'])

                # Golden cross
                if sma_5 > sma_20:
                    technical_setups.add(ticker)

            # Check for RSI conditions
            if b'rsi' in data:
                rsi = float(data[b'rsi'])

                # Oversold condition (RSI < 30)
                if rsi < 30:
                    technical_setups.add(ticker)

                # Overbought condition (RSI > 70)
                elif rsi > 70:
                    technical_setups.add(ticker)

        return technical_setups
    except Exception as e:
        logger.error(f"Error getting technical setups: {str(e)}")
        return set()


async def get_market_regime(redis_client):
    """Get current market regime"""
    try:
        # Use market data to classify regime
        # Get SPY and VIX data from Redis (stored by Polygon API client)
        spy_data = redis_client.hgetall("stock:SPY:technical")
        vix_data = redis_client.hgetall("stock:VIX:technical")

        if spy_data and vix_data:
            # Get SPY trend (above or below 20-day SMA)
            spy_price = float(spy_data.get(b'last_price', 0))
            spy_sma20 = float(spy_data.get(b'sma_20', 0))

            # Get VIX level
            vix_price = float(vix_data.get(b'last_price', 0))

            # Classify regime
            if spy_price > spy_sma20 and vix_price < 20:
                return "bullish"
            elif spy_price < spy_sma20 and vix_price > 30:
                return "bearish"
            elif vix_price > 25:
                return "volatile"
            else:
                return "normal"
        else:
            return "normal"
    except Exception as e:
        logger.error(f"Error determining market regime: {str(e)}")
        return "normal"


async def get_market_volatility(redis_client, polygon_api):
    """Get current market volatility (VIX)"""
    try:
        # Get VIX data from Redis (stored by Polygon API client)
        vix_data = redis_client.hgetall("stock:VIX:last_price")

        if vix_data and b'price' in vix_data:
            return float(vix_data[b'price'])
        else:
            # Query VIX data directly if not in Redis
            vix_response = await polygon_api.get_aggregates(
                ticker="VIX",
                multiplier=1,
                timespan="day",
                from_date=(datetime.datetime.now() -
                           datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
                to_date=datetime.datetime.now().strftime("%Y-%m-%d"),
                limit=1
            )

            if isinstance(vix_response, pd.DataFrame) and not vix_response.empty:
                return float(vix_response['close'].iloc[0])
            return 15.0  # Default value if not available
    except Exception as e:
        logger.error(f"Error getting market volatility: {str(e)}")
        return 15.0


async def get_volume_data(redis_client, polygon_api, ticker):
    """Get historical volume data for a ticker"""
    try:
        # Check cache first
        cache_key = f"volume_data:{ticker}"
        cached_data = redis_client.get(cache_key)
        if cached_data:
            try:
                return json.loads(cached_data)
            except Exception:
                pass  # Continue to fetch from API if cache read fails

        # Use Polygon API to get volume data
        response = await polygon_api.get_aggregates(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_date=(datetime.datetime.now() -
                       datetime.timedelta(days=15)).strftime("%Y-%m-%d"),
            to_date=datetime.datetime.now().strftime("%Y-%m-%d"),
            limit=15
        )

        if isinstance(response, pd.DataFrame) and not response.empty:
            # Extract volume data
            volumes = response['volume'].tolist()

            # Cache the result for 1 hour
            try:
                redis_client.setex(cache_key, 3600, json.dumps(volumes))
            except Exception:
                pass  # Ignore cache write failures

            return volumes
        else:
            logger.warning(f"No volume data returned for {ticker}")
            return None
    except Exception as e:
        logger.error(f"Error getting volume data for {ticker}: {str(e)}")
        return None


async def get_atr_data(redis_client, polygon_api, ticker):
    """Get ATR (Average True Range) for a ticker"""
    try:
        # Check cache first
        cache_key = f"atr_data:{ticker}"
        cached_data = redis_client.get(cache_key)
        if cached_data:
            try:
                return float(cached_data)
            except Exception:
                pass  # Continue to fetch from API if cache read fails

        # Use Polygon API to get OHLC data
        response = await polygon_api.get_aggregates(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_date=(datetime.datetime.now() -
                       datetime.timedelta(days=15)).strftime("%Y-%m-%d"),
            to_date=datetime.datetime.now().strftime("%Y-%m-%d"),
            limit=15
        )

        if isinstance(response, pd.DataFrame) and not response.empty and len(response) >= 14:
            # Calculate ATR
            high = response['high'].values
            low = response['low'].values
            close = response['close'].values

            # Calculate True Range
            tr1 = np.abs(high[1:] - low[1:])
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])

            # True Range is the maximum of the three
            tr = np.maximum(np.maximum(tr1, tr2), tr3)

            # ATR is the average of TR over 14 periods
            atr = np.mean(tr[-14:])

            # Cache the result for 1 hour
            try:
                redis_client.setex(cache_key, 3600, str(float(atr)))
            except Exception:
                pass  # Ignore cache write failures

            return float(atr)
        else:
            logger.warning(
                f"Insufficient data to calculate ATR for {ticker}")
            return None
    except Exception as e:
        logger.error(f"Error calculating ATR for {ticker}: {str(e)}")
        return None


async def get_price_data(redis_client, polygon_api, ticker):
    """Get historical price data for a ticker"""
    try:
        # Check cache first
        cache_key = f"price_data:{ticker}"
        cached_data = redis_client.get(cache_key)
        if cached_data:
            try:
                return json.loads(cached_data)
            except Exception:
                pass  # Continue to fetch from API if cache read fails

        # Use Polygon API to get price data
        response = await polygon_api.get_aggregates(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_date=(datetime.datetime.now() -
                       datetime.timedelta(days=10)).strftime("%Y-%m-%d"),
            to_date=datetime.datetime.now().strftime("%Y-%m-%d"),
            limit=10
        )

        if isinstance(response, pd.DataFrame) and not response.empty:
            # Extract close prices
            prices = response['close'].tolist()

            # Cache the result for 1 hour
            try:
                redis_client.setex(cache_key, 3600, json.dumps(prices))
            except Exception:
                pass  # Ignore cache write failures

            return prices
        else:
            logger.warning(f"No price data returned for {ticker}")
            return None
    except Exception as e:
        logger.error(f"Error getting price data for {ticker}: {str(e)}")
        return None


async def get_market_momentum(polygon_api):
    """Get market momentum (S&P 500 change)"""
    try:
        # Use Polygon API to get SPY data
        response = await polygon_api.get_aggregates(
            ticker="SPY",
            multiplier=1,
            timespan="day",
            from_date=(datetime.datetime.now() -
                       datetime.timedelta(days=5)).strftime("%Y-%m-%d"),
            to_date=datetime.datetime.now().strftime("%Y-%m-%d"),
            limit=5
        )

        if isinstance(response, pd.DataFrame) and not response.empty and len(response) >= 2:
            # Calculate 1-day percent change
            latest_price = response['close'].iloc[0]
            previous_price = response['close'].iloc[1]

            percent_change = (latest_price / previous_price - 1) * 100
            return float(percent_change)
        else:
            logger.warning(
                "Insufficient SPY data to calculate market momentum")
            # Query market data directly
            return 0.0
    except Exception as e:
        logger.error(f"Error calculating market momentum: {str(e)}")
        return 0.0


async def get_current_price(redis_client, polygon_api, ticker):
    """Get current price for a ticker"""
    try:
        # Check Redis for last price
        price_data = redis_client.hgetall(f"stock:{ticker}:last_price")

        if price_data and b'price' in price_data:
            return float(price_data[b'price'])

        # If not in Redis, get from API
        # Get from Polygon API
        response = await polygon_api.get_aggregates(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_date=datetime.datetime.now().strftime("%Y-%m-%d"),
            to_date=datetime.datetime.now().strftime("%Y-%m-%d"),
            limit=1
        )

        if isinstance(response, pd.DataFrame) and not response.empty:
            return float(response['close'].iloc[0])

        return None

    except Exception as e:
        logger.error(f"Error getting current price for {ticker}: {str(e)}")
        return None


async def get_options_flow(unusual_whales, ticker):
    """Get options flow data for a ticker"""
    try:
        # Use Unusual Whales client to get options flow data
        flow_data = await unusual_whales.get_flow_alerts(ticker, limit=50)

        if flow_data:
            return flow_data
        else:
            return []
    except Exception as e:
        logger.error(f"Error getting options flow for {ticker}: {str(e)}")
        return []


def subscribe_to_watchlist_channels(polygon_ws, redis_client, logger):
    """Subscribe to websocket channels for watchlist stocks"""
    try:
        if not polygon_ws:
            return

        # Get current watchlist
        watchlist_data = redis_client.zrevrange("watchlist:active", 0, -1)
        watchlist = [item.decode(
            'utf-8') if isinstance(item, bytes) else item for item in watchlist_data]

        if not watchlist:
            return

        # Subscribe to trades, quotes, and minute aggregates
        from stock_selection_system.gpu_optimized_polygon_websocket_client import subscribe_to_trades, subscribe_to_quotes, subscribe_to_minute_aggs

        subscribe_to_trades(polygon_ws, watchlist)
        subscribe_to_quotes(polygon_ws, watchlist)
        subscribe_to_minute_aggs(polygon_ws, watchlist)

        logger.info(
            f"Subscribed to websocket channels for {len(watchlist)} watchlist stocks")
    except Exception as e:
        logger.error(f"Error subscribing to websocket channels: {str(e)}")


def should_update_watchlist(redis_client, refresh_interval):
    """Check if watchlist should be updated"""
    # Get last update time
    last_update = redis_client.get("watchlist:active:last_update")

    if not last_update:
        return True

    # Convert to datetime
    last_update_time = datetime.datetime.fromisoformat(
        last_update.decode('utf-8'))
    now = datetime.datetime.now()

    # Update if more than 15 minutes since last update
    elapsed_seconds = (now - last_update_time).total_seconds()
    return elapsed_seconds >= refresh_interval


def same_day(timestamp_str):
    """Check if timestamp is from the same day (ET)"""
    import pytz
    # Parse timestamp
    timestamp = datetime.datetime.fromisoformat(timestamp_str)

    # Get current date (ET)
    now = datetime.datetime.now(pytz.timezone('US/Eastern'))

    # Check if same day
    return timestamp.date() == now.date()
