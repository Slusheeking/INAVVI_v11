#!/usr/bin/env python3
"""
Stock Selection Core Module
This module consolidates the functionality from:
- stock_selection_scoring.py: Functions for scoring and ranking stocks
- stock_selection_universe.py: Functions for building and filtering the universe of tradable stocks
- stock_selection_watchlist.py: Functions for managing watchlists of tradable stocks

This consolidation reduces code fragmentation and makes the flow clearer
while maintaining the same functionality.
"""

import ml_system.market_data_helpers as mdh
from gpu_system.gpu_utils import log_memory_usage
import logging
import json
import asyncio
import datetime
import pytz
import numpy as np
import cupy as cp
from typing import List, Tuple, Dict, Any
import sys
import os
from asyncio import Lock

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import from our utility modules

# Configure logging
logger = logging.getLogger('stock_selection_core')


class StockSelectionCore:
    """
    Consolidated stock selection functionality.

    This class combines the functionality from:
    - Stock Selection Scoring: Calculating scores and ranking stocks
    - Stock Selection Universe: Building and filtering the universe of tradable stocks
    - Stock Selection Watchlist: Managing watchlists of tradable stocks
    """

    #
    # Universe Management (from stock_selection_universe.py)
    #
    @staticmethod
    async def build_initial_universe(self) -> None:
        """Build initial universe of tradable stocks with GPU acceleration"""
        async with self._universe_lock:  # Ensure thread safety
            logger.info("Building initial universe with GPU acceleration")
            log_memory_usage("before_build_universe")

            # Get all US equities
            try:
                # Fetch tickers from all market segments to ensure diversity
                tickers_data = []

                # Define market segments to ensure we get a diverse set of tickers
                market_segments = [
                    {"market": "stocks", "exchange": "XNYS"},  # NYSE
                    {"market": "stocks", "exchange": "XNAS"},  # NASDAQ
                    {"market": "stocks", "exchange": "ARCX"},  # NYSE Arca
                    {"market": "stocks", "exchange": "BATS"},  # CBOE BZX
                    {"market": "stocks", "exchange": "XASE"}   # NYSE American
                ]

                # Fetch tickers from each market segment
                for segment in market_segments:
                    # Use pagination to get all tickers
                    next_cursor = None
                    segment_tickers = []

                    # Fetch up to 5 pages per segment (5000 tickers) to ensure we get a diverse set
                    for page in range(5):
                        endpoint = "v3/reference/tickers"
                        params = {
                            "market": segment["market"],
                            "exchange": segment["exchange"],
                            "active": "true",
                            "limit": 1000  # Maximum allowed by the API
                        }

                        if next_cursor:
                            params["cursor"] = next_cursor

                            # Make direct API request to get tickers
                        tickers_response = await self.polygon_api._make_request(endpoint, params)

                        if tickers_response and "results" in tickers_response:
                            page_tickers = tickers_response["results"]
                            segment_tickers.extend(page_tickers)

                            # Check if there are more pages
                            if "next_url" in tickers_response and tickers_response["next_url"]:
                                # Extract cursor from next_url
                                next_url = tickers_response["next_url"]
                                cursor_start = next_url.find("cursor=")
                                if cursor_start != -1:
                                    cursor_start += 7  # Length of "cursor="
                                    cursor_end = next_url.find(
                                        "&", cursor_start)
                                    if cursor_end == -1:
                                        cursor_end = len(next_url)
                                    next_cursor = next_url[cursor_start:cursor_end]
                                else:
                                    break  # No more pages
                            else:
                                break  # No more pages
                        else:
                            logger.warning(
                                f"Failed to retrieve tickers for {segment}")
                            break

                    logger.info(
                        f"Retrieved {len(segment_tickers)} tickers from Polygon API {segment['exchange']}")
                    tickers_data.extend(segment_tickers)

                # Log the total number of tickers retrieved
                logger.info(
                    f"Retrieved a total of {len(tickers_data)} tickers from Polygon API")
                # If we couldn't get any tickers, log an error
                if not tickers_data:
                    logger.error(
                        "Failed to retrieve any tickers from Polygon API")

                if not tickers_data:
                    logger.error("Failed to retrieve tickers from Polygon")
                    return

                # Filter based on basic criteria
                filtered_tickers = []
                for ticker in tickers_data:
                    symbol = ticker.get('ticker', '')

                    # Skip ADRs, ETFs, etc.
                    if not symbol or len(symbol) > 5 or not symbol.isalpha():
                        continue

                    # Only include common stocks (CS) and units (U)
                    ticker_type = ticker.get('type', '').upper()
                    if ticker_type not in ['CS', 'U']:
                        continue

                    primary_exchange = ticker.get('primary_exchange', '')
                    if not primary_exchange or primary_exchange in ['OTC', 'OTCBB']:
                        continue

                    # Add to filtered list
                    filtered_tickers.append(symbol)

                # Log the distribution of tickers by first letter to verify diversity
                letter_counts = {}
                for ticker in filtered_tickers:
                    first_letter = ticker[0] if ticker else ''
                    letter_counts[first_letter] = letter_counts.get(
                        first_letter, 0) + 1

                logger.info(
                    f"Ticker distribution by first letter: {letter_counts}")

                try:
                    # Store in Redis
                    pipeline = self.redis.pipeline()
                    pipeline.delete("market:universe:all")

                    if filtered_tickers:
                        # Store as set in Redis
                        pipeline.sadd("market:universe:all", *filtered_tickers)

                        # Store update timestamp
                        now = datetime.datetime.now().isoformat()
                        pipeline.set("market:universe:last_update", now)

                    pipeline.execute()
                except Exception as e:
                    logger.error(f"Redis pipeline execution failed: {str(e)}")
                    # Fallback to individual operations
                    try:
                        self.redis.delete("market:universe:all")
                        if filtered_tickers:
                            self.redis.sadd("market:universe:all",
                                            *filtered_tickers)
                            now = datetime.datetime.now().isoformat()
                            self.redis.set("market:universe:last_update", now)
                    except Exception as e:
                        logger.error(
                            f"Redis fallback operations failed: {str(e)}")

                # Update local cache
                self.full_universe = set(filtered_tickers)
                logger.info(
                    f"Initial universe built with {len(self.full_universe)} stocks across all exchanges")

                # Apply additional filters to get tradable universe
                await self._apply_tradable_filters_gpu()

                log_memory_usage("after_build_universe")

            except Exception as e:
                logger.error(
                    f"Error building initial universe: {str(e)}", exc_info=True)

    @staticmethod
    async def _apply_tradable_filters_gpu(self):
        """Apply filters to create tradable universe with GPU acceleration"""
        logger.info(
            "Applying filters for tradable universe with GPU acceleration")

        try:
            # Process tickers in batches to avoid API rate limits
            tradable_tickers = []
            batch_size = self.config['batch_size']
            batches = [list(self.full_universe)[i:i+batch_size]
                       for i in range(0, len(self.full_universe), batch_size)]

            total_batches = len(batches)
            for i, batch in enumerate(batches):
                logger.info(
                    f"Processing batch {i+1}/{total_batches} ({len(batch)} tickers)")

                # Process batch in parallel with GPU acceleration
                if self.gpu_available:
                    # Use GPU for batch processing
                    eligible_tickers = await self._check_batch_eligibility_gpu(batch)
                    tradable_tickers.extend(eligible_tickers)
                else:
                    # Fall back to CPU processing
                    futures = []
                    for ticker in batch:
                        futures.append(
                            asyncio.ensure_future(
                                self._check_ticker_eligibility(ticker))
                        )

                    # Wait for all checks to complete
                    results = await asyncio.gather(*futures, return_exceptions=True)

                    # Add eligible tickers
                    for ticker, result in zip(batch, results):
                        if isinstance(result, Exception):
                            logger.error(
                                f"Error checking {ticker}: {str(result)}")
                            continue

                        if result:
                            tradable_tickers.append(ticker)

                # Avoid API rate limits
                await asyncio.sleep(1.0)

            # Store tradable universe in Redis
            pipeline = self.redis.pipeline()
            pipeline.delete("market:universe:tradable")

            if tradable_tickers:
                pipeline.sadd("market:universe:tradable", *tradable_tickers)

                # Store update timestamp
                now = datetime.datetime.now().isoformat()
                pipeline.set("market:universe:tradable:last_update", now)

            pipeline.execute()

            logger.info(
                f"Tradable universe created with {len(tradable_tickers)} stocks")

        except Exception as e:
            logger.error(
                f"Error applying tradable filters: {str(e)}", exc_info=True)

    @staticmethod
    async def _check_batch_eligibility_gpu(self, tickers: List[str]) -> List[str]:
        """Check eligibility for a batch of tickers using GPU acceleration"""
        eligible_tickers = []

        try:
            # Get market data for all tickers in batch
            market_data = {}
            for ticker in tickers:
                # Check cache first
                cache_key = f"eligibility:{ticker}"
                cached_data = self.redis.get(cache_key)

                if cached_data:
                    # Use cached result
                    is_eligible = json.loads(cached_data)
                    if is_eligible:
                        eligible_tickers.append(ticker)
                    continue

                # Get market data
                prev_day = await mdh.get_previous_day_data(self.polygon_api, ticker)
                if prev_day:
                    market_data[ticker] = prev_day

            if not market_data:
                return eligible_tickers

            # Prepare data for GPU processing
            ticker_list = list(market_data.keys())
            prices = np.array([market_data[t].get('close', 0)
                              for t in ticker_list], dtype=np.float32)
            volumes = np.array([market_data[t].get('volume', 0)
                               for t in ticker_list], dtype=np.float32)

            try:
                # Move data to GPU
                if self.gpu_available:
                    cp_prices = cp.asarray(prices)
                    cp_volumes = cp.asarray(volumes)

                    # Apply filters on GPU
                    price_mask = (cp_prices >= self.config['min_price']) & (
                        cp_prices <= self.config['max_price'])
                    volume_mask = cp_volumes >= self.config['min_volume']

                    # Combined mask
                    combined_mask = price_mask & volume_mask

                    # Get indices of eligible tickers
                    eligible_indices = cp.where(combined_mask)[0]

                    try:
                        # Move back to CPU
                        eligible_indices_cpu = cp.asnumpy(eligible_indices)
                    except Exception as e:
                        logger.error(
                            f"Error moving data back to CPU: {str(e)}")
                        eligible_indices_cpu = []
                    finally:
                        # Clean up GPU memory
                        del cp_prices
                        del cp_volumes
                        del price_mask
                        del volume_mask
                        del combined_mask
                        del eligible_indices
                        cp.get_default_memory_pool().free_all_blocks()

                    # Get eligible tickers
                    for idx in eligible_indices_cpu:
                        ticker = ticker_list[idx]
                        eligible_tickers.append(ticker)

                        # Store in cache for 1 day
                        self.redis.setex(
                            f"eligibility:{ticker}", 86400, json.dumps(True))
                else:
                    # CPU fallback
                    for i, ticker in enumerate(ticker_list):
                        price = prices[i]
                        volume = volumes[i]

                        if price >= self.config['min_price'] and price <= self.config['max_price'] and volume >= self.config['min_volume']:
                            eligible_tickers.append(ticker)

                            # Store in cache for 1 day
                            self.redis.setex(
                                f"eligibility:{ticker}", 86400, json.dumps(True))

                return eligible_tickers
            except Exception as e:
                logger.error(f"Error in GPU processing: {str(e)}")
                # Fall back to CPU processing if GPU fails
                for i, ticker in enumerate(ticker_list):
                    price = prices[i]
                    volume = volumes[i]

                    if price >= self.config['min_price'] and price <= self.config['max_price'] and volume >= self.config['min_volume']:
                        eligible_tickers.append(ticker)

                        # Store in cache for 1 day
                        self.redis.setex(
                            f"eligibility:{ticker}", 86400, json.dumps(True))

                return eligible_tickers

        except Exception as e:
            logger.error(f"Error in batch eligibility check: {str(e)}")
            return eligible_tickers

    @staticmethod
    async def _check_ticker_eligibility(self, ticker):
        """Check if a ticker meets eligibility criteria for trading"""
        # Get price and volume data
        try:
            # Check cache first
            cache_key = f"eligibility:{ticker}"
            cached_data = self.redis.get(cache_key)

            if cached_data:
                # Use cached result
                return json.loads(cached_data)

            # Get market data
            prev_day = await mdh.get_previous_day_data(self.polygon_api, ticker)
            if not prev_day:
                return False

            # Basic filters
            price = prev_day.get('close', 0)
            volume = prev_day.get('volume', 0)

            if price < self.config['min_price'] or price > self.config['max_price']:
                return False

            if volume < self.config['min_volume']:
                return False

            # Check for options availability
            options_available = await mdh.check_options_availability(self.unusual_whales, ticker)
            if not options_available:
                return False

            # Store in cache for 1 day
            self.redis.setex(cache_key, 86400, json.dumps(True))
            return True

        except Exception as e:
            logger.error(f"Error checking eligibility for {ticker}: {str(e)}")
            return False

    @staticmethod
    async def _universe_refresh_task(self):
        """Task to periodically refresh the universe"""
        logger.info("Starting universe refresh task")

        # Daily refresh (midnight ET)
        while self.running:
            try:
                # Get current time (Eastern)
                now = datetime.datetime.now(pytz.timezone('US/Eastern'))

                # Refresh at 12:30 AM ET
                if now.hour == 0 and now.minute >= 30:
                    # Check if already refreshed today
                    last_refresh = self.redis.get(
                        "market:universe:last_update")

                    if not last_refresh or not mdh.same_day(last_refresh.decode('utf-8')):
                        logger.info("Performing daily universe refresh")
                        await self.build_initial_universe()

                # Wait 15 minutes before checking again
                await asyncio.sleep(900)

            except asyncio.CancelledError:
                logger.info("Universe refresh task cancelled")
                break
            except Exception as e:
                logger.error(
                    f"Error in universe refresh task: {str(e)}", exc_info=True)
                await asyncio.sleep(60)

    #
    # Watchlist Management (from stock_selection_watchlist.py)
    #
    @staticmethod
    async def update_watchlist(self) -> None:
        """Update the active watchlist with highest potential stocks using GPU acceleration"""
        logger.info("Updating active watchlist with GPU acceleration")
        log_memory_usage("before_update_watchlist")

        try:
            # Get current market conditions
            market_regime = await mdh.get_market_regime(self.redis)
            logger.info(f"Current market regime: {market_regime}")

            # Get candidate stocks from different sources
            pre_market_movers = await mdh.get_pre_market_movers(self.redis)
            unusual_options = await mdh.get_unusual_options_activity(self.unusual_whales)
            technical_setups = await mdh.get_technical_setups(self.redis)

            # Combine all candidates
            all_candidates = set()
            all_candidates.update(pre_market_movers)
            all_candidates.update(unusual_options)
            all_candidates.update(technical_setups)

            # Get tradable universe
            tradable_universe = self.redis.smembers("market:universe:tradable")
            tradable_universe = {
                t.decode('utf-8') if isinstance(t, bytes) else t for t in tradable_universe}

            # Filter by tradable universe
            candidates = all_candidates.intersection(tradable_universe)

            if not candidates:
                logger.warning("No valid candidates found for watchlist")
                return

            logger.info(f"Found {len(candidates)} potential candidates")

            # Calculate ranking factors for all candidates
            if self.gpu_available and len(candidates) > 50:
                # Use GPU for batch processing
                ranked_tickers = await self._calculate_batch_scores_gpu(list(candidates))
            else:
                # Process in parallel with CPU
                futures = []
                for ticker in candidates:
                    futures.append(
                        asyncio.ensure_future(
                            self._calculate_ticker_score(ticker))
                    )

                # Wait for all calculations to complete
                results = await asyncio.gather(*futures, return_exceptions=True)

                # Process results
                ranked_tickers = []
                for ticker, result in zip(candidates, results):
                    if isinstance(result, Exception):
                        logger.error(f"Error ranking {ticker}: {str(result)}")
                        continue

                    if result and result > 0:
                        ranked_tickers.append((ticker, result))

            # Sort by score descending
            ranked_tickers.sort(key=lambda x: x[1], reverse=True)

            # Take top N
            top_tickers = ranked_tickers[:self.config['watchlist_size']]

            try:
                # Update Redis
                pipeline = self.redis.pipeline()
                pipeline.delete("watchlist:active")

                for ticker, score in top_tickers:
                    pipeline.zadd("watchlist:active", {ticker: score})

                # Store last update time
                now = datetime.datetime.now().isoformat()
                pipeline.set("watchlist:active:last_update", now)

                pipeline.execute()
            except Exception as e:
                logger.error(f"Redis pipeline execution failed: {str(e)}")
                # Fallback to individual operations
                try:
                    self.redis.delete("watchlist:active")
                    for ticker, score in top_tickers:
                        self.redis.zadd("watchlist:active", {ticker: score})
                    now = datetime.datetime.now().isoformat()
                    self.redis.set("watchlist:active:last_update", now)
                except Exception as e:
                    logger.error(f"Redis fallback operations failed: {str(e)}")

            # Update local state
            self.active_watchlist = set([t[0] for t in top_tickers])

            logger.info(f"Watchlist updated with {len(top_tickers)} stocks")
            log_memory_usage("after_update_watchlist")

        except Exception as e:
            logger.error(f"Error updating watchlist: {str(e)}", exc_info=True)

    @staticmethod
    async def update_focused_list(self):
        """Update focused watchlist based on time of day and market conditions"""
        logger.info("Updating focused watchlist")

        try:
            # Get current active watchlist
            watchlist_data = self.redis.zrevrange(
                "watchlist:active", 0, -1, withscores=True)

            if not watchlist_data:
                logger.warning("Active watchlist is empty")
                return

            # Convert to list of tuples
            watchlist = [(item[0].decode('utf-8') if isinstance(item[0], bytes) else item[0], item[1])
                         for item in watchlist_data]

            # Determine focus size based on time of day
            now = datetime.datetime.now(pytz.timezone('US/Eastern'))
            market_open = now.replace(
                hour=9, minute=30, second=0, microsecond=0)

            # Calculate hours since market open
            hours_since_open = max(
                0, (now - market_open).total_seconds() / 3600)

            # Determine focus size
            if hours_since_open < 1:  # First hour
                focus_size = min(50, len(watchlist))
            elif hours_since_open < 3:  # Morning session
                focus_size = min(30, len(watchlist))
            elif hours_since_open < 5:  # Mid-day
                focus_size = min(20, len(watchlist))
            else:  # Afternoon
                focus_size = min(10, len(watchlist))

            # Get current market volatility
            volatility = await mdh.get_market_volatility(self.redis, self.polygon_api)

            # Adjust based on volatility
            if volatility > 30:  # High volatility
                focus_size = max(5, int(focus_size * 0.7))
                logger.info(
                    f"High volatility ({volatility}), reducing focus size to {focus_size}")

            # Take top N
            focused_tickers = watchlist[:focus_size]

            # Update Redis
            pipeline = self.redis.pipeline()
            pipeline.delete("watchlist:focused")

            for ticker, score in focused_tickers:
                pipeline.zadd("watchlist:focused", {ticker: score})

            # Store last update time
            now = datetime.datetime.now().isoformat()
            pipeline.set("watchlist:focused:last_update", now)

            pipeline.execute()

            # Update local state
            self.focused_list = set([t[0] for t in focused_tickers])

            logger.info(
                f"Focused list updated with {len(focused_tickers)} stocks")

        except Exception as e:
            logger.error(
                f"Error updating focused list: {str(e)}", exc_info=True)

    @staticmethod
    async def _watchlist_update_task(self):
        """Task to periodically update the watchlist"""
        logger.info("Starting watchlist update task")

        # Regular updates during trading hours
        while self.running:
            try:
                # Get current time (Eastern)
                now = datetime.datetime.now(pytz.timezone('US/Eastern'))

                # Update more frequently near market open
                if 9 <= now.hour < 16:  # 9 AM to 4 PM ET
                    # Calculate update interval
                    if now.hour == 9 and now.minute < 45:
                        # Every 5 minutes near open
                        interval = 300
                    else:
                        # Every 15 minutes during regular hours
                        interval = 900

                    # Update watchlist
                    if mdh.should_update_watchlist(self.redis, self.config['refresh_interval']):
                        await self.update_watchlist()

                    # Wait for next update
                    await asyncio.sleep(interval)
                else:
                    # Outside market hours, check less frequently
                    await asyncio.sleep(1800)  # 30 minutes

            except asyncio.CancelledError:
                logger.info("Watchlist update task cancelled")
                break
            except Exception as e:
                logger.error(
                    f"Error in watchlist update task: {str(e)}", exc_info=True)
                await asyncio.sleep(60)

    @staticmethod
    async def _focus_update_task(self):
        """Task to periodically update the focused list"""
        logger.info("Starting focused list update task")

        # Regular updates during trading hours
        while self.running:
            try:
                # Get current time (Eastern)
                now = datetime.datetime.now(pytz.timezone('US/Eastern'))

                # Only update during market hours
                if 9 <= now.hour < 16:  # 9 AM to 4 PM ET
                    # Update focused list
                    await self.update_focused_list()

                    # Wait 5 minutes
                    await asyncio.sleep(300)
                else:
                    # Outside market hours, check less frequently
                    await asyncio.sleep(1800)  # 30 minutes

            except asyncio.CancelledError:
                logger.info("Focus update task cancelled")
                break
            except Exception as e:
                logger.error(
                    f"Error in focus update task: {str(e)}", exc_info=True)
                await asyncio.sleep(60)

    #
    # Stock Scoring (from stock_selection_scoring.py)
    #
    @staticmethod
    async def _calculate_batch_scores_gpu(self, tickers: List[str]) -> List[Tuple[str, float]]:
        """Calculate scores for a batch of tickers using GPU acceleration"""
        ranked_tickers: List[Tuple[str, float]] = []

        try:
            # Get data for all tickers
            volume_data = {}
            price_data = {}
            volatility_data = {}
            options_data = {}

            # Fetch data in parallel
            volume_futures = [mdh.get_volume_data(self.redis, self.polygon_api, ticker)
                              for ticker in tickers]
            price_futures = [mdh.get_price_data(self.redis, self.polygon_api, ticker)
                             for ticker in tickers]
            volatility_futures = [mdh.get_atr_data(self.redis, self.polygon_api, ticker)
                                  for ticker in tickers]
            options_futures = [mdh.get_options_flow(self.unusual_whales, ticker)
                               for ticker in tickers]

            # Wait for all data to be fetched
            volume_results = await asyncio.gather(*volume_futures, return_exceptions=True)
            price_results = await asyncio.gather(*price_futures, return_exceptions=True)
            volatility_results = await asyncio.gather(*volatility_futures, return_exceptions=True)
            options_results = await asyncio.gather(*options_futures, return_exceptions=True)

            # Process results
            for i, ticker in enumerate(tickers):
                if not isinstance(volume_results[i], Exception) and volume_results[i]:
                    volume_data[ticker] = volume_results[i]

                if not isinstance(price_results[i], Exception) and price_results[i]:
                    price_data[ticker] = price_results[i]

                if not isinstance(volatility_results[i], Exception) and volatility_results[i]:
                    volatility_data[ticker] = volatility_results[i]

                if not isinstance(options_results[i], Exception) and options_results[i]:
                    options_data[ticker] = options_results[i]

            # Get market momentum for relative strength calculation
            market_momentum = await mdh.get_market_momentum(self.polygon_api)

            # Prepare arrays for GPU processing
            valid_tickers = [
                t for t in tickers if t in volume_data and t in price_data]

            if not valid_tickers:
                return ranked_tickers

            # Calculate volume scores
            volume_scores = np.zeros(len(valid_tickers), dtype=np.float32)
            for i, ticker in enumerate(valid_tickers):
                if ticker in volume_data:
                    current_volume = volume_data[ticker][0]
                    avg_volume = sum(
                        volume_data[ticker][1:11]) / 10 if len(volume_data[ticker]) >= 11 else current_volume
                    relative_volume = current_volume / avg_volume if avg_volume > 0 else 1.0

                    # Score based on relative volume (0-100)
                    if relative_volume >= 3.0:
                        volume_scores[i] = 100
                    elif relative_volume >= 2.5:
                        volume_scores[i] = 90
                    elif relative_volume >= 2.0:
                        volume_scores[i] = 80
                    elif relative_volume >= 1.5:
                        volume_scores[i] = 70
                    elif relative_volume >= 1.2:
                        volume_scores[i] = 60
                    elif relative_volume >= 1.0:
                        volume_scores[i] = 50
                    else:
                        volume_scores[i] = max(0, 50 * relative_volume)

            # Calculate volatility scores
            volatility_scores = np.zeros(len(valid_tickers), dtype=np.float32)
            for i, ticker in enumerate(valid_tickers):
                if ticker in volatility_data and ticker in price_data:
                    atr = volatility_data[ticker]
                    current_price = price_data[ticker][0]

                    # Calculate ATR as percentage of price
                    atr_percent = (atr / current_price) * \
                        100 if current_price > 0 else 0

                    # Score based on ATR percentage (0-100)
                    if atr_percent >= 5.0:
                        volatility_scores[i] = 100
                    elif atr_percent >= 4.0:
                        volatility_scores[i] = 90
                    elif atr_percent >= 3.0:
                        volatility_scores[i] = 80
                    elif atr_percent >= 2.0:
                        volatility_scores[i] = 70
                    elif atr_percent >= 1.5:
                        volatility_scores[i] = 60
                    elif atr_percent >= 1.0:
                        volatility_scores[i] = 50
                    else:
                        volatility_scores[i] = max(0, 50 * atr_percent)

            # Calculate momentum scores
            momentum_scores = np.zeros(len(valid_tickers), dtype=np.float32)
            for i, ticker in enumerate(valid_tickers):
                if ticker in price_data and len(price_data[ticker]) >= 2:
                    # Calculate short-term momentum (1-day percent change)
                    short_term = (price_data[ticker][0] / price_data[ticker]
                                  [1] - 1) * 100 if price_data[ticker][1] > 0 else 0

                    # Calculate relative strength (stock momentum vs market momentum)
                    relative_strength = short_term - market_momentum

                    # Score based on relative strength and momentum (0-100)
                    if relative_strength >= 5.0 and short_term > 0:
                        momentum_scores[i] = 100
                    elif relative_strength >= 3.0 and short_term > 0:
                        momentum_scores[i] = 90
                    elif relative_strength >= 2.0 and short_term > 0:
                        momentum_scores[i] = 80
                    elif relative_strength >= 1.0 and short_term > 0:
                        momentum_scores[i] = 70
                    elif relative_strength >= 0.0 and short_term > 0:
                        momentum_scores[i] = 60
                    elif short_term > 0:
                        momentum_scores[i] = 50
                    else:
                        momentum_scores[i] = max(0, 50 + short_term)

            # Calculate options scores
            options_scores = np.zeros(len(valid_tickers), dtype=np.float32)
            for i, ticker in enumerate(valid_tickers):
                if ticker in options_data:
                    # Calculate options volume
                    call_volume = sum(item.get(
                        'volume', 0) for item in options_data[ticker] if item.get('side') == 'call')
                    put_volume = sum(item.get(
                        'volume', 0) for item in options_data[ticker] if item.get('side') == 'put')

                    total_volume = call_volume + put_volume

                    # Calculate premium (dollar value)
                    call_premium = sum(item.get(
                        'premium', 0) for item in options_data[ticker] if item.get('side') == 'call')
                    put_premium = sum(item.get(
                        'premium', 0) for item in options_data[ticker] if item.get('side') == 'put')

                    total_premium = call_premium + put_premium

                    # Calculate put/call ratio
                    put_call_ratio = put_volume / call_volume if call_volume > 0 else 0

                    # Score based on volume and premium (0-100)
                    volume_score = 0
                    if total_volume >= 10000:
                        volume_score = 100
                    elif total_volume >= 5000:
                        volume_score = 80
                    elif total_volume >= 1000:
                        volume_score = 60
                    elif total_volume >= 500:
                        volume_score = 40
                    elif total_volume > 0:
                        volume_score = 20

                    premium_score = 0
                    if total_premium >= 10000000:  # $10M
                        premium_score = 100
                    elif total_premium >= 5000000:  # $5M
                        premium_score = 80
                    elif total_premium >= 1000000:  # $1M
                        premium_score = 60
                    elif total_premium >= 500000:  # $500K
                        premium_score = 40
                    elif total_premium > 0:
                        premium_score = 20

                    # Combined score
                    combined_score = (volume_score + premium_score) / 2

                    # Adjust based on put/call ratio
                    if put_call_ratio >= 3.0:
                        combined_score *= 1.2
                    elif put_call_ratio <= 0.2:
                        combined_score *= 1.1

                    options_scores[i] = min(100, combined_score)

            try:
                # Move data to GPU for final calculation
                if self.gpu_available:
                    cp_volume_scores = cp.asarray(volume_scores)
                    cp_volatility_scores = cp.asarray(volatility_scores)
                    cp_momentum_scores = cp.asarray(momentum_scores)
                    cp_options_scores = cp.asarray(options_scores)

                    # Apply weights
                    weights = self.config['weights']
                    cp_total_scores = (
                        cp_volume_scores * weights['volume'] +
                        cp_volatility_scores * weights['volatility'] +
                        cp_momentum_scores * weights['momentum'] +
                        cp_options_scores * weights['options']
                    )

                    try:
                        # Move back to CPU
                        total_scores = cp.asnumpy(cp_total_scores)
                    except Exception as e:
                        logger.error(
                            f"Error moving scores back to CPU: {str(e)}")
                        total_scores = np.zeros_like(volume_scores)
                    finally:
                        # Clean up GPU memory
                        del cp_volume_scores
                        del cp_volatility_scores
                        del cp_momentum_scores
                        del cp_options_scores
                        del cp_total_scores
                        cp.get_default_memory_pool().free_all_blocks()
                else:
                    # Calculate on CPU
                    weights = self.config['weights']
                    total_scores = (
                        volume_scores * weights['volume'] +
                        volatility_scores * weights['volatility'] +
                        momentum_scores * weights['momentum'] +
                        options_scores * weights['options']
                    )

                # Create ranked list
                for i, ticker in enumerate(valid_tickers):
                    if total_scores[i] > 0:
                        ranked_tickers.append((ticker, float(total_scores[i])))

                        # Store factor breakdown in Redis
                        factors = {
                            'volume': float(volume_scores[i]),
                            'volatility': float(volatility_scores[i]),
                            'momentum': float(momentum_scores[i]),
                            'options': float(options_scores[i]),
                            'total': float(total_scores[i]),
                            'timestamp': datetime.datetime.now().isoformat()
                        }

                        self.redis.hset(
                            f"ticker:{ticker}:factors", mapping=factors)

                return ranked_tickers
            except Exception as e:
                logger.error(f"Error in final score calculation: {str(e)}")
                # Fallback to CPU calculation
                weights = self.config['weights']
                total_scores = (
                    volume_scores * weights['volume'] +
                    volatility_scores * weights['volatility'] +
                    momentum_scores * weights['momentum'] +
                    options_scores * weights['options']
                )

                # Create ranked list
                for i, ticker in enumerate(valid_tickers):
                    if total_scores[i] > 0:
                        ranked_tickers.append((ticker, float(total_scores[i])))

                        # Store factor breakdown in Redis
                        factors = {
                            'volume': float(volume_scores[i]),
                            'volatility': float(volatility_scores[i]),
                            'momentum': float(momentum_scores[i]),
                            'options': float(options_scores[i]),
                            'total': float(total_scores[i]),
                            'timestamp': datetime.datetime.now().isoformat()
                        }

                        self.redis.hset(
                            f"ticker:{ticker}:factors", mapping=factors)

                return ranked_tickers

        except Exception as e:
            logger.error(f"Error in batch score calculation: {str(e)}")
            return ranked_tickers

    @staticmethod
    async def _calculate_ticker_score(self, ticker):
        """Calculate composite ranking score for a ticker"""
        try:
            # Calculate individual factors
            volume_score = await self._calculate_volume_factor(ticker)
            volatility_score = await self._calculate_volatility_factor(ticker)
            momentum_score = await self._calculate_momentum_factor(ticker)
            options_score = await self._calculate_options_factor(ticker)

            # Apply weights
            weights = self.config['weights']
            total_score = (
                volume_score * weights['volume'] +
                volatility_score * weights['volatility'] +
                momentum_score * weights['momentum'] +
                options_score * weights['options']
            )

            # Store factor breakdown in Redis
            factors = {
                'volume': volume_score,
                'volatility': volatility_score,
                'momentum': momentum_score,
                'options': options_score,
                'total': total_score,
                'timestamp': datetime.datetime.now().isoformat()
            }

            self.redis.hset(f"ticker:{ticker}:factors", mapping=factors)

            return total_score

        except Exception as e:
            logger.error(f"Error calculating score for {ticker}: {str(e)}")
            return 0

    @staticmethod
    async def _calculate_volume_factor(self, ticker):
        """Calculate volume factor for a ticker"""
        try:
            # Get volume data
            volume_data = await mdh.get_volume_data(self.redis, self.polygon_api, ticker)
            if not volume_data or len(volume_data) < 2:
                return 0

            # Calculate relative volume
            current_volume = volume_data[0]
            avg_volume = sum(volume_data[1:11]) / \
                10 if len(volume_data) >= 11 else current_volume
            relative_volume = current_volume / avg_volume if avg_volume > 0 else 1.0

            # Score based on relative volume (0-100)
            if relative_volume >= 3.0:
                return 100
            elif relative_volume >= 2.5:
                return 90
            elif relative_volume >= 2.0:
                return 80
            elif relative_volume >= 1.5:
                return 70
            elif relative_volume >= 1.2:
                return 60
            elif relative_volume >= 1.0:
                return 50
            else:
                return max(0, 50 * relative_volume)

        except Exception as e:
            logger.error(
                f"Error calculating volume factor for {ticker}: {str(e)}")
            return 0

    @staticmethod
    async def _calculate_volatility_factor(self, ticker):
        """Calculate volatility factor for a ticker"""
        try:
            # Get ATR and price data
            atr = await mdh.get_atr_data(self.redis, self.polygon_api, ticker)
            price_data = await mdh.get_price_data(self.redis, self.polygon_api, ticker)

            if not atr or not price_data or len(price_data) < 1:
                return 0

            current_price = price_data[0]

            # Calculate ATR as percentage of price
            atr_percent = (atr / current_price) * \
                100 if current_price > 0 else 0

            # Score based on ATR percentage (0-100)
            if atr_percent >= 5.0:
                return 100
            elif atr_percent >= 4.0:
                return 90
            elif atr_percent >= 3.0:
                return 80
            elif atr_percent >= 2.0:
                return 70
            elif atr_percent >= 1.5:
                return 60
            elif atr_percent >= 1.0:
                return 50
            else:
                return max(0, 50 * atr_percent)

        except Exception as e:
            logger.error(
                f"Error calculating volatility factor for {ticker}: {str(e)}")
            return 0

    @staticmethod
    async def _calculate_momentum_factor(self, ticker):
        """Calculate momentum factor for a ticker"""
        try:
            # Get price data
            price_data = await mdh.get_price_data(self.redis, self.polygon_api, ticker)
            if not price_data or len(price_data) < 2:
                return 0

            # Calculate short-term momentum (1-day percent change)
            short_term = (price_data[0] / price_data[1] -
                          1) * 100 if price_data[1] > 0 else 0

            # Get market momentum for relative strength calculation
            market_momentum = await mdh.get_market_momentum(self.polygon_api)

            # Calculate relative strength (stock momentum vs market momentum)
            relative_strength = short_term - market_momentum

            # Score based on relative strength and momentum (0-100)
            if relative_strength >= 5.0 and short_term > 0:
                return 100
            elif relative_strength >= 3.0 and short_term > 0:
                return 90
            elif relative_strength >= 2.0 and short_term > 0:
                return 80
            elif relative_strength >= 1.0 and short_term > 0:
                return 70
            elif relative_strength >= 0.0 and short_term > 0:
                return 60
            elif short_term > 0:
                return 50
            else:
                return max(0, 50 + short_term)

        except Exception as e:
            logger.error(
                f"Error calculating momentum factor for {ticker}: {str(e)}")
            return 0

    @staticmethod
    async def _calculate_options_factor(self, ticker):
        """Calculate options factor for a ticker"""
        try:
            # Get options flow data
            options_data = await mdh.get_options_flow(self.unusual_whales, ticker)
            if not options_data:
                return 0

            # Calculate options volume
            call_volume = sum(item.get('volume', 0)
                              for item in options_data if item.get('side') == 'call')
            put_volume = sum(item.get('volume', 0)
                             for item in options_data if item.get('side') == 'put')

            total_volume = call_volume + put_volume

            # Calculate premium (dollar value)
            call_premium = sum(item.get('premium', 0)
                               for item in options_data if item.get('side') == 'call')
            put_premium = sum(item.get('premium', 0)
                              for item in options_data if item.get('side') == 'put')

            total_premium = call_premium + put_premium

            # Calculate put/call ratio
            put_call_ratio = put_volume / call_volume if call_volume > 0 else 0

            # Score based on volume and premium (0-100)
            volume_score = 0
            if total_volume >= 10000:
                volume_score = 100
            elif total_volume >= 5000:
                volume_score = 80
            elif total_volume >= 1000:
                volume_score = 60
            elif total_volume >= 500:
                volume_score = 40
            elif total_volume > 0:
                volume_score = 20

            premium_score = 0
            if total_premium >= 10000000:  # $10M
                premium_score = 100
            elif total_premium >= 5000000:  # $5M
                premium_score = 80
            elif total_premium >= 1000000:  # $1M
                premium_score = 60
            elif total_premium >= 500000:  # $500K
                premium_score = 40
            elif total_premium > 0:
                premium_score = 20

            # Combined score
            combined_score = (volume_score + premium_score) / 2

            # Adjust based on put/call ratio
            if put_call_ratio >= 3.0:
                combined_score *= 1.2
            elif put_call_ratio <= 0.2:
                combined_score *= 1.1

            return min(100, combined_score)

        except Exception as e:
            logger.error(
                f"Error calculating options factor for {ticker}: {str(e)}")
            return 0
