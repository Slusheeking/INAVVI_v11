#!/usr/bin/env python3
"""
Production-ready dynamic stock selection system for algorithmic trading.
This system integrates market data, options flow, and technical analysis
to dynamically select the most promising stocks for trading.
"""

import logging
import time
import json
import asyncio
import datetime
import pytz
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger('stock_selection')

class StockSelectionSystem:
    """Production-ready dynamic stock selection system"""
    
    def __init__(self, redis_client, polygon_client, unusual_whales_client):
        self.redis = redis_client
        self.polygon = polygon_client
        self.unusual_whales = unusual_whales_client
        
        # Performance optimization with thread pools
        self.executor = ThreadPoolExecutor(max_workers=20)
        
        # Market data cache
        self.cache = {
            'market_data': {},
            'options_data': {},
            'technical_data': {},
            'last_refresh': {}
        }
        
        # Configuration
        self.config = {
            'universe_size': 2000,
            'watchlist_size': 100,
            'focused_list_size': 30,
            'min_price': 5.0,
            'max_price': 200.0,
            'min_volume': 500000,
            'min_relative_volume': 1.5,
            'min_atr_percent': 1.0,
            'refresh_interval': 900,    # 15 minutes
            'cache_expiry': 300,        # 5 minutes
            'weights': {
                'volume': 0.30,
                'volatility': 0.25,
                'momentum': 0.25,
                'options': 0.20
            }
        }
        
        # Internal state
        self.full_universe = set()
        self.active_watchlist = set()
        self.focused_list = set()
        self.running = False
        self.tasks = {}
        
        logger.info("Stock Selection System initialized")
    
    async def start(self):
        """Start the stock selection system"""
        if self.running:
            logger.warning("Stock selection system already running")
            return
            
        self.running = True
        logger.info("Starting stock selection system")
        
        # Initialize universe
        await self.build_initial_universe()
        
        # Start periodic tasks
        self.tasks['universe_refresh'] = asyncio.create_task(self._universe_refresh_task())
        self.tasks['watchlist_update'] = asyncio.create_task(self._watchlist_update_task())
        self.tasks['focus_update'] = asyncio.create_task(self._focus_update_task())
        
        logger.info("Stock selection system started")
    
    async def stop(self):
        """Stop the stock selection system"""
        if not self.running:
            return
            
        logger.info("Stopping stock selection system")
        self.running = False
        
        # Cancel all tasks
        for name, task in self.tasks.items():
            if not task.done():
                logger.info(f"Cancelling task: {name}")
                task.cancel()
                
        # Shutdown thread pool
        self.executor.shutdown(wait=False)
        logger.info("Stock selection system stopped")
    
    async def build_initial_universe(self):
        """Build initial universe of tradable stocks"""
        logger.info("Building initial universe")
        
        # Get all US equities
        try:
            # Get active US stocks
            tickers_data = await self._get_all_active_tickers()
            
            if not tickers_data:
                logger.error("Failed to retrieve tickers from Polygon")
                return
                
            # Filter based on basic criteria
            filtered_tickers = []
            for ticker in tickers_data:
                symbol = ticker.get('ticker')
                
                # Skip ADRs, ETFs, etc.
                if not symbol or len(symbol) > 5 or not symbol.isalpha():
                    continue
                    
                ticker_type = ticker.get('type', '')
                if ticker_type != 'CS':  # Common Stock
                    continue
                    
                primary_exchange = ticker.get('primary_exchange', '')
                if not primary_exchange or primary_exchange in ['OTC', 'OTCBB']:
                    continue
                
                # Add to filtered list
                filtered_tickers.append(symbol)
            
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
            
            # Update local cache
            self.full_universe = set(filtered_tickers)
            logger.info(f"Initial universe built with {len(self.full_universe)} stocks")
            
            # Apply additional filters to get tradable universe
            await self._apply_tradable_filters()
            
        except Exception as e:
            logger.error(f"Error building initial universe: {str(e)}", exc_info=True)
    
    async def _apply_tradable_filters(self):
        """Apply filters to create tradable universe"""
        logger.info("Applying filters for tradable universe")
        
        try:
            # Process tickers in batches to avoid API rate limits
            tradable_tickers = []
            batch_size = 50
            batches = [list(self.full_universe)[i:i+batch_size] for i in range(0, len(self.full_universe), batch_size)]
            
            total_batches = len(batches)
            for i, batch in enumerate(batches):
                logger.info(f"Processing batch {i+1}/{total_batches} ({len(batch)} tickers)")
                
                # Process batch in parallel
                futures = []
                for ticker in batch:
                    futures.append(
                        asyncio.ensure_future(self._check_ticker_eligibility(ticker))
                    )
                
                # Wait for all checks to complete
                results = await asyncio.gather(*futures, return_exceptions=True)
                
                # Add eligible tickers
                for ticker, result in zip(batch, results):
                    if isinstance(result, Exception):
                        logger.error(f"Error checking {ticker}: {str(result)}")
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
            
            logger.info(f"Tradable universe created with {len(tradable_tickers)} stocks")
            
        except Exception as e:
            logger.error(f"Error applying tradable filters: {str(e)}", exc_info=True)
    
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
            prev_day = await self._get_previous_day_data(ticker)
            if not prev_day:
                return False
                
            # Basic filters
            price = prev_day.get('close', 0)
            volume = prev_day.get('volume', 0)
            
            if price < self.config['min_price'] or price > self.config['max_price']:
                return False
                
            if volume < self.config['min_volume']:
                return False
            
            # Check for options availability (real implementation would check actual options chain)
            options_available = await self._check_options_availability(ticker)
            if not options_available:
                return False
            
            # Store in cache for 1 day
            self.redis.setex(cache_key, 86400, json.dumps(True))
            return True
            
        except Exception as e:
            logger.error(f"Error checking eligibility for {ticker}: {str(e)}")
            return False
    
    async def update_watchlist(self):
        """Update the active watchlist with highest potential stocks"""
        logger.info("Updating active watchlist")
        
        try:
            # Get current market conditions
            market_regime = await self._get_market_regime()
            logger.info(f"Current market regime: {market_regime}")
            
            # Get candidate stocks from different sources
            pre_market_movers = await self._get_pre_market_movers()
            unusual_options = await self._get_unusual_options_activity()
            technical_setups = await self._get_technical_setups()
            
            # Combine all candidates
            all_candidates = set()
            all_candidates.update(pre_market_movers)
            all_candidates.update(unusual_options)
            all_candidates.update(technical_setups)
            
            # Get tradable universe
            tradable_universe = self.redis.smembers("market:universe:tradable")
            
            # Filter by tradable universe
            candidates = all_candidates.intersection(tradable_universe)
            
            if not candidates:
                logger.warning("No valid candidates found for watchlist")
                return
                
            logger.info(f"Found {len(candidates)} potential candidates")
            
            # Calculate ranking factors for all candidates
            ranked_tickers = []
            
            # Process in parallel
            futures = []
            for ticker in candidates:
                futures.append(
                    asyncio.ensure_future(self._calculate_ticker_score(ticker))
                )
                
            # Wait for all calculations to complete
            results = await asyncio.gather(*futures, return_exceptions=True)
            
            # Process results
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
            
            # Update Redis
            pipeline = self.redis.pipeline()
            pipeline.delete("watchlist:active")
            
            for ticker, score in top_tickers:
                pipeline.zadd("watchlist:active", {ticker: score})
                
            # Store last update time
            now = datetime.datetime.now().isoformat()
            pipeline.set("watchlist:active:last_update", now)
            
            pipeline.execute()
            
            # Update local state
            self.active_watchlist = set([t[0] for t in top_tickers])
            
            logger.info(f"Watchlist updated with {len(top_tickers)} stocks")
            
        except Exception as e:
            logger.error(f"Error updating watchlist: {str(e)}", exc_info=True)
    
    async def update_focused_list(self):
        """Update focused watchlist based on time of day and market conditions"""
        logger.info("Updating focused watchlist")
        
        try:
            # Get current active watchlist
            watchlist_data = self.redis.zrevrange("watchlist:active", 0, -1, withscores=True)
            
            if not watchlist_data:
                logger.warning("Active watchlist is empty")
                return
                
            # Convert to list of tuples
            watchlist = [(item[0].decode('utf-8') if isinstance(item[0], bytes) else item[0], item[1]) 
                         for item in watchlist_data]
            
            # Determine focus size based on time of day
            now = datetime.datetime.now(pytz.timezone('US/Eastern'))
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            
            # Calculate hours since market open
            hours_since_open = max(0, (now - market_open).total_seconds() / 3600)
            
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
            volatility = await self._get_market_volatility()
            
            # Adjust based on volatility
            if volatility > 30:  # High volatility
                focus_size = max(5, int(focus_size * 0.7))
                logger.info(f"High volatility ({volatility}), reducing focus size to {focus_size}")
            
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
            
            logger.info(f"Focused list updated with {len(focused_tickers)} stocks")
            
        except Exception as e:
            logger.error(f"Error updating focused list: {str(e)}", exc_info=True)
    
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
    
    async def _calculate_volume_factor(self, ticker):
        """Calculate volume factor for ranking"""
        try:
            # Get historical volume data
            volume_data = await self._get_volume_data(ticker)
            
            if not volume_data:
                return 0
                
            # Calculate relative volume (today vs 10-day average)
            current_volume = volume_data[0]
            avg_volume = sum(volume_data[1:11]) / 10 if len(volume_data) >= 11 else current_volume
            
            relative_volume = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Score based on relative volume (0-100)
            if relative_volume >= 3.0:
                score = 100
            elif relative_volume >= 2.5:
                score = 90
            elif relative_volume >= 2.0:
                score = 80
            elif relative_volume >= 1.5:
                score = 70
            elif relative_volume >= 1.2:
                score = 60
            elif relative_volume >= 1.0:
                score = 50
            else:
                score = max(0, 50 * relative_volume)
                
            return score
            
        except Exception as e:
            logger.error(f"Error calculating volume factor for {ticker}: {str(e)}")
            return 0
    
    async def _calculate_volatility_factor(self, ticker):
        """Calculate volatility factor for ranking"""
        try:
            # Get ATR (Average True Range) data
            atr_data = await self._get_atr_data(ticker)
            
            if not atr_data:
                return 0
                
            # Get current price
            price_data = await self._get_price_data(ticker)
            
            if not price_data:
                return 0
                
            current_price = price_data[0]
            
            # Calculate ATR as percentage of price
            atr_percent = (atr_data / current_price) * 100 if current_price > 0 else 0
            
            # Score based on ATR percentage (0-100)
            if atr_percent >= 5.0:  # Fixed typo: was "a5.0"
                score = 100
            elif atr_percent >= 4.0:
                score = 90
            elif atr_percent >= 3.0:
                score = 80
            elif atr_percent >= 2.0:
                score = 70
            elif atr_percent >= 1.5:
                score = 60
            elif atr_percent >= 1.0:
                score = 50
            else:
                score = max(0, 50 * atr_percent)
                
            return score
            
        except Exception as e:
            logger.error(f"Error calculating volatility factor for {ticker}: {str(e)}")
            return 0
    
    async def _calculate_momentum_factor(self, ticker):
        """Calculate momentum factor for ranking"""
        try:
            # Get price data
            price_data = await self._get_price_data(ticker)
            
            if not price_data or len(price_data) < 2:
                return 0
                
            # Calculate short-term momentum (1-day percent change)
            short_term = (price_data[0] / price_data[1] - 1) * 100 if price_data[1] > 0 else 0
            
            # Calculate mid-term momentum (5-day percent change)
            mid_term = (price_data[0] / price_data[5] - 1) * 100 if len(price_data) > 5 and price_data[5] > 0 else short_term
            
            # Get market momentum (S&P 500)
            market_momentum = await self._get_market_momentum()
            
            # Calculate relative strength (stock momentum vs market momentum)
            relative_strength = short_term - market_momentum
            
            # Score based on relative strength and momentum (0-100)
            # Higher score for stocks outperforming the market
            if relative_strength >= 5.0 and short_term > 0:
                score = 100
            elif relative_strength >= 3.0 and short_term > 0:
                score = 90
            elif relative_strength >= 2.0 and short_term > 0:
                score = 80
            elif relative_strength >= 1.0 and short_term > 0:
                score = 70
            elif relative_strength >= 0.0 and short_term > 0:
                score = 60
            elif short_term > 0:
                score = 50
            else:
                score = max(0, 50 + short_term)  # Lower scores for negative momentum
                
            return score
            
        except Exception as e:
            logger.error(f"Error calculating momentum factor for {ticker}: {str(e)}")
            return 0
    
    async def _calculate_options_factor(self, ticker):
        """Calculate options activity factor for ranking"""
        try:
            # Get options flow data from Unusual Whales
            options_data = await self._get_options_flow(ticker)
            
            if not options_data:
                return 0
                
            # Calculate options volume
            call_volume = sum(item.get('volume', 0) for item in options_data if item.get('side') == 'call')
            put_volume = sum(item.get('volume', 0) for item in options_data if item.get('side') == 'put')
            
            total_volume = call_volume + put_volume
            
            # Calculate premium (dollar value)
            call_premium = sum(item.get('premium', 0) for item in options_data if item.get('side') == 'call')
            put_premium = sum(item.get('premium', 0) for item in options_data if item.get('side') == 'put')
            
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
            
            # Adjust based on put/call ratio (higher score for unusual put activity)
            if put_call_ratio >= 3.0:
                combined_score *= 1.2  # Boost score for heavy put activity
            elif put_call_ratio <= 0.2:
                combined_score *= 1.1  # Slight boost for heavy call activity
                
            return min(100, combined_score)
            
        except Exception as e:
            logger.error(f"Error calculating options factor for {ticker}: {str(e)}")
            return 0
    
    # Background tasks
    
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
                    last_refresh = self.redis.get("market:universe:last_update")
                    
                    if not last_refresh or not self._same_day(last_refresh.decode('utf-8')):
                        logger.info("Performing daily universe refresh")
                        await self.build_initial_universe()
                
                # Wait 15 minutes before checking again
                await asyncio.sleep(900)
                
            except asyncio.CancelledError:
                logger.info("Universe refresh task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in universe refresh task: {str(e)}", exc_info=True)
                await asyncio.sleep(60)
    
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
                    if self._should_update_watchlist():
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
                logger.error(f"Error in watchlist update task: {str(e)}", exc_info=True)
                await asyncio.sleep(60)
    
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
                logger.error(f"Error in focus update task: {str(e)}", exc_info=True)
                await asyncio.sleep(60)
    
    # Helper methods for API interactions
    
    async def _get_all_active_tickers(self):
        """Get all active tickers from Polygon"""
        # In production, implement proper pagination to get all tickers
        return await self.polygon.list_tickers(
            market="stocks",
            locale="us",
            active=True,
            limit=1000
        )
    
    async def _get_previous_day_data(self, ticker):
        """Get previous day's trading data"""
        return await self.polygon.get_previous_close(ticker)
    
    async def _check_options_availability(self, ticker):
        """Check if options are available for a ticker"""
        # In production, implement actual check via options chain API
        # For now, assume stocks above $5 with sufficient volume have options
        return True
    
    async def _get_pre_market_movers(self):
        """Get pre-market movers"""
        # Implementation would depend on specific data source
        # Return a set of tickers with significant pre-market movement
        return set()
    
    async def _get_unusual_options_activity(self):
        """Get stocks with unusual options activity"""
        try:
            # Get data from Unusual Whales
            data = self.unusual_whales.get_alerts(limit=100)
            
            if isinstance(data, pd.DataFrame) and not data.empty:
                # Extract tickers
                return set(data['ticker'].unique())
            
            return set()
            
        except Exception as e:
            logger.error(f"Error getting unusual options activity: {str(e)}")
            return set()
    
    async def _get_technical_setups(self):
        """Get stocks with technical setups"""
        # Implementation would scan for technical patterns
        # Return a set of tickers with technical setups
        return set()
    
    async def _get_market_regime(self):
        """Get current market regime"""
        # Implementation would classify market conditions
        return "normal"
    
    async def _get_market_volatility(self):
        """Get current market volatility (VIX)"""
        # Implementation would get VIX or similar volatility metric
        return 15.0
    
    async def _get_volume_data(self, ticker):
        """Get historical volume data for a ticker"""
        # Implementation would get volume data from Polygon
        return [100000, 90000, 95000, 85000, 105000]
    
    async def _get_atr_data(self, ticker):
        """Get ATR (Average True Range) for a ticker"""
        # Implementation would calculate ATR from OHLC data
        return 2.5
    
    async def _get_price_data(self, ticker):
        """Get historical price data for a ticker"""
        # Implementation would get price data from Polygon
        return [100.0, 99.0, 101.0, 98.0, 102.0, 97.0]
    
    async def _get_market_momentum(self):
        """Get market momentum (S&P 500 change)"""
        # Implementation would get S&P 500 performance
        return 0.5
    
    async def _get_options_flow(self, ticker):
        """Get options flow data for a ticker"""
        # Implementation would get options flow from Unusual Whales
        return []
    
    # Utility methods
    
    def _should_update_watchlist(self):
        """Check if watchlist should be updated"""
        # Get last update time
        last_update = self.redis.get("watchlist:active:last_update")
        
        if not last_update:
            return True
            
        # Convert to datetime
        last_update_time = datetime.datetime.fromisoformat(last_update.decode('utf-8'))
        now = datetime.datetime.now()
        
        # Update if more than 15 minutes since last update
        elapsed_seconds = (now - last_update_time).total_seconds()
        return elapsed_seconds >= self.config['refresh_interval']
    
    def _same_day(self, timestamp_str):
        """Check if timestamp is from the same day (ET)"""
        # Parse timestamp
        timestamp = datetime.datetime.fromisoformat(timestamp_str)
        
        # Get current date (ET)
        now = datetime.datetime.now(pytz.timezone('US/Eastern'))
        
        # Check if same day
        return timestamp.date() == now.date()