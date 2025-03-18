#!/usr/bin/env python3
"""
GPU-Optimized Stock Selection System for NVIDIA GH200
This implementation enhances the stock selection system with GPU acceleration
for faster data processing and analysis.
"""

import logging
import time
import json
import asyncio
import datetime
import pytz
import numpy as np
import pandas as pd
import cupy as cp
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import psutil
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gpu_stock_selection')

def log_memory_usage(location_tag):
    """Log CPU and GPU memory usage"""
    try:
        # Log CPU memory
        process = psutil.Process()
        cpu_mem = process.memory_info().rss / (1024 * 1024)
        
        # Log GPU memory
        mem_info = cp.cuda.runtime.memGetInfo()
        free, total = mem_info[0], mem_info[1]
        used = total - free
        
        logger.info(f"[{location_tag}] Memory Usage - CPU: {cpu_mem:.2f}MB, GPU: Used={used/(1024**2):.2f}MB, Free={free/(1024**2):.2f}MB, Total={total/(1024**2):.2f}MB")
    except Exception as e:
        logger.error(f"Failed to log memory usage at {location_tag}: {e}")

def configure_gpu():
    """Configure GPU for optimal performance"""
    try:
        # Configure TensorFlow
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Memory growth enabled for {len(gpus)} GPUs")
                
                # Set TensorFlow to use mixed precision
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                logger.info("TensorFlow configured with mixed precision")
            except RuntimeError as e:
                logger.warning(f"Memory growth configuration failed: {e}")
        
        # Configure CuPy
        try:
            # Use unified memory for better performance
            cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
            logger.info("CuPy configured with unified memory")
            
            # Get device count and properties
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count > 0:
                for i in range(device_count):
                    device_props = cp.cuda.runtime.getDeviceProperties(i)
                    device_name = device_props["name"].decode()
                    logger.info(f"GPU {i}: {device_name}")
                    
                    # If GH200 is available, use it
                    if "GH200" in device_name:
                        cp.cuda.Device(i).use()
                        logger.info(f"Using GH200 GPU at index {i}")
                        break
            
            return True
        except Exception as e:
            logger.error(f"CuPy configuration failed: {e}")
            return False
            
    except Exception as e:
        logger.error(f"GPU configuration failed: {e}")
        return False

class GPUStockSelectionSystem:
    """GPU-Optimized Stock Selection System for NVIDIA GH200"""
    
    def __init__(self, redis_client, polygon_client, unusual_whales_client):
        self.redis = redis_client
        self.polygon = polygon_client
        self.unusual_whales = unusual_whales_client
        
        # Configure GPU
        self.gpu_available = configure_gpu()
        if not self.gpu_available:
            logger.warning("GPU acceleration not available, falling back to CPU")
        
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
            },
            'batch_size': 1024,         # Batch size for GPU processing
            'max_workers': min(os.cpu_count(), 8)  # Limit workers to avoid overloading
        }
        
        # Internal state
        self.full_universe = set()
        self.active_watchlist = set()
        self.focused_list = set()
        self.running = False
        self.tasks = {}
        
        # Shared memory for inter-process communication
        self.shared_data = {}
        
        # Log initial memory usage
        log_memory_usage("initialization")
        
        logger.info("GPU-Optimized Stock Selection System initialized")
    
    async def start(self):
        """Start the stock selection system"""
        if self.running:
            logger.warning("Stock selection system already running")
            return
            
        self.running = True
        logger.info("Starting GPU-optimized stock selection system")
        
        # Initialize universe
        await self.build_initial_universe()
        
        # Start periodic tasks
        self.tasks['universe_refresh'] = asyncio.create_task(self._universe_refresh_task())
        self.tasks['watchlist_update'] = asyncio.create_task(self._watchlist_update_task())
        self.tasks['focus_update'] = asyncio.create_task(self._focus_update_task())
        self.tasks['memory_monitor'] = asyncio.create_task(self._memory_monitor_task())
        
        logger.info("GPU-optimized stock selection system started")
    
    async def stop(self):
        """Stop the stock selection system"""
        if not self.running:
            return
            
        logger.info("Stopping GPU-optimized stock selection system")
        self.running = False
        
        # Cancel all tasks
        for name, task in self.tasks.items():
            if not task.done():
                logger.info(f"Cancelling task: {name}")
                task.cancel()
                
        # Shutdown thread pool
        self.executor.shutdown(wait=False)
        
        # Clean up GPU resources
        if self.gpu_available:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                logger.info("CuPy memory pool cleared")
            except Exception as e:
                logger.warning(f"Error clearing CuPy memory pool: {e}")
        
        logger.info("GPU-optimized stock selection system stopped")
    
    async def build_initial_universe(self):
        """Build initial universe of tradable stocks with GPU acceleration"""
        logger.info("Building initial universe with GPU acceleration")
        log_memory_usage("before_build_universe")
        
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
            await self._apply_tradable_filters_gpu()
            
            log_memory_usage("after_build_universe")
            
        except Exception as e:
            logger.error(f"Error building initial universe: {str(e)}", exc_info=True)
    
    async def _apply_tradable_filters_gpu(self):
        """Apply filters to create tradable universe with GPU acceleration"""
        logger.info("Applying filters for tradable universe with GPU acceleration")
        
        try:
            # Process tickers in batches to avoid API rate limits
            tradable_tickers = []
            batch_size = self.config['batch_size']
            batches = [list(self.full_universe)[i:i+batch_size] for i in range(0, len(self.full_universe), batch_size)]
            
            total_batches = len(batches)
            for i, batch in enumerate(batches):
                logger.info(f"Processing batch {i+1}/{total_batches} ({len(batch)} tickers)")
                
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
    
    async def _check_batch_eligibility_gpu(self, tickers):
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
                prev_day = await self._get_previous_day_data(ticker)
                if prev_day:
                    market_data[ticker] = prev_day
            
            if not market_data:
                return eligible_tickers
            
            # Prepare data for GPU processing
            ticker_list = list(market_data.keys())
            prices = np.array([market_data[t].get('close', 0) for t in ticker_list], dtype=np.float32)
            volumes = np.array([market_data[t].get('volume', 0) for t in ticker_list], dtype=np.float32)
            
            # Move data to GPU
            if self.gpu_available:
                cp_prices = cp.asarray(prices)
                cp_volumes = cp.asarray(volumes)
                
                # Apply filters on GPU
                price_mask = (cp_prices >= self.config['min_price']) & (cp_prices <= self.config['max_price'])
                volume_mask = cp_volumes >= self.config['min_volume']
                
                # Combined mask
                combined_mask = price_mask & volume_mask
                
                # Get indices of eligible tickers
                eligible_indices = cp.where(combined_mask)[0]
                
                # Move back to CPU
                eligible_indices_cpu = cp.asnumpy(eligible_indices)
                
                # Get eligible tickers
                for idx in eligible_indices_cpu:
                    ticker = ticker_list[idx]
                    eligible_tickers.append(ticker)
                    
                    # Store in cache for 1 day
                    self.redis.setex(f"eligibility:{ticker}", 86400, json.dumps(True))
            else:
                # CPU fallback
                for i, ticker in enumerate(ticker_list):
                    price = prices[i]
                    volume = volumes[i]
                    
                    if price >= self.config['min_price'] and price <= self.config['max_price'] and volume >= self.config['min_volume']:
                        eligible_tickers.append(ticker)
                        
                        # Store in cache for 1 day
                        self.redis.setex(f"eligibility:{ticker}", 86400, json.dumps(True))
            
            return eligible_tickers
            
        except Exception as e:
            logger.error(f"Error in batch eligibility check: {str(e)}")
            return eligible_tickers
    
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
        """Update the active watchlist with highest potential stocks using GPU acceleration"""
        logger.info("Updating active watchlist with GPU acceleration")
        log_memory_usage("before_update_watchlist")
        
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
            tradable_universe = {t.decode('utf-8') if isinstance(t, bytes) else t for t in tradable_universe}
            
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
                        asyncio.ensure_future(self._calculate_ticker_score(ticker))
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
            log_memory_usage("after_update_watchlist")
            
        except Exception as e:
            logger.error(f"Error updating watchlist: {str(e)}", exc_info=True)
    
    async def _calculate_batch_scores_gpu(self, tickers):
        """Calculate scores for a batch of tickers using GPU acceleration"""
        ranked_tickers = []
        
        try:
            # Get data for all tickers
            volume_data = {}
            price_data = {}
            volatility_data = {}
            options_data = {}
            
            # Fetch data in parallel
            volume_futures = [self._get_volume_data(ticker) for ticker in tickers]
            price_futures = [self._get_price_data(ticker) for ticker in tickers]
            volatility_futures = [self._get_atr_data(ticker) for ticker in tickers]
            options_futures = [self._get_options_flow(ticker) for ticker in tickers]
            
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
            market_momentum = await self._get_market_momentum()
            
            # Prepare arrays for GPU processing
            valid_tickers = [t for t in tickers if t in volume_data and t in price_data]
            
            if not valid_tickers:
                return ranked_tickers
            
            # Calculate volume scores
            volume_scores = np.zeros(len(valid_tickers), dtype=np.float32)
            for i, ticker in enumerate(valid_tickers):
                if ticker in volume_data:
                    current_volume = volume_data[ticker][0]
                    avg_volume = sum(volume_data[ticker][1:11]) / 10 if len(volume_data[ticker]) >= 11 else current_volume
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
                    atr_percent = (atr / current_price) * 100 if current_price > 0 else 0
                    
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
                    short_term = (price_data[ticker][0] / price_data[ticker][1] - 1) * 100 if price_data[ticker][1] > 0 else 0
                    
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
                    call_volume = sum(item.get('volume', 0) for item in options_data[ticker] if item.get('side') == 'call')
                    put_volume = sum(item.get('volume', 0) for item in options_data[ticker] if item.get('side') == 'put')
                    
                    total_volume = call_volume + put_volume
                    
                    # Calculate premium (dollar value)
                    call_premium = sum(item.get('premium', 0) for item in options_data[ticker] if item.get('side') == 'call')
                    put_premium = sum(item.get('premium', 0) for item in options_data[ticker] if item.get('side') == 'put')
                    
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
                
                # Move back to CPU
                total_scores = cp.asnumpy(cp_total_scores)
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
                    
                    self.redis.hset(f"ticker:{ticker}:factors", mapping=factors)
            
            return ranked_tickers
            
        except Exception as e:
            logger.error(f"Error in batch score calculation: {str(e)}")
            return ranked_tickers
    
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
    
    async def _memory_monitor_task(self):
        """Task to monitor memory usage"""
        logger.info("Starting memory monitor task")
        
        while self.running:
            try:
                # Log memory usage
                log_memory_usage("periodic_check")
                
                # Clean up GPU memory if usage is high
                if self.gpu_available:
                    try:
                        mem_info = cp.cuda.runtime.memGetInfo()
                        free, total = mem_info[0], mem_info[1]
                        used_percent = (total - free) / total * 100
                        
                        if used_percent > 80:
                            logger.warning(f"GPU memory usage high ({used_percent:.2f}%), cleaning up")
                            cp.get_default_memory_pool().free_all_blocks()
                    except Exception as e:
                        logger.error(f"Error cleaning up GPU memory: {e}")
                
                # Wait 5 minutes
                await asyncio.sleep(300)
                
            except asyncio.CancelledError:
                logger.info("Memory monitor task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in memory monitor task: {str(e)}")
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

# Example usage
if __name__ == "__main__":
    import redis
    import asyncio
    from polygon_data_source_ultra import PolygonDataSourceUltra
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def main():
        # Create clients
        redis_client = redis.Redis(host='redis', port=6379, db=0)
        polygon_client = PolygonDataSourceUltra()
        unusual_whales_client = None  # Replace with actual client
        
        # Create stock selection system
        system = GPUStockSelectionSystem(redis_client, polygon_client, unusual_whales_client)
        
        # Start system
        await system.start()
        
        try:
            # Run for a while
            await asyncio.sleep(3600)  # 1 hour
        finally:
            # Stop system
            await system.stop()
            
            # Close clients
            polygon_client.close()
    
    # Run the main function
    asyncio.run(main())