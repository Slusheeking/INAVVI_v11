#!/usr/bin/env python3
"""
Model Integration System

This module provides a production-ready ML model integration system that works with
the data pipeline to generate trading signals based on machine learning models.
"""

import logging
import time
import json
import threading
import queue
import numpy as np
import pandas as pd
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
import xgboost as xgb
import os
import pickle
from datetime import datetime

# Environment variables
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
REDIS_DB = int(os.environ.get('REDIS_DB', 0))
MODELS_DIR = os.environ.get('MODELS_DIR', '/models')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_integration')

class ModelIntegrationSystem:
    """Production-ready ML model integration system"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        
        # Thread pool for model inference
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Model cache
        self.models = {}
        self.model_version = {}
        
        # Processing queues
        self.data_queue = queue.Queue(maxsize=1000)
        self.signal_queue = queue.Queue(maxsize=1000)

        # Configuration
        self.config = {
            'signal_threshold': 0.7,        # Minimum signal score threshold
            'signal_expiry': 300,           # Signal expiry time (seconds)
            'batch_size': 32,               # Batch size for model inference
            'update_interval': 86400,       # Model update interval (seconds)
            'cache_ttl': 60,                # Feature cache TTL (seconds)
            'max_positions': 5,             # Maximum concurrent positions
            'risk_per_trade': 0.005,        # Maximum risk per trade (0.5%)
            'model_paths': {
                'signal_detection': f'{MODELS_DIR}/signal_detection_model.xgb',
                'price_prediction': f'{MODELS_DIR}/price_prediction_model.h5',
                'risk_assessment': f'{MODELS_DIR}/risk_assessment_model.pkl',
                'exit_strategy': f'{MODELS_DIR}/exit_strategy_model.xgb',
                'market_regime': f'{MODELS_DIR}/market_regime_model.pkl'
            }
        }
        
        # Internal state
        self.running = False
        self.threads = []
        self.current_market_regime = "normal"
        
        logger.info("Model Integration System initialized")
    
    def start(self):
        """Start the model integration system"""
        if self.running:
            logger.warning("Model integration system already running")
            return
            
        self.running = True
        logger.info("Starting model integration system")
        
        # Load models
        self._load_models()
        
        # Start worker threads
        self.threads.append(threading.Thread(target=self._data_listener_worker, daemon=True))
        self.threads.append(threading.Thread(target=self._feature_engineering_worker, daemon=True))
        self.threads.append(threading.Thread(target=self._model_inference_worker, daemon=True))
        self.threads.append(threading.Thread(target=self._signal_publisher_worker, daemon=True))
        self.threads.append(threading.Thread(target=self._model_update_worker, daemon=True))
        
        for thread in self.threads:
            thread.start()
            
        logger.info("Model integration system started")
    
    def stop(self):
        """Stop the model integration system"""
        if not self.running:
            logger.warning("Model integration system not running")
            return
            
        logger.info("Stopping model integration system")
        self.running = False
        
        # Wait for threads to terminate
        for thread in self.threads:
            thread.join(timeout=5.0)
            
        # Release model resources
        self._unload_models()
        
        # Shutdown thread pool
        self.executor.shutdown(wait=False)
        
        logger.info("Model integration system stopped")
    
    def _load_models(self):
        """Load ML models"""
        logger.info("Loading ML models")
        
        try:
            # Load signal detection model (XGBoost)
            model_path = self.config['model_paths']['signal_detection']
            self.models['signal_detection'] = xgb.Booster()
            self.models['signal_detection'].load_model(model_path)
            self.model_version['signal_detection'] = self._get_model_version(model_path)
            logger.info(f"Loaded signal detection model: {model_path}")
            
            # Load price prediction model (LSTM)
            model_path = self.config['model_paths']['price_prediction']
            self.models['price_prediction'] = tf.keras.models.load_model(model_path)
            self.model_version['price_prediction'] = self._get_model_version(model_path)
            logger.info(f"Loaded price prediction model: {model_path}")
            
            # Load risk assessment model (Random Forest)
            model_path = self.config['model_paths']['risk_assessment']
            self.models['risk_assessment'] = self._load_pickle_model(model_path)
            self.model_version['risk_assessment'] = self._get_model_version(model_path)
            logger.info(f"Loaded risk assessment model: {model_path}")
            
            # Load exit strategy model (Gradient Boosting)
            model_path = self.config['model_paths']['exit_strategy']
            self.models['exit_strategy'] = xgb.Booster()
            self.models['exit_strategy'].load_model(model_path)
            self.model_version['exit_strategy'] = self._get_model_version(model_path)
            logger.info(f"Loaded exit strategy model: {model_path}")
            
            # Load market regime model (K-Means)
            model_path = self.config['model_paths']['market_regime']
            self.models['market_regime'] = self._load_pickle_model(model_path)
            self.model_version['market_regime'] = self._get_model_version(model_path)
            logger.info(f"Loaded market regime model: {model_path}")
            
            # Update Redis with model versions
            pipeline = self.redis.pipeline()
            for model_name, version in self.model_version.items():
                pipeline.hset("models:versions", model_name, version)
            pipeline.execute()
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}", exc_info=True)
    
    def _load_pickle_model(self, path):
        """Load a pickled model"""
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def _get_model_version(self, path):
        """Get model version from file timestamp"""
        import os
        try:
            timestamp = os.path.getmtime(path)
            return int(timestamp)
        except Exception:
            return int(time.time())
    
    def _unload_models(self):
        """Unload models and free resources"""
        logger.info("Unloading ML models")
        
        # Clear TensorFlow session
        try:
            tf.keras.backend.clear_session()
        except:
            pass
            
        # Clear model dictionaries
        self.models.clear()
        self.model_version.clear()
    
    def _data_listener_worker(self):
        """Worker thread to listen for new market data"""
        logger.info("Starting data listener worker")
        
        pubsub = self.redis.pubsub()
        pubsub.psubscribe("price_update:*")
        pubsub.subscribe("options:flow:new")
        pubsub.subscribe("darkpool:trade:new")
        
        while self.running:
            try:
                # Get new message with timeout
                message = pubsub.get_message(timeout=1.0)
                
                if message and message['type'] in ('message', 'pmessage'):
                    # Extract data
                    channel = message['channel']
                    data = message['data']
                    
                    if isinstance(channel, bytes):
                        channel = channel.decode('utf-8')
                        
                    if isinstance(data, bytes):
                        data = data.decode('utf-8')
                    
                    # Parse data
                    try:
                        parsed_data = json.loads(data)
                        
                        # Add message metadata
                        parsed_data['channel'] = channel
                        parsed_data['received_at'] = time.time()
                        
                        # Put in processing queue
                        self.data_queue.put(parsed_data, block=False)
                        
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON data from channel {channel}")
                        
            except queue.Full:
                logger.warning("Data queue is full, dropping message")
            except Exception as e:
                logger.error(f"Error in data listener: {str(e)}")
                time.sleep(1.0)
    
    def _feature_engineering_worker(self):
        """Worker thread for feature engineering"""
        logger.info("Starting feature engineering worker")
        
        while self.running:
            try:
                # Get message from queue with timeout
                data = self.data_queue.get(timeout=1.0)
                
                # Process data based on channel
                channel = data.get('channel', '')
                
                if 'price_update:' in channel:
                    # Price update message
                    ticker = channel.split(':')[1]
                    self._process_price_update(ticker, data)
                    
                elif channel == 'options:flow:new':
                    # Options flow message
                    self._process_options_flow(data)
                    
                elif channel == 'darkpool:trade:new':
                    # Dark pool trade message
                    self._process_darkpool_trade(data)
                    
                # Mark task as done
                self.data_queue.task_done()
                
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Error in feature engineering: {str(e)}")
                time.sleep(1.0)
    
    def _process_price_update(self, ticker, data):
        """Process price update data"""
        try:
            # Extract basic price data
            price = data.get('price')
            volume = data.get('size')
            timestamp = data.get('timestamp')
            
            # Skip incomplete data
            if not all([ticker, price, timestamp]):
                return
                
            # Get recent price history
            recent_prices = self._get_recent_prices(ticker)
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(ticker, price, recent_prices)
            
            # Create feature vector
            features = {
                'ticker': ticker,
                'price': price,
                'volume': volume,
                'timestamp': timestamp,
                'indicators': indicators,
                'updated_at': time.time()
            }
            
            # Store in Redis
            self.redis.setex(
                f"features:{ticker}:price",
                self.config['cache_ttl'],
                json.dumps(features)
            )
            
            # Check for potential signals
            self._check_for_signals(ticker, features)
            
        except Exception as e:
            logger.error(f"Error processing price update for {ticker}: {str(e)}")
    
    def _process_options_flow(self, data):
        """Process options flow data"""
        try:
            # Extract ticker
            ticker = data.get('ticker')
            if not ticker:
                return
                
            # Store aggregated options data
            self._update_options_features(ticker, data)
            
            # Check for significant options activity
            self._check_options_signals(ticker, data)
            
        except Exception as e:
            logger.error(f"Error processing options flow: {str(e)}")
    
    def _process_darkpool_trade(self, data):
        """Process dark pool trade data"""
        try:
            # Extract ticker
            ticker = data.get('ticker')
            if not ticker:
                return
                
            # Store dark pool data
            self._update_darkpool_features(ticker, data)
            
            # Check for significant dark pool activity
            self._check_darkpool_signals(ticker, data)
            
        except Exception as e:
            logger.error(f"Error processing dark pool trade: {str(e)}")
    
    def _check_for_signals(self, ticker, features):
        """Check for potential trading signals"""
        try:
            # Simple signal detection
            if not self._is_signal_candidate(ticker, features):
                return
                
            # Create complete feature vector for model inference
            feature_vector = self._create_feature_vector(ticker)
            
            # Queue for model inference
            self.signal_queue.put({
                'ticker': ticker,
                'features': feature_vector,
                'signal_type': 'price',
                'timestamp': time.time()
            })
            
        except Exception as e:
            logger.error(f"Error checking signals for {ticker}: {str(e)}")
    
    def _model_inference_worker(self):
        """Worker thread for model inference"""
        logger.info("Starting model inference worker")
        
        while self.running:
            try:
                # Collect a batch of signals for processing
                signals = []
                signal = self.signal_queue.get(timeout=1.0)
                signals.append(signal)
                
                # Try to get more signals up to batch size
                batch_size = self.config['batch_size']
                while len(signals) < batch_size:
                    try:
                        signal = self.signal_queue.get(block=False)
                        signals.append(signal)
                    except queue.Empty:
                        break
                
                # Skip if no signals
                if not signals:
                    continue
                
                # Process signals by type
                self._process_signal_batch(signals)
                
                # Mark tasks as done
                for _ in range(len(signals)):
                    self.signal_queue.task_done()
                
            except queue.Empty:
                # Update market regime periodically
                self._update_market_regime()
            except Exception as e:
                logger.error(f"Error in model inference: {str(e)}")
                time.sleep(1.0)
    
    def _process_signal_batch(self, signals):
        """Process a batch of signals"""
        # Group signals by type
        grouped_signals = {}
        for signal in signals:
            signal_type = signal.get('signal_type', 'unknown')
            if signal_type not in grouped_signals:
                grouped_signals[signal_type] = []
                
            grouped_signals[signal_type].append(signal)
        
        # Process each group
        for signal_type, group in grouped_signals.items():
            if signal_type == 'price':
                self._process_price_signals(group)
            elif signal_type == 'options':
                self._process_options_signals(group)
            elif signal_type == 'darkpool':
                self._process_darkpool_signals(group)
    
    def _process_price_signals(self, signals):
        """Process price-based signals"""
        try:
            # Extract tickers and features
            tickers = [s['ticker'] for s in signals]
            features = [s['features'] for s in signals]
            
            # Prepare features for model inference
            X = np.array(features)
            
            # Get predictions from signal detection model
            signal_scores = self._predict_with_signal_model(X)
            
            # For signals above threshold, get price predictions
            strong_signals = []
            for i, score in enumerate(signal_scores):
                if score >= self.config['signal_threshold']:
                    strong_signals.append({
                        'ticker': tickers[i],
                        'signal_score': float(score),
                        'features': features[i],
                        'timestamp': signals[i]['timestamp']
                    })
            
            # Skip if no strong signals
            if not strong_signals:
                return
                
            # Get price predictions for strong signals
            self._process_strong_signals(strong_signals)
            
        except Exception as e:
            logger.error(f"Error processing price signals: {str(e)}")
    
    def _predict_with_signal_model(self, features):
        """Get predictions from signal detection model"""
        try:
            # Convert to DMatrix for XGBoost
            dmatrix = xgb.DMatrix(features)
            
            # Get predictions
            predictions = self.models['signal_detection'].predict(dmatrix)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting with signal model: {str(e)}")
            return np.zeros(len(features))
    
    def _process_strong_signals(self, signals):
        """Process strong signals with additional models"""
        try:
            # Extract data
            tickers = [s['ticker'] for s in signals]
            features = [s['features'] for s in signals]
            
            # Prepare features for price prediction
            X = np.array(features)
            
            # Get price predictions
            price_predictions = self._predict_with_price_model(X)
            
            # Calculate risk parameters
            risk_params = self._calculate_risk_parameters(signals)
            
            # Generate trading signals
            for i, ticker in enumerate(tickers):
                # Skip if no price prediction
                if i >= len(price_predictions):
                    continue
                    
                # Get predictions
                price_pred = price_predictions[i]
                risk_param = risk_params[i] if i < len(risk_params) else None
                
                # Create trading signal
                signal = {
                    'ticker': ticker,
                    'signal_score': signals[i]['signal_score'],
                    'direction': 'long' if price_pred['direction'] > 0 else 'short',
                    'price_target': price_pred['target'],
                    'confidence': price_pred['confidence'],
                    'stop_loss': risk_param['stop_loss'] if risk_param else None,
                    'position_size': risk_param['position_size'] if risk_param else None,
                    'timestamp': time.time(),
                    'expiry': time.time() + self.config['signal_expiry']
                }
                
                # Publish signal
                self._publish_trading_signal(signal)
                
        except Exception as e:
            logger.error(f"Error processing strong signals: {str(e)}")
    
    def _predict_with_price_model(self, features):
        """Get predictions from price prediction model"""
        try:
            # Get raw predictions
            raw_predictions = self.models['price_prediction'].predict(features)
            
            # Process predictions
            results = []
            for pred in raw_predictions:
                # Extract direction and magnitude
                direction = 1 if pred[0] > 0.5 else -1
                magnitude = abs(pred[1])
                confidence = pred[2]
                
                results.append({
                    'direction': direction,
                    'magnitude': magnitude,
                    'target': magnitude * direction,
                    'confidence': confidence
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Error predicting with price model: {str(e)}")
            return [{'direction': 0, 'magnitude': 0, 'target': 0, 'confidence': 0}] * len(features)
    
    def _calculate_risk_parameters(self, signals):
        """Calculate risk parameters for signals"""
        try:
            # Extract features
            features = [s['features'] for s in signals]
            
            # Prepare features for risk model
            X = np.array(features)
            
            # Get risk predictions
            risk_predictions = self.models['risk_assessment'].predict(X)
            
            # Process predictions
            results = []
            for i, pred in enumerate(risk_predictions):
                ticker = signals[i]['ticker']
                
                # Get current price
                current_price = self._get_current_price(ticker)
                if not current_price:
                    continue
                
                # Calculate stop loss (as percentage of price)
                stop_loss_pct = max(0.005, min(0.05, pred))  # 0.5% to 5%
                stop_loss_price = current_price * (1 - stop_loss_pct)
                
                # Calculate position size based on risk
                account_size = self._get_account_size()
                risk_amount = account_size * self.config['risk_per_trade']
                price_risk = current_price - stop_loss_price
                position_size = risk_amount / price_risk if price_risk > 0 else 0
                
                results.append({
                    'stop_loss': stop_loss_price,
                    'stop_loss_pct': stop_loss_pct,
                    'position_size': position_size,
                    'risk_score': float(pred)
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Error calculating risk parameters: {str(e)}")
            return []
    
    def _publish_trading_signal(self, signal):
        """Publish trading signal to Redis"""
        try:
            # Add metadata
            signal['source'] = 'ml_model'
            signal['market_regime'] = self.current_market_regime
            
            # Store in Redis
            signal_key = f"signals:{signal['ticker']}:{int(signal['timestamp'] * 1000)}"
            self.redis.setex(
                signal_key,
                self.config['signal_expiry'],
                json.dumps(signal)
            )
            
            # Publish to channel
            self.redis.publish(
                'signals:trading:new',
                json.dumps(signal)
            )
            
            # Add to sorted set for priority queue
            score = signal['signal_score'] * signal['confidence']
            self.redis.zadd('signals:pending', {signal_key: score})
            
            logger.info(f"Published trading signal for {signal['ticker']}: score={score:.2f}")
            
        except Exception as e:
            logger.error(f"Error publishing trading signal: {str(e)}")
    
    def _signal_publisher_worker(self):
        """Worker thread to publish signals to execution system"""
        logger.info("Starting signal publisher worker")
        
        while self.running:
            try:
                # Get pending signals ordered by priority
                signals = self.redis.zrevrange(
                    'signals:pending',
                    0, 9,  # Top 10 signals
                    withscores=True
                )
                
                if not signals:
                    time.sleep(1.0)
                    continue
                
                # Get active positions
                active_positions = self._get_active_positions()
                
                # Check if we can open new positions
                if len(active_positions) >= self.config['max_positions']:
                    time.sleep(1.0)
                    continue
                
                # Process signals
                for signal_key, score in signals:
                    # Skip if already in this position
                    ticker = signal_key.decode('utf-8').split(':')[1]
                    if ticker in active_positions:
                        continue
                        
                    # Get signal data
                    signal_data = self.redis.get(signal_key)
                    if not signal_data:
                        # Remove from pending queue if expired
                        self.redis.zrem('signals:pending', signal_key)
                        continue
                        
                    # Parse signal
                    signal = json.loads(signal_data)
                    
                    # Check if expired
                    if time.time() > signal.get('expiry', 0):
                        # Remove expired signal
                        self.redis.zrem('signals:pending', signal_key)
                        self.redis.delete(signal_key)
                        continue
                        
                    # Send to execution system
                    self._send_to_execution(signal)
                    
                    # Remove from pending queue
                    self.redis.zrem('signals:pending', signal_key)
                    
                    # Check if we've reached max positions
                    if len(active_positions) + 1 >= self.config['max_positions']:
                        break
                        
                # Sleep before next check
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in signal publisher: {str(e)}")
                time.sleep(1.0)
    
    def _update_market_regime(self):
        """Update current market regime classification"""
        try:
            # Get market indices data
            market_data = self._get_market_indices_data()
            if not market_data:
                return
                
            # Prepare features for market regime model
            X = np.array([market_data])
            
            # Get regime prediction
            regime = self.models['market_regime'].predict(X)[0]
            
            # Map cluster to regime
            regime_map = {
                0: "trending_up",
                1: "trending_down",
                2: "choppy",
                3: "volatile"
            }
            
            regime_name = regime_map.get(regime, "normal")
            
            # Update current regime if changed
            if regime_name != self.current_market_regime:
                logger.info(f"Market regime changed: {self.current_market_regime} -> {regime_name}")
                self.current_market_regime = regime_name
                
                # Store in Redis
                self.redis.set("market:current_regime", regime_name)
                
        except Exception as e:
            logger.error(f"Error updating market regime: {str(e)}")
    
    def _model_update_worker(self):
        """Worker thread to check for model updates"""
        logger.info("Starting model update worker")
        
        while self.running:
            try:
                # Check for updated models
                updated = self._check_model_updates()
                
                if updated:
                    logger.info("Models updated, reloading...")
                    self._load_models()
                
                # Sleep for a while before next check
                time.sleep(self.config['update_interval'])
                
            except Exception as e:
                logger.error(f"Error in model update worker: {str(e)}")
                time.sleep(60.0)
    
    # Helper methods
    
    def _get_recent_prices(self, ticker):
        """Get recent price history for a ticker"""
        try:
            # Get from Redis
            key = f"stock:{ticker}:candles:1m"
            candles = self.redis.hgetall(key)
            
            if not candles:
                return []
                
            # Parse candles
            parsed_candles = []
            for _, candle_json in candles.items():
                candle = json.loads(candle_json)
                parsed_candles.append(candle)
                
            # Sort by timestamp
            parsed_candles.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Return close prices
            return [c['close'] for c in parsed_candles[:30]]
            
        except Exception as e:
            logger.error(f"Error getting recent prices for {ticker}: {str(e)}")
            return []
    
    def _calculate_indicators(self, ticker, price, recent_prices):
        """Calculate technical indicators"""
        try:
            # Add current price to history
            prices = [price] + recent_prices
            
            # Calculate indicators
            indicators = {}
            
            # SMA-5
            if len(prices) >= 5:
                indicators['sma5'] = sum(prices[:5]) / 5
            
            # SMA-10
            if len(prices) >= 10:
                indicators['sma10'] = sum(prices[:10]) / 10
            
            # SMA-20
            if len(prices) >= 20:
                indicators['sma20'] = sum(prices[:20]) / 20
            
            # Price relative to SMAs
            if 'sma5' in indicators:
                indicators['price_rel_sma5'] = price / indicators['sma5'] - 1
                
            if 'sma10' in indicators:
                indicators['price_rel_sma10'] = price / indicators['sma10'] - 1
                
            if 'sma20' in indicators:
                indicators['price_rel_sma20'] = price / indicators['sma20'] - 1
            
            # Price momentum (% change)
            if len(prices) >= 2:
                indicators['mom1'] = price / prices[1] - 1
                
            if len(prices) >= 5:
                indicators['mom5'] = price / prices[4] - 1
                
            if len(prices) >= 10:
                indicators['mom10'] = price / prices[9] - 1
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {ticker}: {str(e)}")
            return {}
    
    def _update_options_features(self, ticker, data):
        """Update options features for a ticker"""
        try:
            # Get existing options data
            key = f"options:features:{ticker}"
            existing_data = self.redis.get(key)
            
            if existing_data:
                features = json.loads(existing_data)
            else:
                features = {
                    'ticker': ticker,
                    'call_volume': 0,
                    'put_volume': 0,
                    'call_oi': 0,
                    'put_oi': 0,
                    'total_premium': 0,
                    'unusual_activity': []
                }
            
            # Update with new data
            side = data.get('side', '').lower()
            volume = data.get('volume', 0)
            open_interest = data.get('open_interest', 0)
            premium = data.get('premium', 0)
            
            if side == 'call':
                features['call_volume'] += volume
                features['call_oi'] += open_interest
            elif side == 'put':
                features['put_volume'] += volume
                features['put_oi'] += open_interest
                
            features['total_premium'] += premium
            
            # Add to unusual activity if significant
            if premium > 100000:  # $100K premium
                features['unusual_activity'].append({
                    'side': side,
                    'volume': volume,
                    'premium': premium,
                    'timestamp': time.time()
                })
                
            # Keep only recent activity
            now = time.time()
            features['unusual_activity'] = [
                a for a in features['unusual_activity']
                if now - a['timestamp'] < 3600  # Last hour
            ]
            
            # Store updated features
            self.redis.setex(
                key,
                3600,  # 1 hour TTL
                json.dumps(features)
            )
            
        except Exception as e:
            logger.error(f"Error updating options features for {ticker}: {str(e)}")
    
    def _update_darkpool_features(self, ticker, data):
        """Update dark pool features for a ticker"""
        try:
            # Get existing dark pool data
            key = f"darkpool:features:{ticker}"
            existing_data = self.redis.get(key)
            
            if existing_data:
                features = json.loads(existing_data)
            else:
                features = {
                    'ticker': ticker,
                    'total_volume': 0,
                    'total_value': 0,
                    'trade_count': 0,
                    'large_trades': []
                }
            
            # Update with new data
            volume = data.get('size', 0)
            price = data.get('price', 0)
            value = volume * price
            
            features['total_volume'] += volume
            features['total_value'] += value
            features['trade_count'] += 1
            
            # Add to large trades if significant
            if value > 1000000:  # $1M value
                features['large_trades'].append({
                    'volume': volume,
                    'price': price,
                    'value': value,
                    'timestamp': time.time()
                })
                
            # Keep only recent large trades
            now = time.time()
            features['large_trades'] = [
                t for t in features['large_trades']
                if now - t['timestamp'] < 3600  # Last hour
            ]
            
            # Store updated features
            self.redis.setex(
                key,
                3600,  # 1 hour TTL
                json.dumps(features)
            )
            
        except Exception as e:
            logger.error(f"Error updating dark pool features for {ticker}: {str(e)}")
    
    def _is_signal_candidate(self, ticker, features):
        """Check if ticker is a candidate for signal generation"""
        try:
            # Check if in focused watchlist
            in_focus = self.redis.zscore("watchlist:focused", ticker) is not None
            if not in_focus:
                return False
                
            # Check indicators
            indicators = features.get('indicators', {})
            
            # Check if price crossed SMA
            crossed_sma = False
            if 'price_rel_sma5' in indicators and 'price_rel_sma10' in indicators:
                # Crossed from below to above
                if indicators['price_rel_sma5'] > 0 and indicators['price_rel_sma10'] < 0:
                    crossed_sma = True
                    
            # Check momentum
            strong_momentum = False
            if 'mom1' in indicators:
                if abs(indicators['mom1']) > 0.01:  # 1% move
                    strong_momentum = True
            
            # Check volume (would need volume data in features)
            high_volume = True  # Simplified for this example
            
            # Return True if any condition met
            return crossed_sma or strong_momentum or high_volume
            
        except Exception as e:
            logger.error(f"Error checking signal candidate for {ticker}: {str(e)}")
            return False
    
    def _create_feature_vector(self, ticker):
        """Create complete feature vector for model inference"""
        try:
            # Get price features
            price_features = self._get_cached_features(f"features:{ticker}:price")
            
            # Get options features
            options_features = self._get_cached_features(f"options:features:{ticker}")
            
            # Get dark pool features
            darkpool_features = self._get_cached_features(f"darkpool:features:{ticker}")
            
            # Get market features
            market_features = self._get_market_features()
            
            # Create numerical feature array
            features = []
            
            # Add price features
            if price_features:
                indicators = price_features.get('indicators', {})
                features.extend([
                    indicators.get('sma5', 0),
                    indicators.get('sma10', 0),
                    indicators.get('sma20', 0),
                    indicators.get('price_rel_sma5', 0),
                    indicators.get('price_rel_sma10', 0),
                    indicators.get('price_rel_sma20', 0),
                    indicators.get('mom1', 0),
                    indicators.get('mom5', 0),
                    indicators.get('mom10', 0)
                ])
            else:
                features.extend([0] * 9)  # Padding for missing features
            
            # Add options features
            if options_features:
                put_call_ratio = options_features.get('put_volume', 0) / max(1, options_features.get('call_volume', 1))
                features.extend([
                    options_features.get('call_volume', 0),
                    options_features.get('put_volume', 0),
                    put_call_ratio,
                    options_features.get('total_premium', 0),
                    len(options_features.get('unusual_activity', []))
                ])
            else:
                features.extend([0] * 5)  # Padding for missing features
            
            # Add dark pool features
            if darkpool_features:
                features.extend([
                    darkpool_features.get('total_volume', 0),
                    darkpool_features.get('trade_count', 0),
                    len(darkpool_features.get('large_trades', []))
                ])
            else:
                features.extend([0] * 3)  # Padding for missing features
            
            # Add market features
            if market_features:
                features.extend([
                    market_features.get('spy_change', 0),
                    market_features.get('vix', 0),
                    market_features.get('sector_rel_strength', 0)
                ])
            else:
                features.extend([0] * 3)  # Padding for missing features
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating feature vector for {ticker}: {str(e)}")
            return [0] * 20  # Return zero vector as fallback
    
    def _get_cached_features(self, key):
        """Get cached features from Redis"""
        try:
            data = self.redis.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error getting cached features for {key}: {str(e)}")
            return None
    
    def _get_market_features(self):
        """Get current market features"""
        try:
            # Get market data from Redis
            spy_data = self.redis.get("market:SPY:data")
            vix_data = self.redis.get("market:VIX:data")
            
            if not spy_data or not vix_data:
                return None
                
            spy = json.loads(spy_data)
            vix = json.loads(vix_data)
            
            # Create market features
            features = {
                'spy_change': spy.get('daily_change', 0),
                'vix': vix.get('last', 0),
                'sector_rel_strength': 0  # Would need sector data
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting market features: {str(e)}")
            return None
    
    def _check_options_signals(self, ticker, data):
        """Check for significant options activity"""
        try:
            # Check premium amount
            premium = data.get('premium', 0)
            if premium < 100000:  # Less than $100K
                return
                
            # Create feature vector
            feature_vector = self._create_feature_vector(ticker)
            
            # Queue for model inference
            self.signal_queue.put({
                'ticker': ticker,
                'features': feature_vector,
                'signal_type': 'options',
                'options_data': data,
                'timestamp': time.time()
            })
            
        except Exception as e:
            logger.error(f"Error checking options signals for {ticker}: {str(e)}")
    
    def _check_darkpool_signals(self, ticker, data):
        """Check for significant dark pool activity"""
        try:
            # Check trade value
            size = data.get('size', 0)
            price = data.get('price', 0)
            value = size * price
            
            if value < 1000000:  # Less than $1M
                return
                
            # Create feature vector
            feature_vector = self._create_feature_vector(ticker)
            
            # Queue for model inference
            self.signal_queue.put({
                'ticker': ticker,
                'features': feature_vector,
                'signal_type': 'darkpool',
                'darkpool_data': data,
                'timestamp': time.time()
            })
            
        except Exception as e:
            logger.error(f"Error checking dark pool signals for {ticker}: {str(e)}")
    
    def _process_options_signals(self, signals):
        """Process options-based signals"""
        # Implementation similar to _process_price_signals
        pass
    
    def _process_darkpool_signals(self, signals):
        """Process dark pool-based signals"""
        # Implementation similar to _process_price_signals
        pass
    
    def _get_current_price(self, ticker):
        """Get current price for a ticker"""
        try:
            price_data = self.redis.hgetall(f"stock:{ticker}:last_trade")
            if price_data and b'price' in price_data:
                return float(price_data[b'price'])
            return None
        except Exception as e:
            logger.error(f"Error getting current price for {ticker}: {str(e)}")
            return None
    
    def _get_account_size(self):
        """Get current account size"""
        try:
            account_data = self.redis.hgetall("account:status")
            if account_data and b'equity' in account_data:
                return float(account_data[b'equity'])
            return 5000.0  # Default to $5000
        except Exception as e:
            logger.error(f"Error getting account size: {str(e)}")
            return 5000.0
    
    def _get_active_positions(self):
        """Get currently active positions"""
        try:
            positions = self.redis.hgetall("positions:active")
            return set(k.decode('utf-8').split(':')[0] for k in positions.keys())
        except Exception as e:
            logger.error(f"Error getting active positions: {str(e)}")
            return set()
    
    def _send_to_execution(self, signal):
        """Send signal to execution system"""
        try:
            # Publish to execution channel
            self.redis.publish(
                'execution:new_signal',
                json.dumps(signal)
            )
            
            logger.info(f"Sent signal to execution for {signal['ticker']}")
            
        except Exception as e:
            logger.error(f"Error sending to execution: {str(e)}")
    
    def _get_market_indices_data(self):
        """Get market indices data for regime classification"""
        try:
            # Get data for key indices
            spy_data = self.redis.get("market:SPY:data")
            qqq_data = self.redis.get("market:QQQ:data")
            iwm_data = self.redis.get("market:IWM:data")
            vix_data = self.redis.get("market:VIX:data")
            
            if not all([spy_data, qqq_data, iwm_data, vix_data]):
                return None
                
            # Parse data
            spy = json.loads(spy_data)
            qqq = json.loads(qqq_data)
            iwm = json.loads(iwm_data)
            vix = json.loads(vix_data)
            
            # Create feature vector for regime model
            features = [
                spy.get('daily_change', 0),
                qqq.get('daily_change', 0),
                iwm.get('daily_change', 0),
                vix.get('last', 0),
                vix.get('daily_change', 0),
                spy.get('atr_ratio', 0),
                qqq.get('atr_ratio', 0),
                iwm.get('atr_ratio', 0)
            ]
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting market indices data: {str(e)}")
            return None
    
    def _check_model_updates(self):
        """Check if models have been updated"""
        try:
            updated = False
            
            for model_name, model_path in self.config['model_paths'].items():
                # Get current version
                current_version = self.model_version.get(model_name, 0)
                
                # Get new version
                new_version = self._get_model_version(model_path)
                
                if new_version > current_version:
                    logger.info(f"Model update detected for {model_name}: {current_version} -> {new_version}")
                    updated = True
                    
            return updated
            
        except Exception as e:
            logger.error(f"Error checking model updates: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    import redis
    
    # Create Redis client
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB
    )
    
    # Create model integration system
    model_system = ModelIntegrationSystem(redis_client)
    
    # Start system
    model_system.start()
    
    try:
        # Run for a while
        print("Model integration system running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Stop system
        model_system.stop()