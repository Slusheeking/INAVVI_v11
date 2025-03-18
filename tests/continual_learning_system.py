#!/usr/bin/env python3
"""
Continual Learning System

This module provides a continual learning system that automatically updates
trading models with new market data. Features include:
1. Daily incremental model updates
2. Scheduled full model retraining
3. Model versioning and rollback capabilities
4. Performance monitoring and validation
5. Automatic model deployment

The system ensures models stay relevant as market conditions evolve.
"""

import os
import time
import json
import logging
import datetime
import threading
import schedule
import numpy as np
import pandas as pd
import joblib
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('continual_learning')

class ContinualLearningSystem:
    """
    Continual Learning System for trading models
    Updates models daily with new market data
    """
    
    def __init__(self, redis_client, data_loader, model_trainer):
        self.redis = redis_client
        self.data_loader = data_loader
        self.model_trainer = model_trainer
        
        # Configuration
        self.config = {
            'models_dir': os.environ.get('MODELS_DIR', '/models'),
            'data_dir': os.environ.get('DATA_DIR', '/data'),
            'training_schedule': {
                'daily_update': '23:30',  # After market close
                'full_retrain': '00:30'   # Overnight full retraining
            },
            'update_window': 5,          # Days of data for incremental updates
            'performance_threshold': 0.8, # Performance threshold for model updating
            'max_versions': 5             # Maximum model versions to keep
        }
        
        # Current models
        self.models = {}
        self.model_versions = {}
        
        # State
        self.running = False
        self.threads = []
        
        logger.info("Continual Learning System initialized")
    
    def start(self):
        """Start the continual learning system"""
        if self.running:
            logger.warning("Continual learning system already running")
            return
            
        self.running = True
        logger.info("Starting continual learning system")
        
        # Load existing models
        self._load_models()
        
        # Schedule tasks
        schedule.every().day.at(self.config['training_schedule']['daily_update']).do(self.daily_model_update)
        schedule.every().day.at(self.config['training_schedule']['full_retrain']).do(self.full_model_retrain)
        
        # Start scheduler thread
        thread = threading.Thread(target=self._scheduler_thread, daemon=True)
        thread.start()
        self.threads.append(thread)
        
        logger.info("Continual learning system started")
    
    def stop(self):
        """Stop the continual learning system"""
        if not self.running:
            logger.warning("Continual learning system not running")
            return
            
        logger.info("Stopping continual learning system")
        self.running = False
        
        # Wait for threads to terminate
        for thread in self.threads:
            thread.join(timeout=5.0)
            
        logger.info("Continual learning system stopped")
    
    def _scheduler_thread(self):
        """Thread to run scheduled tasks"""
        logger.info("Starting scheduler thread")
        
        while self.running:
            try:
                # Run pending scheduled tasks
                schedule.run_pending()
                
                # Sleep for a bit
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in scheduler thread: {str(e)}", exc_info=True)
                time.sleep(60)
    
    def _load_models(self):
        """Load existing models"""
        try:
            logger.info("Loading existing models")
            
            # Model types and their loaders
            model_types = {
                'signal_detection': self._load_xgboost_model,
                'price_prediction': self._load_keras_model,
                'risk_assessment': self._load_sklearn_model,
                'exit_strategy': self._load_xgboost_model,
                'market_regime': self._load_sklearn_model
            }
            
            # Load each model if it exists
            for model_name, loader_func in model_types.items():
                try:
                    model = loader_func(model_name)
                    
                    if model:
                        self.models[model_name] = model
                        logger.info(f"Loaded {model_name} model")
                    else:
                        logger.warning(f"Could not load {model_name} model, will be trained from scratch")
                        
                except Exception as e:
                    logger.error(f"Error loading {model_name} model: {str(e)}")
            
            # Load scalers
            for model_name in model_types:
                scaler_path = os.path.join(self.config['models_dir'], f"{model_name}_scaler.pkl")
                if os.path.exists(scaler_path):
                    try:
                        scaler = joblib.load(scaler_path)
                        self.model_trainer.scalers[model_name] = scaler
                        logger.info(f"Loaded scaler for {model_name} model")
                    except Exception as e:
                        logger.error(f"Error loading scaler for {model_name} model: {str(e)}")
            
            # Get model versions
            self._load_model_versions()
            
            logger.info(f"Loaded {len(self.models)} existing models")
            
        except Exception as e:
            logger.error(f"Error loading existing models: {str(e)}", exc_info=True)
    
    def _load_xgboost_model(self, model_name):
        """Load an XGBoost model"""
        model_path = os.path.join(self.config['models_dir'], f"{model_name}_model.xgb")
        
        if not os.path.exists(model_path):
            return None
            
        try:
            model = xgb.Booster()
            model.load_model(model_path)
            return model
        except Exception as e:
            logger.error(f"Error loading XGBoost model {model_name}: {str(e)}")
            return None
    
    def _load_keras_model(self, model_name):
        """Load a Keras model"""
        model_path = os.path.join(self.config['models_dir'], f"{model_name}_model.h5")
        
        if not os.path.exists(model_path):
            return None
            
        try:
            model = load_model(model_path)
            return model
        except Exception as e:
            logger.error(f"Error loading Keras model {model_name}: {str(e)}")
            return None
    
    def _load_sklearn_model(self, model_name):
        """Load a scikit-learn model"""
        model_path = os.path.join(self.config['models_dir'], f"{model_name}_model.pkl")
        
        if not os.path.exists(model_path):
            return None
            
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            logger.error(f"Error loading scikit-learn model {model_name}: {str(e)}")
            return None
    
    def _load_model_versions(self):
        """Load model versions from disk"""
        try:
            for model_name in self.models.keys():
                versions = []
                
                # Check for versioned models
                base_path = os.path.join(self.config['models_dir'], model_name)
                if os.path.exists(base_path):
                    for filename in os.listdir(base_path):
                        if filename.startswith('v') and filename.endswith('.model'):
                            try:
                                version = int(filename.split('_')[0][1:])
                                timestamp = int(os.path.getmtime(os.path.join(base_path, filename)))
                                versions.append((version, timestamp))
                            except Exception:
                                pass
                
                # Sort by version (descending)
                versions.sort(reverse=True)
                self.model_versions[model_name] = versions
                
                logger.info(f"Loaded {len(versions)} versions for {model_name} model")
                
        except Exception as e:
            logger.error(f"Error loading model versions: {str(e)}", exc_info=True)
    
    def daily_model_update(self):
        """Perform daily incremental update of models"""
        logger.info("Starting daily model update")
        
        try:
            # Check if models exist
            if not self.models:
                logger.warning("No existing models to update, performing full training")
                return self.full_model_retrain()
                
            # Load recent data
            recent_data = self._load_recent_data()
            
            if recent_data is None or len(recent_data) == 0:
                logger.error("Failed to load recent data for model updating")
                return False
                
            # Update each model
            updated_models = {}
            
            for model_name, model in self.models.items():
                try:
                    # Skip models that we're not going to update incrementally
                    if model_name == 'market_regime':
                        logger.info(f"Skipping incremental update for {model_name} model")
                        continue
                        
                    # Update model
                    success, updated_model = self._update_model(model_name, model, recent_data)
                    
                    if success and updated_model:
                        updated_models[model_name] = updated_model
                        logger.info(f"Successfully updated {model_name} model")
                    else:
                        logger.warning(f"Failed to update {model_name} model")
                        
                except Exception as e:
                    logger.error(f"Error updating {model_name} model: {str(e)}", exc_info=True)
            
            # Save updated models
            for model_name, model in updated_models.items():
                try:
                    self._save_model_version(model_name, model)
                    
                    # Replace current model
                    self.models[model_name] = model
                    
                except Exception as e:
                    logger.error(f"Error saving updated {model_name} model: {str(e)}")
            
            # Update model info in Redis
            self._update_models_info()
            
            logger.info(f"Daily model update completed: {len(updated_models)} models updated")
            return True
            
        except Exception as e:
            logger.error(f"Error in daily model update: {str(e)}", exc_info=True)
            return False
    
    def full_model_retrain(self):
        """Perform full retraining of all models"""
        logger.info("Starting full model retraining")
        
        try:
            # Use model trainer for full retraining
            success = self.model_trainer.train_all_models()
            
            if success:
                # Reload models
                self._load_models()
                
                logger.info("Full model retraining completed successfully")
                return True
            else:
                logger.error("Full model retraining failed")
                return False
                
        except Exception as e:
            logger.error(f"Error in full model retraining: {str(e)}", exc_info=True)
            return False
    
    def _load_recent_data(self):
        """Load recent data for incremental updates"""
        try:
            # Get active tickers
            active_tickers = self.model_trainer.get_active_tickers()
            
            if not active_tickers:
                logger.warning("No active tickers for recent data loading")
                return None
                
            logger.info(f"Loading recent data for {len(active_tickers)} tickers")
            
            # Calculate date ranges
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=self.config['update_window'])
            
            # Load price data
            price_data = self.data_loader.load_price_data(
                tickers=active_tickers,
                start_date=start_date,
                end_date=end_date,
                timeframe='1m'
            )
            
            # Load options data
            options_data = self.data_loader.load_options_data(
                tickers=active_tickers,
                start_date=start_date,
                end_date=end_date
            )
            
            # Load market data
            market_data = self.data_loader.load_market_data(
                start_date=start_date,
                end_date=end_date,
                symbols=['SPY', 'QQQ', 'IWM', 'VIX']
            )
            
            # Combine data
            combined_data = self.model_trainer.prepare_training_data(
                price_data=price_data,
                options_data=options_data,
                market_data=market_data
            )
            
            logger.info(f"Loaded {len(combined_data)} recent samples for model updating")
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error loading recent data: {str(e)}", exc_info=True)
            return None
    
    def _update_model(self, model_name, model, recent_data):
        """Update a specific model with recent data"""
        try:
            logger.info(f"Updating {model_name} model")
            
            # Different update approaches based on model type
            if model_name == 'signal_detection':
                return self._update_xgboost_model(model_name, model, recent_data)
            elif model_name == 'price_prediction':
                return self._update_keras_model(model_name, model, recent_data)
            elif model_name == 'risk_assessment':
                return self._update_sklearn_model(model_name, model, recent_data)
            elif model_name == 'exit_strategy':
                return self._update_xgboost_model(model_name, model, recent_data)
            else:
                logger.warning(f"Unsupported model type for incremental update: {model_name}")
                return False, None
                
        except Exception as e:
            logger.error(f"Error updating {model_name} model: {str(e)}", exc_info=True)
            return False, None
    
    def _update_xgboost_model(self, model_name, model, recent_data):
        """Update an XGBoost model"""
        try:
            # Prepare data based on model type
            if model_name == 'signal_detection':
                features, target = self.model_trainer.prepare_signal_detection_data(recent_data)
            elif model_name == 'exit_strategy':
                features, target = self.model_trainer.prepare_exit_strategy_data(recent_data)
            else:
                return False, None
                
            if len(features) == 0 or len(target) == 0:
                logger.warning(f"No valid data for updating {model_name} model")
                return False, None
                
            # Get scaler
            scaler = self.model_trainer.scalers.get(model_name)
            if scaler:
                features_scaled = scaler.transform(features)
            else:
                logger.warning(f"No scaler found for {model_name} model, using unscaled features")
                features_scaled = features
                
            # Create DMatrix
            dtrain = xgb.DMatrix(features_scaled, label=target)
            
            # Get model configuration
            model_config = self.model_trainer.config['model_configs'][model_name]
            
            # Reduce number of iterations for incremental update
            num_rounds = int(model_config['params']['n_estimators'] * 0.3)
            
            # Update model
            updated_model = xgb.train(
                params=model_config['params'],
                dtrain=dtrain,
                num_boost_round=num_rounds,
                xgb_model=model
            )
            
            # Evaluate updated model
            y_pred = updated_model.predict(dtrain)
            
            if model_name == 'signal_detection':
                # Binary classification metrics
                y_pred_binary = (y_pred > 0.5).astype(int)
                accuracy = accuracy_score(target, y_pred_binary)
                
                logger.info(f"Updated {model_name} model accuracy: {accuracy:.4f}")
                
                # Only accept if above threshold
                if accuracy < self.config['performance_threshold']:
                    logger.warning(f"Updated {model_name} model accuracy below threshold: {accuracy:.4f} < {self.config['performance_threshold']}")
                    return False, None
                    
            else:
                # Regression metrics
                mse = np.mean((y_pred - target) ** 2)
                rmse = np.sqrt(mse)
                
                logger.info(f"Updated {model_name} model RMSE: {rmse:.6f}")
            
            return True, updated_model
            
        except Exception as e:
            logger.error(f"Error updating XGBoost model {model_name}: {str(e)}", exc_info=True)
            return False, None
    
    def _update_keras_model(self, model_name, model, recent_data):
        """Update a Keras model"""
        try:
            # Prepare data
            if model_name == 'price_prediction':
                sequences, targets = self.model_trainer.prepare_price_prediction_data(recent_data)
            else:
                return False, None
                
            if len(sequences) == 0 or len(targets) == 0:
                logger.warning(f"No valid data for updating {model_name} model")
                return False, None
                
            # Get model configuration
            model_config = self.model_trainer.config['model_configs'][model_name]
            
            # Reduce epochs for incremental update
            epochs = int(model_config['params']['epochs'] * 0.3)
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
            
            # Split data
            split_idx = int(len(sequences) * 0.8)
            X_train, X_val = sequences[:split_idx], sequences[split_idx:]
            y_train, y_val = targets[:split_idx], targets[split_idx:]
            
            # Fine-tune model
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=model_config['params']['batch_size'],
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate updated model
            val_loss = model.evaluate(X_val, y_val, verbose=0)
            
            logger.info(f"Updated {model_name} model validation loss: {val_loss:.6f}")
            
            return True, model
            
        except Exception as e:
            logger.error(f"Error updating Keras model {model_name}: {str(e)}", exc_info=True)
            return False, None
    
    def _update_sklearn_model(self, model_name, model, recent_data):
        """Update a scikit-learn model"""
        try:
            # Prepare data
            if model_name == 'risk_assessment':
                features, target = self.model_trainer.prepare_risk_assessment_data(recent_data)
            else:
                return False, None
                
            if len(features) == 0 or len(target) == 0:
                logger.warning(f"No valid data for updating {model_name} model")
                return False, None
                
            # Get scaler
            scaler = self.model_trainer.scalers.get(model_name)
            if scaler:
                features_scaled = scaler.transform(features)
            else:
                logger.warning(f"No scaler found for {model_name} model, using unscaled features")
                features_scaled = features
                
            # For RandomForest, we need to create a new model
            # as scikit-learn doesn't support incremental learning for these models
            if isinstance(model, RandomForestClassifier):
                # Get model configuration
                model_config = self.model_trainer.config['model_configs'][model_name]
                
                # Create new model with same parameters
                new_model = RandomForestClassifier(
                    n_estimators=model_config['params']['n_estimators'],
                    max_depth=model_config['params']['max_depth'],
                    min_samples_leaf=model_config['params']['min_samples_leaf'],
                    random_state=self.model_trainer.config['random_state']
                )
                
                # Train on new data
                new_model.fit(features_scaled, target)
                
                # Evaluate
                y_pred = new_model.predict(features_scaled)
                mse = np.mean((y_pred - target) ** 2)
                r2 = new_model.score(features_scaled, target)
                
                logger.info(f"Updated {model_name} model metrics - MSE: {mse:.6f}, R²: {r2:.4f}")
                
                return True, new_model
                
            else:
                logger.warning(f"Unsupported scikit-learn model type for {model_name}")
                return False, None
                
        except Exception as e:
            logger.error(f"Error updating scikit-learn model {model_name}: {str(e)}", exc_info=True)
            return False, None
    
    def _save_model_version(self, model_name, model):
        """Save a versioned copy of the model"""
        try:
            # Create model directory if it doesn't exist
            model_dir = os.path.join(self.config['models_dir'], model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Get current versions
            versions = self.model_versions.get(model_name, [])
            
            # Determine new version number
            new_version = 1
            if versions:
                new_version = versions[0][0] + 1
                
            # Save model based on type
            if model_name in ['signal_detection', 'exit_strategy']:
                # XGBoost model
                model_path = os.path.join(model_dir, f"v{new_version}_model.xgb")
                model.save_model(model_path)
            elif model_name == 'price_prediction':
                # Keras model
                model_path = os.path.join(model_dir, f"v{new_version}_model.h5")
                model.save(model_path)
            else:
                # scikit-learn model
                model_path = os.path.join(model_dir, f"v{new_version}_model.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Update current model
            if model_name in ['signal_detection', 'exit_strategy']:
                current_path = os.path.join(self.config['models_dir'], f"{model_name}_model.xgb")
                model.save_model(current_path)
            elif model_name == 'price_prediction':
                current_path = os.path.join(self.config['models_dir'], f"{model_name}_model.h5")
                model.save(current_path)
            else:
                current_path = os.path.join(self.config['models_dir'], f"{model_name}_model.pkl")
                with open(current_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Update versions list
            timestamp = int(time.time())
            versions.insert(0, (new_version, timestamp))
            
            # Prune old versions
            if len(versions) > self.config['max_versions']:
                old_versions = versions[self.config['max_versions']:]
                versions = versions[:self.config['max_versions']]
                
                # Delete old version files
                for version, _ in old_versions:
                    old_path = os.path.join(model_dir, f"v{version}_model.{'xgb' if model_name in ['signal_detection', 'exit_strategy'] else 'h5' if model_name == 'price_prediction' else 'pkl'}")
                    if os.path.exists(old_path):
                        os.remove(old_path)
            
            # Update model versions
            self.model_versions[model_name] = versions
            
            logger.info(f"Saved version {new_version} of {model_name} model")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving model version for {model_name}: {str(e)}", exc_info=True)
            return False
    
    def _update_models_info(self):
        """Update Redis with model information"""
        try:
            # Collect model info
            models_info = {}
            
            for model_name, model in self.models.items():
                # Get versions
                versions = self.model_versions.get(model_name, [])
                
                # Get metrics if available
                metrics_path = os.path.join(self.config['models_dir'], f"{model_name}_metrics.json")
                metrics = {}
                
                if os.path.exists(metrics_path):
                    try:
                        with open(metrics_path, 'r') as f:
                            metrics = json.load(f)
                    except Exception:
                        pass
                
                # Create model info
                models_info[model_name] = {
                    'versions': len(versions),
                    'latest_version': versions[0][0] if versions else 0,
                    'last_updated': versions[0][1] if versions else 0,
                    'last_updated_str': datetime.datetime.fromtimestamp(versions[0][1]).isoformat() if versions else None,
                    'metrics': metrics
                }
            
            # Update Redis
            self.redis.set("models:info", json.dumps(models_info))
            
            logger.info(f"Updated model info for {len(models_info)} models")
            
        except Exception as e:
            logger.error(f"Error updating model info: {str(e)}", exc_info=True)
    
    def rollback_model(self, model_name, version=None):
        """Rollback a model to a previous version"""
        try:
            # Check if model exists
            if model_name not in self.model_versions:
                logger.error(f"Model {model_name} not found or has no versions")
                return False
                
            # Get versions
            versions = self.model_versions[model_name]
            
            if not versions:
                logger.error(f"No versions found for model {model_name}")
                return False
                
            # Determine version to rollback to
            target_version = None
            
            if version is None:
                # Rollback to previous version
                if len(versions) < 2:
                    logger.error(f"No previous version available for model {model_name}")
                    return False
                    
                target_version = versions[1][0]
            else:
                # Rollback to specific version
                for v, _ in versions:
                    if v == version:
                        target_version = v
                        break
                        
                if target_version is None:
                    logger.error(f"Version {version} not found for model {model_name}")
                    return False
            
            # Load model from version
            model_dir = os.path.join(self.config['models_dir'], model_name)
            
            if model_name in ['signal_detection', 'exit_strategy']:
                # XGBoost model
                model_path = os.path.join(model_dir, f"v{target_version}_model.xgb")
                model = xgb.Booster()
                model.load_model(model_path)
                
                # Update current model
                current_path = os.path.join(self.config['models_dir'], f"{model_name}_model.xgb")
                model.save_model(current_path)
                
            elif model_name == 'price_prediction':
                # Keras model
                model_path = os.path.join(model_dir, f"v{target_version}_model.h5")
                model = load_model(model_path)
                
                # Update current model
                current_path = os.path.join(self.config['models_dir'], f"{model_name}_model.h5")
                model.save(current_path)
                
            else:
                # scikit-learn model
                model_path = os.path.join(model_dir, f"v{target_version}_model.pkl")
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                    
                # Update current model
                current_path = os.path.join(self.config['models_dir'], f"{model_name}_model.pkl")
                with open(current_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Update model in memory
            self.models[model_name] = model
            
            logger.info(f"Rolled back {model_name} model to version {target_version}")
            
            # Update model info in Redis
            self._update_models_info()
            
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back model {model_name}: {str(e)}", exc_info=True)
            return False


# Example usage
if __name__ == "__main__":
    import redis
    from tests.data_pipeline_integration import DataPipelineIntegration
    from tests.ml_model_trainer import MLModelTrainer
    
    # Create Redis client
    redis_client = redis.Redis(
        host=os.environ.get('REDIS_HOST', 'localhost'),
        port=int(os.environ.get('REDIS_PORT', 6379)),
        db=int(os.environ.get('REDIS_DB', 0))
    )
    
    # Create data loader
    data_loader = DataPipelineIntegration(
        redis_host=os.environ.get('REDIS_HOST', 'localhost'),
        redis_port=int(os.environ.get('REDIS_PORT', 6379)),
        redis_db=int(os.environ.get('REDIS_DB', 0)),
        polygon_api_key=os.environ.get('POLYGON_API_KEY', ''),
        unusual_whales_api_key=os.environ.get('UNUSUAL_WHALES_API_KEY', ''),
        use_gpu=os.environ.get('USE_GPU', 'true').lower() == 'true'
    )
    
    # Create model trainer
    model_trainer = MLModelTrainer(redis_client, data_loader)
    
    # Create continual learning system
    continual_learning = ContinualLearningSystem(redis_client, data_loader, model_trainer)
    
    # Start system
    continual_learning.start()
    
    try:
        # Run for a while
        print("Continual learning system running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Stop system
        continual_learning.stop()