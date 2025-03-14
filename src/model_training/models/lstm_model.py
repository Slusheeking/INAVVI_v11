#!/usr/bin/env python3
"""
LSTM Model for Trading

This module implements an LSTM (Long Short-Term Memory) neural network model
optimized for trading applications. It includes multi-timeframe inputs,
dollar profit optimization, and GPU acceleration.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
# TensorFlow imports - Pylance may show errors but these will work at runtime
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

# Import standard libraries only

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("lstm_model")


class LSTMModel:
    """
    LSTM model optimized for trading applications.

    This class implements an LSTM neural network with multi-timeframe inputs,
    dollar profit optimization, and GPU acceleration.
    """

    def __init__(
        self,
        model_id: str = None,
        input_shape: tuple[int, int] = (60, 20),  # (sequence_length, n_features)
        multi_timeframe: bool = False,
        timeframe_shapes: dict[str, tuple[int, int]] | None = None,
        output_type: str = "regression",
        use_gpu: bool = True,
        use_mixed_precision: bool = True,
        lstm_units: list[int] = [128, 64],
        dense_units: list[int] = [32, 16],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        l1_reg: float = 0.0,
        l2_reg: float = 0.001,
        batch_size: int = 64,
        epochs: int = 100,
        early_stopping_patience: int = 20,
        reduce_lr_patience: int = 10,
        use_dollar_profit_loss: bool = True,
        dollar_profit_params: dict[str, Any] | None = None,
    ):
        """
        Initialize the LSTM model.

        Args:
            model_id: Unique identifier for the model
            input_shape: Shape of input data (sequence_length, n_features)
            multi_timeframe: Whether to use multi-timeframe inputs
            timeframe_shapes: Dictionary mapping timeframe names to input shapes
            output_type: Type of output ('regression' or 'classification')
            use_gpu: Whether to use GPU for training
            use_mixed_precision: Whether to use mixed precision training
            lstm_units: List of units for LSTM layers
            dense_units: List of units for dense layers
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            l1_reg: L1 regularization factor
            l2_reg: L2 regularization factor
            batch_size: Batch size for training
            epochs: Maximum number of epochs for training
            early_stopping_patience: Patience for early stopping
            reduce_lr_patience: Patience for learning rate reduction
            use_dollar_profit_loss: Whether to use dollar profit loss function
            dollar_profit_params: Parameters for dollar profit loss function
        """
        self.model_id = model_id or f"lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.input_shape = input_shape
        self.multi_timeframe = multi_timeframe
        self.timeframe_shapes = timeframe_shapes or {}
        self.output_type = output_type
        self.use_gpu = use_gpu
        self.use_mixed_precision = use_mixed_precision
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience
        self.use_dollar_profit_loss = use_dollar_profit_loss
        self.dollar_profit_params = dollar_profit_params or {}

        # Initialize model
        self.model = None
        self.feature_names = None
        self.training_history = None
        self.metadata = {
            "model_id": self.model_id,
            "model_type": "lstm",
            "created_at": datetime.now().isoformat(),
            "input_shape": self.input_shape,
            "multi_timeframe": self.multi_timeframe,
            "timeframe_shapes": self.timeframe_shapes,
            "output_type": self.output_type,
            "parameters": {
                "lstm_units": self.lstm_units,
                "dense_units": self.dense_units,
                "dropout_rate": self.dropout_rate,
                "learning_rate": self.learning_rate,
                "l1_reg": self.l1_reg,
                "l2_reg": self.l2_reg,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "early_stopping_patience": self.early_stopping_patience,
                "reduce_lr_patience": self.reduce_lr_patience,
                "use_dollar_profit_loss": self.use_dollar_profit_loss,
                "dollar_profit_params": self.dollar_profit_params,
            },
            "performance": {},
            "feature_importance": {},
        }

        # Configure GPU and mixed precision
        self._configure_gpu()

        logger.info(f"Initialized LSTMModel with ID {self.model_id}")

    def _configure_gpu(self) -> None:
        """Configure GPU and mixed precision settings optimized for NVIDIA containers."""
        if self.use_gpu:
            # Check if running in NVIDIA container
            is_nvidia_container = os.path.exists("/.dockerenv") and os.environ.get("NVIDIA_BUILD_ID")
            if is_nvidia_container:
                logger.info("Running in NVIDIA TensorFlow container")
            
            # Check if GPU is available
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                try:
                    # Configure memory growth to avoid allocating all memory at once
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)

                    logger.info(f"Using GPU: {gpus}")

                    # Configure memory growth
                    logger.info("GPU memory growth enabled")
                    
                    # Set visible devices if specific GPUs should be used
                    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES")
                    if gpu_ids:
                        logger.info(f"Using GPUs: {gpu_ids}")
                    
                    # Enable mixed precision if requested - optimized for Tensor Cores
                    if self.use_mixed_precision:
                        # Use TensorFlow's built-in mixed precision
                        policy = tf.keras.mixed_precision.Policy('mixed_float16')
                        tf.keras.mixed_precision.set_global_policy(policy)
                        logger.info("Mixed precision training enabled (optimized for Tensor Cores)")
                    
                    # Enable XLA compilation for better performance
                    tf.config.optimizer.set_jit(True)
                    logger.info("XLA compilation enabled")
                    
                    # Apply optimizations for NVIDIA GPUs
                    os.environ["TF_USE_CUDNN"] = "1"
                    os.environ["TF_CUDNN_DETERMINISTIC"] = "0"  # Disable for better performance
                    os.environ["TF_CUDNN_USE_AUTOTUNE"] = "1"   # Enable autotuning
                    os.environ["TF_CUDNN_RESET_RNN_DESCRIPTOR"] = "1"  # Better memory usage for RNNs
                    
                    # Additional optimizations for NVIDIA containers
                    if is_nvidia_container:
                        os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"  # Dedicate GPU threads
                        os.environ["TF_GPU_THREAD_COUNT"] = "2"  # Number of GPU threads
                        os.environ["TF_USE_CUDA_MALLOC_ASYNC"] = "1"  # Async memory allocation
                        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"  # Enable oneDNN optimizations
                        
                        logger.info("NVIDIA container optimizations applied")
                    
                    # Log GPU information
                    try:
                        for i, gpu in enumerate(gpus):
                            gpu_details = tf.config.experimental.get_device_details(gpu)
                            logger.info(f"GPU {i}: {gpu_details}")
                    except Exception as e:
                        logger.debug(f"Could not get detailed GPU information: {e}")

                except RuntimeError as e:
                    logger.warning(f"Error configuring GPU: {e}")
            else:
                logger.warning("No GPU found, using CPU instead")
                self.use_gpu = False

    def build_model(self) -> tf.keras.Model:
        """
        Build the LSTM model architecture.

        Returns:
            TensorFlow Keras model
        """
        regularizer = l1_l2(l1=self.l1_reg, l2=self.l2_reg)

        if self.multi_timeframe:
            # Multi-timeframe model with separate inputs for each timeframe
            inputs = {}
            lstm_outputs = []

            for timeframe, shape in self.timeframe_shapes.items():
                # Create input for this timeframe
                inputs[timeframe] = Input(shape=shape, name=f"input_{timeframe}")

                # LSTM layers for this timeframe
                x = inputs[timeframe]
                for i, units in enumerate(self.lstm_units):
                    return_sequences = i < len(self.lstm_units) - 1
                    x = Bidirectional(
                        LSTM(
                            units,
                            return_sequences=return_sequences,
                            kernel_regularizer=regularizer,
                            recurrent_regularizer=regularizer,
                            name=f"lstm_{timeframe}_{i}",
                        )
                    )(x)
                    x = BatchNormalization(name=f"bn_{timeframe}_{i}")(x)
                    x = Dropout(self.dropout_rate, name=f"dropout_{timeframe}_{i}")(x)

                # Add to outputs
                lstm_outputs.append(x)

            # Combine outputs from all timeframes
            if len(lstm_outputs) > 1:
                combined = Concatenate(name="timeframe_concat")(lstm_outputs)
            else:
                combined = lstm_outputs[0]

            # Dense layers
            for i, units in enumerate(self.dense_units):
                combined = Dense(
                    units,
                    activation="relu",
                    kernel_regularizer=regularizer,
                    name=f"dense_{i}",
                )(combined)
                combined = BatchNormalization(name=f"bn_dense_{i}")(combined)
                combined = Dropout(self.dropout_rate, name=f"dropout_dense_{i}")(
                    combined
                )

            # Output layer
            if self.output_type == "regression":
                output = Dense(1, activation="linear", name="output")(combined)
            else:  # classification
                output = Dense(1, activation="sigmoid", name="output")(combined)

            # Create model
            model = Model(inputs=list(inputs.values()), outputs=output)

        else:
            # Single timeframe model
            model = Sequential()

            # LSTM layers
            for i, units in enumerate(self.lstm_units):
                return_sequences = i < len(self.lstm_units) - 1
                if i == 0:
                    # First layer needs input shape
                    model.add(
                        Bidirectional(
                            LSTM(
                                units,
                                return_sequences=return_sequences,
                                kernel_regularizer=regularizer,
                                recurrent_regularizer=regularizer,
                                input_shape=self.input_shape,
                                name=f"lstm_{i}",
                            )
                        )
                    )
                else:
                    model.add(
                        Bidirectional(
                            LSTM(
                                units,
                                return_sequences=return_sequences,
                                kernel_regularizer=regularizer,
                                recurrent_regularizer=regularizer,
                                name=f"lstm_{i}",
                            )
                        )
                    )
                model.add(BatchNormalization(name=f"bn_{i}"))
                model.add(Dropout(self.dropout_rate, name=f"dropout_{i}"))

            # Dense layers
            for i, units in enumerate(self.dense_units):
                model.add(
                    Dense(
                        units,
                        activation="relu",
                        kernel_regularizer=regularizer,
                        name=f"dense_{i}",
                    )
                )
                model.add(BatchNormalization(name=f"bn_dense_{i}"))
                model.add(Dropout(self.dropout_rate, name=f"dropout_dense_{i}"))

            # Output layer
            if self.output_type == "regression":
                model.add(Dense(1, activation="linear", name="output"))
            else:  # classification
                model.add(Dense(1, activation="sigmoid", name="output"))

        # Compile model
        # Use standard loss function
        if self.output_type == "regression":
            loss = "mse"
        else:  # classification
            loss = "binary_crossentropy"

        # Metrics
        if self.output_type == "regression":
            metrics = ["mae", "mse"]
        else:  # classification
            metrics = ["accuracy", tf.keras.metrics.AUC()]

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), loss=loss, metrics=metrics
        )

        return model

    def train(
        self,
        X_train: np.ndarray | dict[str, np.ndarray],
        y_train: np.ndarray,
        X_val: np.ndarray | dict[str, np.ndarray] | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | dict[str, list[str]] | None = None,
        sample_weight_train: np.ndarray | None = None,
        sample_weight_val: np.ndarray | None = None,
        additional_train_data: dict[str, np.ndarray] | None = None,
        additional_val_data: dict[str, np.ndarray] | None = None,
        callbacks: list[tf.keras.callbacks.Callback] | None = None,
    ) -> dict[str, Any]:
        """
        Train the LSTM model.

        Args:
            X_train: Training features (array or dict of arrays for multi-timeframe)
            y_train: Training labels
            X_val: Validation features (array or dict of arrays for multi-timeframe)
            y_val: Validation labels
            feature_names: Names of features (list or dict of lists for multi-timeframe)
            sample_weight_train: Sample weights for training data
            sample_weight_val: Sample weights for validation data
            additional_train_data: Additional data for training (e.g., price data for dollar profit)
            additional_val_data: Additional data for validation
            callbacks: Additional callbacks for training

        Returns:
            Dictionary with training results
        """
        start_time = time.time()

        # Store feature names
        self.feature_names = feature_names

        # Create validation set if not provided
        if X_val is None or y_val is None:
            if self.multi_timeframe:
                # Split each timeframe separately
                X_val = {}
                for timeframe, X_timeframe in X_train.items():
                    (
                        X_train_split,
                        X_val_split,
                        y_train_split,
                        y_val_split,
                    ) = train_test_split(
                        X_timeframe, y_train, test_size=0.2, random_state=42
                    )
                    X_train[timeframe] = X_train_split
                    X_val[timeframe] = X_val_split

                # Use the last split for y_train and y_val
                y_train = y_train_split
                y_val = y_val_split

                # Split sample weights if provided
                if sample_weight_train is not None:
                    _, sample_weight_val = train_test_split(
                        sample_weight_train, test_size=0.2, random_state=42
                    )
            else:
                # Simple split for single timeframe
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42
                )

                # Split sample weights if provided
                if sample_weight_train is not None:
                    _, sample_weight_val = train_test_split(
                        sample_weight_train, test_size=0.2, random_state=42
                    )

        # Build model if not already built
        if self.model is None:
            self.model = self.build_model()

        # Set up callbacks
        if callbacks is None:
            callbacks = []

        # Add standard callbacks
        model_checkpoint = ModelCheckpoint(
            filepath=f"/tmp/{self.model_id}_best.h5",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )
        callbacks.append(model_checkpoint)

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        )
        callbacks.append(early_stopping)

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=self.reduce_lr_patience,
            min_lr=1e-6,
            verbose=1,
        )
        callbacks.append(reduce_lr)

        # Add TensorBoard callback
        tensorboard = TensorBoard(
            log_dir=f"/tmp/logs/{self.model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            histogram_freq=1,
            write_graph=True,
            update_freq="epoch",
        )
        callbacks.append(tensorboard)

        # Prepare sample weights for dollar profit loss
        if self.use_dollar_profit_loss and additional_train_data:
            # If using dollar profit loss, pass price data as sample weights
            train_sample_weight = {
                "price": additional_train_data.get("price"),
                "atr": additional_train_data.get("atr"),
            }

            if additional_val_data:
                val_sample_weight = {
                    "price": additional_val_data.get("price"),
                    "atr": additional_val_data.get("atr"),
                }
            else:
                val_sample_weight = None
        else:
            train_sample_weight = sample_weight_train
            val_sample_weight = sample_weight_val

        # Train model
        history = self.model.fit(
            X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val, val_sample_weight)
            if val_sample_weight is not None
            else (X_val, y_val),
            callbacks=callbacks,
            sample_weight=train_sample_weight,
            verbose=2,
        )

        # Store training history
        self.training_history = history.history

        # Calculate performance metrics
        train_predictions = self.predict(X_train)
        val_predictions = self.predict(X_val)

        train_metrics = self._calculate_metrics(y_train, train_predictions)
        val_metrics = self._calculate_metrics(y_val, val_predictions)

        # Calculate additional metrics if price data is provided
        if additional_train_data and "price" in additional_train_data:
            # Calculate simple profit metrics
            train_profit = np.sum(np.sign(train_predictions) * y_train * additional_train_data["price"])
            train_metrics["estimated_profit"] = float(train_profit)
            
            if additional_val_data and "price" in additional_val_data:
                val_profit = np.sum(np.sign(val_predictions) * y_val * additional_val_data["price"])
                val_metrics["estimated_profit"] = float(val_profit)

        # Update metadata
        self.metadata["performance"] = {
            "train": train_metrics,
            "validation": val_metrics,
            "training_time": time.time() - start_time,
            "epochs_trained": len(history.history["loss"]),
            "final_loss": history.history["loss"][-1],
            "final_val_loss": history.history["val_loss"][-1],
        }

        self.metadata["updated_at"] = datetime.now().isoformat()

        logger.info(
            f"Trained LSTMModel {self.model_id} in {time.time() - start_time:.2f} seconds"
        )
        logger.info(
            f"Final loss: {history.history['loss'][-1]:.6f}, Final val_loss: {history.history['val_loss'][-1]:.6f}"
        )

        return {
            "model_id": self.model_id,
            "training_time": time.time() - start_time,
            "epochs_trained": len(history.history["loss"]),
            "final_loss": float(history.history["loss"][-1]),
            "final_val_loss": float(history.history["val_loss"][-1]),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }

    def predict(self, X: np.ndarray | dict[str, np.ndarray]) -> np.ndarray:
        """
        Make predictions with the model.

        Args:
            X: Features (array or dict of arrays for multi-timeframe)

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Make predictions
        predictions = self.model.predict(X, batch_size=self.batch_size)

        # Reshape if needed
        if len(predictions.shape) > 1 and predictions.shape[1] == 1:
            predictions = predictions.flatten()

        return predictions

    def save(self, directory: str) -> str:
        """
        Save the model to a directory.

        Args:
            directory: Directory to save the model

        Returns:
            Path to the saved model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Save model
        model_path = os.path.join(directory, f"{self.model_id}.h5")
        self.model.save(model_path)

        # Save metadata
        metadata_path = os.path.join(directory, f"{self.model_id}_metadata.json")
        with open(metadata_path, "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            metadata_json = self._prepare_metadata_for_json()
            json.dump(metadata_json, f, indent=2)

        # Save feature names
        if self.feature_names:
            feature_path = os.path.join(directory, f"{self.model_id}_features.json")
            with open(feature_path, "w") as f:
                json.dump(self.feature_names, f, indent=2)

        # Save training history
        if self.training_history:
            history_path = os.path.join(directory, f"{self.model_id}_history.json")
            with open(history_path, "w") as f:
                # Convert numpy values to Python types for JSON serialization
                history_dict = {}
                for key, values in self.training_history.items():
                    history_dict[key] = [float(v) for v in values]
                json.dump(history_dict, f, indent=2)

        logger.info(f"Saved LSTMModel {self.model_id} to {directory}")

        return model_path

    def load(self, model_path: str, metadata_path: str | None = None) -> None:
        """
        Load the model from a file.

        Args:
            model_path: Path to the model file
            metadata_path: Path to the metadata file (optional)
        """
        # Configure GPU before loading
        self._configure_gpu()

        # Load model
        self.model = tf.keras.models.load_model(model_path)

        # Load metadata if provided
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path) as f:
                self.metadata = json.load(f)

            # Update instance variables from metadata
            self.model_id = self.metadata.get("model_id", self.model_id)
            self.input_shape = tuple(self.metadata.get("input_shape", self.input_shape))
            self.multi_timeframe = self.metadata.get(
                "multi_timeframe", self.multi_timeframe
            )
            self.timeframe_shapes = self.metadata.get(
                "timeframe_shapes", self.timeframe_shapes
            )
            self.output_type = self.metadata.get("output_type", self.output_type)

            params = self.metadata.get("parameters", {})
            self.lstm_units = params.get("lstm_units", self.lstm_units)
            self.dense_units = params.get("dense_units", self.dense_units)
            self.dropout_rate = params.get("dropout_rate", self.dropout_rate)
            self.learning_rate = params.get("learning_rate", self.learning_rate)
            self.l1_reg = params.get("l1_reg", self.l1_reg)
            self.l2_reg = params.get("l2_reg", self.l2_reg)
            self.batch_size = params.get("batch_size", self.batch_size)
            self.epochs = params.get("epochs", self.epochs)
            self.use_dollar_profit_loss = params.get(
                "use_dollar_profit_loss", self.use_dollar_profit_loss
            )
            self.dollar_profit_params = params.get(
                "dollar_profit_params", self.dollar_profit_params
            )

        # Try to load feature names from separate file if not in metadata
        feature_path = os.path.join(
            os.path.dirname(model_path), f"{self.model_id}_features.json"
        )
        if os.path.exists(feature_path):
            with open(feature_path) as f:
                self.feature_names = json.load(f)

        # Try to load training history from separate file
        history_path = os.path.join(
            os.path.dirname(model_path), f"{self.model_id}_history.json"
        )
        if os.path.exists(history_path):
            with open(history_path) as f:
                self.training_history = json.load(f)

        logger.info(f"Loaded LSTMModel {self.model_id} from {model_path}")

    def get_metadata(self) -> dict[str, Any]:
        """
        Get model metadata.

        Returns:
            Dictionary with model metadata
        """
        return self._prepare_metadata_for_json()

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float]:
        """
        Calculate performance metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary with performance metrics
        """
        metrics = {}

        # Regression metrics
        metrics["mse"] = float(mean_squared_error(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        metrics["mae"] = float(np.mean(np.abs(y_true - y_pred)))

        # Direction accuracy
        direction_true = np.sign(y_true)
        direction_pred = np.sign(y_pred)
        metrics["direction_accuracy"] = float(np.mean(direction_true == direction_pred))

        # Classification metrics if binary
        if self.output_type == "classification":
            y_pred_binary = (y_pred > 0.5).astype(int)
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred_binary))
            metrics["precision"] = float(precision_score(y_true, y_pred_binary))
            metrics["recall"] = float(recall_score(y_true, y_pred_binary))
            metrics["f1"] = float(f1_score(y_true, y_pred_binary))

        return metrics

    def _prepare_metadata_for_json(self) -> dict[str, Any]:
        """
        Prepare metadata for JSON serialization.

        Returns:
            JSON-serializable metadata dictionary
        """
        # Create a copy of metadata
        metadata_json = self.metadata.copy()

        # Convert numpy values to Python types
        if "performance" in metadata_json:
            for dataset in ["train", "validation"]:
                if dataset in metadata_json["performance"]:
                    for key, value in metadata_json["performance"][dataset].items():
                        if isinstance(
                            value, (np.float32, np.float64, np.int32, np.int64)
                        ):
                            metadata_json["performance"][dataset][key] = float(value)

        return metadata_json


if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Create synthetic data
    sequence_length = 60
    n_features = 20
    n_samples = 1000

    # Single timeframe data
    X = np.random.rand(n_samples, sequence_length, n_features)
    y = np.random.normal(0, 0.01, n_samples)  # Small price changes
    price_data = np.random.uniform(100, 200, n_samples)  # Price data
    atr_data = np.random.uniform(1, 5, n_samples)  # ATR data

    # Create model
    model = LSTMModel(
        input_shape=(sequence_length, n_features),
        use_gpu=True,
        use_mixed_precision=True,
        lstm_units=[64, 32],
        dense_units=[16],
        use_dollar_profit_loss=True,
        dollar_profit_params={
            "position_sizing_method": "volatility_adjusted",
            "risk_per_trade": 0.01,
            "account_size": 100000.0,
        },
    )

    # Train model
    result = model.train(
        X_train=X,
        y_train=y,
        feature_names=[f"feature_{i}" for i in range(n_features)],
        additional_train_data={"price": price_data, "atr": atr_data},
    )

    print(f"Training result: {result}")

    # Make predictions
    predictions = model.predict(X[:10])
    print(f"Predictions: {predictions}")

    # Save model
    model_path = model.save("/tmp/lstm_models")
    print(f"Model saved to {model_path}")

    # Multi-timeframe example
    print("\nMulti-timeframe example:")

    # Create multi-timeframe data
    timeframes = {
        "1m": (30, 10),  # 30 timesteps, 10 features
        "5m": (20, 10),  # 20 timesteps, 10 features
        "1h": (10, 10),  # 10 timesteps, 10 features
    }

    X_multi = {}
    for timeframe, (seq_len, feat) in timeframes.items():
        X_multi[timeframe] = np.random.rand(n_samples, seq_len, feat)

    # Create multi-timeframe model
    model_multi = LSTMModel(
        multi_timeframe=True,
        timeframe_shapes={tf: (seq, feat) for tf, (seq, feat) in timeframes.items()},
        use_gpu=True,
        use_mixed_precision=True,
        lstm_units=[64, 32],
        dense_units=[16],
        use_dollar_profit_loss=True,
    )

    # Train multi-timeframe model
    result_multi = model_multi.train(
        X_train=X_multi,
        y_train=y,
        feature_names={
            tf: [f"{tf}_feature_{i}" for i in range(feat)]
            for tf, (_, feat) in timeframes.items()
        },
        additional_train_data={"price": price_data, "atr": atr_data},
    )

    print(f"Multi-timeframe training result: {result_multi}")
