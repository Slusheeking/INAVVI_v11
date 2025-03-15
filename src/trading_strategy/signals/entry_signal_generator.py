"""
Entry Signal Generator

This module provides the EntrySignalGenerator class for generating entry signals
based on model predictions and market conditions.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Any

from src.utils.logging import get_logger

logger = get_logger("trading_strategy.signals.entry_signal_generator")


class EntrySignalGenerator:
    """
    Generates entry signals based on model predictions and market conditions.

    The entry signal generation process includes:
    1. Evaluating model predictions across multiple timeframes
    2. Calculating signal strength and conviction
    3. Applying market regime filters
    4. Implementing entry conditions and filters
    5. Generating final entry signals with metadata

    This approach ensures that entry signals are generated only when there is
    sufficient evidence of a profitable trading opportunity.
    """

    def __init__(
        self,
        model_registry=None,
        feature_pipeline=None,
        min_signal_strength: float = 0.6,  # Minimum signal strength (0-1)
        min_conviction: float = 0.7,  # Minimum conviction score (0-1)
        use_multi_timeframe_confirmation: bool = True,  # Whether to require multi-timeframe confirmation
        use_market_regime_filter: bool = True,  # Whether to filter based on market regime
        max_signals_per_day: int = 10,  # Maximum number of signals per day
        signal_expiration_hours: int = 24,  # Signal expiration time in hours
    ):
        """
        Initialize the EntrySignalGenerator.

        Args:
            model_registry: Model registry for loading models
            feature_pipeline: Feature pipeline for generating features
            min_signal_strength: Minimum signal strength required (0-1)
            min_conviction: Minimum conviction score required (0-1)
            use_multi_timeframe_confirmation: Whether to require multi-timeframe confirmation
            use_market_regime_filter: Whether to filter based on market regime
            max_signals_per_day: Maximum number of signals to generate per day
            signal_expiration_hours: Signal expiration time in hours
        """
        self.model_registry = model_registry
        self.feature_pipeline = feature_pipeline
        self.min_signal_strength = min_signal_strength
        self.min_conviction = min_conviction
        self.use_multi_timeframe_confirmation = use_multi_timeframe_confirmation
        self.use_market_regime_filter = use_market_regime_filter
        self.max_signals_per_day = max_signals_per_day
        self.signal_expiration_hours = signal_expiration_hours

        # Store generated signals
        self.signals: list[dict[str, Any]] = []

        # Store market regime information
        self.market_regime: dict[str, Any] = {}

        # Store model predictions
        self.model_predictions: dict[str, dict[str, Any]] = {}

        logger.info(
            f"Initialized EntrySignalGenerator with min_signal_strength={min_signal_strength:.1f}, "
            f"min_conviction={min_conviction:.1f}, "
            f"use_multi_timeframe_confirmation={use_multi_timeframe_confirmation}"
        )

    def update_market_regime(self, market_regime: dict[str, Any]) -> None:
        """
        Update market regime information.

        Args:
            market_regime: Dictionary with market regime information
        """
        self.market_regime = market_regime

        logger.info(
            f"Updated market regime: {market_regime.get('regime_type', 'unknown')}, "
            f"volatility={market_regime.get('volatility', 'unknown')}"
        )

    def update_model_predictions(
        self, ticker: str, timeframe: str, predictions: dict[str, Any]
    ) -> None:
        """
        Update model predictions for a ticker and timeframe.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            predictions: Dictionary with model predictions
        """
        if ticker not in self.model_predictions:
            self.model_predictions[ticker] = {}

        self.model_predictions[ticker][timeframe] = predictions

        logger.debug(
            f"Updated model predictions for {ticker} ({timeframe}): "
            f"direction={predictions.get('direction', 'unknown')}, "
            f"probability={predictions.get('probability', 0):.2f}"
        )

    def generate_entry_signal(
        self,
        ticker: str,
        timeframe: str,
        current_price: float,
        timestamp: datetime | None = None,
    ) -> dict[str, Any] | None:
        """
        Generate an entry signal for a ticker and timeframe.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            current_price: Current price of the asset
            timestamp: Timestamp for the signal

        Returns:
            Entry signal dictionary or None if no signal is generated
        """
        # Set timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now()

        # Check if we have predictions for this ticker and timeframe
        if (
            ticker not in self.model_predictions
            or timeframe not in self.model_predictions[ticker]
        ):
            logger.warning(f"No model predictions found for {ticker} ({timeframe})")
            return None

        # Get predictions
        predictions = self.model_predictions[ticker][timeframe]

        # Check signal strength
        signal_strength = predictions.get("probability", 0)
        if signal_strength < self.min_signal_strength:
            logger.debug(
                f"Signal strength too low for {ticker} ({timeframe}): "
                f"{signal_strength:.2f} < {self.min_signal_strength:.2f}"
            )
            return None

        # Check conviction
        conviction = predictions.get("conviction", 0)
        if conviction < self.min_conviction:
            logger.debug(
                f"Conviction too low for {ticker} ({timeframe}): "
                f"{conviction:.2f} < {self.min_conviction:.2f}"
            )
            return None

        # Check multi-timeframe confirmation if enabled
        if (
            self.use_multi_timeframe_confirmation
            and not self._check_multi_timeframe_confirmation(ticker, timeframe)
        ):
            logger.debug(
                f"Multi-timeframe confirmation failed for {ticker} ({timeframe})"
            )
            return None

        # Check market regime filter if enabled
        if self.use_market_regime_filter and not self._check_market_regime_filter(
            ticker, predictions.get("direction", "")
        ):
            logger.debug(f"Market regime filter failed for {ticker} ({timeframe})")
            return None

        # Generate signal ID
        signal_id = f"{ticker}_{timeframe}_{timestamp.strftime('%Y%m%d%H%M%S')}"

        # Calculate expiration time
        expiration_time = timestamp + timedelta(hours=self.signal_expiration_hours)

        # Create signal
        signal = {
            "id": signal_id,
            "ticker": ticker,
            "timeframe": timeframe,
            "direction": predictions.get("direction", ""),
            "signal_strength": signal_strength,
            "conviction": conviction,
            "current_price": current_price,
            "timestamp": timestamp,
            "expiration_time": expiration_time,
            "status": "active",
            "metadata": {
                "model_id": predictions.get("model_id", ""),
                "model_version": predictions.get("model_version", ""),
                "features": predictions.get("features", {}),
                "market_regime": self.market_regime.get("regime_type", "unknown"),
                "volatility": self.market_regime.get("volatility", "unknown"),
                "multi_timeframe_confirmation": self._get_multi_timeframe_confirmation_details(
                    ticker, timeframe
                ),
            },
        }

        # Add to signals list
        self.signals.append(signal)

        logger.info(
            f"Generated entry signal for {ticker} ({timeframe}): "
            f"direction={signal['direction']}, "
            f"strength={signal_strength:.2f}, "
            f"conviction={conviction:.2f}"
        )

        return signal

    def generate_signal(self, ticker: str, df: pd.DataFrame) -> dict[str, Any] | None:
        """
        Generate a trading signal using model inference.
        
        Args:
            ticker: Ticker symbol
            df: DataFrame with features
            
        Returns:
            Signal dictionary or None if no signal
        """
        if self.model_registry is None or self.feature_pipeline is None:
            logger.error("Model registry and feature pipeline are required for signal generation")
            return None
            
        try:
            # Generate features if needed
            if len(df.columns) < 10:  # Assuming raw data has fewer columns than feature-rich data
                logger.info(f"Generating features for {ticker}")
                df = self.feature_pipeline.generate_technical_indicators(df)
                df = self.feature_pipeline.generate_price_features(df)
                df = self.feature_pipeline.generate_volume_features(df)
                df = self.feature_pipeline.generate_volatility_features(df)
            
            # Get latest price
            if 'close' in df.columns:
                current_price = df['close'].iloc[-1]
            else:
                current_price = 0.0
                logger.warning(f"No close price found for {ticker}, using default value {current_price}")
            
            # Create model inference engine
            from src.model_training.models.inference.model_inference import ModelInference
            model_inference = ModelInference(model_registry=self.model_registry)
            
            # Generate signal
            signal = model_inference.generate_signal(ticker, df, confidence_threshold=self.min_signal_strength)
            
            if signal:
                # Add additional metadata
                signal['conviction'] = signal.get('confidence', 0.0)
                signal['signal_strength'] = signal.get('confidence', 0.0)
                signal['timestamp'] = pd.Timestamp.now()
                signal['expiration_time'] = signal['timestamp'] + timedelta(hours=self.signal_expiration_hours)
                signal['status'] = 'active'
                
                # Add to signals list
                self.signals.append(signal)
                
                logger.info(f"Generated {signal['direction']} signal for {ticker} with confidence {signal['confidence']:.4f}")
                
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {ticker}: {e}")
            return None
    
    def generate_entry_signals_batch(
        self,
        tickers: list[str],
        timeframe: str,
        prices: dict[str, float],
        timestamp: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generate entry signals for a batch of tickers.

        Args:
            tickers: List of ticker symbols
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            prices: Dictionary mapping tickers to current prices
            timestamp: Timestamp for the signals

        Returns:
            List of entry signal dictionaries
        """
        signals = []

        # Set timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now()

        # Generate signals for each ticker
        for ticker in tickers:
            # Skip if we don't have a price for this ticker
            if ticker not in prices:
                continue

            # Generate signal
            signal = self.generate_entry_signal(
                ticker=ticker,
                timeframe=timeframe,
                current_price=prices[ticker],
                timestamp=timestamp,
            )

            # Add to signals list if generated
            if signal:
                signals.append(signal)

        # Limit to max_signals_per_day
        if len(signals) > self.max_signals_per_day:
            # Sort by conviction (highest first)
            signals.sort(key=lambda x: x["conviction"], reverse=True)
            signals = signals[: self.max_signals_per_day]

        logger.info(
            f"Generated {len(signals)} entry signals for {len(tickers)} tickers"
        )

        return signals

    def get_active_signals(self) -> list[dict[str, Any]]:
        """
        Get active signals (not expired or executed).

        Returns:
            List of active signal dictionaries
        """
        current_time = datetime.now()

        # Filter active signals
        active_signals = [
            signal
            for signal in self.signals
            if signal["status"] == "active" and signal["expiration_time"] > current_time
        ]

        return active_signals

    def mark_signal_executed(self, signal_id: str) -> None:
        """
        Mark a signal as executed.

        Args:
            signal_id: Signal ID
        """
        # Find signal
        for signal in self.signals:
            if signal["id"] == signal_id:
                signal["status"] = "executed"
                signal["execution_time"] = datetime.now()

                logger.info(f"Marked signal {signal_id} as executed")
                break

    def mark_signal_expired(self, signal_id: str) -> None:
        """
        Mark a signal as expired.

        Args:
            signal_id: Signal ID
        """
        # Find signal
        for signal in self.signals:
            if signal["id"] == signal_id:
                signal["status"] = "expired"

                logger.info(f"Marked signal {signal_id} as expired")
                break

    def cleanup_expired_signals(self) -> int:
        """
        Clean up expired signals.

        Returns:
            Number of signals cleaned up
        """
        current_time = datetime.now()

        # Count expired signals
        expired_count = sum(
            1
            for signal in self.signals
            if signal["status"] == "active"
            and signal["expiration_time"] <= current_time
        )

        # Mark expired signals
        for signal in self.signals:
            if (
                signal["status"] == "active"
                and signal["expiration_time"] <= current_time
            ):
                signal["status"] = "expired"

        logger.info(f"Cleaned up {expired_count} expired signals")

        return expired_count

    def _check_multi_timeframe_confirmation(self, ticker: str, timeframe: str) -> bool:
        """
        Check if there is multi-timeframe confirmation for a signal.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe (e.g., '1h', '4h', '1d')

        Returns:
            True if there is multi-timeframe confirmation, False otherwise
        """
        # Get predictions for this ticker
        ticker_predictions = self.model_predictions.get(ticker, {})

        # Get direction for this timeframe
        direction = ticker_predictions.get(timeframe, {}).get("direction", "")

        # If no direction, return False
        if not direction:
            return False

        # Check if there is confirmation from at least one other timeframe
        for tf, predictions in ticker_predictions.items():
            # Skip the current timeframe
            if tf == timeframe:
                continue

            # Check if direction matches and signal strength is sufficient
            if (
                predictions.get("direction", "") == direction
                and predictions.get("probability", 0) >= self.min_signal_strength
            ):
                return True

        return False

    def _get_multi_timeframe_confirmation_details(
        self, ticker: str, timeframe: str
    ) -> dict[str, Any]:
        """
        Get details of multi-timeframe confirmation.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe (e.g., '1h', '4h', '1d')

        Returns:
            Dictionary with multi-timeframe confirmation details
        """
        # Get predictions for this ticker
        ticker_predictions = self.model_predictions.get(ticker, {})

        # Get direction for this timeframe
        direction = ticker_predictions.get(timeframe, {}).get("direction", "")

        # Initialize confirmation details
        confirmation_details = {
            "confirmed": False,
            "confirming_timeframes": [],
            "conflicting_timeframes": [],
        }

        # If no direction, return empty details
        if not direction:
            return confirmation_details

        # Check each timeframe
        for tf, predictions in ticker_predictions.items():
            # Skip the current timeframe
            if tf == timeframe:
                continue

            # Get direction and probability
            tf_direction = predictions.get("direction", "")
            tf_probability = predictions.get("probability", 0)

            # Check if direction matches and signal strength is sufficient
            if tf_direction == direction and tf_probability >= self.min_signal_strength:
                confirmation_details["confirming_timeframes"].append(
                    {
                        "timeframe": tf,
                        "direction": tf_direction,
                        "probability": tf_probability,
                    }
                )
            elif tf_direction and tf_direction != direction:
                confirmation_details["conflicting_timeframes"].append(
                    {
                        "timeframe": tf,
                        "direction": tf_direction,
                        "probability": tf_probability,
                    }
                )

        # Set confirmed flag
        confirmation_details["confirmed"] = (
            len(confirmation_details["confirming_timeframes"]) > 0
        )

        return confirmation_details

    def _check_market_regime_filter(self, ticker: str, direction: str) -> bool:
        """
        Check if a signal passes the market regime filter.

        Args:
            ticker: Ticker symbol
            direction: Signal direction ('long' or 'short')

        Returns:
            True if the signal passes the filter, False otherwise
        """
        # If no market regime information, allow the signal
        if not self.market_regime:
            return True

        # Get market regime type
        regime_type = self.market_regime.get("regime_type", "unknown")

        # Apply regime-specific filters
        if regime_type == "bullish":
            # In bullish regime, favor long signals
            return direction == "long"
        elif regime_type == "bearish":
            # In bearish regime, favor short signals
            return direction == "short"
        elif regime_type == "neutral":
            # In neutral regime, allow both directions
            return True
        elif regime_type == "volatile":
            # In volatile regime, require higher signal strength
            ticker_predictions = self.model_predictions.get(ticker, {})
            for tf, predictions in ticker_predictions.items():
                if predictions.get("direction", "") == direction:
                    if (
                        predictions.get("probability", 0)
                        >= self.min_signal_strength * 1.2
                    ):
                        return True
            return False
        else:
            # Unknown regime, allow the signal
            return True
