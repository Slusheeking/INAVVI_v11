"""
Peak Detector

This module provides the PeakDetector class for detecting peaks and troughs in price data
to identify optimal exit points.
"""

from typing import Any, Dict, List

import numpy as np
from scipy.signal import find_peaks

from src.utils.metrics.metrics_utils import calculate_trading_metrics
from src.utils.serialization.serialization_utils import save_json, load_json
from src.utils.logging import get_logger

logger = get_logger("trading_strategy.signals.peak_detector")


class PeakDetector:
    """
    Detects peaks and troughs in price data to identify optimal exit points.

    The peak detection process includes:
    1. Smoothing price data to reduce noise
    2. Detecting peaks and troughs using signal processing techniques
    3. Calculating peak/trough prominence and width
    4. Filtering peaks/troughs based on significance criteria
    5. Identifying potential exit points

    This approach helps identify optimal exit points to maximize profits and minimize
    drawdowns.
    """

    def __init__(
        self,
        smoothing_window: int = 5,  # Window size for smoothing
        peak_distance: int = 5,  # Minimum distance between peaks
        prominence_threshold: float = 0.01,  # Minimum prominence as percentage of price
        width_threshold: int = 3,  # Minimum width of peaks
        use_adaptive_thresholds: bool = True,  # Whether to adapt thresholds to volatility
        lookback_periods: int = 100,  # Number of periods to look back for detection
    ):
        """
        Initialize the PeakDetector.

        Args:
            smoothing_window: Window size for smoothing price data
            peak_distance: Minimum distance between peaks
            prominence_threshold: Minimum prominence as percentage of price
            width_threshold: Minimum width of peaks
            use_adaptive_thresholds: Whether to adapt thresholds to volatility
            lookback_periods: Number of periods to look back for detection
        """
        self.smoothing_window = smoothing_window
        self.peak_distance = peak_distance
        self.prominence_threshold = prominence_threshold
        self.width_threshold = width_threshold
        self.use_adaptive_thresholds = use_adaptive_thresholds
        self.lookback_periods = lookback_periods

        logger.info(
            f"Initialized PeakDetector with smoothing_window={smoothing_window}, "
            f"peak_distance={peak_distance}, prominence_threshold={prominence_threshold:.1%}"
        )

    def detect_peaks(
        self,
        prices: np.ndarray,
        timestamps: np.ndarray | None = None,
        volatility: float | None = None,
    ) -> dict[str, Any]:
        """
        Detect peaks in price data.

        Args:
            prices: Array of price data
            timestamps: Array of timestamps corresponding to prices
            volatility: Volatility measure (e.g., ATR percentage)

        Returns:
            Dictionary with peak information
        """
        # Ensure we have enough data
        if len(prices) < self.smoothing_window + 1:
            logger.warning(
                f"Not enough data for peak detection: {len(prices)} < {self.smoothing_window + 1}"
            )
            return {
                "peaks": [],
                "peak_indices": [],
                "peak_prominences": [],
                "peak_widths": [],
            }

        # Create timestamps if not provided
        if timestamps is None:
            timestamps = np.array([i for i in range(len(prices))])

        # Smooth prices
        smoothed_prices = self._smooth_prices(prices)

        # Adjust prominence threshold based on volatility if enabled
        prominence_threshold = self._adjust_prominence_threshold(volatility)

        # Calculate absolute prominence threshold
        abs_prominence_threshold = np.mean(prices) * prominence_threshold

        # Find peaks
        peak_indices, peak_properties = find_peaks(
            smoothed_prices,
            distance=self.peak_distance,
            prominence=abs_prominence_threshold,
            width=self.width_threshold,
        )

        # Get peak prominences and widths
        peak_prominences = peak_properties["prominences"]
        peak_widths = peak_properties["widths"]

        # Create peak information
        peaks = []
        for i, peak_idx in enumerate(peak_indices):
            if peak_idx < len(prices) and peak_idx < len(timestamps):
                peaks.append(
                    {
                        "index": int(peak_idx),
                        "timestamp": timestamps[peak_idx],
                        "price": prices[peak_idx],
                        "prominence": peak_prominences[i],
                        "width": peak_widths[i],
                        "prominence_pct": peak_prominences[i] / prices[peak_idx]
                        if prices[peak_idx] > 0
                        else 0,
                    }
                )

        logger.info(f"Detected {len(peaks)} peaks in price data")

        return {
            "peaks": peaks,
            "peak_indices": peak_indices,
            "peak_prominences": peak_prominences,
            "peak_widths": peak_widths,
        }

    def detect_troughs(
        self,
        prices: np.ndarray,
        timestamps: np.ndarray | None = None,
        volatility: float | None = None,
    ) -> dict[str, Any]:
        """
        Detect troughs in price data.

        Args:
            prices: Array of price data
            timestamps: Array of timestamps corresponding to prices
            volatility: Volatility measure (e.g., ATR percentage)

        Returns:
            Dictionary with trough information
        """
        # Ensure we have enough data
        if len(prices) < self.smoothing_window + 1:
            logger.warning(
                f"Not enough data for trough detection: {len(prices)} < {self.smoothing_window + 1}"
            )
            return {
                "troughs": [],
                "trough_indices": [],
                "trough_prominences": [],
                "trough_widths": [],
            }

        # Create timestamps if not provided
        if timestamps is None:
            timestamps = np.array([i for i in range(len(prices))])

        # Smooth prices
        smoothed_prices = self._smooth_prices(prices)

        # Adjust prominence threshold based on volatility if enabled
        prominence_threshold = self._adjust_prominence_threshold(volatility)

        # Calculate absolute prominence threshold
        abs_prominence_threshold = np.mean(prices) * prominence_threshold

        # Find troughs (peaks in negative prices)
        trough_indices, trough_properties = find_peaks(
            -smoothed_prices,  # Negate prices to find troughs
            distance=self.peak_distance,
            prominence=abs_prominence_threshold,
            width=self.width_threshold,
        )

        # Get trough prominences and widths
        trough_prominences = trough_properties["prominences"]
        trough_widths = trough_properties["widths"]

        # Create trough information
        troughs = []
        for i, trough_idx in enumerate(trough_indices):
            if trough_idx < len(prices) and trough_idx < len(timestamps):
                troughs.append(
                    {
                        "index": int(trough_idx),
                        "timestamp": timestamps[trough_idx],
                        "price": prices[trough_idx],
                        "prominence": trough_prominences[i],
                        "width": trough_widths[i],
                        "prominence_pct": trough_prominences[i] / prices[trough_idx]
                        if prices[trough_idx] > 0
                        else 0,
                    }
                )

        logger.info(f"Detected {len(troughs)} troughs in price data")

        return {
            "troughs": troughs,
            "trough_indices": trough_indices,
            "trough_prominences": trough_prominences,
            "trough_widths": trough_widths,
        }

    def detect_peaks_and_troughs(
        self,
        prices: np.ndarray,
        timestamps: np.ndarray | None = None,
        volatility: float | None = None,
    ) -> dict[str, Any]:
        """
        Detect both peaks and troughs in price data.

        Args:
            prices: Array of price data
            timestamps: Array of timestamps corresponding to prices
            volatility: Volatility measure (e.g., ATR percentage)

        Returns:
            Dictionary with peak and trough information
        """
        # Detect peaks
        peak_results = self.detect_peaks(prices, timestamps, volatility)

        # Detect troughs
        trough_results = self.detect_troughs(prices, timestamps, volatility)

        # Combine results
        results = {
            "peaks": peak_results["peaks"],
            "troughs": trough_results["troughs"],
            "peak_indices": peak_results["peak_indices"],
            "trough_indices": trough_results["trough_indices"],
            "peak_prominences": peak_results["peak_prominences"],
            "trough_prominences": trough_results["trough_prominences"],
            "peak_widths": peak_results["peak_widths"],
            "trough_widths": trough_results["trough_widths"],
        }

        return results

    def identify_exit_points(
        self,
        prices: np.ndarray,
        timestamps: np.ndarray | None = None,
        direction: str = "long",
        volatility: float | None = None,
        current_index: int | None = None,
    ) -> dict[str, Any]:
        """
        Identify potential exit points based on peaks and troughs.

        Args:
            prices: Array of price data
            timestamps: Array of timestamps corresponding to prices
            direction: Trade direction ('long' or 'short')
            volatility: Volatility measure (e.g., ATR percentage)
            current_index: Current index in the price array

        Returns:
            Dictionary with exit point information
        """
        # Set current index to last index if not provided
        if current_index is None:
            current_index = len(prices) - 1

        # Ensure current index is valid
        current_index = min(current_index, len(prices) - 1)

        # Detect peaks and troughs
        results = self.detect_peaks_and_troughs(prices, timestamps, volatility)

        # Identify exit points based on direction
        if direction.lower() == "long":
            # For long positions, look for peaks
            exit_points = results["peaks"]

            # Sort by prominence (highest first)
            exit_points.sort(key=lambda x: x["prominence"], reverse=True)
        else:
            # For short positions, look for troughs
            exit_points = results["troughs"]

            # Sort by prominence (highest first)
            exit_points.sort(key=lambda x: x["prominence"], reverse=True)

        # Filter exit points that are in the future (after current index)
        future_exit_points = [
            point for point in exit_points if point["index"] > current_index
        ]

        # Filter exit points that are in the past (before current index)
        past_exit_points = [
            point for point in exit_points if point["index"] <= current_index
        ]

        # Get most recent past exit point
        most_recent_exit_point = past_exit_points[0] if past_exit_points else None

        # Get next future exit point
        next_exit_point = future_exit_points[0] if future_exit_points else None

        logger.info(
            f"Identified exit points for {direction} position: "
            f"{len(future_exit_points)} future, {len(past_exit_points)} past"
        )

        return {
            "direction": direction,
            "exit_points": exit_points,
            "future_exit_points": future_exit_points,
            "past_exit_points": past_exit_points,
            "most_recent_exit_point": most_recent_exit_point,
            "next_exit_point": next_exit_point,
        }

    def _smooth_prices(self, prices: np.ndarray) -> np.ndarray:
        """
        Smooth price data using a moving average.

        Args:
            prices: Array of price data

        Returns:
            Smoothed price data
        """
        # Use simple moving average for smoothing
        smoothed = np.convolve(
            prices, np.ones(self.smoothing_window) / self.smoothing_window, mode="valid"
        )

        # Pad the beginning to maintain the same length
        padding = np.full(self.smoothing_window - 1, smoothed[0])
        smoothed = np.concatenate((padding, smoothed))

        return smoothed

    def _adjust_prominence_threshold(self, volatility: float | None = None) -> float:
        """
        Adjust prominence threshold based on volatility.

        Args:
            volatility: Volatility measure (e.g., ATR percentage)

        Returns:
            Adjusted prominence threshold
        """
        if not self.use_adaptive_thresholds or volatility is None:
            return self.prominence_threshold

        # Adjust threshold based on volatility
        # Higher volatility = higher threshold to filter out noise
        if volatility < 0.01:  # Low volatility
            return self.prominence_threshold * 0.8  # Lower threshold
        elif volatility > 0.03:  # High volatility
            return self.prominence_threshold * 1.5  # Higher threshold
        else:
            # Linear scaling between 0.8 and 1.5
            factor = 0.8 + (volatility - 0.01) / 0.02 * 0.7
            return self.prominence_threshold * factor

    def analyze_price_structure(
        self,
        prices: np.ndarray,
        timestamps: np.ndarray | None = None,
        volatility: float | None = None,
    ) -> dict[str, Any]:
        """
        Analyze price structure to identify patterns.

        Args:
            prices: Array of price data
            timestamps: Array of timestamps corresponding to prices
            volatility: Volatility measure (e.g., ATR percentage)

        Returns:
            Dictionary with price structure analysis
        """
        # Detect peaks and troughs
        results = self.detect_peaks_and_troughs(prices, timestamps, volatility)

        # Get peaks and troughs
        peaks = results["peaks"]
        troughs = results["troughs"]

        # Sort by index
        peaks.sort(key=lambda x: x["index"])
        troughs.sort(key=lambda x: x["index"])

        # Analyze higher highs and lower lows
        higher_highs = self._analyze_higher_highs(peaks)
        lower_lows = self._analyze_lower_lows(troughs)

        # Analyze peak-to-trough ratios
        peak_trough_ratios = self._analyze_peak_trough_ratios(peaks, troughs)

        # Determine trend direction
        trend_direction = self._determine_trend_direction(higher_highs, lower_lows)

        # Determine trend strength
        trend_strength = self._determine_trend_strength(
            higher_highs, lower_lows, peak_trough_ratios
        )

        logger.info(
            f"Analyzed price structure: trend_direction={trend_direction}, "
            f"trend_strength={trend_strength:.2f}"
        )

        return {
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "higher_highs": higher_highs,
            "lower_lows": lower_lows,
            "peak_trough_ratios": peak_trough_ratios,
        }

    def _analyze_higher_highs(self, peaks: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Analyze higher highs in peak data.

        Args:
            peaks: List of peak dictionaries

        Returns:
            Dictionary with higher highs analysis
        """
        if len(peaks) < 2:
            return {
                "has_higher_highs": False,
                "higher_high_count": 0,
                "higher_high_percentage": 0.0,
            }

        # Count higher highs
        higher_high_count = 0
        for i in range(1, len(peaks)):
            if peaks[i]["price"] > peaks[i - 1]["price"]:
                higher_high_count += 1

        # Calculate percentage
        higher_high_percentage = higher_high_count / (len(peaks) - 1)

        return {
            "has_higher_highs": higher_high_count > 0,
            "higher_high_count": higher_high_count,
            "higher_high_percentage": higher_high_percentage,
        }

    def _analyze_lower_lows(self, troughs: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Analyze lower lows in trough data.

        Args:
            troughs: List of trough dictionaries

        Returns:
            Dictionary with lower lows analysis
        """
        if len(troughs) < 2:
            return {
                "has_lower_lows": False,
                "lower_low_count": 0,
                "lower_low_percentage": 0.0,
            }

        # Count lower lows
        lower_low_count = 0
        for i in range(1, len(troughs)):
            if troughs[i]["price"] < troughs[i - 1]["price"]:
                lower_low_count += 1

        # Calculate percentage
        lower_low_percentage = lower_low_count / (len(troughs) - 1)

        return {
            "has_lower_lows": lower_low_count > 0,
            "lower_low_count": lower_low_count,
            "lower_low_percentage": lower_low_percentage,
        }

    def _analyze_peak_trough_ratios(
        self, peaks: list[dict[str, Any]], troughs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Analyze peak-to-trough ratios.

        Args:
            peaks: List of peak dictionaries
            troughs: List of trough dictionaries

        Returns:
            Dictionary with peak-to-trough ratio analysis
        """
        if len(peaks) == 0 or len(troughs) == 0:
            return {"average_ratio": 0.0, "max_ratio": 0.0, "min_ratio": 0.0}

        # Calculate ratios
        ratios = []
        for peak in peaks:
            peak_idx = peak["index"]
            peak_price = peak["price"]

            # Find closest trough before peak
            prev_troughs = [t for t in troughs if t["index"] < peak_idx]
            if prev_troughs:
                closest_trough = max(prev_troughs, key=lambda t: t["index"])
                trough_price = closest_trough["price"]

                # Calculate ratio
                ratio = (
                    (peak_price - trough_price) / trough_price
                    if trough_price > 0
                    else 0
                )
                ratios.append(ratio)

        if not ratios:
            return {"average_ratio": 0.0, "max_ratio": 0.0, "min_ratio": 0.0}

        # Calculate statistics
        average_ratio = sum(ratios) / len(ratios)
        max_ratio = max(ratios)
        min_ratio = min(ratios)

        return {
            "average_ratio": average_ratio,
            "max_ratio": max_ratio,
            "min_ratio": min_ratio,
            "ratios": ratios,
        }

    def _determine_trend_direction(
        self, higher_highs: dict[str, Any], lower_lows: dict[str, Any]
    ) -> str:
        """
        Determine trend direction based on higher highs and lower lows.

        Args:
            higher_highs: Higher highs analysis
            lower_lows: Lower lows analysis

        Returns:
            Trend direction ('uptrend', 'downtrend', or 'sideways')
        """
        higher_high_pct = higher_highs.get("higher_high_percentage", 0)
        lower_low_pct = lower_lows.get("lower_low_percentage", 0)

        # Determine trend direction
        if higher_high_pct > 0.7 and lower_low_pct < 0.3:
            return "uptrend"
        elif lower_low_pct > 0.7 and higher_high_pct < 0.3:
            return "downtrend"
        else:
            return "sideways"

    def _determine_trend_strength(
        self,
        higher_highs: dict[str, Any],
        lower_lows: dict[str, Any],
        peak_trough_ratios: dict[str, Any],
    ) -> float:
        """
        Determine trend strength based on higher highs, lower lows, and peak-trough ratios.

        Args:
            higher_highs: Higher highs analysis
            lower_lows: Lower lows analysis
            peak_trough_ratios: Peak-to-trough ratio analysis

        Returns:
            Trend strength (0-1)
        """
        higher_high_pct = higher_highs.get("higher_high_percentage", 0)
        lower_low_pct = lower_lows.get("lower_low_percentage", 0)
        avg_ratio = peak_trough_ratios.get("average_ratio", 0)

        # Calculate trend strength
        if higher_high_pct > lower_low_pct:
            # Uptrend
            trend_strength = higher_high_pct * 0.7 + min(avg_ratio, 0.1) * 3.0
        elif lower_low_pct > higher_high_pct:
            # Downtrend
            trend_strength = lower_low_pct * 0.7 + min(avg_ratio, 0.1) * 3.0
        else:
            # Sideways
            trend_strength = 0.5 - abs(higher_high_pct - lower_low_pct)

        # Ensure trend strength is between 0 and 1
        trend_strength = max(0.0, min(1.0, trend_strength))

        return trend_strength
        
    def calculate_performance_metrics(
        self,
        prices: np.ndarray,
        exit_points: List[Dict[str, Any]],
        entry_price: float,
        direction: str = "long",
        risk_free_rate: float = 0.0,
    ) -> Dict[str, float]:
        """
        Calculate performance metrics for a trading strategy based on detected exit points.
        
        Args:
            prices: Array of price data
            exit_points: List of exit points (peaks or troughs)
            entry_price: Entry price for the trade
            direction: Trade direction ('long' or 'short')
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            
        Returns:
            Dictionary of performance metrics
        """
        if not exit_points or len(prices) == 0:
            logger.warning("No exit points or prices provided for performance metrics calculation")
            return {}
        
        # Calculate returns based on exit points
        returns = []
        
        for exit_point in exit_points:
            exit_price = exit_point["price"]
            
            if direction.lower() == "long":
                # For long positions, return = (exit_price - entry_price) / entry_price
                ret = (exit_price - entry_price) / entry_price
            else:
                # For short positions, return = (entry_price - exit_price) / entry_price
                ret = (entry_price - exit_price) / entry_price
                
            returns.append(ret)
        
        # Use metrics_utils to calculate trading metrics
        metrics = calculate_trading_metrics(
            returns=returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=252  # Assuming daily data
        )
        
        logger.info(f"Calculated performance metrics: win_rate={metrics['win_rate']:.2f}, "
                   f"profit_factor={metrics['profit_factor']:.2f}, "
                   f"expectancy={metrics['expectancy']:.4f}")
        
        return metrics
    
    def save_analysis(self, analysis: Dict[str, Any], file_path: str) -> None:
        """
        Save price structure analysis to a file.
        
        Args:
            analysis: Analysis results from analyze_price_structure
            file_path: Path to save the analysis
        """
        try:
            save_json(analysis, file_path, indent=2)
            logger.info(f"Saved price structure analysis to {file_path}")
        except Exception as e:
            logger.error(f"Error saving price structure analysis: {e}")
            raise
    
    def load_analysis(self, file_path: str) -> Dict[str, Any]:
        """
        Load price structure analysis from a file.
        
        Args:
            file_path: Path to load the analysis from
            
        Returns:
            Analysis results
        """
        try:
            analysis = load_json(file_path)
            logger.info(f"Loaded price structure analysis from {file_path}")
            return analysis
        except Exception as e:
            logger.error(f"Error loading price structure analysis: {e}")
            raise
