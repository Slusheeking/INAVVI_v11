"""
Signal generation for the Autonomous Trading System.

This module provides utilities for generating trading signals,
including entry signals and peak detection.
"""

from src.trading_strategy.signals.entry_signal_generator import (
    EntrySignalGenerator,
    generate_entry_signals,
    filter_signals,
    rank_signals,
)

from src.trading_strategy.signals.peak_detector import (
    PeakDetector,
    detect_peaks,
    detect_troughs,
    calculate_peak_metrics,
    filter_peaks,
)

__all__ = [
    "EntrySignalGenerator",
    "generate_entry_signals",
    "filter_signals",
    "rank_signals",
    "PeakDetector",
    "detect_peaks",
    "detect_troughs",
    "calculate_peak_metrics",
    "filter_peaks",
]