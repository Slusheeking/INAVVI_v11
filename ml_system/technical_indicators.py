#!/usr/bin/env python3
"""
Technical Indicators Module
Contains functions for calculating various technical indicators used in stock analysis.
"""

import numpy as np


def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    if len(data) < period:
        return np.array([np.nan] * len(data))

    alpha = 2.0 / (period + 1)
    ema = np.zeros_like(data)
    ema[period-1] = np.mean(data[:period])

    for i in range(period, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]

    return ema


def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    if len(data) < slow_period + signal_period:
        return np.array([np.nan] * len(data)), np.array([np.nan] * len(data)), np.array([np.nan] * len(data))

    # Calculate fast and slow EMAs
    fast_ema = calculate_ema(data, fast_period)
    slow_ema = calculate_ema(data, slow_period)

    # Calculate MACD line
    macd_line = fast_ema - slow_ema

    # Calculate signal line (EMA of MACD line)
    signal_line = calculate_ema(macd_line, signal_period)

    # Calculate histogram (MACD line - signal line)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_bollinger_bands(data, period=20, num_std=2):
    """Calculate Bollinger Bands"""
    if len(data) < period:
        return np.array([np.nan] * len(data)), np.array([np.nan] * len(data)), np.array([np.nan] * len(data))

    # Calculate rolling mean (middle band)
    rolling_mean = np.array([np.nan] * len(data))
    for i in range(period-1, len(data)):
        rolling_mean[i] = np.mean(data[i-period+1:i+1])

    # Calculate rolling standard deviation
    rolling_std = np.array([np.nan] * len(data))
    for i in range(period-1, len(data)):
        rolling_std[i] = np.std(data[i-period+1:i+1])

    # Calculate upper and lower bands
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)

    return upper_band, rolling_mean, lower_band


def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)"""
    if len(close) < period + 1:
        return np.array([np.nan] * len(close))

    # Calculate True Range (TR)
    tr = np.zeros(len(close))
    for i in range(1, len(close)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)

    # Calculate +DM and -DM
    plus_dm = np.zeros(len(close))
    minus_dm = np.zeros(len(close))

    for i in range(1, len(close)):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0

        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0

    # Calculate smoothed TR, +DM, and -DM
    smoothed_tr = np.zeros(len(close))
    smoothed_plus_dm = np.zeros(len(close))
    smoothed_minus_dm = np.zeros(len(close))

    # Initialize with simple averages
    smoothed_tr[period] = np.sum(tr[1:period+1])
    smoothed_plus_dm[period] = np.sum(plus_dm[1:period+1])
    smoothed_minus_dm[period] = np.sum(minus_dm[1:period+1])

    # Calculate smoothed values
    for i in range(period+1, len(close)):
        smoothed_tr[i] = smoothed_tr[i-1] - (smoothed_tr[i-1] / period) + tr[i]
        smoothed_plus_dm[i] = smoothed_plus_dm[i-1] - \
            (smoothed_plus_dm[i-1] / period) + plus_dm[i]
        smoothed_minus_dm[i] = smoothed_minus_dm[i-1] - \
            (smoothed_minus_dm[i-1] / period) + minus_dm[i]

    # Calculate +DI and -DI
    # Add small epsilon (1e-8) to prevent division by zero
    plus_di = 100 * (smoothed_plus_dm / (smoothed_tr + 1e-8))
    minus_di = 100 * (smoothed_minus_dm / (smoothed_tr + 1e-8))

    # Calculate DX and ADX
    # Add small epsilon (1e-8) to prevent division by zero
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)

    # Calculate ADX (smoothed DX)
    adx = np.zeros(len(close))
    adx[2*period-1] = np.mean(dx[period:2*period])

    for i in range(2*period, len(close)):
        adx[i] = ((adx[i-1] * (period-1)) + dx[i]) / period

    return adx


def calculate_obv(close, volume):
    """Calculate On-Balance Volume (OBV)"""
    obv = np.zeros(len(close))

    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]

    return obv
