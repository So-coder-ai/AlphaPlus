"""
Technical Analysis Feature Engineering for Crypto Trading

This module provides comprehensive feature engineering functions for cryptocurrency
trading data, including technical indicators, momentum measures, and volume analysis.
All functions are designed to work with pandas DataFrames containing OHLCV data.

Author: AlphaPulse Team
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Optional


def klines_to_dataframe(klines: list) -> pd.DataFrame:
    """
    Convert Binance kline data to a clean pandas DataFrame.
    
    Args:
        klines (list): Raw kline data from Binance API
        
    Returns:
        pd.DataFrame: DataFrame with OHLCV data indexed by timestamp
        
    Example:
        >>> klines = client.get_klines("BTCUSDT", "1h", 100)
        >>> df = klines_to_dataframe(klines)
        >>> print(df.head())
    """
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(klines, columns=columns)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    return df[["open", "high", "low", "close", "volume"]]


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log returns and percentage changes for price data.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with added 'log_return' and 'pct_change' columns
        
    Note:
        - Log returns are useful for statistical modeling
        - Percentage changes are more interpretable for humans
    """
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["pct_change"] = df["close"].pct_change()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Relative Strength Index (RSI) momentum indicator.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        period (int): Lookback period for RSI calculation. Default: 14
        
    Returns:
        pd.DataFrame: DataFrame with added 'rsi' column
        
    Note:
        RSI values range from 0 to 100:
        - Above 70: Overbought condition
        - Below 30: Oversold condition
        - Around 50: Neutral momentum
    """
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence) indicator.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        fast (int): Fast EMA period. Default: 12
        slow (int): Slow EMA period. Default: 26
        signal (int): Signal line EMA period. Default: 9
        
    Returns:
        pd.DataFrame: DataFrame with 'macd', 'macd_signal', and 'macd_hist' columns
        
    Note:
        - MACD line: Fast EMA - Slow EMA
        - Signal line: EMA of MACD line
        - Histogram: MACD line - Signal line
    """
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    df["bb_upper"] = sma + std_dev * std
    df["bb_lower"] = sma - std_dev * std
    df["bb_mid"] = sma
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (sma + 1e-9)
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
    return df


def add_ema_crossover(df: pd.DataFrame, short: int = 9, long_: int = 21) -> pd.DataFrame:
    df[f"ema_{short}"] = df["close"].ewm(span=short, adjust=False).mean()
    df[f"ema_{long_}"] = df["close"].ewm(span=long_, adjust=False).mean()
    df["ema_cross"] = (df[f"ema_{short}"] - df[f"ema_{long_}"]) / (df["close"] + 1e-9)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    h_l = df["high"] - df["low"]
    h_pc = (df["high"] - df["close"].shift(1)).abs()
    l_pc = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(period).mean()
    df["atr_pct"] = df["atr"] / (df["close"] + 1e-9)
    return df


def add_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-9)
    df["stoch_d"] = df["stoch_k"].rolling(d_period).mean()
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    direction = np.sign(df["close"].diff()).fillna(0)
    df["obv"] = (direction * df["volume"]).cumsum()
    df["obv_change"] = df["obv"].pct_change()
    return df


def add_volume_zscore(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    mean_vol = df["volume"].rolling(period).mean()
    std_vol = df["volume"].rolling(period).std()
    df["volume_zscore"] = (df["volume"] - mean_vol) / (std_vol + 1e-9)
    return df


def add_target(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.0) -> pd.DataFrame:
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    df["target"] = (future_return > threshold).astype(int)
    return df


def build_features(
    df: pd.DataFrame,
    target_horizon: int = 1,
    drop_na: bool = True,
    include_target: bool = True,
) -> pd.DataFrame:
    """
    Build comprehensive feature set from OHLCV data for machine learning.
    
    This is the main feature engineering pipeline that applies all technical
    indicators and prepares the data for ML model training.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        target_horizon (int): Number of periods ahead for target variable. Default: 1
        drop_na (bool): Whether to drop rows with NaN values. Default: True
        include_target (bool): Whether to include target column. Default: True
        
    Returns:
        pd.DataFrame: DataFrame with all engineered features
        
    Features Included:
        - Returns: log_return, pct_change
        - Momentum: rsi, macd, macd_signal, macd_hist
        - Trend: bb_width, bb_position, ema_cross
        - Volatility: atr_pct
        - Oscillators: stoch_k, stoch_d
        - Volume: obv_change, volume_zscore
        - Target: binary direction (if include_target=True)
        
    Example:
        >>> df_raw = klines_to_dataframe(klines)
        >>> df_features = build_features(df_raw, target_horizon=1)
        >>> X = df_features[FEATURE_COLS]
        >>> y = df_features['target']
    """
    df = df.copy()
    df = add_returns(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_ema_crossover(df)
    df = add_atr(df)
    df = add_stochastic(df)
    df = add_obv(df)
    df = add_volume_zscore(df)
    if include_target:
        df = add_target(df, horizon=target_horizon)
    if drop_na:
        df.dropna(inplace=True)
    return df


FEATURE_COLS = [
    "log_return", "pct_change",
    "rsi",
    "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_position",
    "ema_cross",
    "atr_pct",
    "stoch_k", "stoch_d",
    "obv_change",
    "volume_zscore",
]
