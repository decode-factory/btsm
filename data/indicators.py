import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List
import logging

class Indicators:
    """Technical indicators calculation class."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def sma(self, series: pd.Series, window: int) -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Args:
            series: Price series (typically close prices)
            window: Window size for calculation
        
        Returns:
            Series containing SMA values
        """
        return series.rolling(window=window).mean()
    
    def ema(self, series: pd.Series, window: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            series: Price series (typically close prices)
            window: Window size for calculation
        
        Returns:
            Series containing EMA values
        """
        return series.ewm(span=window, adjust=False).mean()
    
    def macd(self, series: pd.Series, 
            fast_period: int = 12, 
            slow_period: int = 26, 
            signal_period: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            series: Price series (typically close prices)
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
        
        Returns:
            Dictionary containing MACD line, signal line, and histogram
        """
        fast_ema = self.ema(series, fast_period)
        slow_ema = self.ema(series, slow_period)
        
        macd_line = fast_ema - slow_ema
        signal_line = self.ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            series: Price series (typically close prices)
            window: Window size for calculation
        
        Returns:
            Series containing RSI values
        """
        # Calculate price changes
        delta = series.diff()
        
        # Separate gains and losses
        gain = delta.copy()
        loss = delta.copy()
        
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = -loss  # Convert to positive values
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def bollinger_bands(self, series: pd.Series, 
                       window: int = 20, 
                       num_std: float = 2.0) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            series: Price series (typically close prices)
            window: Window size for calculation
            num_std: Number of standard deviations for the bands
        
        Returns:
            Dictionary containing upper, middle, and lower bands
        """
        middle = self.sma(series, window)
        std_dev = series.rolling(window=window).std()
        
        upper = middle + (std_dev * num_std)
        lower = middle - (std_dev * num_std)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    
    def atr(self, high: pd.Series, 
           low: pd.Series, 
           close: pd.Series, 
           window: int = 14) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Window size for calculation
        
        Returns:
            Series containing ATR values
        """
        # Calculate True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        tr = pd.DataFrame({
            'tr1': tr1,
            'tr2': tr2,
            'tr3': tr3
        }).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=window).mean()
        
        return atr