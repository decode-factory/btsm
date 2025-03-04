import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union
import logging
from .indicators import Indicators

class DataProcessor:
    """Process market data and calculate technical indicators."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.indicators = Indicators()
        
    def process(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Process market data for all symbols.
        
        Args:
            data: Dictionary mapping symbols to their respective DataFrames
            
        Returns:
            Processed data with added indicators
        """
        result = {}
        
        for symbol, df in data.items():
            try:
                # Process individual DataFrame
                processed_df = self.process_dataframe(df)
                result[symbol] = processed_df
            except Exception as e:
                self.logger.error(f"Error processing data for {symbol}: {str(e)}")
                # Keep original data
                result[symbol] = df
                
        return result
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a single DataFrame, adding technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Processed DataFrame with indicators
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Standardize column names
        if 'date' in result.columns and 'timestamp' not in result.columns:
            result['timestamp'] = result['date']
            
        # Sort by timestamp
        result = result.sort_values('timestamp')
        
        # Calculate technical indicators
        try:
            # Moving Averages - add all variations needed for strategies
            # Standard SMA periods
            result['sma_5'] = self.indicators.sma(result['close'], 5)
            result['sma_10'] = self.indicators.sma(result['close'], 10)
            result['sma_15'] = self.indicators.sma(result['close'], 15)
            result['sma_20'] = self.indicators.sma(result['close'], 20)
            result['sma_30'] = self.indicators.sma(result['close'], 30)
            result['sma_40'] = self.indicators.sma(result['close'], 40)
            result['sma_50'] = self.indicators.sma(result['close'], 50)
            result['sma_60'] = self.indicators.sma(result['close'], 60)
            result['sma_80'] = self.indicators.sma(result['close'], 80)
            result['sma_100'] = self.indicators.sma(result['close'], 100)
            result['sma_120'] = self.indicators.sma(result['close'], 120)
            result['sma_140'] = self.indicators.sma(result['close'], 140)
            result['sma_160'] = self.indicators.sma(result['close'], 160)
            result['sma_180'] = self.indicators.sma(result['close'], 180)
            result['sma_200'] = self.indicators.sma(result['close'], 200)
            
            # Standard EMA periods
            result['ema_5'] = self.indicators.ema(result['close'], 5)
            result['ema_10'] = self.indicators.ema(result['close'], 10)
            result['ema_15'] = self.indicators.ema(result['close'], 15)
            result['ema_20'] = self.indicators.ema(result['close'], 20)
            result['ema_30'] = self.indicators.ema(result['close'], 30)
            result['ema_40'] = self.indicators.ema(result['close'], 40)
            result['ema_50'] = self.indicators.ema(result['close'], 50)
            result['ema_60'] = self.indicators.ema(result['close'], 60)
            result['ema_80'] = self.indicators.ema(result['close'], 80)
            result['ema_100'] = self.indicators.ema(result['close'], 100)
            result['ema_120'] = self.indicators.ema(result['close'], 120)
            result['ema_140'] = self.indicators.ema(result['close'], 140)
            result['ema_160'] = self.indicators.ema(result['close'], 160)
            result['ema_180'] = self.indicators.ema(result['close'], 180)
            result['ema_200'] = self.indicators.ema(result['close'], 200)
            
            # MACD standard inputs
            result['ema_12'] = self.indicators.ema(result['close'], 12)
            result['ema_26'] = self.indicators.ema(result['close'], 26)
            
            # MACD
            macd_data = self.indicators.macd(result['close'])
            result['macd'] = macd_data['macd']
            result['macd_signal'] = macd_data['signal']
            result['macd_hist'] = macd_data['histogram']
            
            # RSI periods
            result['rsi_7'] = self.indicators.rsi(result['close'], 7)
            result['rsi_10'] = self.indicators.rsi(result['close'], 10)
            result['rsi_13'] = self.indicators.rsi(result['close'], 13)
            result['rsi_14'] = self.indicators.rsi(result['close'], 14)
            result['rsi_16'] = self.indicators.rsi(result['close'], 16)
            result['rsi_19'] = self.indicators.rsi(result['close'], 19)
            
            # Bollinger Bands
            bb_data = self.indicators.bollinger_bands(result['close'])
            result['bb_upper'] = bb_data['upper']
            result['bb_middle'] = bb_data['middle']
            result['bb_lower'] = bb_data['lower']
            
            # Additional calculations
            # Returns
            result['returns'] = result['close'].pct_change() * 100
            result['log_returns'] = np.log(result['close'] / result['close'].shift(1)) * 100
            
            # Volatility (20-day rolling standard deviation of returns)
            result['volatility_20'] = result['returns'].rolling(window=20).std()
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            
        return result