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
            # Moving Averages
            result['sma_20'] = self.indicators.sma(result['close'], 20)
            result['sma_50'] = self.indicators.sma(result['close'], 50)
            result['sma_200'] = self.indicators.sma(result['close'], 200)
            
            result['ema_12'] = self.indicators.ema(result['close'], 12)
            result['ema_26'] = self.indicators.ema(result['close'], 26)
            
            # MACD
            macd_data = self.indicators.macd(result['close'])
            result['macd'] = macd_data['macd']
            result['macd_signal'] = macd_data['signal']
            result['macd_hist'] = macd_data['histogram']
            
            # RSI
            result['rsi_14'] = self.indicators.rsi(result['close'], 14)
            
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