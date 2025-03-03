import logging
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from bs4 import BeautifulSoup

class DataCollector:
    """Data collector for NSE/BSE markets with historical and real-time capabilities."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cache = {}
        self.api_key = config.get('data_api_key')
        
    def fetch_data(self, symbols: List[str], 
                  timeframe: str = 'day', 
                  start_date: Optional[str] = None, 
                  end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch market data for the given symbols and timeframe.
        
        Args:
            symbols: List of stock symbols
            timeframe: Time interval ('1m', '5m', '15m', '30m', '1h', 'day', 'week')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary mapping symbols to their respective DataFrames
        """
        if timeframe == 'day':
            return self.fetch_historical_data(symbols, start_date, end_date)
        else:
            return self.fetch_intraday_data(symbols, timeframe)
    
    def fetch_historical_data(self, symbols: List[str], 
                             start_date: Optional[str] = None, 
                             end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Fetch historical daily data for the given symbols."""
        result = {}
        
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            # Default to 1 year of data
            start = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365))
            start_date = start.strftime('%Y-%m-%d')
            
        for symbol in symbols:
            try:
                # Cache key
                cache_key = f"{symbol}_{start_date}_{end_date}_day"
                
                # Check if data is in cache
                if cache_key in self.cache:
                    result[symbol] = self.cache[cache_key]
                    continue
                
                # Construct API URL based on config
                if self.config.get('data_source') == 'nse':
                    data = self._fetch_nse_data(symbol, start_date, end_date)
                elif self.config.get('data_source') == 'bse':
                    data = self._fetch_bse_data(symbol, start_date, end_date)
                else:
                    # Use a generic data provider as fallback
                    data = self._fetch_generic_data(symbol, start_date, end_date)
                
                # Store in cache
                self.cache[cache_key] = data
                result[symbol] = data
                
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
                
        return result
    
    def fetch_intraday_data(self, symbols: List[str], 
                           timeframe: str) -> Dict[str, pd.DataFrame]:
        """Fetch intraday data for the given symbols and timeframe."""
        result = {}
        
        for symbol in symbols:
            try:
                # For real implementation, this would connect to streaming API
                # or fetch the latest bars based on the timeframe
                if self.config.get('data_source') == 'nse':
                    data = self._fetch_nse_intraday(symbol, timeframe)
                elif self.config.get('data_source') == 'bse':
                    data = self._fetch_bse_intraday(symbol, timeframe)
                else:
                    data = self._fetch_generic_intraday(symbol, timeframe)
                    
                result[symbol] = data
                
            except Exception as e:
                self.logger.error(f"Error fetching intraday data for {symbol}: {str(e)}")
                
        return result
    
    def _fetch_nse_data(self, symbol: str, 
                       start_date: str, 
                       end_date: str) -> pd.DataFrame:
        """Fetch historical data from NSE."""
        # In a real implementation, this would use the NSE API
        # For demo, create synthetic data
        return self._create_synthetic_data(symbol, start_date, end_date)
    
    def _fetch_bse_data(self, symbol: str, 
                       start_date: str, 
                       end_date: str) -> pd.DataFrame:
        """Fetch historical data from BSE."""
        # In a real implementation, this would use the BSE API
        # For demo, create synthetic data
        return self._create_synthetic_data(symbol, start_date, end_date)
    
    def _fetch_generic_data(self, symbol: str, 
                           start_date: str, 
                           end_date: str) -> pd.DataFrame:
        """Fetch historical data from a generic provider."""
        # In a real implementation, this would use a data provider API
        # For demo, create synthetic data
        return self._create_synthetic_data(symbol, start_date, end_date)
    
    def _fetch_nse_intraday(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch intraday data from NSE."""
        # In a real implementation, this would use the NSE API
        # For demo, create synthetic data
        return self._create_synthetic_intraday_data(symbol, timeframe)
    
    def _fetch_bse_intraday(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch intraday data from BSE."""
        # In a real implementation, this would use the BSE API
        # For demo, create synthetic data
        return self._create_synthetic_intraday_data(symbol, timeframe)
    
    def _fetch_generic_intraday(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch intraday data from a generic provider."""
        # In a real implementation, this would use a data provider API
        # For demo, create synthetic data
        return self._create_synthetic_intraday_data(symbol, timeframe)
    
    def _create_synthetic_data(self, symbol: str, 
                              start_date: str, 
                              end_date: str) -> pd.DataFrame:
        """Create synthetic data for testing/demo purposes."""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        days = (end - start).days + 1
        
        # Create date range
        dates = [start + timedelta(days=i) for i in range(days)]
        
        # Filter out weekends
        dates = [date for date in dates if date.weekday() < 5]
        
        # Generate synthetic prices
        base_price = 1000 + hash(symbol) % 9000  # Different base price for each symbol
        price_factor = 0.01  # Daily volatility factor
        
        prices = [base_price]
        for i in range(1, len(dates)):
            # Random walk with drift
            change = prices[-1] * price_factor * (np.random.randn() + 0.001)
            new_price = max(prices[-1] + change, 1)  # Ensure price doesn't go below 1
            prices.append(new_price)
            
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * (1 + abs(0.01 * np.random.randn())) for p in prices],
            'low': [p * (1 - abs(0.01 * np.random.randn())) for p in prices],
            'close': [p * (1 + 0.001 * np.random.randn()) for p in prices],
            'volume': [int(1000000 * abs(np.random.randn())) for _ in prices]
        })
        
        # Ensure high > low
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df
    
    def _create_synthetic_intraday_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Create synthetic intraday data for testing/demo purposes."""
        # Parse timeframe
        if timeframe.endswith('m'):
            minutes = int(timeframe[:-1])
        elif timeframe.endswith('h'):
            minutes = int(timeframe[:-1]) * 60
        else:
            minutes = 1
            
        # Market hours: 9:15 AM to 3:30 PM (375 minutes)
        intervals = 375 // minutes
        
        # Current date
        today = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
        
        # Generate timestamps
        timestamps = [today + timedelta(minutes=i*minutes) for i in range(intervals)]
        
        # Generate synthetic prices similar to historical data
        base_price = 1000 + hash(symbol) % 9000
        price_factor = 0.002  # Higher volatility for intraday
        
        prices = [base_price]
        for i in range(1, len(timestamps)):
            change = prices[-1] * price_factor * np.random.randn()
            new_price = max(prices[-1] + change, 1)
            prices.append(new_price)
            
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * (1 + abs(0.003 * np.random.randn())) for p in prices],
            'low': [p * (1 - abs(0.003 * np.random.randn())) for p in prices],
            'close': [p * (1 + 0.001 * np.random.randn()) for p in prices],
            'volume': [int(100000 * abs(np.random.randn())) for _ in prices]
        })
        
        # Ensure high > low
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df