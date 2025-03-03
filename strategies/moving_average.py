# strategies/moving_average.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from .base import Strategy

class MovingAverageCrossover(Strategy):
    """
    Moving Average Crossover Strategy.
    
    This strategy generates buy signals when the faster moving average crosses above
    the slower moving average, and generates sell signals when the faster moving
    average crosses below the slower moving average.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'fast_ma_type': 'SMA',    # 'SMA' or 'EMA'
            'fast_ma_period': 20,     # Period for fast moving average
            'slow_ma_type': 'SMA',    # 'SMA' or 'EMA'
            'slow_ma_period': 50,     # Period for slow moving average
            'position_size_pct': 5,    # Percentage of capital per position
            'stop_loss_pct': 5,        # Stop loss percentage
            'take_profit_pct': 10      # Take profit percentage
        }
        
        # Merge default parameters with provided parameters
        merged_params = default_params.copy()
        if params:
            merged_params.update(params)
            
        super().__init__('MA_Crossover', merged_params)
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on moving average crossovers.
        
        Args:
            data: Dictionary mapping symbols to their respective DataFrames
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        for symbol, df in data.items():
            # Check if we have enough data
            required_periods = max(self.params['fast_ma_period'], self.params['slow_ma_period']) + 5
            if len(df) < required_periods:
                self.logger.warning(f"Not enough data for {symbol}, skipping signal generation")
                continue
                
            # Get the latest data point
            current = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Determine column names based on parameters
            fast_col = f"{'ema' if self.params['fast_ma_type'] == 'EMA' else 'sma'}_{self.params['fast_ma_period']}"
            slow_col = f"{'ema' if self.params['slow_ma_type'] == 'EMA' else 'sma'}_{self.params['slow_ma_period']}"
            
            # Check if required indicators are available
            if fast_col not in df.columns or slow_col not in df.columns:
                self.logger.warning(f"Required indicators not found for {symbol}")
                continue
                
            # Check for crossovers
            current_fast = current[fast_col]
            current_slow = current[slow_col]
            previous_fast = previous[fast_col]
            previous_slow = previous[slow_col]
            
            # Buy signal: Fast MA crosses above Slow MA
            if previous_fast <= previous_slow and current_fast > current_slow:
                # Calculate position size based on current price
                quantity = self._calculate_position_size(current['close'])
                
                signal = {
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': current['close'],
                    'quantity': quantity,
                    'timestamp': current['timestamp'],
                    'reason': f"{self.params['fast_ma_type']}{self.params['fast_ma_period']} crossed above "
                             f"{self.params['slow_ma_type']}{self.params['slow_ma_period']}",
                    'stop_loss': current['close'] * (1 - self.params['stop_loss_pct'] / 100),
                    'take_profit': current['close'] * (1 + self.params['take_profit_pct'] / 100)
                }
                signals.append(signal)
                
            # Sell signal: Fast MA crosses below Slow MA
            elif previous_fast >= previous_slow and current_fast < current_slow:
                # For simplicity, assume we sell the entire position
                signal = {
                    'symbol': symbol,
                    'action': 'SELL',
                    'price': current['close'],
                    'quantity': None,  # Will be determined by current position
                    'timestamp': current['timestamp'],
                    'reason': f"{self.params['fast_ma_type']}{self.params['fast_ma_period']} crossed below "
                             f"{self.params['slow_ma_type']}{self.params['slow_ma_period']}"
                }
                signals.append(signal)
                
        return signals
    
    def _calculate_position_size(self, price: float) -> int:
        """
        Calculate position size based on parameters.
        
        Args:
            price: Current price of the asset
            
        Returns:
            Quantity to buy/sell
        """
        # In a real system, this would access the account equity
        # For demo purposes, assume a fixed equity
        equity = self.params.get('equity', 1000000)  # Default 10 lakh INR
        
        # Calculate position value based on percentage of equity
        position_value = equity * (self.params['position_size_pct'] / 100)
        
        # Calculate quantity (round down to nearest integer)
        quantity = int(position_value / price)
        
        return max(1, quantity)  # Ensure at least 1 share