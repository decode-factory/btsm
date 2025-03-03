# strategies/rsi_strategy.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from .base import Strategy

class RSIStrategy(Strategy):
    """
    RSI-based Strategy for oversold/overbought conditions.
    
    This strategy generates buy signals when RSI falls below the oversold threshold,
    and generates sell signals when RSI rises above the overbought threshold.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'rsi_period': 14,         # Period for RSI calculation
            'oversold_threshold': 30,  # RSI threshold for oversold condition
            'overbought_threshold': 70, # RSI threshold for overbought condition
            'position_size_pct': 5,    # Percentage of capital per position
            'stop_loss_pct': 5,        # Stop loss percentage
            'take_profit_pct': 10      # Take profit percentage
        }
        
        # Merge default parameters with provided parameters
        merged_params = default_params.copy()
        if params:
            merged_params.update(params)
            
        super().__init__('RSI_Strategy', merged_params)
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on RSI values.
        
        Args:
            data: Dictionary mapping symbols to their respective DataFrames
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        for symbol, df in data.items():
            # Check if we have enough data
            required_periods = self.params['rsi_period'] + 5
            if len(df) < required_periods:
                self.logger.warning(f"Not enough data for {symbol}, skipping signal generation")
                continue
                
            # Get the latest data points
            current = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Determine RSI column name
            rsi_col = f"rsi_{self.params['rsi_period']}"
            
            # Check if required indicators are available
            if rsi_col not in df.columns:
                self.logger.warning(f"Required indicator {rsi_col} not found for {symbol}")
                continue
                
            # Get current and previous RSI values
            current_rsi = current[rsi_col]
            previous_rsi = previous[rsi_col]
            
            # Buy signal: RSI crosses above oversold threshold
            if previous_rsi <= self.params['oversold_threshold'] and current_rsi > self.params['oversold_threshold']:
                # Calculate position size based on current price
                quantity = self._calculate_position_size(current['close'])
                
                signal = {
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': current['close'],
                    'quantity': quantity,
                    'timestamp': current['timestamp'],
                    'reason': f"RSI({self.params['rsi_period']}) crossed above oversold threshold {self.params['oversold_threshold']}",
                    'stop_loss': current['close'] * (1 - self.params['stop_loss_pct'] / 100),
                    'take_profit': current['close'] * (1 + self.params['take_profit_pct'] / 100)
                }
                signals.append(signal)
                
            # Sell signal: RSI crosses below overbought threshold
            elif previous_rsi >= self.params['overbought_threshold'] and current_rsi < self.params['overbought_threshold']:
                # For simplicity, assume we sell the entire position
                signal = {
                    'symbol': symbol,
                    'action': 'SELL',
                    'price': current['close'],
                    'quantity': None,  # Will be determined by current position
                    'timestamp': current['timestamp'],
                    'reason': f"RSI({self.params['rsi_period']}) crossed below overbought threshold {self.params['overbought_threshold']}"
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