# broker_interface.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging
import pandas as pd
from datetime import datetime, time
import pytz

class BrokerInterface(ABC):
    """Base abstract class for broker interfaces."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"broker.{name}")
        self.connected = False
        self.available_symbols = set()
        
        # Get timezone from config or default to Asia/Kolkata (IST)
        tz_str = config.get('timezone', 'Asia/Kolkata')
        self.timezone = pytz.timezone(tz_str)
        
        # Market timing (default: NSE market hours)
        self.market_open = time(9, 15)  # 9:15 AM
        self.market_close = time(15, 30)  # 3:30 PM
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the broker API.
        
        Returns:
            True if connection is successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the broker API.
        
        Returns:
            True if disconnection is successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dictionary with account details
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get currently open positions.
        
        Returns:
            List of position dictionaries
        """
        pass
    
    @abstractmethod
    def get_orders(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get orders with optional status filter.
        
        Args:
            status: Optional order status filter
            
        Returns:
            List of order dictionaries
        """
        pass
    
    @abstractmethod
    def place_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Place a new order.
        
        Args:
            order_params: Order parameters
            
        Returns:
            Dictionary with order details
        """
        pass
    
    @abstractmethod
    def modify_order(self, order_id: str, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modify an existing order.
        
        Args:
            order_id: Order ID to modify
            order_params: Updated order parameters
            
        Returns:
            Dictionary with order details
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation is successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_historical_data(self, symbol: str, 
                          timeframe: str, 
                          start_date: str, 
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get historical price data.
        
        Args:
            symbol: Instrument symbol
            timeframe: Timeframe (e.g., '1m', '5m', '1h', 'day')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (optional)
            
        Returns:
            DataFrame with historical data
        """
        pass
    
    def execute_order(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trading signal by converting it to an order.
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            Dictionary with order details and execution status
        """
        # Validate signal
        required_fields = ['symbol', 'action', 'price']
        if not all(field in signal for field in required_fields):
            raise ValueError(f"Signal missing required fields: {required_fields}")
        
        # Check if market is open
        if not self.is_market_open():
            self.logger.warning(f"Market is closed, cannot execute order for {signal['symbol']}")
            return {
                'status': 'rejected',
                'reason': 'Market closed',
                'signal': signal
            }
        
        # Convert signal to order parameters
        order_params = self._signal_to_order(signal)
        
        # Place the order
        try:
            order_result = self.place_order(order_params)
            
            return {
                'status': 'executed',
                'order': order_result,
                'signal': signal
            }
        except Exception as e:
            self.logger.error(f"Error executing order: {str(e)}")
            
            return {
                'status': 'failed',
                'reason': str(e),
                'signal': signal
            }
    
    def _signal_to_order(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert trading signal to order parameters.
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            Dictionary with order parameters
        """
        # Base order parameters
        order_params = {
            'symbol': signal['symbol'],
            'order_type': signal.get('order_type', 'MARKET'),
            'quantity': signal.get('quantity', 1),
            'price': signal.get('price'),
            'trigger_price': signal.get('trigger_price'),
            'disclosed_quantity': signal.get('disclosed_quantity'),
            'validity': signal.get('validity', 'DAY')
        }
        
        # Set buy/sell direction
        if signal['action'] == 'BUY':
            order_params['transaction_type'] = 'BUY'
        elif signal['action'] == 'SELL':
            order_params['transaction_type'] = 'SELL'
        else:
            raise ValueError(f"Invalid action in signal: {signal['action']}")
        
        # Add stop loss and take profit if available
        if 'stop_loss' in signal:
            order_params['stop_loss'] = signal['stop_loss']
        
        if 'take_profit' in signal:
            order_params['take_profit'] = signal['take_profit']
        
        return order_params
    
    def is_market_open(self) -> bool:
        """
        Check if the market is currently open.
        
        Returns:
            True if the market is open, False otherwise
        """
        # Get current time in configured timezone
        now = datetime.now(self.timezone)
        current_time = now.time()
        
        # Check if today is a weekday (0=Monday, 6=Sunday)
        is_weekday = now.weekday() < 5
        
        # Check if current time is within market hours
        is_trading_hours = self.market_open <= current_time <= self.market_close
        
        # Additional check for market holidays could be implemented here
        
        return is_weekday and is_trading_hours
    
    def get_market_status(self) -> Dict[str, Any]:
        """
        Get current market status.
        
        Returns:
            Dictionary with market status information
        """
        now = datetime.now(self.timezone)
        
        return {
            'is_open': self.is_market_open(),
            'current_time': now.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'market_open': self.market_open.strftime('%H:%M'),
            'market_close': self.market_close.strftime('%H:%M'),
            'timezone': str(self.timezone),
            'connected': self.connected
        }