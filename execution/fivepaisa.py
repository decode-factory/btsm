# fivepaisa.py
from typing import Dict, List, Any, Optional
import logging
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import hashlib
import json
import numpy as np
from .broker_interface import BrokerInterface

class FivePaisaBroker(BrokerInterface):
    """5paisa API implementation of the broker interface."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("5paisa", config)
        
        # API credentials
        self.app_name = config.get('fivepaisa_app_name', '')
        self.client_id = config.get('fivepaisa_client_id', '')
        self.client_secret = config.get('fivepaisa_client_secret', '')
        
        # API URLs
        self.base_url = "https://Openapi.5paisa.com/VendorsAPI/Service1.svc"
        
        # Session and auth tokens
        self.session = requests.Session()
        self.client_code = None
        self.jwt_token = None
        
        # Instrument cache
        self.instruments = {}
    
    def connect(self) -> bool:
        """
        Connect to 5paisa API.
        
        Returns:
            True if connection is successful, False otherwise
        """
        if not all([self.app_name, self.client_id, self.client_secret]):
            self.logger.error("API credentials are missing")
            return False
        
        try:
            # In a real implementation, this would use the 5paisa login flow
            # For demonstration, we're simulating a successful connection
            
            # Simulate successful authentication
            self.client_code = "12345678"
            self.jwt_token = "simulated_jwt_token"
            
            # Set default headers for API requests
            self.session.headers.update({
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.jwt_token}'
            })
            
            # Load instruments
            self._load_instruments()
            
            self.connected = True
            self.logger.info("Connected to 5paisa API")
            return True
            
        except Exception as e:
            self.logger.error(f"Error connecting to 5paisa API: {str(e)}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from 5paisa API.
        
        Returns:
            True if disconnection is successful, False otherwise
        """
        try:
            if self.connected:
                # Clear session data
                self.client_code = None
                self.jwt_token = None
                self.session = requests.Session()
                self.connected = False
                
                self.logger.info("Disconnected from 5paisa API")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from 5paisa API: {str(e)}")
            return False
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dictionary with account details
        """
        if not self.connected:
            raise ConnectionError("Not connected to 5paisa API")
        
        try:
            # In a real implementation, this would call the 5paisa API
            # request_body = {
            #     "head": {
            #         "key": self.client_id
            #     },
            #     "body": {
            #         "ClientCode": self.client_code
            #     }
            # }
            # response = self.session.post(f"{self.base_url}/V2/MarginV3", json=request_body)
            # return response.json()['body']
            
            # Return simulated account info
            return {
                'client_code': self.client_code,
                'name': 'Demo User',
                'email': 'demo@example.com',
                'broker': '5paisa',
                'margin': {
                    'cash': 750000.0,
                    'margin_used': 125000.0,
                    'margin_available': 625000.0,
                    'collateral': 0.0
                },
                'limits': {
                    'equity_delivery': 750000.0,
                    'equity_intraday': 3000000.0,
                    'derivatives': 1500000.0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {str(e)}")
            raise
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get currently open positions.
        
        Returns:
            List of position dictionaries
        """
        if not self.connected:
            raise ConnectionError("Not connected to 5paisa API")
        
        try:
            # In a real implementation, this would call the 5paisa API
            # request_body = {
            #     "head": {
            #         "key": self.client_id
            #     },
            #     "body": {
            #         "ClientCode": self.client_code
            #     }
            # }
            # response = self.session.post(f"{self.base_url}/V3/Holding", json=request_body)
            # return response.json()['body']['EquityHolding']
            
            # Return simulated positions
            return [
                {
                    'symbol': 'BHARTIARTL',
                    'exchange': 'NSE',
                    'isin': 'INE397D01024',
                    'product': 'C',  # CNC (Delivery)
                    'quantity': 15,
                    'average_price': 865.50,
                    'last_price': 880.25,
                    'pnl': 221.25,
                    'day_change': 10.75,
                    'day_change_percentage': 1.24
                },
                {
                    'symbol': 'KOTAKBANK',
                    'exchange': 'NSE',
                    'isin': 'INE237A01028',
                    'product': 'C',  # CNC (Delivery)
                    'quantity': 10,
                    'average_price': 1785.25,
                    'last_price': 1795.50,
                    'pnl': 102.50,
                    'day_change': 5.75,
                    'day_change_percentage': 0.32
                }
            ]
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {str(e)}")
            raise
    
    def get_orders(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get orders with optional status filter.
        
        Args:
            status: Optional order status filter
            
        Returns:
            List of order dictionaries
        """
        if not self.connected:
            raise ConnectionError("Not connected to 5paisa API")
        
        try:
            # In a real implementation, this would call the 5paisa API
            # request_body = {
            #     "head": {
            #         "key": self.client_id
            #     },
            #     "body": {
            #         "ClientCode": self.client_code
            #     }
            # }
            # response = self.session.post(f"{self.base_url}/V2/OrderBook", json=request_body)
            # orders = response.json()['body']['OrderBookDetail']
            
            # Simulate orders
            orders = [
                {
                    'order_id': '5P123456789',
                    'exchange_order_id': 'N123456789',
                    'symbol': 'HDFCBANK',
                    'exchange': 'NSE',
                    'transaction_type': 'B',  # Buy
                    'order_type': 'MKT',  # Market
                    'quantity': 10,
                    'price': 0,
                    'status': 'Executed',
                    'filled_quantity': 10,
                    'average_price': 1650.75,
                    'order_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                },
                {
                    'order_id': '5P123456790',
                    'exchange_order_id': 'N123456790',
                    'symbol': 'LT',
                    'exchange': 'NSE',
                    'transaction_type': 'B',  # Buy
                    'order_type': 'L',  # Limit
                    'quantity': 5,
                    'price': 2750.0,
                    'status': 'Pending',
                    'filled_quantity': 0,
                    'average_price': 0,
                    'order_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            ]
            
            # Filter by status if provided
            if status:
                status_map = {
                    'PENDING': ['pending', 'open', 'after market order req received'],
                    'COMPLETE': ['executed', 'complete', 'filled'],
                    'REJECTED': ['rejected', 'cancelled', 'expired']
                }
                
                # Map generic status to 5paisa statuses
                five_paisa_statuses = []
                for key, values in status_map.items():
                    if status.upper() == key:
                        five_paisa_statuses.extend(values)
                
                if five_paisa_statuses:
                    orders = [order for order in orders if order['status'].lower() in five_paisa_statuses]
            
            return orders
            
        except Exception as e:
            self.logger.error(f"Error getting orders: {str(e)}")
            raise
    
    def place_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Place a new order.
        
        Args:
            order_params: Order parameters
            
        Returns:
            Dictionary with order details
        """
        if not self.connected:
            raise ConnectionError("Not connected to 5paisa API")
        
        required_params = ['symbol', 'transaction_type', 'quantity']
        if not all(param in order_params for param in required_params):
            raise ValueError(f"Missing required parameters: {required_params}")
        
        try:
            # In a real implementation, this would call the 5paisa API
            # Map order parameters to 5paisa format
            # order_type_map = {
            #     'MARKET': 'MKT',
            #     'LIMIT': 'L',
            #     'SL': 'SL',
            #     'SL-M': 'SL-M'
            # }
            
            # transaction_type_map = {
            #     'BUY': 'B',
            #     'SELL': 'S'
            # }
            
            # request_body = {
            #     "head": {
            #         "key": self.client_id
            #     },
            #     "body": {
            #         "ClientCode": self.client_code,
            #         "OrderFor": "P",  # P: Place Order
            #         "Exchange": order_params.get('exchange', 'N'),  # N: NSE
            #         "ExchangeType": "C",  # C: Cash
            #         "Price": order_params.get('price', 0),
            #         "OrderID": 0,  # 0 for new order
            #         "OrderType": order_type_map.get(order_params.get('order_type', 'MARKET'), 'MKT'),
            #         "Qty": order_params['quantity'],
            #         "ScripCode": self._get_scrip_code(order_params['symbol']),
            #         "DisQty": order_params.get('disclosed_quantity', 0),
            #         "StopLossPrice": order_params.get('trigger_price', 0),
            #         "IsStopLossOrder": "N",
            #         "IOCOrder": "N",
            #         "RemoteOrderID": str(int(time.time())),
            #         "ExchOrderID": 0,
            #         "AtMarket": "Y" if order_params.get('order_type', 'MARKET') == 'MARKET' else "N",
            #         "TradePassword": "",
            #         "TradedQty": 0
            #     }
            # }
            
            # response = self.session.post(f"{self.base_url}/V1/PlaceOrderV1", json=request_body)
            # return response.json()['body']
            
            # Generate a simulated order ID
            order_id = f"5P{int(time.time() * 1000) % 10000000000}"
            
            # Map transaction type
            transaction_type_map = {
                'BUY': 'B',
                'SELL': 'S'
            }
            
            # Map order type
            order_type_map = {
                'MARKET': 'MKT',
                'LIMIT': 'L',
                'SL': 'SL',
                'SL-M': 'SL-M'
            }
            
            # Simulate order placement
            order = {
                'order_id': order_id,
                'exchange_order_id': f"N{int(time.time() * 1000) % 1000000}",
                'symbol': order_params['symbol'],
                'exchange': order_params.get('exchange', 'NSE'),
                'transaction_type': transaction_type_map.get(order_params['transaction_type'], 'B'),
                'order_type': order_type_map.get(order_params.get('order_type', 'MARKET'), 'MKT'),
                'quantity': order_params['quantity'],
                'price': order_params.get('price', 0),
                'trigger_price': order_params.get('trigger_price', 0),
                'status': 'Pending',
                'filled_quantity': 0,
                'average_price': 0,
                'order_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.logger.info(f"Order placed: {order_id} for {order_params['symbol']}")
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            raise
    
    def modify_order(self, order_id: str, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modify an existing order.
        
        Args:
            order_id: Order ID to modify
            order_params: Updated order parameters
            
        Returns:
            Dictionary with order details
        """
        if not self.connected:
            raise ConnectionError("Not connected to 5paisa API")
        
        try:
            # In a real implementation, this would call the 5paisa API
            # request_body = {
            #     "head": {
            #         "key": self.client_id
            #     },
            #     "body": {
            #         "ClientCode": self.client_code,
            #         "OrderFor": "M",  # M: Modify Order
            #         "Exchange": order_params.get('exchange', 'N'),  # N: NSE
            #         "ExchangeType": "C",  # C: Cash
            #         "Price": order_params.get('price', 0),
            #         "OrderID": order_id,
            #         "OrderType": order_params.get('order_type', 'LIMIT'),
            #         "Qty": order_params.get('quantity', 0),
            #         "ScripCode": self._get_scrip_code(order_params.get('symbol', '')),
            #         "DisQty": order_params.get('disclosed_quantity', 0),
            #         "StopLossPrice": order_params.get('trigger_price', 0),
            #         "RemoteOrderID": str(int(time.time()))
            #     }
            # }
            
            # response = self.session.post(f"{self.base_url}/V1/ModifyOrderV1", json=request_body)
            # return response.json()['body']
            
            # Simulate order modification
            # In a real implementation, we would first fetch the current order
            current_order = {
                'order_id': order_id,
                'exchange_order_id': 'N123456789',
                'symbol': order_params.get('symbol', 'HDFCBANK'),
                'exchange': 'NSE',
                'transaction_type': 'B',  # Buy
                'order_type': 'L',  # Limit
                'quantity': 10,
                'price': 1650.0,
                'status': 'Pending',
                'filled_quantity': 0,
                'average_price': 0,
                'order_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Update order with new parameters
            if 'quantity' in order_params:
                current_order['quantity'] = order_params['quantity']
                
            if 'price' in order_params:
                current_order['price'] = order_params['price']
                
            if 'trigger_price' in order_params:
                current_order['trigger_price'] = order_params['trigger_price']
            
            # Update timestamp
            current_order['order_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            self.logger.info(f"Order modified: {order_id}")
            
            return current_order
            
        except Exception as e:
            self.logger.error(f"Error modifying order: {str(e)}")
            raise
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation is successful, False otherwise
        """
        if not self.connected:
            raise ConnectionError("Not connected to 5paisa API")
        
        try:
            # In a real implementation, this would call the 5paisa API
            # request_body = {
            #     "head": {
            #         "key": self.client_id
            #     },
            #     "body": {
            #         "ClientCode": self.client_code,
            #         "OrderFor": "C",  # C: Cancel Order
            #         "Exchange": "N",  # N: NSE
            #         "ExchangeType": "C",  # C: Cash
            #         "OrderID": order_id,
            #         "RemoteOrderID": str(int(time.time()))
            #     }
            # }
            
            # response = self.session.post(f"{self.base_url}/V1/CancelOrderV1", json=request_body)
            # return response.json()['body']['Status'] == 0
            
            # Simulate successful cancellation
            self.logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order: {str(e)}")
            raise
    
    def get_historical_data(self, symbol: str, 
                          timeframe: str, 
                          start_date: str, 
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get historical price data from 5paisa.
        
        Args:
            symbol: Instrument symbol
            timeframe: Timeframe (e.g., '1m', '5m', '1h', 'day')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (optional)
            
        Returns:
            DataFrame with historical data
        """
        if not self.connected:
            raise ConnectionError("Not connected to 5paisa API")
        
        try:
            # Validate timeframe
            valid_timeframes = ['1m', '5m', '15m', '30m', '60m', 'day', 'week']
            if timeframe not in valid_timeframes:
                raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {valid_timeframes}")
            
            # Map timeframe to 5paisa format
            timeframe_map = {
                '1m': '1',
                '5m': '5',
                '15m': '15',
                '30m': '30',
                '60m': '60',
                'day': 'D',
                'week': 'W'
            }
            five_paisa_timeframe = timeframe_map[timeframe]
            
            # Get scrip code
            scrip_code = self._get_scrip_code(symbol)
            if not scrip_code:
                raise ValueError(f"Invalid symbol: {symbol}")
            
            # Parse dates
            from_date = datetime.strptime(start_date, '%Y-%m-%d')
            to_date = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
            
            # In a real implementation, this would call the 5paisa API
            # request_body = {
            #     "head": {
            #         "key": self.client_id
            #     },
            #     "body": {
            #         "ClientCode": self.client_code,
            #         "FromDate": from_date.strftime('%Y-%m-%d'),
            #         "ToDate": to_date.strftime('%Y-%m-%d'),
            #         "Time": five_paisa_timeframe,
            #         "ScripCode": scrip_code,
            #         "Exchange": "N",  # N: NSE
            #         "ExchangeType": "C"  # C: Cash
            #     }
            # }
            
            # response = self.session.post(f"{self.base_url}/V2/HistoricalData", json=request_body)
            # data = response.json()['body']['History']
            
            # Generate synthetic data for demonstration
            data = self._generate_synthetic_data(symbol, from_date, to_date, timeframe)
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting historical data: {str(e)}")
            raise
    
    def _load_instruments(self) -> None:
        """Load instrument details from 5paisa."""
        try:
            # In a real implementation, this would call the 5paisa API
            # request_body = {
            #     "head": {
            #         "key": self.client_id
            #     },
            #     "body": {
            #         "Exchange": "N"  # N: NSE
            #     }
            # }
            
            # response = self.session.post(f"{self.base_url}/V2/MarketFeed", json=request_body)
            # instruments_data = response.json()['body']
            
            # For demonstration, we'll use a small set of instruments
            instruments_data = [
                {"ScripCode": 500325, "Symbol": "RELIANCE", "Name": "Reliance Industries Limited", "Series": "EQ", "Exchange": "NSE"},
                {"ScripCode": 500180, "Symbol": "HDFCBANK", "Name": "HDFC Bank Limited", "Series": "EQ", "Exchange": "NSE"},
                {"ScripCode": 532540, "Symbol": "TCS", "Name": "Tata Consultancy Services Limited", "Series": "EQ", "Exchange": "NSE"},
                {"ScripCode": 500209, "Symbol": "INFY", "Name": "Infosys Limited", "Series": "EQ", "Exchange": "NSE"},
                {"ScripCode": 500875, "Symbol": "ITC", "Name": "ITC Limited", "Series": "EQ", "Exchange": "NSE"},
                {"ScripCode": 532174, "Symbol": "ICICIBANK", "Name": "ICICI Bank Limited", "Series": "EQ", "Exchange": "NSE"},
                {"ScripCode": 500258, "Symbol": "BHARTIARTL", "Name": "Bharti Airtel Limited", "Series": "EQ", "Exchange": "NSE"},
                {"ScripCode": 500112, "Symbol": "SBIN", "Name": "State Bank of India", "Series": "EQ", "Exchange": "NSE"},
                {"ScripCode": 500247, "Symbol": "KOTAKBANK", "Name": "Kotak Mahindra Bank Limited", "Series": "EQ", "Exchange": "NSE"},
                {"ScripCode": 500510, "Symbol": "LT", "Name": "Larsen & Toubro Limited", "Series": "EQ", "Exchange": "NSE"}
            ]
            
            # Process instrument data
            for instrument in instruments_data:
                symbol = instrument['Symbol']
                self.instruments[symbol] = instrument
                self.available_symbols.add(symbol)
            
            self.logger.info(f"Loaded {len(self.instruments)} instruments")
            
        except Exception as e:
            self.logger.error(f"Error loading instruments: {str(e)}")
    
    def _get_scrip_code(self, symbol: str) -> Optional[int]:
        """Get scrip code for a symbol."""
        if symbol in self.instruments:
            return self.instruments[symbol]['ScripCode']
        return None
    
    def _generate_synthetic_data(self, symbol: str, 
                               from_date: datetime, 
                               to_date: datetime, 
                               timeframe: str) -> List[List[Any]]:
        """Generate synthetic OHLCV data for testing."""
        # Determine date range and interval
        if timeframe == 'day':
            # Daily data
            dates = pd.date_range(from_date, to_date, freq='B')  # Business days
            interval_minutes = 24 * 60
        elif timeframe == 'week':
            # Weekly data
            dates = pd.date_range(from_date, to_date, freq='W')
            interval_minutes = 7 * 24 * 60
        else:
            # Intraday data
            minutes_map = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '60m': 60}
            interval_minutes = minutes_map.get(timeframe, 5)
            
            # Generate only during trading hours
            market_hours = []
            current_date = from_date
            while current_date <= to_date:
                if current_date.weekday() < 5:  # Monday to Friday
                    # Trading hours: 9:15 AM to 3:30 PM
                    start_time = current_date.replace(hour=9, minute=15)
                    end_time = current_date.replace(hour=15, minute=30)
                    
                    # Generate timestamps during market hours
                    timestamps = pd.date_range(start_time, end_time, freq=f'{interval_minutes}min')
                    market_hours.extend(timestamps)
                
                current_date += timedelta(days=1)
            
            dates = market_hours
        
        # Symbol-specific base prices
        symbol_prices = {
            'RELIANCE': 2500.0,
            'HDFCBANK': 1650.0,
            'TCS': 3400.0,
            'INFY': 1550.0,
            'ITC': 400.0,
            'ICICIBANK': 950.0,
            'BHARTIARTL': 875.0,
            'SBIN': 550.0,
            'KOTAKBANK': 1800.0,
            'LT': 2800.0
        }
        
        base_price = symbol_prices.get(symbol, 1000.0)
        
        # Generate OHLCV data
        data = []
        volatility = 0.01  # 1% daily volatility
        trend = 0.0002  # Small upward trend
        
        # Adjust volatility based on timeframe
        if timeframe in ['1m', '5m', '15m', '30m', '60m']:
            # Scale volatility for intraday data
            volatility = volatility * (interval_minutes / (24 * 60)) ** 0.5
            trend = trend * (interval_minutes / (24 * 60))
        
        # Random walk
        price = base_price
        
        for date in dates:
            # Calculate price movement
            movement = price * (volatility * np.random.randn() + trend)
            price += movement
            
            # Generate OHLCV
            open_price = price
            high_price = open_price * (1 + abs(0.5 * volatility * np.random.randn()))
            low_price = open_price * (1 - abs(0.5 * volatility * np.random.randn()))
            close_price = (open_price + high_price + low_price + open_price) / 4 + (0.1 * volatility * np.random.randn())
            
            # Ensure high is highest and low is lowest
            high_price = max(open_price, high_price, close_price)
            low_price = min(open_price, low_price, close_price)
            
            # Volume (higher for higher volatility)
            volume = int(1000 * (1 + abs(np.random.randn())))
            
            data.append([date, open_price, high_price, low_price, close_price, volume])
        
        return data