# paper_trading.py
from typing import Dict, List, Any, Optional
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import uuid
import json
import os
from .broker_interface import BrokerInterface

class PaperTradingBroker(BrokerInterface):
    """Paper trading implementation of the broker interface."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("PaperTrading", config)
        
        # Set initial account balance
        self.initial_balance = float(config.get('paper_trading_initial_balance', 1000000.0))
        self.current_balance = self.initial_balance
        
        # Storage for orders and positions
        self.orders = []
        self.positions = {}
        self.order_history = []
        
        # Slippage and fees settings
        self.slippage_pct = float(config.get('paper_trading_slippage_pct', 0.05))
        self.brokerage_pct = float(config.get('paper_trading_brokerage_pct', 0.05))
        self.min_brokerage = float(config.get('paper_trading_min_brokerage', 20.0))
        self.max_brokerage = float(config.get('paper_trading_max_brokerage', 100.0))
        
        # Data cache for market simulation
        self.market_data = {}
        self.last_prices = {}
        
        # Storage directory
        self.storage_dir = config.get('paper_trading_storage_dir', 'data/paper_trading')
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Load existing state if available
        self._load_state()
    
    def connect(self) -> bool:
        """
        Connect to paper trading system.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            self.connected = True
            self.logger.info("Connected to paper trading system")
            return True
            
        except Exception as e:
            self.logger.error(f"Error connecting to paper trading system: {str(e)}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from paper trading system.
        
        Returns:
            True if disconnection is successful, False otherwise
        """
        try:
            # Save state before disconnecting
            self._save_state()
            
            self.connected = False
            self.logger.info("Disconnected from paper trading system")
            return True
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from paper trading system: {str(e)}")
            return False
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dictionary with account details
        """
        # Calculate equity (cash + positions value)
        positions_value = self._calculate_positions_value()
        equity = self.current_balance + positions_value
        
        # Calculate daily P&L
        daily_pnl = self._calculate_daily_pnl()
        
        return {
            'broker': 'PaperTrading',
            'cash_balance': self.current_balance,
            'positions_value': positions_value,
            'equity': equity,
            'initial_balance': self.initial_balance,
            'total_pnl': equity - self.initial_balance,
            'total_pnl_pct': ((equity / self.initial_balance) - 1) * 100,
            'daily_pnl': daily_pnl,
            'daily_pnl_pct': (daily_pnl / (equity - daily_pnl)) * 100 if (equity - daily_pnl) != 0 else 0,
            'available_margin': self.current_balance
        }
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get currently open positions.
        
        Returns:
            List of position dictionaries
        """
        result = []
        
        for symbol, position in self.positions.items():
            # Get current price
            current_price = self._get_current_price(symbol)
            
            # Calculate P&L
            cost_basis = position['quantity'] * position['average_price']
            market_value = position['quantity'] * current_price
            pnl = market_value - cost_basis
            pnl_pct = (pnl / cost_basis) * 100 if cost_basis != 0 else 0
            
            position_data = {
                'symbol': symbol,
                'exchange': position.get('exchange', 'NSE'),
                'product': position.get('product', 'CNC'),
                'quantity': position['quantity'],
                'average_price': position['average_price'],
                'last_price': current_price,
                'market_value': market_value,
                'cost_basis': cost_basis,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'day_change': position.get('day_change', 0),
                'day_change_percentage': position.get('day_change_percentage', 0)
            }
            
            result.append(position_data)
        
        return result
    
    def get_orders(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get orders with optional status filter.
        
        Args:
            status: Optional order status filter
            
        Returns:
            List of order dictionaries
        """
        if status:
            return [order for order in self.orders if order['status'] == status]
        return self.orders
    
    def place_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Place a new order.
        
        Args:
            order_params: Order parameters
            
        Returns:
            Dictionary with order details
        """
        required_params = ['symbol', 'transaction_type', 'quantity']
        if not all(param in order_params for param in required_params):
            raise ValueError(f"Missing required parameters: {required_params}")
        
        # Generate order ID
        order_id = f"PT{int(time.time())}{uuid.uuid4().hex[:6]}"
        
        # Get current price for market orders
        if order_params.get('order_type', 'MARKET') == 'MARKET':
            price = self._get_current_price(order_params['symbol'])
        else:
            price = order_params.get('price', 0)
        
        # Create order
        order = {
            'order_id': order_id,
            'symbol': order_params['symbol'],
            'exchange': order_params.get('exchange', 'NSE'),
            'transaction_type': order_params['transaction_type'],
            'order_type': order_params.get('order_type', 'MARKET'),
            'quantity': order_params['quantity'],
            'price': price,
            'trigger_price': order_params.get('trigger_price', 0),
            'status': 'PENDING',
            'filled_quantity': 0,
            'average_price': 0,
            'order_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'execution_timestamp': None,
            'product': order_params.get('product', 'CNC')
        }
        
        # Add to orders list
        self.orders.append(order)
        
        # Process order immediately for market orders or based on trigger for limit/stop orders
        if order['order_type'] == 'MARKET':
            self._execute_order(order_id)
        
        self.logger.info(f"Order placed: {order_id} for {order_params['symbol']}")
        
        # Save state after order placement
        self._save_state()
        
        return order
    
    def modify_order(self, order_id: str, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modify an existing order.
        
        Args:
            order_id: Order ID to modify
            order_params: Updated order parameters
            
        Returns:
            Dictionary with order details
        """
        # Find the order
        order_index = next((i for i, order in enumerate(self.orders) if order['order_id'] == order_id), None)
        
        if order_index is None:
            raise ValueError(f"Order not found: {order_id}")
        
        order = self.orders[order_index]
        
        # Check if order can be modified
        if order['status'] not in ['PENDING', 'OPEN']:
            raise ValueError(f"Cannot modify order with status: {order['status']}")
        
        # Update modifiable parameters
        for param in ['quantity', 'price', 'trigger_price', 'order_type']:
            if param in order_params:
                order[param] = order_params[param]
        
        # Update timestamp
        order['order_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save state after order modification
        self._save_state()
        
        self.logger.info(f"Order modified: {order_id}")
        
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation is successful, False otherwise
        """
        # Find the order
        order_index = next((i for i, order in enumerate(self.orders) if order['order_id'] == order_id), None)
        
        if order_index is None:
            raise ValueError(f"Order not found: {order_id}")
        
        order = self.orders[order_index]
        
        # Check if order can be cancelled
        if order['status'] not in ['PENDING', 'OPEN']:
            raise ValueError(f"Cannot cancel order with status: {order['status']}")
        
        # Update order status
        order['status'] = 'CANCELLED'
        
        # Move to order history
        self.order_history.append(order)
        self.orders.pop(order_index)
        
        # Save state after order cancellation
        self._save_state()
        
        self.logger.info(f"Order cancelled: {order_id}")
        
        return True
    
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
        # In paper trading, we need to delegate to a data source
        # This could be a data collector from the data module
        if hasattr(self, 'data_collector') and self.data_collector:
            result = self.data_collector.fetch_historical_data(symbol, start_date, end_date)
            
            # Cache the data for market simulation
            if symbol in result:
                self.market_data[symbol] = result[symbol]
                
                # Update last price
                if not result[symbol].empty:
                    last_row = result[symbol].iloc[-1]
                    self.last_prices[symbol] = last_row['close']
            
            return result.get(symbol, pd.DataFrame())
        else:
            raise NotImplementedError("Historical data access requires a data collector to be set")
    
    def set_data_collector(self, data_collector: Any) -> None:
        """
        Set the data collector for historical data access.
        
        Args:
            data_collector: Data collector instance
        """
        self.data_collector = data_collector
    
    def update_market_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Update market data for paper trading simulation.
        
        Args:
            data: Dictionary of DataFrames with market data
        """
        # Update market data cache
        self.market_data.update(data)
        
        # Update last prices
        for symbol, df in data.items():
            if not df.empty:
                last_row = df.iloc[-1]
                self.last_prices[symbol] = last_row['close']
        
        # Process pending orders based on new prices
        self._process_pending_orders()
    
    def _execute_order(self, order_id: str) -> Dict[str, Any]:
        """
        Execute a pending order.
        
        Args:
            order_id: Order ID to execute
            
        Returns:
            Dictionary with execution details
        """
        # Find the order
        order_index = next((i for i, order in enumerate(self.orders) if order['order_id'] == order_id), None)
        
        if order_index is None:
            raise ValueError(f"Order not found: {order_id}")
        
        order = self.orders[order_index]
        
        # Check if order can be executed
        if order['status'] != 'PENDING':
            return order
        
        # Get current price
        current_price = self._get_current_price(order['symbol'])
        
        # Check if limit/stop conditions are met
        if order['order_type'] == 'LIMIT' and order['transaction_type'] == 'BUY' and current_price > order['price']:
            return order  # Don't execute buy limit order if price is higher
        
        if order['order_type'] == 'LIMIT' and order['transaction_type'] == 'SELL' and current_price < order['price']:
            return order  # Don't execute sell limit order if price is lower
        
        if order['order_type'] == 'SL' and order['transaction_type'] == 'BUY' and current_price < order['trigger_price']:
            return order  # Don't execute buy stop order if price is lower than trigger
        
        if order['order_type'] == 'SL' and order['transaction_type'] == 'SELL' and current_price > order['trigger_price']:
            return order  # Don't execute sell stop order if price is higher than trigger
        
        # Apply slippage
        if order['transaction_type'] == 'BUY':
            execution_price = current_price * (1 + self.slippage_pct / 100)
        else:
            execution_price = current_price * (1 - self.slippage_pct / 100)
        
        # Check if we have enough balance for buy orders
        if order['transaction_type'] == 'BUY':
            order_value = order['quantity'] * execution_price
            brokerage = self._calculate_brokerage(order_value)
            total_cost = order_value + brokerage
            
            if total_cost > self.current_balance:
                # Insufficient funds
                order['status'] = 'REJECTED'
                order['reject_reason'] = 'Insufficient funds'
                
                # Move to order history
                self.order_history.append(order)
                self.orders.pop(order_index)
                
                self.logger.warning(f"Order {order_id} rejected: Insufficient funds")
                return order
        
        # Check if we have enough quantity for sell orders
        if order['transaction_type'] == 'SELL':
            position = self.positions.get(order['symbol'])
            
            if not position or position['quantity'] < order['quantity']:
                # Insufficient position
                order['status'] = 'REJECTED'
                order['reject_reason'] = 'Insufficient position'
                
                # Move to order history
                self.order_history.append(order)
                self.orders.pop(order_index)
                
                self.logger.warning(f"Order {order_id} rejected: Insufficient position")
                return order
        
        # Execute the order
        order['status'] = 'EXECUTED'
        order['filled_quantity'] = order['quantity']
        order['average_price'] = execution_price
        order['execution_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Update positions
        self._update_positions(order)
        
        # Calculate and apply brokerage
        order_value = order['quantity'] * execution_price
        brokerage = self._calculate_brokerage(order_value)
        order['brokerage'] = brokerage
        
        # Update balance
        if order['transaction_type'] == 'BUY':
            self.current_balance -= (order_value + brokerage)
        else:
            self.current_balance += (order_value - brokerage)
        
        # Move to order history
        self.order_history.append(order)
        self.orders.pop(order_index)
        
        self.logger.info(f"Order executed: {order_id} at price {execution_price}")
        
        # Save state after order execution
        self._save_state()
        
        return order
    
    def _update_positions(self, order: Dict[str, Any]) -> None:
        """
        Update positions based on executed order.
        
        Args:
            order: Executed order dictionary
        """
        symbol = order['symbol']
        quantity = order['quantity']
        price = order['average_price']
        
        if order['transaction_type'] == 'BUY':
            if symbol in self.positions:
                # Update existing position with average price
                current_qty = self.positions[symbol]['quantity']
                current_avg_price = self.positions[symbol]['average_price']
                
                new_qty = current_qty + quantity
                new_avg_price = ((current_qty * current_avg_price) + (quantity * price)) / new_qty
                
                self.positions[symbol]['quantity'] = new_qty
                self.positions[symbol]['average_price'] = new_avg_price
            else:
                # Create new position
                self.positions[symbol] = {
                    'symbol': symbol,
                    'exchange': order['exchange'],
                    'product': order['product'],
                    'quantity': quantity,
                    'average_price': price,
                    'buy_date': datetime.now().strftime('%Y-%m-%d')
                }
        
        elif order['transaction_type'] == 'SELL':
            if symbol in self.positions:
                current_qty = self.positions[symbol]['quantity']
                
                # Reduce position quantity
                new_qty = current_qty - quantity
                
                if new_qty <= 0:
                    # Position closed
                    del self.positions[symbol]
                else:
                    # Update position quantity
                    self.positions[symbol]['quantity'] = new_qty
    
    def _process_pending_orders(self) -> None:
        """Process pending orders based on current market prices."""
        # Create a copy of orders list as it may change during iteration
        orders_to_process = self.orders.copy()
        
        for order in orders_to_process:
            if order['status'] == 'PENDING':
                try:
                    self._execute_order(order['order_id'])
                except Exception as e:
                    self.logger.error(f"Error processing order {order['order_id']}: {str(e)}")
    
    def _get_current_price(self, symbol: str) -> float:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Instrument symbol
            
        Returns:
            Current price
        """
        # Use last price if available
        if symbol in self.last_prices:
            return self.last_prices[symbol]
        
        # Otherwise try to get from market data
        if symbol in self.market_data and not self.market_data[symbol].empty:
            last_row = self.market_data[symbol].iloc[-1]
            self.last_prices[symbol] = last_row['close']
            return last_row['close']
        
        # If no data available, raise error
        raise ValueError(f"No price data available for {symbol}")
    
    def _calculate_brokerage(self, order_value: float) -> float:
        """
        Calculate brokerage for an order.
        
        Args:
            order_value: Order value in currency
            
        Returns:
            Brokerage amount
        """
        # Calculate percentage-based brokerage
        brokerage = order_value * (self.brokerage_pct / 100)
        
        # Apply min/max limits
        brokerage = max(brokerage, self.min_brokerage)
        brokerage = min(brokerage, self.max_brokerage)
        
        return brokerage
    
    def _calculate_positions_value(self) -> float:
        """
        Calculate total value of all positions.
        
        Returns:
            Total positions value
        """
        total_value = 0.0
        
        for symbol, position in self.positions.items():
            try:
                current_price = self._get_current_price(symbol)
                position_value = position['quantity'] * current_price
                total_value += position_value
            except Exception as e:
                self.logger.warning(f"Error calculating position value for {symbol}: {str(e)}")
        
        return total_value
    
    def _calculate_daily_pnl(self) -> float:
        """
        Calculate daily profit and loss.
        
        Returns:
            Daily P&L amount
        """
        daily_pnl = 0.0
        
        # Calculate P&L for current positions
        for symbol, position in self.positions.items():
            try:
                current_price = self._get_current_price(symbol)
                previous_price = self._get_previous_day_price(symbol)
                
                if previous_price:
                    # Calculate daily change
                    price_change = current_price - previous_price
                    position_pnl = position['quantity'] * price_change
                    daily_pnl += position_pnl
            except Exception as e:
                self.logger.warning(f"Error calculating daily P&L for {symbol}: {str(e)}")
        
        # Add P&L from today's closed positions
        today = datetime.now().strftime('%Y-%m-%d')
        
        for order in self.order_history:
            if order['status'] == 'EXECUTED' and order['execution_timestamp'].startswith(today):
                if order['transaction_type'] == 'SELL':
                    # Calculate profit from sell orders
                    symbol = order['symbol']
                    sell_value = order['quantity'] * order['average_price']
                    
                    # Find matching buy position if available
                    position = self.positions.get(symbol)
                    if position:
                        buy_price = position['average_price']
                        buy_value = order['quantity'] * buy_price
                        order_pnl = sell_value - buy_value
                        daily_pnl += order_pnl
        
        return daily_pnl
    
    def _get_previous_day_price(self, symbol: str) -> Optional[float]:
        """
        Get previous day's closing price for a symbol.
        
        Args:
            symbol: Instrument symbol
            
        Returns:
            Previous day's price or None if not available
        """
        # If we have market data, get previous day's close
        if symbol in self.market_data and len(self.market_data[symbol]) > 1:
            df = self.market_data[symbol]
            
            # Check if we have daily data
            if 'date' in df.columns:
                today = datetime.now().date()
                yesterday = today - timedelta(days=1)
                
                # Find yesterday's data
                prev_data = df[df['date'].dt.date == yesterday]
                if not prev_data.empty:
                    return prev_data.iloc[-1]['close']
            
            # Otherwise just use the second-to-last row
            if len(df) >= 2:
                return df.iloc[-2]['close']
        
        return None
    
    def _save_state(self) -> None:
        """Save current state to disk."""
        try:
            state = {
                'current_balance': self.current_balance,
                'positions': self.positions,
                'orders': self.orders,
                'order_history': self.order_history[-100:],  # Keep last 100 orders
                'last_prices': self.last_prices,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save to file
            state_file = os.path.join(self.storage_dir, 'paper_trading_state.json')
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.debug("Paper trading state saved")
            
        except Exception as e:
            self.logger.error(f"Error saving paper trading state: {str(e)}")
    
    def _load_state(self) -> None:
        """Load state from disk if available."""
        try:
            state_file = os.path.join(self.storage_dir, 'paper_trading_state.json')
            
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                self.current_balance = state.get('current_balance', self.initial_balance)
                self.positions = state.get('positions', {})
                self.orders = state.get('orders', [])
                self.order_history = state.get('order_history', [])
                self.last_prices = state.get('last_prices', {})
                
                self.logger.info(f"Paper trading state loaded from {state_file}")
                self.logger.info(f"Current balance: {self.current_balance}")
                self.logger.info(f"Active positions: {len(self.positions)}")
                self.logger.info(f"Pending orders: {len(self.orders)}")
            
        except Exception as e:
            self.logger.error(f"Error loading paper trading state: {str(e)}")