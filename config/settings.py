# settings.py
import os
import configparser
from typing import Dict, Any, Optional
import logging
from dotenv import load_dotenv
import json
import sys

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from environment variables and config files.
    
    Args:
        config_path: Path to config INI file (optional)
    
    Returns:
        Dictionary with configuration parameters
    """
    # Load environment variables from .env file
    load_dotenv()
    
    config = {}
    
    # Default config values
    default_config = {
        # Core settings
        'app_name': 'IndiaTrader',
        'log_level': 'INFO',
        'log_file': 'trading.log',
        'timezone': 'Asia/Kolkata',
        
        # Data settings
        'data_source': 'nse',  # 'nse', 'bse', or provider name
        'data_api_key': '',
        'historical_data_path': 'data/historical',
        'cache_enabled': 'true',
        'cache_expiry': '3600',  # seconds
        
        # Trading settings
        'paper_trading': 'true',
        'max_positions': '10',
        'max_position_size_pct': '10',  # percentage of equity
        'default_stop_loss_pct': '3',
        'default_take_profit_pct': '6',
        'max_daily_drawdown_pct': '5',
        
        # API settings
        'zerodha_api_key': '',
        'zerodha_api_secret': '',
        'upstox_api_key': '',
        'upstox_api_secret': '',
        'fivepaisa_app_name': '',
        'fivepaisa_client_id': '',
        'fivepaisa_client_secret': '',
        
        # Analysis settings
        'sentiment_analysis_enabled': 'true',
        'prediction_models_path': 'models',
        'prediction_forecast_days': '5',
        
        # Notification settings
        'notifications_enabled': 'true',
        'email_notifications': 'false',
        'email_from': '',
        'email_to': '',
        'smtp_server': '',
        'smtp_port': '587',
        'smtp_username': '',
        'smtp_password': '',
        
        # Performance settings
        'performance_benchmark': 'NIFTY50',
        'performance_metrics_file': 'reports/performance.json'
    }
    
    # Load default config
    config.update(default_config)
    
    # Load configuration from INI file if provided
    if config_path and os.path.exists(config_path):
        parser = configparser.ConfigParser()
        parser.read(config_path)
        
        # Process each section in the config file
        for section in parser.sections():
            for key, value in parser.items(section):
                config[f"{section}.{key}"] = value
    
    # Override with environment variables
    for key, value in os.environ.items():
        if key.startswith('TRADING_'):
            # Remove prefix and convert to lowercase
            clean_key = key[8:].lower()
            config[clean_key] = value
    
    # Type conversion for boolean and numeric values
    _convert_types(config)
    
    # Set up logging
    setup_logging(config.get('log_level', 'INFO'), config.get('log_file'))
    
    return config

def _convert_types(config: Dict[str, Any]) -> None:
    """
    Convert string values to appropriate types.
    
    Args:
        config: Configuration dictionary to modify in-place
    """
    # Boolean values
    bool_keys = [
        'cache_enabled', 'paper_trading', 'sentiment_analysis_enabled',
        'notifications_enabled', 'email_notifications'
    ]
    
    # Integer values
    int_keys = [
        'cache_expiry', 'max_positions', 'smtp_port'
    ]
    
    # Float values
    float_keys = [
        'max_position_size_pct', 'default_stop_loss_pct', 
        'default_take_profit_pct', 'max_daily_drawdown_pct',
        'prediction_forecast_days'
    ]
    
    # Process boolean values
    for key in bool_keys:
        if key in config:
            value = config[key].lower()
            config[key] = value in ('true', 'yes', '1', 'on')
    
    # Process integer values
    for key in int_keys:
        if key in config:
            try:
                config[key] = int(config[key])
            except ValueError:
                # Keep as string if conversion fails
                pass
    
    # Process float values
    for key in float_keys:
        if key in config:
            try:
                config[key] = float(config[key])
            except ValueError:
                # Keep as string if conversion fails
                pass

def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Path to log file (optional)
    """
    # Convert log level string to numeric value
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    # Basic configuration
    logging_config = {
        'level': numeric_level,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S',
    }
    
    # Add file handler if log file is specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        logging_config['filename'] = log_file
        logging_config['filemode'] = 'a'  # append mode
    
    # Configure logging
    logging.basicConfig(**logging_config)
    
    # Add console handler if logging to file
    if log_file:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(numeric_level)
        formatter = logging.Formatter(logging_config['format'], logging_config['datefmt'])
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at level {log_level}")