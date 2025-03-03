# logger.py
import os
import logging
import sys
from typing import Optional

def configure_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """
    Configure logging with specified level and optional file output.
    
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
    
    # Log Python version and environment info
    logger.debug(f"Python version: {sys.version}")
    logger.debug(f"Running on: {sys.platform}")

def get_logger(name: str, log_level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with the specified name and optional level override.
    
    Args:
        name: Logger name (typically __name__)
        log_level: Optional logging level override
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    if log_level:
        numeric_level = getattr(logging, log_level.upper(), None)
        if isinstance(numeric_level, int):
            logger.setLevel(numeric_level)
    
    return logger