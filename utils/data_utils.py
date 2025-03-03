# data_utils.py
import pandas as pd
import numpy as np
import os
import json
import logging
from typing import Dict, List, Any, Union, Optional, Tuple
from datetime import datetime, timedelta
import requests
from io import StringIO
import hashlib

def resample_ohlc(df: pd.DataFrame, 
                timeframe: str, 
                date_column: str = 'timestamp',
                open_column: str = 'open',
                high_column: str = 'high',
                low_column: str = 'low',
                close_column: str = 'close',
                volume_column: str = 'volume') -> pd.DataFrame:
    """
    Resample OHLCV data to a different timeframe.
    
    Args:
        df: DataFrame with OHLCV data
        timeframe: Target timeframe (e.g., '1h', '4h', '1d')
        date_column: Column name for datetime values
        open_column: Column name for open prices
        high_column: Column name for high prices
        low_column: Column name for low prices
        close_column: Column name for close prices
        volume_column: Column name for volume
        
    Returns:
        Resampled DataFrame
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Ensure date column is datetime type
    df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    
    # Set date column as index
    df_copy = df_copy.set_index(date_column)
    
    # Map timeframe string to pandas offset
    timeframe_map = {
        '1m': '1min',
        '5m': '5min',
        '15m': '15min',
        '30m': '30min',
        '1h': '1H',
        '4h': '4H',
        '1d': 'D',
        'day': 'D',
        '1w': 'W',
        'week': 'W',
        '1M': 'M',
        'month': 'M'
    }
    
    resample_rule = timeframe_map.get(timeframe, timeframe)
    
    # Define aggregation functions
    agg_dict = {
        open_column: 'first',
        high_column: 'max',
        low_column: 'min',
        close_column: 'last'
    }
    
    # Add volume if it exists
    if volume_column in df_copy.columns:
        agg_dict[volume_column] = 'sum'
    
    # Resample data
    resampled = df_copy.resample(resample_rule).agg(agg_dict)
    
    # Reset index to convert back to column
    resampled = resampled.reset_index()
    
    return resampled

def merge_dataframes_by_time(dfs: List[pd.DataFrame], 
                           date_column: str = 'timestamp',
                           how: str = 'outer') -> pd.DataFrame:
    """
    Merge multiple DataFrames by time column.
    
    Args:
        dfs: List of DataFrames to merge
        date_column: Column name for datetime values
        how: Type of merge ('inner', 'outer', 'left', 'right')
        
    Returns:
        Merged DataFrame
    """
    if not dfs:
        return pd.DataFrame()
    
    # Make copies to avoid modifying originals
    dfs_copy = [df.copy() for df in dfs]
    
    # Ensure date columns are datetime type and set as index
    for i, df in enumerate(dfs_copy):
        df[date_column] = pd.to_datetime(df[date_column])
        dfs_copy[i] = df.set_index(date_column)
    
    # Merge DataFrames
    result = pd.concat(dfs_copy, axis=1, join=how)
    
    # Reset index to convert back to column
    result = result.reset_index()
    
    return result

def load_csv_data(file_path: str, date_column: Optional[str] = None) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        date_column: Optional column name to parse as dates
        
    Returns:
        DataFrame with loaded data
    """
    try:
        if date_column:
            df = pd.read_csv(file_path, parse_dates=[date_column])
        else:
            df = pd.read_csv(file_path)
        
        return df
    
    except Exception as e:
        logging.error(f"Error loading CSV data from {file_path}: {str(e)}")
        return pd.DataFrame()

def save_csv_data(df: pd.DataFrame, file_path: str, index: bool = False) -> bool:
    """
    Save DataFrame to a CSV file.
    
    Args:
        df: DataFrame to save
        file_path: Path to save the CSV file
        index: Whether to include the index in the CSV
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(file_path, index=index)
        
        return True
    
    except Exception as e:
        logging.error(f"Error saving CSV data to {file_path}: {str(e)}")
        return False

def load_json_data(file_path: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary with loaded data
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return data
    
    except Exception as e:
        logging.error(f"Error loading JSON data from {file_path}: {str(e)}")
        return {}

def save_json_data(data: Dict[str, Any], file_path: str, indent: int = 2) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save the JSON file
        indent: Indentation level for pretty printing
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save to JSON
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
        
        return True
    
    except Exception as e:
        logging.error(f"Error saving JSON data to {file_path}: {str(e)}")
        return False

def download_csv_data(url: str, date_column: Optional[str] = None) -> pd.DataFrame:
    """
    Download CSV data from a URL.
    
    Args:
        url: URL to download the CSV from
        date_column: Optional column name to parse as dates
        
    Returns:
        DataFrame with downloaded data
    """
    try:
        # Download CSV data
        response = requests.get(url)
        response.raise_for_status()  # Raise error for 4xx/5xx responses
        
        # Parse CSV data
        csv_data = StringIO(response.text)
        
        if date_column:
            df = pd.read_csv(csv_data, parse_dates=[date_column])
        else:
            df = pd.read_csv(csv_data)
        
        return df
    
    except Exception as e:
        logging.error(f"Error downloading CSV data from {url}: {str(e)}")
        return pd.DataFrame()

def get_cached_data(key: str, cache_dir: str = 'data/cache') -> Optional[Dict[str, Any]]:
    """
    Get data from cache if available and not expired.
    
    Args:
        key: Cache key
        cache_dir: Directory to store cache files
        
    Returns:
        Cached data or None if not available
    """
    try:
        # Create cache key hash
        key_hash = hashlib.md5(key.encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"{key_hash}.json")
        
        # Check if cache file exists
        if not os.path.exists(cache_file):
            return None
        
        # Load cache data
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        
        # Check if cache is expired
        expiry = cache_data.get('expiry')
        if expiry and datetime.now().timestamp() > expiry:
            # Cache expired
            return None
        
        return cache_data.get('data')
    
    except Exception as e:
        logging.error(f"Error reading from cache for key {key}: {str(e)}")
        return None

def save_to_cache(key: str, data: Dict[str, Any], expiry_seconds: int = 3600, cache_dir: str = 'data/cache') -> bool:
    """
    Save data to cache with expiry time.
    
    Args:
        key: Cache key
        data: Data to cache
        expiry_seconds: Cache expiry time in seconds
        cache_dir: Directory to store cache files
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create cache key hash
        key_hash = hashlib.md5(key.encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"{key_hash}.json")
        
        # Calculate expiry timestamp
        expiry = (datetime.now() + timedelta(seconds=expiry_seconds)).timestamp()
        
        # Create cache entry
        cache_entry = {
            'key': key,
            'expiry': expiry,
            'timestamp': datetime.now().timestamp(),
            'data': data
        }
        
        # Save to file
        with open(cache_file, 'w') as f:
            json.dump(cache_entry, f)
        
        return True
    
    except Exception as e:
        logging.error(f"Error saving to cache for key {key}: {str(e)}")
        return False

def clean_cache(cache_dir: str = 'data/cache') -> int:
    """
    Clean expired cache entries.
    
    Args:
        cache_dir: Directory storing cache files
        
    Returns:
        Number of cache entries deleted
    """
    try:
        if not os.path.exists(cache_dir):
            return 0
        
        # Get current timestamp
        current_time = datetime.now().timestamp()
        
        # Count deleted files
        deleted_count = 0
        
        # Iterate through cache files
        for filename in os.listdir(cache_dir):
            if not filename.endswith('.json'):
                continue
            
            cache_file = os.path.join(cache_dir, filename)
            
            try:
                # Load cache data
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Check if cache is expired
                expiry = cache_data.get('expiry')
                if expiry and current_time > expiry:
                    # Delete expired cache file
                    os.remove(cache_file)
                    deleted_count += 1
            
            except Exception as e:
                logging.warning(f"Error processing cache file {filename}: {str(e)}")
                continue
        
        return deleted_count
    
    except Exception as e:
        logging.error(f"Error cleaning cache: {str(e)}")
        return 0