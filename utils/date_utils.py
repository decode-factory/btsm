# date_utils.py
from datetime import datetime, timedelta, date
import pandas as pd
import pytz
from typing import Union, List, Tuple, Optional

def get_market_calendar(year: int, country: str = 'IN') -> List[date]:
    """
    Get trading calendar for a specific market.
    
    Args:
        year: Calendar year
        country: Country code ('IN' for India, 'US' for United States)
        
    Returns:
        List of trading days
    """
    # For real implementation, this would use exchange-specific calendar data
    # For now, we'll create a synthetic calendar excluding weekends and common holidays
    
    # Start with all days in the year
    start_date = date(year, 1, 1)
    end_date = date(year, 12, 31)
    
    all_days = []
    current_date = start_date
    
    while current_date <= end_date:
        all_days.append(current_date)
        current_date += timedelta(days=1)
    
    # Filter out weekends
    trading_days = [d for d in all_days if d.weekday() < 5]  # 0-4 are Monday to Friday
    
    # Filter out common holidays
    holidays = get_market_holidays(year, country)
    trading_days = [d for d in trading_days if d not in holidays]
    
    return trading_days

def get_market_holidays(year: int, country: str = 'IN') -> List[date]:
    """
    Get list of market holidays for a given year and country.
    
    Args:
        year: Calendar year
        country: Country code ('IN' for India, 'US' for United States)
        
    Returns:
        List of holiday dates
    """
    holidays = []
    
    if country == 'IN':
        # Common Indian market holidays
        holidays = [
            date(year, 1, 26),  # Republic Day
            date(year, 8, 15),  # Independence Day
            date(year, 10, 2),  # Gandhi Jayanti
            date(year, 12, 25),  # Christmas
        ]
        
        # Add other holidays based on year
        # Note: This is not comprehensive - a real implementation would use exchange data
        if year == 2023:
            holidays.extend([
                date(2023, 1, 26),  # Republic Day
                date(2023, 3, 7),   # Holi
                date(2023, 4, 7),   # Good Friday
                date(2023, 4, 14),  # Dr. Ambedkar Jayanti
                date(2023, 4, 22),  # Eid-ul-Fitr
                date(2023, 5, 1),   # Maharashtra Day
                date(2023, 6, 29),  # Bakri Eid
                date(2023, 8, 15),  # Independence Day
                date(2023, 10, 2),  # Gandhi Jayanti
                date(2023, 10, 24), # Diwali
                date(2023, 11, 14), # Gurunanak Jayanti
                date(2023, 12, 25), # Christmas
            ])
    
    elif country == 'US':
        # Common US market holidays
        holidays = [
            date(year, 1, 1),   # New Year's Day
            date(year, 12, 25), # Christmas
        ]
        
        # Add other holidays based on year
        # Note: This is not comprehensive - a real implementation would use exchange data
        if year == 2023:
            holidays.extend([
                date(2023, 1, 2),   # New Year's Day (observed)
                date(2023, 1, 16),  # Martin Luther King, Jr. Day
                date(2023, 2, 20),  # Presidents' Day
                date(2023, 4, 7),   # Good Friday
                date(2023, 5, 29),  # Memorial Day
                date(2023, 6, 19),  # Juneteenth
                date(2023, 7, 4),   # Independence Day
                date(2023, 9, 4),   # Labor Day
                date(2023, 11, 23), # Thanksgiving
                date(2023, 12, 25), # Christmas
            ])
    
    return holidays

def get_next_trading_day(current_date: Union[str, date, datetime], country: str = 'IN') -> date:
    """
    Get the next trading day from the given date.
    
    Args:
        current_date: Current date
        country: Country code for the market
        
    Returns:
        Next trading day
    """
    # Convert string to date if needed
    if isinstance(current_date, str):
        current_date = pd.to_datetime(current_date).date()
    elif isinstance(current_date, datetime):
        current_date = current_date.date()
    
    # Start with the next day
    next_date = current_date + timedelta(days=1)
    
    # Get the year's trading calendar
    year = next_date.year
    trading_days = get_market_calendar(year, country)
    
    # Find the next date in the trading calendar
    while next_date not in trading_days:
        next_date += timedelta(days=1)
        
        # If we crossed into the next year, get that year's calendar
        if next_date.year > year:
            year = next_date.year
            trading_days = get_market_calendar(year, country)
    
    return next_date

def get_previous_trading_day(current_date: Union[str, date, datetime], country: str = 'IN') -> date:
    """
    Get the previous trading day from the given date.
    
    Args:
        current_date: Current date
        country: Country code for the market
        
    Returns:
        Previous trading day
    """
    # Convert string to date if needed
    if isinstance(current_date, str):
        current_date = pd.to_datetime(current_date).date()
    elif isinstance(current_date, datetime):
        current_date = current_date.date()
    
    # Start with the previous day
    prev_date = current_date - timedelta(days=1)
    
    # Get the year's trading calendar
    year = prev_date.year
    trading_days = get_market_calendar(year, country)
    
    # Find the previous date in the trading calendar
    while prev_date not in trading_days:
        prev_date -= timedelta(days=1)
        
        # If we crossed into the previous year, get that year's calendar
        if prev_date.year < year:
            year = prev_date.year
            trading_days = get_market_calendar(year, country)
    
    return prev_date

def convert_timezone(dt: datetime, from_tz: str, to_tz: str) -> datetime:
    """
    Convert datetime from one timezone to another.
    
    Args:
        dt: Datetime to convert
        from_tz: Source timezone
        to_tz: Target timezone
        
    Returns:
        Converted datetime
    """
    # Make sure dt is timezone-aware
    if dt.tzinfo is None:
        dt = pytz.timezone(from_tz).localize(dt)
    else:
        # If it already has a timezone, convert it to the from_tz
        dt = dt.astimezone(pytz.timezone(from_tz))
    
    # Convert to the target timezone
    return dt.astimezone(pytz.timezone(to_tz))

def is_market_open(dt: Optional[datetime] = None, country: str = 'IN', exchange: str = 'NSE') -> bool:
    """
    Check if the market is currently open.
    
    Args:
        dt: Datetime to check (defaults to current time)
        country: Country code for the market
        exchange: Exchange code
        
    Returns:
        True if market is open, False otherwise
    """
    # Default to current time
    if dt is None:
        dt = datetime.now(pytz.timezone('UTC'))
    
    # Make sure dt is timezone-aware
    if dt.tzinfo is None:
        dt = pytz.timezone('UTC').localize(dt)
    
    # Convert to the market's timezone
    if country == 'IN':
        market_tz = 'Asia/Kolkata'
    elif country == 'US':
        market_tz = 'America/New_York'
    else:
        market_tz = 'UTC'
    
    market_dt = dt.astimezone(pytz.timezone(market_tz))
    
    # Check if it's a trading day
    if market_dt.date() not in get_market_calendar(market_dt.year, country):
        return False
    
    # Check if it's during trading hours
    if country == 'IN' and exchange == 'NSE':
        # NSE trading hours: 9:15 AM to 3:30 PM IST
        market_open = market_dt.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = market_dt.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= market_dt <= market_close
    
    elif country == 'US':
        # US market hours: 9:30 AM to 4:00 PM EST/EDT
        market_open = market_dt.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = market_dt.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= market_dt <= market_close
    
    # Default fallback
    return False

def get_time_to_market_open(dt: Optional[datetime] = None, country: str = 'IN', exchange: str = 'NSE') -> Optional[timedelta]:
    """
    Get time remaining until market opens.
    
    Args:
        dt: Datetime to check from (defaults to current time)
        country: Country code for the market
        exchange: Exchange code
        
    Returns:
        Timedelta until market open, or None if market is already open or won't open today
    """
    # Default to current time
    if dt is None:
        dt = datetime.now(pytz.timezone('UTC'))
    
    # Make sure dt is timezone-aware
    if dt.tzinfo is None:
        dt = pytz.timezone('UTC').localize(dt)
    
    # Convert to the market's timezone
    if country == 'IN':
        market_tz = 'Asia/Kolkata'
    elif country == 'US':
        market_tz = 'America/New_York'
    else:
        market_tz = 'UTC'
    
    market_dt = dt.astimezone(pytz.timezone(market_tz))
    
    # Check if it's a trading day
    if market_dt.date() not in get_market_calendar(market_dt.year, country):
        next_trading_day = get_next_trading_day(market_dt.date(), country)
        # Set time to market open on the next trading day
        if country == 'IN' and exchange == 'NSE':
            next_market_open = datetime.combine(next_trading_day, datetime.min.time()).replace(hour=9, minute=15)
        elif country == 'US':
            next_market_open = datetime.combine(next_trading_day, datetime.min.time()).replace(hour=9, minute=30)
        else:
            next_market_open = datetime.combine(next_trading_day, datetime.min.time())
        
        next_market_open = pytz.timezone(market_tz).localize(next_market_open)
        return next_market_open - market_dt
    
    # Check if market is already open
    if is_market_open(market_dt, country, exchange):
        return None
    
    # Check if market has already closed for the day
    if country == 'IN' and exchange == 'NSE':
        market_open = market_dt.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = market_dt.replace(hour=15, minute=30, second=0, microsecond=0)
        
        if market_dt > market_close:
            # Market closed, calculate time to next market open
            next_trading_day = get_next_trading_day(market_dt.date(), country)
            next_market_open = datetime.combine(next_trading_day, datetime.min.time()).replace(hour=9, minute=15)
            next_market_open = pytz.timezone(market_tz).localize(next_market_open)
            return next_market_open - market_dt
        else:
            # Market not yet open today
            return market_open - market_dt
    
    elif country == 'US':
        market_open = market_dt.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = market_dt.replace(hour=16, minute=0, second=0, microsecond=0)
        
        if market_dt > market_close:
            # Market closed, calculate time to next market open
            next_trading_day = get_next_trading_day(market_dt.date(), country)
            next_market_open = datetime.combine(next_trading_day, datetime.min.time()).replace(hour=9, minute=30)
            next_market_open = pytz.timezone(market_tz).localize(next_market_open)
            return next_market_open - market_dt
        else:
            # Market not yet open today
            return market_open - market_dt
    
    # Default fallback
    return None

def get_time_to_market_close(dt: Optional[datetime] = None, country: str = 'IN', exchange: str = 'NSE') -> Optional[timedelta]:
    """
    Get time remaining until market closes.
    
    Args:
        dt: Datetime to check from (defaults to current time)
        country: Country code for the market
        exchange: Exchange code
        
    Returns:
        Timedelta until market close, or None if market is already closed or won't open today
    """
    # Default to current time
    if dt is None:
        dt = datetime.now(pytz.timezone('UTC'))
    
    # Make sure dt is timezone-aware
    if dt.tzinfo is None:
        dt = pytz.timezone('UTC').localize(dt)
    
    # Convert to the market's timezone
    if country == 'IN':
        market_tz = 'Asia/Kolkata'
    elif country == 'US':
        market_tz = 'America/New_York'
    else:
        market_tz = 'UTC'
    
    market_dt = dt.astimezone(pytz.timezone(market_tz))
    
    # Check if it's a trading day
    if market_dt.date() not in get_market_calendar(market_dt.year, country):
        return None
    
    # Check if market is open
    if not is_market_open(market_dt, country, exchange):
        return None
    
    # Calculate time to market close
    if country == 'IN' and exchange == 'NSE':
        market_close = market_dt.replace(hour=15, minute=30, second=0, microsecond=0)
        return market_close - market_dt
    
    elif country == 'US':
        market_close = market_dt.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_close - market_dt
    
    # Default fallback
    return None