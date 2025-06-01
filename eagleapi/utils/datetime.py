from datetime import datetime
from typing import Optional
import pytz
from eagleapi.core.config import settings

def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a datetime object in a readable way.
    
    Args:
        dt: The datetime object to format
        format_str: The format string to use (default: YYYY-MM-DD HH:MM:SS)
    
    Returns:
        str: The formatted datetime string
    """
    if not dt:
        return ""
    
    # Convert to local timezone if timezone is enabled
    if settings.USE_TIMEZONE:
        local_tz = pytz.timezone(settings.TIMEZONE)
        dt = dt.astimezone(local_tz)
    
    return dt.strftime(format_str)

def get_current_time() -> datetime:
    """
    Get the current time in UTC.
    
    Returns:
        datetime: The current time in UTC
    """
    return datetime.utcnow()

def parse_datetime(dt_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> Optional[datetime]:
    """
    Parse a datetime string into a datetime object.
    
    Args:
        dt_str: The datetime string to parse
        format_str: The format string to use
    
    Returns:
        datetime: The parsed datetime object, or None if parsing fails
    """
    try:
        return datetime.strptime(dt_str, format_str)
    except (ValueError, TypeError):
        return None

def format_time_ago(dt: datetime) -> str:
    """
    Format a datetime as "X time ago".
    
    Args:
        dt: The datetime to format
    
    Returns:
        str: A string like "5 minutes ago", "1 hour ago", etc.
    """
    if not dt:
        return ""
    
    now = get_current_time()
    diff = now - dt
    
    seconds = diff.total_seconds()
    minutes = seconds // 60
    hours = minutes // 60
    days = hours // 24
    
    if seconds < 60:
        return f"{int(seconds)} seconds ago"
    elif minutes < 60:
        return f"{int(minutes)} minutes ago"
    elif hours < 24:
        return f"{int(hours)} hours ago"
    else:
        return f"{int(days)} days ago"
