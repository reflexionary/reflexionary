"""
Fi-MCP (Financial Market Connector Protocol) Configuration

This module contains configuration for the Fi-MCP integration, which provides
access to financial market data, including real-time prices, historical data,
and other market information.
"""

import os
from typing import Dict, Any, Optional, List, Union
from datetime import time, timedelta

# --- API Connection Settings ---
FI_MCP_SIMULATOR_BASE_URL: str = os.getenv('FI_MCP_SIMULATOR_BASE_URL', 'http://localhost:5000')
FI_MCP_SIMULATOR_API_KEY: str = os.getenv('FI_MCP_SIMULATOR_API_KEY', '')

# --- Rate Limiting ---
RATE_LIMIT_THRESHOLD: int = int(os.getenv('FI_MCP_RATE_LIMIT_THRESHOLD', '1000'))  # Max requests per minute
RATE_LIMIT_WINDOW: int = 60  # seconds
RATE_LIMIT_BACKOFF_FACTOR: float = 1.5  # Exponential backoff factor
MAX_RETRIES: int = 3

# --- Timeout Settings ---
REQUEST_TIMEOUT: int = 30  # seconds
CONNECT_TIMEOUT: int = 10  # seconds
READ_TIMEOUT: int = 30  # seconds

# --- Data Refresh Intervals (in seconds) ---
REFRESH_INTERVALS: Dict[str, int] = {
    'realtime': 5,  # Real-time data refresh interval
    'intraday': 60,  # Intraday data refresh interval
    'eod': 3600,  # End-of-day data refresh interval
    'fundamentals': 86400,  # Fundamental data refresh interval (daily)
}

# --- Market Data Configuration ---
SUPPORTED_MARKETS: List[str] = ['NSE', 'BSE', 'NFO', 'MCX', 'NCDEX']
DEFAULT_MARKET: str = 'NSE'

# Trading Hours (in IST)
TRADING_HOURS: Dict[str, Dict[str, time]] = {
    'NSE': {
        'pre_open_start': time(9, 0),    # 9:00 AM
        'pre_open_end': time(9, 15),     # 9:15 AM
        'market_open': time(9, 15),      # 9:15 AM
        'market_close': time(15, 30),    # 3:30 PM
        'post_market_end': time(16, 0),  # 4:00 PM
    },
    'BSE': {
        'pre_open_start': time(9, 0),    # 9:00 AM
        'pre_open_end': time(9, 15),     # 9:15 AM
        'market_open': time(9, 15),      # 9:15 AM
        'market_close': time(15, 30),    # 3:30 PM
        'post_market_end': time(16, 0),  # 4:00 PM
    },
    # Add other markets as needed
}

# --- Data Retention Policy ---
DATA_RETENTION_DAYS: Dict[str, int] = {
    'tick_data': 7,         # 7 days of tick data
    'minute_data': 30,      # 30 days of 1-minute data
    'hourly_data': 90,      # 90 days of hourly data
    'daily_data': 365 * 3,  # 3 years of daily data
    'fundamentals': 0,      # Keep all fundamental data (0 = unlimited)
}

# --- Mock Data Configuration ---
MOCK_DATA_PATHS: Dict[str, str] = {
    'historical': 'data/mock_historical_prices_cache.csv',
    'fi_mcp': 'data/mock_fi_mcp_data_cache.json',
    'corporate_actions': 'data/mock_corporate_actions.json',
    'fundamentals': 'data/mock_fundamentals.json',
}

# --- Cache Configuration ---
CACHE_CONFIG: Dict[str, Any] = {
    'enabled': True,
    'ttl': 300,  # 5 minutes
    'max_size': 1000,
    'strategy': 'lru',  # Least Recently Used
}

# --- API Endpoints ---
ENDPOINTS: Dict[str, str] = {
    'auth': '/api/v1/auth',
    'quotes': '/api/v1/market/quote',
    'historical': '/api/v1/market/historical',
    'intraday': '/api/v1/market/intraday',
    'search': '/api/v1/search',
    'portfolio': '/api/v1/portfolio',
    'orders': '/api/v1/orders',
    'fundamentals': '/api/v1/fundamentals',
}

# --- Logging Configuration ---
LOG_CONFIG: Dict[str, Any] = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/fi_mcp.log',
    'max_size': 10485760,  # 10MB
    'backup_count': 5,
}

# --- WebSocket Configuration ---
WEBSOCKET_CONFIG: Dict[str, Any] = {
    'enabled': True,
    'url': FI_MCP_SIMULATOR_BASE_URL.replace('http', 'ws') + '/ws',
    'ping_interval': 30,  # seconds
    'ping_timeout': 10,   # seconds
    'reconnect_attempts': 5,
    'reconnect_delay': 5,  # seconds
}

# --- Instrument Mappings ---
INSTRUMENT_TYPES: Dict[str, List[str]] = {
    'equity': ['EQ', 'BE', 'SM', 'ST', 'TT'],
    'derivatives': ['FUT', 'OPT', 'FUTIDX', 'OPTIDX', 'FUTSTK', 'OPTSTK'],
    'debt': ['TB', 'SD', 'TB', 'CB', 'GILT', 'CP', 'TB', 'TBILL', 'SDI', 'CD', 'CP', 'TB', 'TBILL'],
    'mf': ['MF', 'MFSS'],
    'others': ['REPO', 'GOLD', 'SILVER', 'COMMODITY', 'CURRENCY'],
}

# --- Error Configuration ---
ERROR_CODES: Dict[int, str] = {
    400: 'Bad Request',
    401: 'Unauthorized',
    403: 'Forbidden',
    404: 'Not Found',
    429: 'Too Many Requests',
    500: 'Internal Server Error',
    502: 'Bad Gateway',
    503: 'Service Unavailable',
    504: 'Gateway Timeout',
}

def get_market_hours(market: str = 'NSE') -> Dict[str, time]:
    """
    Get the trading hours for a specific market.
    
    Args:
        market: Market identifier (e.g., 'NSE', 'BSE')
        
    Returns:
        Dict containing market hours as time objects
    """
    return TRADING_HOURS.get(market.upper(), TRADING_HOURS[DEFAULT_MARKET])

def is_market_open(market: str = 'NSE') -> bool:
    """
    Check if the specified market is currently open for trading.
    
    Args:
        market: Market identifier (e.g., 'NSE', 'BSE')
        
    Returns:
        bool: True if market is open, False otherwise
    """
    from datetime import datetime, time
    
    market_hours = get_market_hours(market)
    now = datetime.now().time()
    
    # Check if current time is within market hours
    return (market_hours['market_open'] <= now <= market_hours['market_close'])

def get_endpoint_url(endpoint_key: str) -> str:
    """
    Get the full URL for an API endpoint.
    
    Args:
        endpoint_key: Key from the ENDPOINTS dictionary
        
    Returns:
        str: Full URL for the endpoint
    """
    if endpoint_key not in ENDPOINTS:
        raise ValueError(f"Unknown endpoint: {endpoint_key}")
    
    base_url = FI_MCP_SIMULATOR_BASE_URL.rstrip('/')
    endpoint = ENDPOINTS[endpoint_key].lstrip('/')
    return f"{base_url}/{endpoint}"

def get_headers() -> Dict[str, str]:
    """
    Get the default headers for API requests.
    
    Returns:
        Dict containing the request headers
    """
    return {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {FI_MCP_SIMULATOR_API_KEY}',
        'User-Agent': 'Reflexary/1.0',
    }
