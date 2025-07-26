"""
Reflexionary - Financial Intelligence Base Module

This module contains shared utilities and data loaders for the financial intelligence layer.
It serves as the foundation for all quant-related functionality in Reflexionary.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

# Import core components
from config import settings

# Configure logging
import logging
logger = logging.getLogger(__name__)

# Type aliases
Numeric = Union[int, float]
DateLike = Union[str, pd.Timestamp]

# Module-level variables to store loaded data
_HISTORICAL_PRICES_DF: Optional[pd.DataFrame] = None
_FI_MCP_DATA: Optional[Dict[str, Any]] = None


def _load_historical_prices_df() -> pd.DataFrame:
    """
    Load historical price data from the cache file.
    
    Returns:
        pd.DataFrame: DataFrame containing historical price data with columns:
                     - date (datetime)
                     - symbol (str)
                     - open (float)
                     - high (float)
                     - low (float)
                     - close (float)
                     - volume (float)
    """
    cache_path = settings.DATA_DIR / "mock_historical_prices_cache.csv"
    try:
        if not cache_path.exists():
            logger.warning(f"Historical prices cache not found at {cache_path}")
            return pd.DataFrame()
            
        df = pd.read_csv(cache_path, parse_dates=['date'])
        logger.info(f"Loaded historical prices with shape {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading historical prices: {str(e)}")
        return pd.DataFrame()


def _load_all_mock_fi_mcp_data() -> Dict:
    """
    Load all mock Fi-MCP data from the cache file.
    
    Returns:
        dict: Dictionary containing the parsed mock Fi-MCP data
    """
    cache_path = settings.DATA_DIR / "mock_fi_mcp_data_cache.json"
    try:
        if not cache_path.exists():
            logger.warning(f"Mock Fi-MCP data cache not found at {cache_path}")
            return {}
            
        with open(cache_path, 'r') as f:
            data = json.load(f)
            
        logger.info(f"Loaded mock Fi-MCP data with {len(data.get('accounts', []))} accounts")
        return data
        
    except Exception as e:
        logger.error(f"Error loading mock Fi-MCP data: {str(e)}")
        return {}


def _get_portfolio_returns_series(portfolio_id: str, lookback_days: int = 365) -> Optional[pd.Series]:
    """
    Get the returns series for a specific portfolio.
    
    Args:
        portfolio_id: The ID of the portfolio
        lookback_days: Number of days to look back for returns data
        
    Returns:
        Optional[pd.Series]: Series of daily returns, or None if not found
    """
    try:
        # Load historical data
        prices = _load_historical_prices_df()
        if prices.empty:
            return None
            
        # Filter for the specific portfolio's holdings
        # This is a simplified example - in reality, you'd map portfolio_id to symbols
        portfolio_symbols = ['AAPL', 'MSFT', 'GOOGL']  # Example symbols
        portfolio_prices = prices[prices['symbol'].isin(portfolio_symbols)]
        
        # Calculate returns (simplified)
        returns = portfolio_prices.pivot(index='date', columns='symbol', values='close')
        returns = returns.pct_change().dropna()
        
        # Equal-weighted portfolio returns
        portfolio_returns = returns.mean(axis=1)
        
        # Apply lookback window
        return portfolio_returns.tail(lookback_days)
        
    except Exception as e:
        logger.error(f"Error calculating portfolio returns: {str(e)}")
        return None


def _get_user_holdings_from_loaded_data(user_id: str, data: Dict) -> Dict:
    """
    Extract and format a user's holdings from the loaded Fi-MCP data.
    
    Args:
        user_id: The ID of the user
        data: The loaded Fi-MCP data dictionary
        
    Returns:
        dict: Formatted holdings data with positions, accounts, and totals
    """
    try:
        # Find the user's account in the data
        user_account = next(
            (acc for acc in data.get('accounts', []) 
             if acc.get('user_id') == user_id),
            None
        )
        
        if not user_account:
            logger.warning(f"No account found for user {user_id}")
            return {
                'positions': [],
                'accounts': [],
                'total_value': 0.0,
                'total_gain_loss': 0.0,
                'last_updated': pd.Timestamp.now().isoformat()
            }
            
        # Extract and format positions
        positions = []
        for position in user_account.get('positions', []):
            positions.append({
                'symbol': position.get('symbol'),
                'quantity': position.get('quantity', 0),
                'average_price': position.get('average_price', 0.0),
                'current_price': position.get('current_price', 0.0),
                'market_value': position.get('market_value', 0.0),
                'unrealized_gain_loss': position.get('unrealized_gain_loss', 0.0),
                'unrealized_gain_loss_pct': position.get('unrealized_gain_loss_pct', 0.0)
            })
            
        # Calculate totals
        total_value = sum(pos['market_value'] for pos in positions)
        total_gain_loss = sum(pos['unrealized_gain_loss'] for pos in positions)
        
        return {
            'positions': positions,
            'accounts': [{
                'account_id': user_account.get('account_id'),
                'account_type': user_account.get('account_type'),
                'balance': user_account.get('balance', 0.0),
                'available_cash': user_account.get('available_cash', 0.0)
            }],
            'total_value': total_value,
            'total_gain_loss': total_gain_loss,
            'total_gain_loss_pct': (total_gain_loss / (total_value - total_gain_loss)) * 100 if (total_value - total_gain_loss) > 0 else 0.0,
            'last_updated': user_account.get('last_updated', pd.Timestamp.now().isoformat())
        }
        
    except Exception as e:
        logger.error(f"Error formatting user holdings: {str(e)}")
        return {
            'positions': [],
            'accounts': [],
            'total_value': 0.0,
            'total_gain_loss': 0.0,
            'last_updated': pd.Timestamp.now().isoformat()
        }
