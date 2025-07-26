"""
Risk Analysis Module

This module provides functions for calculating various risk metrics including
Value at Risk (VaR), Expected Shortfall, and other risk-related statistics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import stats
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases
Numeric = Union[int, float]
DateLike = Union[str, pd.Timestamp]


def calculate_value_at_risk(
    user_id: str,
    confidence_level: float = 0.95,
    time_horizon_days: int = 1,
    method: str = 'historical'
) -> Dict[str, Any]:
    """
    Calculate Value at Risk (VaR) for a user's portfolio.
    
    Args:
        user_id: The ID of the user
        confidence_level: The confidence level for VaR (e.g., 0.95 for 95%)
        time_horizon_days: The time horizon in days
        method: The method to use for calculation ('historical' or 'parametric')
        
    Returns:
        Dictionary containing VaR and related metrics
    """
    logger.info(f"Calculating {confidence_level*100}% {time_horizon_days}-day VaR for user {user_id}")
    
    try:
        # Get the returns series for the user's portfolio
        from .._base import _get_portfolio_returns_series
        returns_series = _get_portfolio_returns_series(user_id)
        
        if returns_series is None or len(returns_series) < 2:
            raise ValueError("Insufficient data for VaR calculation")
        
        # Calculate daily returns if not already in returns format
        if not np.all(returns_series.between(-1, 1)):
            returns_series = returns_series.pct_change().dropna()
        
        # Calculate VaR based on the selected method
        if method == 'historical':
            # Historical VaR
            var = -np.percentile(returns_series, (1 - confidence_level) * 100)
            
            # Calculate Expected Shortfall (CVaR)
            cvar = -returns_series[returns_series <= -var].mean()
            
        elif method == 'parametric':
            # Parametric (Normal Distribution) VaR
            mean = returns_series.mean()
            std_dev = returns_series.std()
            
            # Calculate VaR using normal distribution
            var = -(mean + std_dev * stats.norm.ppf(1 - confidence_level)) * np.sqrt(time_horizon_days)
            
            # Calculate Expected Shortfall (CVaR)
            cvar = (mean - std_dev * (stats.norm.pdf(stats.norm.ppf(1 - confidence_level)) / (1 - confidence_level))) * np.sqrt(time_horizon_days)
        else:
            raise ValueError(f"Unsupported VaR method: {method}")
        
        # Calculate some additional risk metrics
        max_drawdown = (1 - (1 + returns_series).cumprod().div((1 + returns_series).cumprod().cummax())).max()
        volatility = returns_series.std() * np.sqrt(252)  # Annualized volatility
        
        return {
            'value_at_risk': var,
            'expected_shortfall': cvar,
            'confidence_level': confidence_level,
            'time_horizon_days': time_horizon_days,
            'calculation_method': method,
            'max_drawdown': max_drawdown,
            'annualized_volatility': volatility,
            'data_points_used': len(returns_series),
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"Error calculating VaR: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e)
        }


def analyze_return_distribution(
    user_id: str,
    ticker: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze the return distribution for a user's portfolio or a specific asset.
    
    Args:
        user_id: The ID of the user
        ticker: Optional ticker symbol to analyze a specific asset
        
    Returns:
        Dictionary containing return distribution statistics
    """
    logger.info(f"Analyzing return distribution for user {user_id}" + (f" and ticker {ticker}" if ticker else ""))
    
    try:
        from .._base import _get_portfolio_returns_series, _HISTORICAL_PRICES_DF
        
        if ticker:
            # Get returns for a specific ticker
            if _HISTORICAL_PRICES_DF is None or ticker not in _HISTORICAL_PRICES_DF.columns:
                raise ValueError(f"No data available for ticker: {ticker}")
                
            prices = _HISTORICAL_PRICES_DF[ticker].dropna()
            returns = prices.pct_change().dropna()
            
        else:
            # Get returns for the user's portfolio
            returns_series = _get_portfolio_returns_series(user_id)
            if returns_series is None or len(returns_series) < 5:  # Need at least 5 data points
                raise ValueError("Insufficient data for return distribution analysis")
                
            returns = returns_series if returns_series.between(-1, 1).all() else returns_series.pct_change().dropna()
        
        # Calculate distribution statistics
        stats_dict = {
            'mean': float(returns.mean()),
            'std_dev': float(returns.std()),
            'skewness': float(returns.skew()),
            'kurtosis': float(returns.kurtosis()),
            'min': float(returns.min()),
            'max': float(returns.max()),
            'median': float(returns.median()),
            'data_points': len(returns),
            'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
            'sortino_ratio': calculate_sortino_ratio(returns),
            'positive_ratio': float((returns > 0).mean()),
            'status': 'success'
        }
        
        # Add normality test
        try:
            from scipy.stats import normaltest
            _, p_value = normaltest(returns)
            stats_dict['normality_test_p_value'] = float(p_value)
            stats_dict['is_normal'] = p_value > 0.05
        except Exception as e:
            logger.warning(f"Could not perform normality test: {str(e)}")
            stats_dict['normality_test_p_value'] = None
            stats_dict['is_normal'] = None
        
        return stats_dict
        
    except Exception as e:
        logger.error(f"Error analyzing return distribution: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e)
        }


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sortino ratio for a return series.
    
    Args:
        returns: Series of returns
        risk_free_rate: The risk-free rate of return
        
    Returns:
        The Sortino ratio
    """
    downside_returns = returns[returns < risk_free_rate]
    if len(downside_returns) < 1:
        return float('inf')
    
    downside_std = downside_returns.std()
    if downside_std == 0:
        return float('inf')
        
    excess_returns = returns.mean() - risk_free_rate
    return float(excess_returns / downside_std * np.sqrt(252))  # Annualized


def detect_financial_anomaly(
    user_id: str,
    transaction_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Detect potential financial anomalies in user transactions.
    
    Args:
        user_id: The ID of the user
        transaction_data: Dictionary containing transaction data
        
    Returns:
        Dictionary containing anomaly detection results
    """
    logger.info(f"Detecting financial anomalies for user {user_id}")
    
    try:
        # This is a simplified implementation - in a real system, you would use more sophisticated
        # anomaly detection techniques (e.g., isolation forest, one-class SVM, etc.)
        
        anomalies = []
        
        # Check for unusually large transactions
        transaction_amounts = [float(tx.get('amount', 0)) for tx in transaction_data.get('transactions', [])]
        if transaction_amounts:
            amounts_series = pd.Series(transaction_amounts)
            q75, q25 = np.percentile(amounts_series, [75, 25])
            iqr = q75 - q25
            threshold = q75 + (1.5 * iqr)
            
            large_txs = [i for i, amt in enumerate(transaction_amounts) if amt > threshold]
            if large_txs:
                anomalies.append({
                    'type': 'large_transaction',
                    'description': f"Found {len(large_txs)} unusually large transactions",
                    'details': {
                        'threshold': threshold,
                        'transaction_indices': large_txs
                    }
                })
        
        # Check for unusual transaction frequency
        if 'transactions' in transaction_data and len(transaction_data['transactions']) > 0:
            # Convert to DataFrame for easier manipulation
            import pandas as pd
            df = pd.DataFrame(transaction_data['transactions'])
            
            # Ensure we have a datetime column
            if 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp']).dt.date
                daily_counts = df.groupby('date').size()
                
                # Simple check for unusual transaction frequency
                if len(daily_counts) > 5:  # Need at least 5 days of data
                    mean_tx = daily_counts.mean()
                    std_tx = daily_counts.std()
                    
                    if std_tx > 0:  # Avoid division by zero
                        z_scores = (daily_counts - mean_tx) / std_tx
                        unusual_days = z_scores[z_scores > 3]  # More than 3 standard deviations
                        
                        if not unusual_days.empty:
                            anomalies.append({
                                'type': 'unusual_activity',
                                'description': f"Unusually high transaction frequency on {len(unusual_days)} days",
                                'details': {
                                    'unusual_days': unusual_days.to_dict(),
                                    'mean_daily_transactions': mean_tx,
                                    'std_daily_transactions': std_tx
                                }
                            })
        
        return {
            'anomalies_detected': len(anomalies) > 0,
            'anomaly_count': len(anomalies),
            'anomalies': anomalies,
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"Error detecting financial anomalies: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e)
        }