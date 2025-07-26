"""
Tacit - Financial Intelligence Core Metrics Module

This module provides comprehensive performance and risk metrics calculation
for investment portfolios. It serves as the core metrics engine for the
financial intelligence layer.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple

# Import core components
import quantstats as qs
import statsmodels.api as sm

# Import shared base utilities
from financial_intelligence._base import _get_portfolio_returns_series, _HISTORICAL_PRICES_DF

# Configure logging
logger = logging.getLogger(__name__)

# Configure quantstats
qs.extend_pandas()

class PerformanceAnalyzer:
    """
    A comprehensive performance analysis toolkit for investment portfolios.
    
    This class provides methods to calculate various performance metrics
    including risk-adjusted returns, drawdown analysis, and benchmark comparison.
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize the PerformanceAnalyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 0.05 or 5%)
        """
        self.risk_free_rate = risk_free_rate
        self.metrics = {}
    
    def calculate_metrics(self, user_id: str) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics for a user's portfolio.
        
        Args:
            user_id: The ID of the user
            
        Returns:
            Dict containing various performance metrics
        """
        logger.info(f"Calculating performance metrics for user {user_id}")
        
        # Get portfolio returns
        returns = _get_portfolio_returns_series(user_id)
        if returns is None or returns.empty:
            return {"error": "Insufficient data to calculate performance metrics"}
        
        # Calculate metrics
        self._calculate_return_metrics(returns)
        self._calculate_risk_metrics(returns)
        self._calculate_risk_adjusted_returns(returns)
        self._calculate_drawdown_metrics(returns)
        self._calculate_benchmark_metrics(returns)
        
        return self.metrics
    
    def _calculate_return_metrics(self, returns: pd.Series) -> None:
        """Calculate return-related metrics."""
        # Cumulative return
        cum_return = qs.stats.comp(returns)
        self.metrics["cumulative_return"] = float(cum_return)
        
        # Annualized return
        annual_return = qs.stats.cagr(returns)
        self.metrics["annualized_return"] = float(annual_return)
        
        # Rolling returns
        rolling_1y = qs.stats.rolling_returns(returns, "1y")
        self.metrics["rolling_1y_return"] = float(rolling_1y.iloc[-1]) if not rolling_1y.empty else None
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> None:
        """Calculate risk-related metrics."""
        # Volatility
        vol = qs.stats.volatility(returns)
        self.metrics["volatility"] = float(vol)
        
        # Downside risk
        downside_risk = qs.stats.downside_risk(returns)
        self.metrics["downside_risk"] = float(downside_risk)
        
        # Value at Risk (VaR)
        var_95 = qs.stats.var(returns, method="historical")
        self.metrics["var_95"] = float(var_95)
    
    def _calculate_risk_adjusted_returns(self, returns: pd.Series) -> None:
        """Calculate risk-adjusted return metrics."""
        # Sharpe Ratio
        sharpe = qs.stats.sharpe(returns, rf=self.risk_free_rate/252)  # Daily risk-free rate
        self.metrics["sharpe_ratio"] = float(sharpe)
        
        # Sortino Ratio
        sortino = qs.stats.sortino(returns, rf=self.risk_free_rate/252)
        self.metrics["sortino_ratio"] = float(sortino)
        
        # Information Ratio
        if _HISTORICAL_PRICES_DF is not None and 'NIFTYBEES.NS' in _HISTORICAL_PRICES_DF.columns:
            benchmark_returns = _HISTORICAL_PRICES_DF['NIFTYBEES.NS'].pct_change().dropna()
            info_ratio = qs.stats.information_ratio(returns, benchmark_returns)
            self.metrics["information_ratio"] = float(info_ratio)
    
    def _calculate_drawdown_metrics(self, returns: pd.Series) -> None:
        """Calculate drawdown-related metrics."""
        # Maximum Drawdown
        max_dd = qs.stats.max_drawdown(returns)
        self.metrics["max_drawdown"] = float(abs(max_dd))
        
        # Calmar Ratio
        calmar = qs.stats.calmar(returns)
        self.metrics["calmar_ratio"] = float(calmar)
        
        # Drawdown duration
        dd_duration = qs.stats.drawdown_details(qs.stats.to_drawdown_series(returns))
        if not dd_duration.empty:
            self.metrics["avg_drawdown_duration"] = float(dd_duration["days"].mean())
            self.metrics["max_drawdown_duration"] = float(dd_duration["days"].max())
    
    def _calculate_benchmark_metrics(self, returns: pd.Series) -> None:
        """Calculate benchmark comparison metrics."""
        if _HISTORICAL_PRICES_DF is None or 'NIFTYBEES.NS' not in _HISTORICAL_PRICES_DF.columns:
            return
            
        benchmark_returns = _HISTORICAL_PRICES_DF['NIFTYBEES.NS'].pct_change().dropna()
        
        # Align returns with benchmark
        aligned_returns = pd.DataFrame({
            'portfolio': returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if aligned_returns.empty:
            return
        
        # Alpha and Beta
        try:
            X = sm.add_constant(aligned_returns['benchmark'])
            model = sm.OLS(aligned_returns['portfolio'], X).fit()
            self.metrics["alpha"] = float(model.params['const'] * 252)  # Annualized
            self.metrics["beta"] = float(model.params['benchmark'])
            self.metrics["r_squared"] = float(model.rsquared)
        except Exception as e:
            logger.warning(f"Error calculating alpha/beta: {e}")
        
        # Tracking Error
        active_returns = aligned_returns['portfolio'] - aligned_returns['benchmark']
        self.metrics["tracking_error"] = float(np.std(active_returns, ddof=1) * np.sqrt(252))
        
        # Up/Down Capture Ratio
        up_capture, down_capture = self._calculate_capture_ratios(
            aligned_returns['portfolio'], 
            aligned_returns['benchmark']
        )
        self.metrics["up_capture_ratio"] = float(up_capture)
        self.metrics["down_capture_ratio"] = float(down_capture)
    
    @staticmethod
    def _calculate_capture_ratios(returns: pd.Series, benchmark: pd.Series) -> Tuple[float, float]:
        """Calculate up and down capture ratios."""
        # Align returns and benchmark
        df = pd.DataFrame({'returns': returns, 'benchmark': benchmark}).dropna()
        
        # Up market periods (benchmark return > 0)
        up_market = df[df['benchmark'] > 0]
        up_capture = (up_market['returns'] + 1).prod() ** (252/len(up_market)) - 1
        up_benchmark = (up_market['benchmark'] + 1).prod() ** (252/len(up_market)) - 1
        
        # Down market periods (benchmark return < 0)
        down_market = df[df['benchmark'] < 0]
        down_capture = (down_market['returns'] + 1).prod() ** (252/len(down_market)) - 1
        down_benchmark = (down_market['benchmark'] + 1).prod() ** (252/len(down_market)) - 1
        
        # Calculate ratios
        up_ratio = (up_capture / up_benchmark) if up_benchmark != 0 else np.nan
        down_ratio = (down_capture / down_benchmark) if down_benchmark != 0 else np.nan
        
        return up_ratio, down_ratio

def calculate_comprehensive_portfolio_metrics(user_id: str) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics for a user's portfolio.
    
    This is a convenience function that creates and uses a PerformanceAnalyzer instance.
    
    Args:
        user_id: The ID of the user
        
    Returns:
        Dict containing various performance metrics
    """
    analyzer = PerformanceAnalyzer()
    return analyzer.calculate_metrics(user_id)

# Example usage
if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path to import financial_intelligence._base
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # This will trigger the data loading in _base.py
    from financial_intelligence._base import _load_historical_prices_df, _load_all_mock_fi_mcp_data
    _load_historical_prices_df()
    _load_all_mock_fi_mcp_data()
    
    logger.info("\n--- Performance Analyzer Self-Test ---")
    
    # Test with a sample user ID (assuming this exists in your mock data)
    test_user_id = "user_0000000"
    
    logger.info(f"\n[Test] Calculating metrics for user {test_user_id}...")
    metrics = calculate_comprehensive_portfolio_metrics(test_user_id)
    
    # Print results
    import pprint
    logger.info("\nPerformance Metrics:")
    pprint.pprint(metrics, indent=2)
    
    logger.info("\n--- Performance Analyzer Self-Test Complete ---")
