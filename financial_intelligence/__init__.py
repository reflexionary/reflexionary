"""
Financial Intelligence Module

This package contains the core financial intelligence and quantitative analysis
capabilities of the Reflexionary system. It provides tools for portfolio analysis,
risk assessment, time series forecasting, and more.
"""

# Import key functions to make them available at the package level
from .financial_quant_tools import (
    get_portfolio_performance_metrics,
    get_optimal_portfolio_allocation,
    get_value_at_risk,
    get_return_distribution_analysis,
    detect_user_financial_anomaly,
    get_volatility_clustering_status,
    get_stationarity_check,
    get_asset_price_forecast,
    get_market_order_imbalance,
    get_estimated_slippage,
    get_black_scholes_option_price
)

# Version information
__version__ = '0.1.0'
__all__ = [
    'get_portfolio_performance_metrics',
    'get_optimal_portfolio_allocation',
    'get_value_at_risk',
    'get_return_distribution_analysis',
    'detect_user_financial_anomaly',
    'get_volatility_clustering_status',
    'get_stationarity_check',
    'get_asset_price_forecast',
    'get_market_order_imbalance',
    'get_estimated_slippage',
    'get_black_scholes_option_price'
]