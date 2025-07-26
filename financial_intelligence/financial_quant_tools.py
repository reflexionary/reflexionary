"""
Financial Quant Tools - The Unified Facade for Financial Intelligence

This module serves as the main entry point for all financial quantitative analysis
in Reflexionary. It provides a clean, unified interface to the various specialized
quant modules, making it easy to access advanced financial analysis capabilities.
"""

import os
import sys
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root to Python path: {project_root}")

# Import shared base utilities
from financial_intelligence._base import (
    _load_historical_prices_df, 
    _load_all_mock_fi_mcp_data,
    _get_portfolio_returns_series,
    _get_user_holdings_from_loaded_data
)

# Import functions from core_metrics
from financial_intelligence.core_metrics.performance_analyzer import calculate_comprehensive_portfolio_metrics

# Import functions from portfolio_opt
from financial_intelligence.portfolio_opt.optimizer import PortfolioOptimizer, optimize_portfolio_allocation

# Import functions from risk_analysis
from financial_intelligence.risk_analysis.risk_calculator import calculate_value_at_risk, analyze_return_distribution
from financial_intelligence.risk_analysis.anomaly_detector import detect_financial_anomaly

# Import functions from ts_analysis
from financial_intelligence.ts_analysis.time_series_analyzer import (
    detect_volatility_clustering, 
    check_stationarity, 
    forecast_asset_prices,
    detect_market_regime
)

# Import functions from advanced_concepts
from financial_intelligence.advanced_concepts.market_microstructure import get_order_imbalance_score, estimate_slippage
from financial_intelligence.advanced_concepts.derivatives_math import calculate_black_scholes_option_price
from financial_intelligence.advanced_concepts.strategy_evaluator import (
    evaluate_trading_signal, 
    calculate_hurst_exponent, 
    check_granger_causality, 
    run_monte_carlo_portfolio_paths, 
    run_scenario_analysis,
    run_strategy_backtest
)

# Import functions from ml_quant_models
from financial_intelligence.ml_quant_models.factor_models import get_bayesian_factor_exposure_conceptual
from financial_intelligence.ml_quant_models.rl_execution_policies import get_rl_trade_execution_policy_conceptual
from financial_intelligence.ml_quant_models.deep_hedging import get_deep_hedging_strategy_conceptual
from financial_intelligence.ml_quant_models.tft_forecaster import get_tft_price_forecast_conceptual

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Ensure base data is loaded when this facade is imported
_load_historical_prices_df()
_load_all_mock_fi_mcp_data()
logger.info("Financial Quant Tools Facade: Base data loaders triggered.")

# --- Facade Functions ---
# These functions provide a clean interface to the underlying quant modules

def get_portfolio_performance_metrics(user_id: str) -> Dict[str, Any]:
    """Get comprehensive performance metrics for a user's portfolio."""
    return calculate_comprehensive_portfolio_metrics(user_id)

def get_optimal_portfolio_allocation(
    user_id: str, 
    risk_tolerance: str, 
    total_investment_value: float
) -> Dict[str, Any]:
    """Get optimal portfolio allocation based on risk tolerance."""
    return optimize_portfolio_allocation(
        user_id=user_id,
        risk_tolerance=risk_tolerance,
        total_investment_value=total_investment_value
    )

def get_value_at_risk(
    user_id: str, 
    confidence_level: float = 0.95, 
    time_horizon_days: int = 1
) -> Dict[str, Any]:
    """Calculate Value at Risk for a user's portfolio."""
    return calculate_value_at_risk(user_id, confidence_level, time_horizon_days)

def get_return_distribution_analysis(
    user_id: str, 
    ticker: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze return distribution for a user's portfolio or specific asset."""
    return analyze_return_distribution(user_id, ticker)

def detect_user_financial_anomaly(
    user_id: str, 
    transaction_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Detect potential financial anomalies in user transactions."""
    return detect_financial_anomaly(user_id, transaction_data)

def get_volatility_clustering_status(
    user_id: str, 
    ticker: str
) -> Dict[str, Any]:
    """Check for volatility clustering in an asset's returns."""
    return detect_volatility_clustering(user_id, ticker)

def get_stationarity_check(
    user_id: str, 
    ticker: str
) -> Dict[str, Any]:
    """Check if a time series is stationary."""
    return check_stationarity(user_id, ticker)

def get_asset_price_forecast(
    user_id: str, 
    ticker: str, 
    forecast_days: int = 30
) -> Dict[str, Any]:
    """Generate price forecasts for an asset."""
    return forecast_asset_prices(user_id, ticker, forecast_days)

def get_market_order_imbalance(ticker: str) -> Dict[str, Any]:
    """Calculate order imbalance for a given ticker."""
    return get_order_imbalance_score(ticker)

def get_estimated_slippage(
    ticker: str, 
    order_size: float
) -> Dict[str, Any]:
    """Estimate slippage for a given order size."""
    return estimate_slippage(ticker, order_size)

def get_black_scholes_option_price(
    S: float, 
    K: float, 
    T: float, 
    r: float, 
    sigma: float, 
    option_type: str
) -> Dict[str, Any]:
    """Calculate Black-Scholes option price."""
    return calculate_black_scholes_option_price(S, K, T, r, sigma, option_type)

# --- Self-Verification Block ---
if __name__ == "__main__":
    import time
    import uuid

    # Configure logging for self-test
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("\n--- Financial Quant Tools Facade: Self-Test Initiated ---")
    
    # Test user ID (should exist in mock data)
    test_user_id = "user_0000000"
    
    # Run a subset of tests to verify integration
    tests = [
        ("Portfolio Metrics", lambda: get_portfolio_performance_metrics(test_user_id)),
        ("Portfolio Optimization", lambda: get_optimal_portfolio_allocation(test_user_id, "medium", 100000)),
        ("Value at Risk", lambda: get_value_at_risk(test_user_id)),
        ("Black-Scholes Option Pricing", lambda: get_black_scholes_option_price(
            S=1500, K=1550, T=0.5, r=0.07, sigma=0.25, option_type="call"
        )),
        ("TFT Price Forecast", lambda: get_asset_price_forecast(test_user_id, "RELIANCE.NS", 7))
    ]
    
    # Run tests and collect results
    results = {}
    for test_name, test_func in tests:
        try:
            logger.info(f"\n[Test] {test_name}")
            start_time = time.time()
            result = test_func()
            exec_time = time.time() - start_time
            results[test_name] = {
                "status": "PASSED",
                "execution_time": f"{exec_time:.2f}s",
                "result_keys": list(result.keys()) if isinstance(result, dict) else str(type(result))
            }
            logger.info(f"  ✓ {test_name} completed in {exec_time:.2f}s")
        except Exception as e:
            results[test_name] = {
                "status": "FAILED",
                "error": str(e)
            }
            logger.error(f"  ✗ {test_name} failed: {str(e)}", exc_info=True)
    
    # Print test summary
    logger.info("\n--- Test Summary ---")
    for test_name, result in results.items():
        status = result['status']
        if status == "PASSED":
            logger.info(f"{status}: {test_name} ({result['execution_time']})")
        else:
            logger.error(f"{status}: {test_name} - {result['error']}")
    
    logger.info("\n--- Financial Quant Tools Facade: Self-Test Completed ---")
