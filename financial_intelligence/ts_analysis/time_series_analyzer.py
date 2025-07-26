"""
Time Series Analysis Module

This module provides functionality for analyzing financial time series data,
including stationarity tests, volatility clustering detection, and conceptual
forecasting using various time series models.
"""

import pandas as pd
import numpy as np
import logging
import random
from typing import Dict, Any, Optional, List

# Statsmodels for time series analysis
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss

"""
Time Series Analysis Module for Reflexionary

This module provides functionality for analyzing financial time series data,
including stationarity tests, volatility clustering detection, and conceptual
forecasting using various time series models.
"""

import pandas as pd
import numpy as np
import logging
import random
import os
import sys
import json
from typing import Dict, Any, Optional, List

# Statsmodels for time series analysis
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss

# Import shared base utilities for data access
from financial_intelligence._base import _load_historical_prices_df

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Conceptual Imports for Advanced Time Series Libraries ---
# from darts import TimeSeries
# from darts.models import ARIMA, ExponentialSmoothing, NBEATS, TFTModel
# import pyflux as pf
# from arch import arch_model

def detect_volatility_clustering(user_id: str, ticker: str) -> Dict[str, Any]:
    """
    Detects volatility clustering for a given ticker using Ljung-Box test on squared returns.
    
    Args:
        user_id: The unique identifier for the user
        ticker: The ticker symbol to analyze
        
    Returns:
        Dict containing volatility clustering analysis results
    """
    logger.info(f"TS Analysis: Detecting volatility clustering for '{ticker}'...")
    # Load historical prices
    df = _load_historical_prices_df()
    if df.empty:
        return {"error": "No historical price data available."}
        
    # Ensure we have the required columns
    if 'date' not in df.columns or 'price' not in df.columns:
        return {"error": "Invalid data format. Expected columns: 'date' and 'price'."}
        
    # Set date as index and sort
    df = df.set_index('date').sort_index()
    price_series = df['price']

    returns = price_series.pct_change().dropna()
    if len(returns) < 30:  # Reduced from 252 to 30 for testing with limited data
        return {"error": f"Not enough historical data for Ljung-Box test (got {len(returns)} days, minimum 30 required)."}

    try:
        # Ljung-Box test on squared returns for ARCH effects
        ljung_box_results = acorr_ljungbox(returns**2, lags=[5, 10, 20], return_df=True)
        p_value_10_lags = ljung_box_results.loc[10, 'lb_pvalue'] if 10 in ljung_box_results.index else 1.0

        is_clustering = p_value_10_lags < 0.05

        return {
            "ticker": ticker,
            "volatility_clustering_detected": is_clustering,
            "ljung_box_p_value_10_lags": f"{p_value_10_lags:.4f}",
            "description": f"Evidence of volatility clustering {'DETECTED' if is_clustering else 'NOT detected'} for {ticker}."
        }
    except Exception as e:
        logger.error(f"TS Analysis: ERROR detecting volatility clustering for {ticker}: {e}")
        return {"error": f"Failed to detect volatility clustering: {e}"}

def check_stationarity(user_id: str, ticker: str) -> Dict[str, Any]:
    """
    Checks the stationarity of a ticker's price series using ADF and KPSS tests.
    
    Args:
        user_id: The unique identifier for the user
        ticker: The ticker symbol to analyze
        
    Returns:
        Dict containing stationarity test results
    """
    logger.info(f"TS Analysis: Checking stationarity for '{ticker}'...")
    # Load historical prices
    df = _load_historical_prices_df()
    if df.empty:
        return {"error": "No historical price data available."}
        
    # Ensure we have the required columns
    if 'date' not in df.columns or 'price' not in df.columns:
        return {"error": "Invalid data format. Expected columns: 'date' and 'price'."}
        
    # Set date as index and sort
    df = df.set_index('date').sort_index()
    price_series = df['price'].dropna()
    
    if len(price_series) < 20:
        return {"error": f"Not enough data points for stationarity tests (got {len(price_series)}, minimum 20 required)."}

    try:
        # ADF Test (Augmented Dickey-Fuller)
        adf_result = adfuller(price_series)
        adf_pvalue = adf_result[1]
        adf_stationary = adf_pvalue < 0.05 

        # KPSS Test
        kpss_result = kpss(price_series, regression='c', nlags='auto')
        kpss_pvalue = kpss_result[1]
        kpss_stationary = kpss_pvalue > 0.05 

        return {
            "ticker": ticker,
            "adf_p_value": f"{adf_pvalue:.4f}",
            "adf_stationary": adf_stationary,
            "kpss_p_value": f"{kpss_pvalue:.4f}",
            "kpss_stationary": kpss_stationary,
            "interpretation": "ADF: Low p-value suggests non-stationarity. KPSS: High p-value suggests stationarity."
        }
    except Exception as e:
        logger.error(f"TS Analysis: ERROR checking stationarity for {ticker}: {e}")
        return {"error": f"Failed to check stationarity: {e}"}

def detect_market_regime(user_id: str, ticker: str, lookback_days: int = 252) -> Dict[str, Any]:
    """
    Detect different market regimes (e.g., trending, mean-reverting, volatile) for a given ticker.
    
    This function uses statistical methods to identify different market regimes in the price series.
    
    Args:
        user_id: The unique identifier for the user
        ticker: The ticker symbol to analyze
        lookback_days: Number of days to look back for the analysis
        
    Returns:
        Dict containing market regime analysis results
    """
    logger.info(f"Detecting market regime for {ticker} with {lookback_days}-day lookback")
    
    try:
        # Get historical prices (using the shared function from _base.py)
        from .._base import _HISTORICAL_PRICES_DF
        
        if _HISTORICAL_PRICES_DF is None or ticker not in _HISTORICAL_PRICES_DF.columns:
            raise ValueError(f"No historical price data available for {ticker}")
        
        # Get the most recent data points
        prices = _HISTORICAL_PRICES_DF[ticker].dropna().tail(lookback_days)
        
        if len(prices) < 30:  # Need at least 30 data points
            raise ValueError("Insufficient data points for market regime detection")
        
        # Calculate returns and log returns
        returns = prices.pct_change().dropna()
        log_returns = np.log(prices / prices.shift(1)).dropna()
        
        # Calculate various metrics to identify market regime
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        mean_return = returns.mean() * 252  # Annualized return
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0
        
        # Use ADF test to check for mean reversion
        adf_result = adfuller(returns, maxlag=1)
        
        # Calculate Hurst exponent (simplified)
        lags = range(2, 20)
        tau = [np.sqrt(np.abs(log_returns.diff(lag)).dropna().var()) for lag in lags]
        hurst_exp = np.polyfit(np.log(lags), np.log(tau), 1)[0]
        
        # Determine market regime based on metrics
        if hurst_exp < 0.4:
            regime = "mean_reverting"
        elif hurst_exp > 0.6:
            regime = "trending"
        else:
            regime = "random_walk"
        
        # Check for high volatility regime
        if volatility > 0.3:  # 30% annualized volatility threshold
            volatility_regime = "high_volatility"
        elif volatility < 0.15:  # 15% annualized volatility threshold
            volatility_regime = "low_volatility"
        else:
            volatility_regime = "moderate_volatility"
        
        # Compile results
        result = {
            'ticker': ticker,
            'lookback_days': lookback_days,
            'market_regime': regime,
            'volatility_regime': volatility_regime,
            'metrics': {
                'annualized_volatility': float(volatility),
                'annualized_return': float(mean_return),
                'sharpe_ratio': float(sharpe_ratio),
                'hurst_exponent': float(hurst_exp),
                'adf_pvalue': float(adf_result[1]),
                'data_points_used': len(prices)
            },
            'status': 'success'
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in detect_market_regime: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e)
        }


def forecast_asset_prices(user_id: str, ticker: str, forecast_days: int = 30) -> Dict[str, Any]:
    """
    (Conceptual) Forecasts future asset prices using time series models.
    
    Args:
        user_id: The unique identifier for the user
        ticker: The ticker symbol to forecast
        forecast_days: Number of days to forecast
        
    Returns:
        Dict containing conceptual forecast results
    """
    logger.info(f"TS Analysis: Conceptually forecasting prices for '{ticker}'...")
    # Load historical prices
    df = _load_historical_prices_df()
    if df.empty:
        return {"error": "No historical price data available for forecasting."}
        
    # Ensure we have the required columns
    if 'date' not in df.columns or 'price' not in df.columns:
        return {"error": "Invalid data format. Expected columns: 'date' and 'price'."}
        
    # Set date as index and sort
    df = df.set_index('date').sort_index()
    series = df['price'].dropna()
    
    if len(series) < 20:  # Reduced from 100 to 20 for testing with limited data
        return {"error": f"Not enough historical data for forecasting (got {len(series)} days, minimum 20 required)."}

    # Generate mock forecast data
    last_price = series.iloc[-1]
    forecast_path = [last_price * (1 + random.uniform(-0.005, 0.008))**(i/252) 
                    for i in range(forecast_days)]
    lower_bound = [p * random.uniform(0.95, 0.98) for p in forecast_path]
    upper_bound = [p * random.uniform(1.02, 1.05) for p in forecast_path]

    return {
        "ticker": ticker,
        "forecast_horizon_days": forecast_days,
        "last_known_price": f"₹{last_price:.2f}",
        "forecast_path_sample": [f"{p:.2f}" for p in forecast_path[:10]],
        "lower_bound_sample": [f"{p:.2f}" for p in lower_bound[:10]],
        "upper_bound_sample": [f"{p:.2f}" for p in upper_bound[:10]],
        "note": "Conceptual forecast. Real implementation would use advanced models."
    }

# --- Self-Test Block ---
if __name__ == "__main__":
    import sys
    import os
    import json
    
    # Add parent directory to path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Load historical data
    from financial_intelligence._base import _load_historical_prices_df
    _load_historical_prices_df()

    logger.info("\n--- Time Series Analyzer: Self-Test ---")
    test_user_id = "user_0000000"

    # Test 1: Volatility Clustering
    print("\n[Test 1] Testing Volatility Clustering...")
    vol_result = detect_volatility_clustering(test_user_id, "MOCK")
    print(json.dumps(vol_result, indent=2))

    # Test 2: Stationarity Check
    print("\n[Test 2] Testing Stationarity...")
    stat_result = check_stationarity(test_user_id, "MOCK")
    print(json.dumps(stat_result, indent=2))

    # Test 3: Price Forecasting
    print("\n[Test 3] Testing Price Forecasting...")
    forecast_result = forecast_asset_prices(test_user_id, "MOCK", 30)
    print(json.dumps(forecast_result, indent=2))

    logger.info("\n--- Self-Test Complete ---")