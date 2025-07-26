"""
Tacit - Portfolio Optimization Module

This module provides portfolio optimization and allocation functionality
using various optimization techniques and constraints. It supports both
detailed optimization methods and simplified risk tolerance-based optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import json

# Import optimization libraries
from pypfopt import EfficientFrontier, objective_functions
from pypfopt import risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# Import shared base utilities
from financial_intelligence._base import _HISTORICAL_PRICES_DF, _get_user_holdings_from_loaded_data

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Conceptual imports for advanced optimization libraries
# from riskfolio.Portfolio import Portfolio  # For Riskfolio-Lib specific metrics/optimization
# import cvxportfolio  # For advanced convex optimization with frictions

class PortfolioOptimizer:
    """
    A comprehensive portfolio optimization toolkit that supports multiple optimization
    objectives and constraints.
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.05,
                 weight_bounds: Tuple[float, float] = (0, 1),
                 target_return: Optional[float] = None,
                 target_volatility: Optional[float] = None):
        """
        Initialize the PortfolioOptimizer.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 0.05 or 5%)
            weight_bounds: Minimum and maximum weight for each asset (default: (0, 1))
            target_return: Target return for optimization (if None, max Sharpe is used)
            target_volatility: Target volatility for optimization (if None, max Sharpe is used)
        """
        self.risk_free_rate = risk_free_rate
        self.weight_bounds = weight_bounds
        self.target_return = target_return
        self.target_volatility = target_volatility
        self.weights = None
        self.performance = None
    
    def optimize_portfolio(self, 
                         user_id: str,
                         method: str = 'max_sharpe',
                         market_neutral: bool = False,
                         risk_tolerance: Optional[str] = None,
                         total_investment_value: Optional[float] = None) -> Dict[str, Any]:
        """
        Optimize portfolio allocation based on historical price data.
        
        Args:
            user_id: The ID of the user
            method: Optimization method ('max_sharpe', 'min_volatility', 'efficient_risk', 'efficient_return')
            market_neutral: Whether to allow short positions (market neutral)
            risk_tolerance: Risk tolerance level ('low', 'medium', 'high'). If provided, overrides method.
            total_investment_value: Total monetary value to be invested (required for discrete allocation)
            
        Returns:
            Dict containing optimization results including weights and performance metrics
        """
        logger.info(f"Optimizing portfolio for user {user_id} using {method} method")
        
        # Get historical prices
        df_prices = _HISTORICAL_PRICES_DF
        if df_prices is None or df_prices.empty:
            return {"error": "No historical price data available for optimization"}
        
        # Get user's current holdings
        user_holdings = _get_user_holdings_from_loaded_data(user_id, {})  # Pass empty dict as second argument
        if not user_holdings:
            return {"error": "No holdings data available for user"}
        
        # Filter prices for assets in user's portfolio
        user_assets = [ticker for ticker in user_holdings.keys() if ticker in df_prices.columns]
        if not user_assets:
            return {"error": "No matching assets found in historical data"}
            
        df_user_prices = df_prices[user_assets]
        
        try:
            # Calculate expected returns and sample covariance
            mu = expected_returns.mean_historical_return(df_user_prices)
            S = risk_models.sample_cov(df_user_prices)
            
            # Optimize portfolio
            ef = EfficientFrontier(mu, S, weight_bounds=self.weight_bounds)
            
            # Add constraints (if any)
            # Example: ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
            
            # Optimize based on risk tolerance if specified, otherwise use method
            if risk_tolerance is not None:
                if risk_tolerance.lower() == "high":
                    ef.max_sharpe(risk_free_rate=self.risk_free_rate/252)
                    method = "max_sharpe (high risk)"
                elif risk_tolerance.lower() == "medium":
                    ef.max_quadratic_utility(risk_aversion=2)
                    method = "quadratic_utility (medium risk)"
                elif risk_tolerance.lower() == "low":
                    ef.min_volatility()
                    method = "min_volatility (low risk)"
                else:
                    return {"error": "Invalid risk tolerance. Please choose 'low', 'medium', or 'high'."}
            else:
                # Use specified method if risk_tolerance is not provided
                if method == 'max_sharpe':
                    ef.max_sharpe(risk_free_rate=self.risk_free_rate/252)  # Daily rate
                elif method == 'min_volatility':
                    ef.min_volatility()
                elif method == 'efficient_risk':
                    if self.target_volatility is None:
                        return {"error": "Target volatility must be specified for efficient_risk method"}
                    ef.efficient_risk(target_volatility=self.target_volatility/np.sqrt(252))  # Annual to daily
                elif method == 'efficient_return':
                    if self.target_return is None:
                        return {"error": "Target return must be specified for efficient_return method"}
                    ef.efficient_return(target_return=self.target_return/252)  # Annual to daily
                else:
                    return {"error": f"Unknown optimization method: {method}"}
            
            # Get optimized weights
            self.weights = ef.clean_weights()
            
            # Get performance metrics
            self.performance = ef.portfolio_performance(
                verbose=True,
                risk_free_rate=self.risk_free_rate/252  # Daily rate
            )
            
            # Prepare results
            result = {
                'optimization_method': method,
                'market_neutral': market_neutral,
                'assets': user_assets,
                'weights': {k: float(v) for k, v in self.weights.items() if v > 1e-6},
                'expected_return': float(self.performance[0]),
                'expected_volatility': float(self.performance[1]),
                'sharpe_ratio': float(self.performance[2]),
                'status': 'success',
                'advanced_optimization_note': "Optimization can be extended with CVaR, robust optimization (Riskfolio-Lib) and transaction cost modeling (cvxportfolio)."
            }
            
            # Calculate discrete allocation if needed and total_investment_value is provided
            if not market_neutral and total_investment_value is not None and total_investment_value > 0:
                self._add_discrete_allocation(result, df_user_prices, user_holdings, total_investment_value)
            
            return result
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {str(e)}", exc_info=True)
            return {"error": f"Portfolio optimization failed: {str(e)}"}
    
    def _add_discrete_allocation(self, 
                               result: Dict,
                               df_prices: pd.DataFrame,
                               user_holdings: Dict,
                               total_investment_value: float) -> None:
        """
        Add discrete allocation to the optimization result.
        
        Args:
            result: Dictionary to update with discrete allocation
            df_prices: DataFrame of historical prices
            user_holdings: User's current holdings
        """
        try:
            # Get latest prices
            latest_prices = get_latest_prices(df_prices)
            
            # Use the provided total_investment_value for allocation
            total_portfolio_value = total_investment_value
            if total_portfolio_value <= 0:
                logger.warning("Cannot calculate discrete allocation: Invalid portfolio value")
                return
            
            # Calculate discrete allocation
            da = DiscreteAllocation(
                self.weights,
                latest_prices,
                total_portfolio_value=total_portfolio_value
            )
            
            # Get allocation
            alloc, leftover = da.greedy_portfolio()
            
            # Add to results
            result['discrete_allocation'] = {
                'allocation': alloc,
                'leftover_cash': float(leftover),
                'total_invested': float(total_portfolio_value - leftover)
            }
            
        except Exception as e:
            logger.warning(f"Discrete allocation failed: {str(e)}")
    
    def calculate_efficient_frontier(self, 
                                   user_id: str,
                                   points: int = 20) -> Dict[str, List[float]]:
        """
        Calculate the efficient frontier for a user's portfolio.
        
        Args:
            user_id: The ID of the user
            points: Number of points on the efficient frontier
            
        Returns:
            Dict containing efficient frontier data points
        """
        logger.info(f"Calculating efficient frontier for user {user_id}")
        
        # Get historical prices
        df_prices = _HISTORICAL_PRICES_DF
        if df_prices is None or df_prices.empty:
            return {"error": "No historical price data available"}
        
        # Get user's current holdings
        user_holdings = _get_user_holdings_from_loaded_data(user_id, {})  # Pass empty dict as second argument
        if not user_holdings:
            return {"error": "No holdings data available for user"}
        
        # Filter prices for assets in user's portfolio
        user_assets = [ticker for ticker in user_holdings.keys() if ticker in df_prices.columns]
        if not user_assets:
            return {"error": "No matching assets found in historical data"}
            
        df_user_prices = df_prices[user_assets]
        
        try:
            # Calculate expected returns and sample covariance
            mu = expected_returns.mean_historical_return(df_user_prices)
            S = risk_models.sample_cov(df_user_prices)
            
            # Calculate efficient frontier
            ef = EfficientFrontier(mu, S, weight_bounds=self.weight_bounds)
            
            # Get min and max volatility points
            min_vol = ef.min_volatility()
            min_vol_ret = ef.portfolio_performance()[0]
            max_ret = ef.portfolio_performance(risk_free_rate=0)[0]  # Max return portfolio
            
            # Generate target returns between min vol and max return
            target_returns = np.linspace(min_vol_ret, max_ret, points)
            
            # Calculate points on the efficient frontier
            frontier = {
                'returns': [],
                'volatility': [],
                'sharpe': []
            }
            
            for target_return in target_returns:
                try:
                    ef = EfficientFrontier(mu, S, weight_bounds=self.weight_bounds)
                    ef.efficient_return(target_return=target_return)
                    ret, vol, sharpe = ef.portfolio_performance(
                        risk_free_rate=self.risk_free_rate/252  # Daily rate
                    )
                    frontier['returns'].append(float(ret))
                    frontier['volatility'].append(float(vol))
                    frontier['sharpe'].append(float(sharpe))
                except Exception as e:
                    logger.warning(f"Could not calculate point on efficient frontier: {e}")
            
            return {
                'frontier': frontier,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Efficient frontier calculation failed: {str(e)}", exc_info=True)
            return {"error": f"Efficient frontier calculation failed: {str(e)}"}

def optimize_portfolio_allocation(user_id: str, 
                                method: str = 'max_sharpe',
                                risk_free_rate: float = 0.05,
                                risk_tolerance: Optional[str] = None,
                                total_investment_value: Optional[float] = None) -> Dict[str, Any]:
    """
    Optimize portfolio allocation for a user.
    
    This is a convenience function that creates and uses a PortfolioOptimizer instance.
    
    Args:
        user_id: The ID of the user
        method: Optimization method ('max_sharpe', 'min_volatility', 'efficient_risk', 'efficient_return')
        risk_free_rate: Annual risk-free rate (default: 0.05 or 5%)
        risk_tolerance: Risk tolerance level ('low', 'medium', 'high'). If provided, overrides method.
        total_investment_value: Total monetary value to be invested (required for discrete allocation)
        
    Returns:
        Dict containing optimization results
    """
    optimizer = PortfolioOptimizer(risk_free_rate=risk_free_rate)
    return optimizer.optimize_portfolio(
        user_id, 
        method=method, 
        risk_tolerance=risk_tolerance,
        total_investment_value=total_investment_value
    )

# Example usage
if __name__ == "__main__":
    import sys
    import os
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add parent directory to path to import financial_intelligence._base
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # This will trigger the data loading in _base.py
    from financial_intelligence._base import _load_historical_prices_df, _load_all_mock_fi_mcp_data
    _load_historical_prices_df()
    _load_all_mock_fi_mcp_data()
    
    logger.info("\n--- Portfolio Optimizer Self-Test ---")
    
    # Test with a sample user ID (assuming this exists in your mock data)
    test_user_id = "user_0000000"
    
    # Test 1: Risk tolerance-based optimization (High Risk)
    logger.info("\n[Test 1] High Risk Tolerance Optimization:")
    optimizer = PortfolioOptimizer(risk_free_rate=0.05)
    result_high_risk = optimizer.optimize_portfolio(
        test_user_id, 
        risk_tolerance="high",
        total_investment_value=100000
    )
    
    # Print results
    import pprint
    logger.info("\nOptimization Results (High Risk):")
    pprint.pprint(result_high_risk, indent=2)
    
    # Test 2: Risk tolerance-based optimization (Medium Risk)
    logger.info("\n[Test 2] Medium Risk Tolerance Optimization:")
    result_medium_risk = optimizer.optimize_portfolio(
        test_user_id, 
        risk_tolerance="medium",
        total_investment_value=100000
    )
    logger.info("\nOptimization Results (Medium Risk):")
    pprint.pprint(result_medium_risk, indent=2)
    
    # Test 3: Risk tolerance-based optimization (Low Risk)
    logger.info("\n[Test 3] Low Risk Tolerance Optimization:")
    result_low_risk = optimizer.optimize_portfolio(
        test_user_id, 
        risk_tolerance="low",
        total_investment_value=100000
    )
    logger.info("\nOptimization Results (Low Risk):")
    pprint.pprint(result_low_risk, indent=2)
    
    # Test 4: Efficient Frontier
    logger.info("\n[Test 4] Calculating Efficient Frontier:")
    frontier = optimizer.calculate_efficient_frontier(test_user_id, points=10)
    
    if 'frontier' in frontier:
        logger.info(f"Calculated {len(frontier['frontier']['returns'])} points on the efficient frontier")
    
    logger.info("\n--- Portfolio Optimizer Self-Test Complete ---")
