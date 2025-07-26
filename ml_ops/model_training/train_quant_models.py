"""
Quantitative Finance Model Training Script

This script trains and manages models for Tethys's Mathematical Intelligence Layer, including:
- Time series forecasting models (TFT, N-BEATS, LSTM)
- Portfolio optimization models
- Risk assessment models
- Factor models
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Import financial intelligence components
from financial_intelligence.ml_quant_models.time_series_forecasting import TimeSeriesForecaster
from financial_intelligence.ml_quant_models.factor_models import get_bayesian_factor_exposure_conceptual
from financial_intelligence.portfolio_opt.optimizer import PortfolioOptimizer
from financial_intelligence.risk_analysis.risk_calculator import calculate_value_at_risk
from financial_intelligence._base import _load_historical_prices_df

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantModelTrainer:
    """Trainer for quantitative finance models."""
    
    def __init__(self, model_dir: str = "../../models"):
        self.model_dir = os.path.abspath(model_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        
    def train_time_series_models(self, ticker: str, lookback: int = 30, horizon: int = 5) -> Dict[str, Any]:
        """
        Train time series forecasting models.
        
        Args:
            ticker: Stock ticker symbol
            lookback: Number of historical periods to use
            horizon: Number of periods to forecast
            
        Returns:
            Training results
        """
        logger.info(f"Training time series models for {ticker}")
        
        # Load historical data
        df_prices = _load_historical_prices_df()
        if df_prices is None or df_prices.empty:
            return {"error": "No historical price data available"}
        
        # Prepare data for the specific ticker
        if ticker not in df_prices.columns:
            return {"error": f"Ticker {ticker} not found in historical data"}
        
        # Create price series
        price_data = df_prices[ticker].dropna()
        if len(price_data) < lookback + horizon:
            return {"error": f"Insufficient data for {ticker}. Need at least {lookback + horizon} periods"}
        
        # Convert to DataFrame with datetime index
        price_df = pd.DataFrame({
            'price': price_data.values
        }, index=pd.to_datetime(price_data.index))
        
        # Train different model types
        model_types = ['tft', 'nbeats', 'lstm']
        results = {}
        
        for model_type in model_types:
            try:
                forecaster = TimeSeriesForecaster(
                    model_type=model_type,
                    lookback=lookback,
                    horizon=horizon
                )
                
                # Train the model
                train_result = forecaster.fit(price_df, target_column='price')
                
                # Generate predictions
                predictions = forecaster.predict(
                    historical_data=price_df,
                    n_steps=horizon,
                    return_confidence=True
                )
                
                # Save model
                model_path = os.path.join(self.model_dir, f"timeseries_{ticker}_{model_type}.pkl")
                joblib.dump(forecaster, model_path)
                
                results[model_type] = {
                    "status": "success",
                    "model_path": model_path,
                    "training_metrics": train_result.get("training_metrics", {}),
                    "predictions": predictions
                }
                
            except Exception as e:
                logger.error(f"Error training {model_type} model: {e}")
                results[model_type] = {"status": "error", "error": str(e)}
        
        return {
            "ticker": ticker,
            "lookback": lookback,
            "horizon": horizon,
            "models": results
        }
    
    def train_portfolio_optimizer(self, user_id: str) -> Dict[str, Any]:
        """
        Train portfolio optimization models.
        
        Args:
            user_id: User identifier
            
        Returns:
            Training results
        """
        logger.info(f"Training portfolio optimizer for user {user_id}")
        
        try:
            # Initialize optimizer
            optimizer = PortfolioOptimizer()
            
            # Optimize portfolio
            result = optimizer.optimize_portfolio(
                user_id=user_id,
                method='max_sharpe',
                risk_tolerance='medium'
            )
            
            if result.get("error"):
                return {"error": result["error"]}
            
            # Save optimization results
            opt_path = os.path.join(self.model_dir, f"portfolio_opt_{user_id}.json")
            with open(opt_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            return {
                "status": "success",
                "optimization_result": result,
                "model_path": opt_path
            }
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            return {"error": str(e)}
    
    def train_risk_models(self, user_id: str) -> Dict[str, Any]:
        """
        Train risk assessment models.
        
        Args:
            user_id: User identifier
            
        Returns:
            Training results
        """
        logger.info(f"Training risk models for user {user_id}")
        
        try:
            # Calculate VaR
            var_result = calculate_value_at_risk(
                user_id=user_id,
                confidence_level=0.95,
                time_horizon_days=1
            )
            
            # Train anomaly detection model
            df_prices = _load_historical_prices_df()
            if df_prices is not None and not df_prices.empty:
                # Use returns for anomaly detection
                returns = df_prices.pct_change().dropna()
                
                # Train isolation forest
                iso_forest = IsolationForest(
                    n_estimators=100,
                    contamination=0.05,
                    random_state=42
                )
                
                # Fit on returns (flatten if multiple columns)
                if returns.shape[1] > 1:
                    returns_flat = returns.values.flatten()
                else:
                    returns_flat = returns.values
                
                iso_forest.fit(returns_flat.reshape(-1, 1))
                
                # Save model
                risk_model_path = os.path.join(self.model_dir, f"risk_anomaly_{user_id}.pkl")
                joblib.dump(iso_forest, risk_model_path)
                
                return {
                    "status": "success",
                    "var_result": var_result,
                    "anomaly_model_path": risk_model_path,
                    "anomaly_contamination": 0.05
                }
            else:
                return {
                    "status": "partial",
                    "var_result": var_result,
                    "error": "No price data available for anomaly detection"
                }
                
        except Exception as e:
            logger.error(f"Error in risk model training: {e}")
            return {"error": str(e)}
    
    def train_factor_models(self, user_id: str) -> Dict[str, Any]:
        """
        Train factor exposure models.
        
        Args:
            user_id: User identifier
            
        Returns:
            Training results
        """
        logger.info(f"Training factor models for user {user_id}")
        
        try:
            # Get factor exposure (conceptual for now)
            factor_result = get_bayesian_factor_exposure_conceptual(user_id)
            
            # Save factor model results
            factor_path = os.path.join(self.model_dir, f"factor_exposure_{user_id}.json")
            with open(factor_path, 'w') as f:
                json.dump(factor_result, f, indent=2)
            
            return {
                "status": "success",
                "factor_exposure": factor_result,
                "model_path": factor_path
            }
            
        except Exception as e:
            logger.error(f"Error in factor model training: {e}")
            return {"error": str(e)}
    
    def evaluate_quant_models(self, user_id: str, ticker: str) -> Dict[str, Any]:
        """
        Evaluate quantitative model performance.
        
        Args:
            user_id: User identifier
            ticker: Stock ticker for evaluation
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating quant models for user {user_id}, ticker {ticker}")
        
        results = {}
        
        # Evaluate time series models
        ts_results = self.train_time_series_models(ticker)
        results["time_series"] = ts_results
        
        # Evaluate portfolio optimization
        portfolio_results = self.train_portfolio_optimizer(user_id)
        results["portfolio_optimization"] = portfolio_results
        
        # Evaluate risk models
        risk_results = self.train_risk_models(user_id)
        results["risk_models"] = risk_results
        
        # Evaluate factor models
        factor_results = self.train_factor_models(user_id)
        results["factor_models"] = factor_results
        
        return results

def main():
    """Main training function."""
    trainer = QuantModelTrainer()
    
    # Training parameters
    user_id = "test_user_quant_training"
    ticker = "RELIANCE.NS"  # Example ticker
    
    # Train and evaluate all quant models
    results = trainer.evaluate_quant_models(user_id, ticker)
    
    # Save comprehensive results
    summary = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "ticker": ticker,
        "results": results
    }
    
    summary_path = os.path.join(trainer.model_dir, "quant_training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Quantitative model training completed. Summary saved to {summary_path}")
    
    # Print summary
    print("\n=== Training Summary ===")
    for model_type, result in results.items():
        status = result.get("status", "unknown")
        print(f"{model_type}: {status}")

if __name__ == "__main__":
    main() 