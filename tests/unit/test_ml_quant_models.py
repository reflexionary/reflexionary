"""
Unit Tests for ML Quant Models

This module contains unit tests for the ML Quant Models components.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import the modules to be tested
from financial_intelligence.ml_quant_models.factor_models import get_bayesian_factor_exposure_conceptual
from financial_intelligence.ml_quant_models.rl_execution_policies import get_rl_trade_execution_policy_conceptual
from financial_intelligence.ml_quant_models.time_series_forecasting import TimeSeriesForecaster

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TestMLQuantModels(unittest.TestCase):
    """Test cases for ML Quant Models."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once before all tests."""
        logger.info("\n=== Setting up ML Quant Models tests ===")
        
        # Generate mock historical data for testing
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=100)
        cls.mock_data = pd.DataFrame({
            'price': 100 + np.cumsum(np.random.normal(0.1, 1, len(dates)))
        }, index=dates)
    
    def test_factor_models(self):
        """Test the Bayesian factor exposure function."""
        logger.info("\n--- Testing Factor Models ---")
        result = get_bayesian_factor_exposure_conceptual("test_user")
        
        # Basic assertions
        self.assertIn('factor_exposures', result)
        self.assertIn('factor_uncertainty', result)
        self.assertEqual(result['user_id'], "test_user")
        
        # Check factor exposure ranges
        for factor, exposure in result['factor_exposures'].items():
            self.assertTrue(-1 <= exposure <= 1, 
                          f"Factor exposure for {factor} is out of range: {exposure}")
        
        logger.info("Factor models test passed.")
    
    def test_rl_execution_policies(self):
        """Test the RL execution policy function."""
        logger.info("\n--- Testing RL Execution Policies ---")
        
        # Test with different order sizes
        order_sizes = [50000.0, 500000.0, 5000000.0]
        urgencies = [0.3, 0.7, 0.9]
        
        for size, urgency in zip(order_sizes, urgencies):
            result = get_rl_trade_execution_policy_conceptual(
                "test_user", size, "RELIANCE.NS", urgency=urgency
            )
            
            self.assertIn('execution_policy', result)
            self.assertIn('execution_schedule', result)
            logger.info(f"Order size: {size}, Urgency: {urgency}, Policy: {result['execution_policy']}")
    
    def test_time_series_forecaster(self):
        """Test the time series forecaster."""
        logger.info("\n--- Testing Time Series Forecaster ---")
        
        # Initialize and test the forecaster
        forecaster = TimeSeriesForecaster(
            model_type="tft",
            lookback=30,
            horizon=5
        )
        
        # Test fitting
        train_result = forecaster.fit(self.mock_data, target_column="price")
        self.assertTrue(forecaster.is_fitted)
        self.assertIn('training_metrics', train_result)
        
        # Test prediction
        predictions = forecaster.predict(
            historical_data=self.mock_data,
            n_steps=5,
            return_confidence=True
        )
        
        # Verify predictions
        self.assertEqual(len(predictions['predictions']), 5)
        self.assertIn('prediction_intervals', predictions)
        
        logger.info("Time series forecaster test passed.")

if __name__ == "__main__":
    # Run the tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    logger.info("\n=== All ML Quant Models tests completed ===")
