"""
Anomaly Detection Module

This module provides functions for detecting anomalies in financial data,
such as unusual transactions, market manipulation patterns, and other
suspicious activities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases
Numeric = Union[int, float]
DateLike = Union[str, pd.Timestamp]

def detect_financial_anomaly(
    user_id: str,
    transaction_data: Dict[str, Any],
    sensitivity: float = 1.0
) -> Dict[str, Any]:
    """
    Detect potential financial anomalies in user transactions.
    
    This function analyzes transaction patterns to identify unusual activities
    that might indicate fraud, market manipulation, or other suspicious behavior.
    
    Args:
        user_id: The ID of the user
        transaction_data: Dictionary containing transaction data with 'transactions' key
        sensitivity: Sensitivity parameter (0.5 to 3.0) - higher values catch more anomalies
        
    Returns:
        Dictionary containing anomaly detection results
    """
    logger.info(f"Detecting financial anomalies for user {user_id} with sensitivity {sensitivity}")
    
    try:
        # Validate input
        if 'transactions' not in transaction_data or not transaction_data['transactions']:
            return {
                'status': 'error',
                'error': 'No transaction data provided',
                'anomalies_detected': False
            }
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(transaction_data['transactions'])
        
        # Initialize results
        anomalies = []
        
        # 1. Detect unusually large transactions (using IQR method)
        if 'amount' in df.columns and not df['amount'].empty:
            amounts = pd.to_numeric(df['amount'], errors='coerce').dropna()
            if len(amounts) > 0:
                q1 = amounts.quantile(0.25)
                q3 = amounts.quantile(0.75)
                iqr = q3 - q1
                threshold = q3 + (sensitivity * 1.5 * iqr)
                
                large_txs = df[df['amount'] > threshold]
                if not large_txs.empty:
                    anomalies.append({
                        'type': 'large_transaction',
                        'description': f"Found {len(large_txs)} unusually large transactions",
                        'details': {
                            'threshold': float(threshold),
                            'transactions': large_txs.to_dict('records')
                        }
                    })
        
        # 2. Detect unusual transaction frequency (if timestamps are available)
        if 'timestamp' in df.columns and not df['timestamp'].empty:
            try:
                df['datetime'] = pd.to_datetime(df['timestamp'])
                df['date'] = df['datetime'].dt.date
                
                # Group by date and count transactions
                daily_counts = df.groupby('date').size()
                
                if len(daily_counts) > 5:  # Need sufficient data
                    # Calculate z-scores for daily transaction counts
                    mean_tx = daily_counts.mean()
                    std_tx = daily_counts.std()
                    
                    if std_tx > 0:  # Avoid division by zero
                        z_scores = (daily_counts - mean_tx) / std_tx
                        unusual_days = z_scores[z_scores > (3 * sensitivity)]
                        
                        if not unusual_days.empty:
                            anomalies.append({
                                'type': 'unusual_activity',
                                'description': f"Unusually high transaction frequency on {len(unusual_days)} days",
                                'details': {
                                    'unusual_days': unusual_days.to_dict(),
                                    'mean_daily_transactions': float(mean_tx),
                                    'std_daily_transactions': float(std_tx)
                                }
                            })
            except Exception as e:
                logger.warning(f"Error analyzing transaction frequency: {str(e)}")
        
        # 3. Detect round number transactions (potential indicator of manipulation)
        if 'amount' in df.columns:
            round_amounts = df[pd.to_numeric(df['amount'], errors='coerce') % 100 == 0]
            if len(round_amounts) > 0:
                anomalies.append({
                    'type': 'round_number_transactions',
                    'description': f"Found {len(round_amounts)} round number transactions",
                    'details': {
                        'count': len(round_amounts),
                        'percentage': len(round_amounts) / len(df) * 100,
                        'sample': round_amounts.head().to_dict('records')
                    }
                })
        
        # 4. Check for rapid succession transactions (potential testing of limits)
        if 'timestamp' in df.columns and len(df) > 3:
            try:
                df = df.sort_values('timestamp')
                time_deltas = pd.to_datetime(df['timestamp']).diff().dt.total_seconds()
                rapid_succession = time_deltas[time_deltas < (3600 / sensitivity)]  # Transactions within 1h/sensitivity
                
                if len(rapid_succession) > 0:
                    anomalies.append({
                        'type': 'rapid_succession_transactions',
                        'description': f"Found {len(rapid_succession)} transactions in rapid succession",
                        'details': {
                            'time_threshold_seconds': 3600 / sensitivity,
                            'count': int(len(rapid_succession)),
                            'time_deltas_seconds': rapid_succession.tolist()
                        }
                    })
            except Exception as e:
                logger.warning(f"Error analyzing transaction timing: {str(e)}")
        
        # Prepare final results
        result = {
            'user_id': user_id,
            'anomalies_detected': len(anomalies) > 0,
            'anomaly_count': len(anomalies),
            'anomalies': anomalies,
            'transaction_count': len(df),
            'time_period': {
                'start': str(df['timestamp'].min()) if 'timestamp' in df.columns else None,
                'end': str(df['timestamp'].max()) if 'timestamp' in df.columns else None
            },
            'status': 'success'
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in detect_financial_anomaly: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e),
            'user_id': user_id,
            'anomalies_detected': False
        }


def detect_market_manipulation(
    price_data: pd.DataFrame,
    volume_data: pd.DataFrame,
    threshold: float = 2.0
) -> Dict[str, Any]:
    """
    Detect potential market manipulation patterns in price and volume data.
    
    Args:
        price_data: DataFrame with datetime index and price column
        volume_data: DataFrame with datetime index and volume column
        threshold: Standard deviation threshold for anomaly detection
        
    Returns:
        Dictionary containing manipulation detection results
    """
    logger.info("Detecting potential market manipulation patterns")
    
    try:
        # Ensure we have the required data
        if price_data.empty or volume_data.empty:
            return {
                'status': 'error',
                'error': 'Missing price or volume data'
            }
        
        # Merge price and volume data
        merged = pd.concat([price_data, volume_data], axis=1).dropna()
        
        if len(merged) < 10:  # Need sufficient data points
            return {
                'status': 'error',
                'error': 'Insufficient data points for analysis'
            }
        
        # Calculate price and volume z-scores
        price_returns = merged['price'].pct_change().dropna()
        volume_returns = merged['volume'].pct_change().dropna()
        
        price_z = (price_returns - price_returns.mean()) / price_returns.std()
        volume_z = (volume_returns - volume_returns.mean()) / volume_returns.std()
        
        # Detect unusual price/volume relationships
        unusual_activity = (price_z.abs() > threshold) & (volume_z.abs() > threshold)
        
        # Prepare results
        results = {
            'status': 'success',
            'anomaly_count': int(unusual_activity.sum()),
            'anomaly_percentage': float(unusual_activity.mean() * 100),
            'time_period': {
                'start': str(merged.index.min()),
                'end': str(merged.index.max())
            },
            'metrics': {
                'price_volatility': float(price_returns.std() * np.sqrt(252)),  # Annualized
                'volume_volatility': float(volume_returns.std() * np.sqrt(252)),
                'price_volume_correlation': float(price_returns.corr(volume_returns))
            }
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error in detect_market_manipulation: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e)
        }