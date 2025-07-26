"""
Machine Learning Configuration for Reflexionary

This module contains configuration for all machine learning models used in the application,
including model paths, training parameters, feature engineering settings, and deployment configs.
"""

import os
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

# --- Model Storage Configuration ---
MODEL_REGISTRY_PATH: str = os.getenv('MODEL_REGISTRY_PATH', 'models/registry')
MODEL_ARTIFACTS_PATH: str = os.getenv('MODEL_ARTIFACTS_PATH', 'models/artifacts')
FEATURE_STORE_PATH: str = os.getenv('FEATURE_STORE_PATH', 'data/feature_store')

# --- Model Versioning ---
MODEL_VERSION: str = '1.0.0'
MODEL_STAGING: str = 'staging'  # 'staging', 'production', 'archived'

# --- Model Types ---
class ModelType(str, Enum):
    ANOMALY_DETECTION = 'anomaly_detection'
    SENTIMENT_ANALYSIS = 'sentiment_analysis'
    PRICE_PREDICTION = 'price_prediction'
    PORTFOLIO_OPTIMIZATION = 'portfolio_optimization'
    RISK_ASSESSMENT = 'risk_assessment'
    CHATBOT_NLP = 'chatbot_nlp'

# --- Model Paths ---
ML_MODEL_PATHS: Dict[str, str] = {
    'anomaly_detection': os.path.join(MODEL_ARTIFACTS_PATH, 'anomaly_detection'),
    'sentiment_analysis': os.path.join(MODEL_ARTIFACTS_PATH, 'sentiment_analysis'),
    'price_prediction': os.path.join(MODEL_ARTIFACTS_PATH, 'price_prediction'),
    'portfolio_optimization': os.path.join(MODEL_ARTIFACTS_PATH, 'portfolio_optimization'),
    'risk_assessment': os.path.join(MODEL_ARTIFACTS_PATH, 'risk_assessment'),
    'chatbot_nlp': os.path.join(MODEL_ARTIFACTS_PATH, 'chatbot_nlp'),
}

# --- Training Configuration ---
TRAINING_PARAMETERS: Dict[str, Dict[str, Any]] = {
    'default': {
        'random_seed': 42,
        'test_size': 0.2,
        'validation_split': 0.1,
        'early_stopping_patience': 10,
        'max_epochs': 100,
        'batch_size': 32,
        'learning_rate': 1e-3,
    },
    'anomaly_detection': {
        'contamination': 0.01,  # Expected proportion of anomalies in the data
        'n_estimators': 100,
        'max_samples': 'auto',  # Number of samples to draw for training each base estimator
        'contamination': 'auto',  # Automatically determine contamination
    },
    'sentiment_analysis': {
        'max_sequence_length': 128,
        'embedding_dim': 100,
        'lstm_units': 64,
        'dropout_rate': 0.2,
    },
    'price_prediction': {
        'lookback_window': 30,  # Number of previous time steps to use for prediction
        'forecast_horizon': 5,  # Number of time steps to predict into the future
        'lstm_units': [64, 32],  # Number of units in each LSTM layer
        'dropout_rate': 0.2,
    },
}

# --- Feature Engineering ---
FEATURE_CONFIG: Dict[str, Any] = {
    'technical_indicators': [
        'sma_7', 'sma_14', 'sma_21',
        'ema_12', 'ema_26',
        'rsi_14',
        'macd', 'macd_signal', 'macd_hist',
        'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
        'atr_14',
        'obv',
    ],
    'time_features': [
        'day_of_week', 'day_of_month', 'month', 'quarter', 'year',
        'is_weekend', 'is_month_start', 'is_month_end',
    ],
    'sentiment_features': [
        'vader_compound', 'vader_neg', 'vader_neu', 'vader_pos',
        'textblob_polarity', 'textblob_subjectivity',
    ],
    'market_features': [
        'volume', 'returns', 'volatility', 'spread',
        'high_low_ratio', 'close_open_ratio',
    ],
}

# --- Anomaly Detection ---
ANOMALY_DETECTION_THRESHOLDS: Dict[str, float] = {
    'default': 0.95,  # 95th percentile
    'financial_transaction': 0.99,  # 99th percentile for financial transactions
    'user_behavior': 0.90,  # 90th percentile for user behavior
    'market_data': 0.99,  # 99th percentile for market data
}

# --- Model Evaluation Metrics ---
EVALUATION_METRICS: Dict[str, List[str]] = {
    'regression': ['mse', 'mae', 'r2', 'explained_variance'],
    'classification': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    'anomaly_detection': ['precision', 'recall', 'f1', 'roc_auc', 'average_precision'],
    'clustering': ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score'],
}

# --- Hyperparameter Tuning ---
HYPERPARAM_GRID: Dict[str, Dict[str, Any]] = {
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    },
    'xgboost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0],
    },
    'lstm': {
        'units': [[32], [64], [32, 16]],
        'dropout_rate': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.0001],
        'batch_size': [16, 32, 64],
    },
}

# --- Model Deployment ---
DEPLOYMENT_CONFIG: Dict[str, Any] = {
    'api_endpoint': '/api/v1/predict',
    'input_schema': {
        'features': 'List[float]',
        'timestamp': 'str',
        'model_version': 'str',
    },
    'output_schema': {
        'prediction': 'Union[float, int, str]',
        'confidence': 'float',
        'model_version': 'str',
        'timestamp': 'str',
    },
    'monitoring': {
        'enabled': True,
        'metrics': ['latency', 'throughput', 'error_rate', 'data_drift'],
        'alert_thresholds': {
            'latency_ms': 1000,  # 1 second
            'error_rate': 0.05,  # 5%
            'data_drift': 0.1,   # 10%
        },
    },
}

# --- Model Monitoring ---
MONITORING_CONFIG: Dict[str, Any] = {
    'enabled': True,
    'metrics_interval': 3600,  # seconds
    'drift_detection': {
        'enabled': True,
        'window_size': 1000,  # Number of samples to consider for drift detection
        'threshold': 0.5,     # Threshold for statistical test p-value
    },
    'performance': {
        'enabled': True,
        'metrics': ['accuracy', 'precision', 'recall', 'f1'],
        'degradation_threshold': 0.05,  # 5% degradation
    },
    'data_quality': {
        'enabled': True,
        'checks': ['missing_values', 'outliers', 'data_types', 'value_ranges'],
    },
}

# --- Experiment Tracking ---
EXPERIMENT_TRACKING: Dict[str, Any] = {
    'enabled': True,
    'backend': 'mlflow',  # Options: 'mlflow', 'tensorboard', 'wandb'
    'tracking_uri': os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'),
    'experiment_name': 'reflexionary',
    'log_artifacts': True,
    'log_models': True,
    'log_metrics': True,
    'log_params': True,
}

# --- Data Validation ---
DATA_VALIDATION: Dict[str, Any] = {
    'enabled': True,
    'schema': {
        'required_columns': ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
        'value_ranges': {
            'open': {'min': 0, 'max': 1e6},
            'high': {'min': 0, 'max': 1e6},
            'low': {'min': 0, 'max': 1e6},
            'close': {'min': 0, 'max': 1e6},
            'volume': {'min': 0, 'max': 1e12},
        },
        'data_types': {
            'timestamp': 'datetime64[ns]',
            'open': 'float64',
            'high': 'float64',
            'low': 'float64',
            'close': 'float64',
            'volume': 'int64',
        },
    },
    'checks': [
        'missing_values',
        'duplicate_rows',
        'outliers',
        'data_drift',
        'concept_drift',
    ],
}

# --- Model Explainability ---
EXPLAINABILITY_CONFIG: Dict[str, Any] = {
    'enabled': True,
    'methods': ['shap', 'lime', 'permutation_importance'],
    'top_features': 10,  # Number of top features to show in explanations
    'sample_size': 1000,  # Number of samples to use for global explanations
    'class_names': None,  # For classification tasks
}

def get_model_path(model_type: Union[str, ModelType]) -> str:
    """
    Get the filesystem path for a specific model type.
    
    Args:
        model_type: Type of the model (e.g., 'anomaly_detection' or ModelType.ANOMALY_DETECTION)
        
    Returns:
        str: Path to the model directory
    """
    if isinstance(model_type, ModelType):
        model_type = model_type.value
    
    if model_type not in ML_MODEL_PATHS:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return ML_MODEL_PATHS[model_type]

def get_training_params(model_type: Union[str, ModelType]) -> Dict[str, Any]:
    """
    Get training parameters for a specific model type.
    
    Args:
        model_type: Type of the model
        
    Returns:
        Dict containing training parameters
    """
    if isinstance(model_type, ModelType):
        model_type = model_type.value
    
    # Merge default params with model-specific params
    params = TRAINING_PARAMETERS['default'].copy()
    if model_type in TRAINING_PARAMETERS:
        params.update(TRAINING_PARAMETERS[model_type])
    
    return params

def get_anomaly_threshold(anomaly_type: str = 'default') -> float:
    """
    Get the anomaly detection threshold for a specific type of anomaly.
    
    Args:
        anomaly_type: Type of anomaly to get threshold for
        
    Returns:
        float: Threshold value (0-1)
    """
    return ANOMALY_DETECTION_THRESHOLDS.get(anomaly_type, ANOMALY_DETECTION_THRESHOLDS['default'])
