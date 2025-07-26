"""
Tethys - Application Settings

Centralized configuration for the Tethys Financial Co-Pilot application.
This module contains all configuration settings, environment variables,
and constants used throughout the application.
"""

import os
from pathlib import Path
from typing import Optional

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# API Keys and External Services
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FIREBASE_SERVICE_ACCOUNT_KEY_PATH = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY_PATH")

# Fi-MCP Simulator Configuration
FI_MCP_SIMULATOR_BASE_URL = os.getenv("FI_MCP_SIMULATOR_BASE_URL", "http://localhost:3000")
FI_MCP_SIMULATOR_API_KEY = os.getenv("FI_MCP_SIMULATOR_API_KEY")

# Memory Layer Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
ANNOY_INDEX_DIR = os.path.join(BASE_DIR, os.getenv("ANNOY_INDEX_DIR", "annoy_indexes"))

# Vector Index Configuration
ANNOY_N_TREES = int(os.getenv("ANNOY_N_TREES", "100"))
ANNOY_SEARCH_K = int(os.getenv("ANNOY_SEARCH_K", "10"))

# Memory Management Configuration
MEMORY_RETENTION_DAYS = int(os.getenv("MEMORY_RETENTION_DAYS", "365"))
MAX_MEMORIES_PER_USER = int(os.getenv("MAX_MEMORIES_PER_USER", "10000"))
MEMORY_BATCH_SIZE = int(os.getenv("MEMORY_BATCH_SIZE", "100"))

# Financial Intelligence Configuration
DEFAULT_RISK_FREE_RATE = float(os.getenv("DEFAULT_RISK_FREE_RATE", "0.05"))
DEFAULT_CONFIDENCE_LEVEL = float(os.getenv("DEFAULT_CONFIDENCE_LEVEL", "0.95"))
DEFAULT_TIME_HORIZON_DAYS = int(os.getenv("DEFAULT_TIME_HORIZON_DAYS", "1"))

# Portfolio Optimization Configuration
DEFAULT_TARGET_VOLATILITY = float(os.getenv("DEFAULT_TARGET_VOLATILITY", "0.15"))
DEFAULT_TARGET_RETURN = float(os.getenv("DEFAULT_TARGET_RETURN", "0.10"))
OPTIMIZATION_METHOD = os.getenv("OPTIMIZATION_METHOD", "max_sharpe")

# Anomaly Detection Configuration
ANOMALY_DETECTION_THRESHOLD = float(os.getenv("ANOMALY_DETECTION_THRESHOLD", "0.8"))
ANOMALY_CONTAMINATION = float(os.getenv("ANOMALY_CONTAMINATION", "0.05"))
TRANSACTION_ANOMALY_THRESHOLD = float(os.getenv("TRANSACTION_ANOMALY_THRESHOLD", "1000"))

# Goal Planning Configuration
DEFAULT_GOAL_TIMELINE_MONTHS = int(os.getenv("DEFAULT_GOAL_TIMELINE_MONTHS", "12"))
MIN_GOAL_AMOUNT = float(os.getenv("MIN_GOAL_AMOUNT", "100"))
MAX_GOAL_AMOUNT = float(os.getenv("MAX_GOAL_AMOUNT", "10000000"))

# User Interaction Configuration
DEFAULT_COMMUNICATION_STYLE = os.getenv("DEFAULT_COMMUNICATION_STYLE", "friendly")
DEFAULT_DETAIL_LEVEL = os.getenv("DEFAULT_DETAIL_LEVEL", "medium")
DEFAULT_NOTIFICATION_FREQUENCY = os.getenv("DEFAULT_NOTIFICATION_FREQUENCY", "daily")

# Performance and Monitoring Configuration
METRICS_RETENTION_DAYS = int(os.getenv("METRICS_RETENTION_DAYS", "90"))
PERFORMANCE_MONITORING_ENABLED = os.getenv("PERFORMANCE_MONITORING_ENABLED", "true").lower() == "true"
HEALTH_CHECK_INTERVAL_SECONDS = int(os.getenv("HEALTH_CHECK_INTERVAL_SECONDS", "300"))

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "4"))
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))

# Security Configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
API_RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "100"))  # requests per minute
API_RATE_LIMIT_WINDOW = int(os.getenv("API_RATE_LIMIT_WINDOW", "60"))  # seconds

# Data Synchronization Configuration
DATA_SYNC_INTERVAL_MINUTES = int(os.getenv("DATA_SYNC_INTERVAL_MINUTES", "60"))
DATA_SYNC_BATCH_SIZE = int(os.getenv("DATA_SYNC_BATCH_SIZE", "1000"))
DATA_SYNC_TIMEOUT_SECONDS = int(os.getenv("DATA_SYNC_TIMEOUT_SECONDS", "300"))

# Model Training Configuration
MODEL_TRAINING_ENABLED = os.getenv("MODEL_TRAINING_ENABLED", "true").lower() == "true"
MODEL_TRAINING_INTERVAL_HOURS = int(os.getenv("MODEL_TRAINING_INTERVAL_HOURS", "24"))
MODEL_BACKUP_ENABLED = os.getenv("MODEL_BACKUP_ENABLED", "true").lower() == "true"

# File Storage Configuration
MODELS_DIR = os.path.join(BASE_DIR, os.getenv("MODELS_DIR", "models"))
DATA_DIR = os.path.join(BASE_DIR, os.getenv("DATA_DIR", "data"))
LOGS_DIR = os.path.join(BASE_DIR, os.getenv("LOGS_DIR", "logs"))
METRICS_DIR = os.path.join(BASE_DIR, os.getenv("METRICS_DIR", "metrics"))
CACHE_DIR = os.path.join(BASE_DIR, os.getenv("CACHE_DIR", "cache"))

# Cache Configuration
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))

# Development and Debug Configuration
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
TESTING_MODE = os.getenv("TESTING_MODE", "false").lower() == "true"
MOCK_DATA_ENABLED = os.getenv("MOCK_DATA_ENABLED", "false").lower() == "true"

# Feature Flags
FEATURE_MEMORY_LAYER = os.getenv("FEATURE_MEMORY_LAYER", "true").lower() == "true"
FEATURE_MATHEMATICAL_INTELLIGENCE = os.getenv("FEATURE_MATHEMATICAL_INTELLIGENCE", "true").lower() == "true"
FEATURE_ANOMALY_DETECTION = os.getenv("FEATURE_ANOMALY_DETECTION", "true").lower() == "true"
FEATURE_GOAL_PLANNING = os.getenv("FEATURE_GOAL_PLANNING", "true").lower() == "true"
FEATURE_PORTFOLIO_OPTIMIZATION = os.getenv("FEATURE_PORTFOLIO_OPTIMIZATION", "true").lower() == "true"

# Error Handling Configuration
MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))
RETRY_DELAY_SECONDS = int(os.getenv("RETRY_DELAY_SECONDS", "1"))
CIRCUIT_BREAKER_ENABLED = os.getenv("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true"
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5"))

# Notification Configuration
NOTIFICATION_EMAIL_ENABLED = os.getenv("NOTIFICATION_EMAIL_ENABLED", "false").lower() == "true"
NOTIFICATION_PUSH_ENABLED = os.getenv("NOTIFICATION_PUSH_ENABLED", "false").lower() == "true"
NOTIFICATION_WEBHOOK_ENABLED = os.getenv("NOTIFICATION_WEBHOOK_ENABLED", "false").lower() == "true"

# Data Privacy Configuration
DATA_ENCRYPTION_ENABLED = os.getenv("DATA_ENCRYPTION_ENABLED", "true").lower() == "true"
PII_MASKING_ENABLED = os.getenv("PII_MASKING_ENABLED", "true").lower() == "true"
AUDIT_LOGGING_ENABLED = os.getenv("AUDIT_LOGGING_ENABLED", "true").lower() == "true"

# Backup and Recovery Configuration
BACKUP_ENABLED = os.getenv("BACKUP_ENABLED", "true").lower() == "true"
BACKUP_INTERVAL_HOURS = int(os.getenv("BACKUP_INTERVAL_HOURS", "24"))
BACKUP_RETENTION_DAYS = int(os.getenv("BACKUP_RETENTION_DAYS", "30"))

# Validation functions
def validate_configuration():
    """Validate the configuration settings."""
    errors = []
    
    # Check required environment variables
    if not GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY is required")
    
    if not FIREBASE_SERVICE_ACCOUNT_KEY_PATH:
        errors.append("FIREBASE_SERVICE_ACCOUNT_KEY_PATH is required")
    elif not os.path.exists(FIREBASE_SERVICE_ACCOUNT_KEY_PATH):
        errors.append(f"FIREBASE_SERVICE_ACCOUNT_KEY_PATH does not exist: {FIREBASE_SERVICE_ACCOUNT_KEY_PATH}")
    
    # Check numeric ranges
    if not (0 < DEFAULT_CONFIDENCE_LEVEL < 1):
        errors.append("DEFAULT_CONFIDENCE_LEVEL must be between 0 and 1")
    
    if DEFAULT_TIME_HORIZON_DAYS <= 0:
        errors.append("DEFAULT_TIME_HORIZON_DAYS must be positive")
    
    if ANOMALY_DETECTION_THRESHOLD < 0 or ANOMALY_DETECTION_THRESHOLD > 1:
        errors.append("ANOMALY_DETECTION_THRESHOLD must be between 0 and 1")
    
    # Check directory permissions
    directories = [MODELS_DIR, DATA_DIR, LOGS_DIR, METRICS_DIR, CACHE_DIR]
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
        except PermissionError:
            errors.append(f"Cannot create directory: {directory}")
    
    return errors

def get_config_summary() -> dict:
    """Get a summary of the current configuration."""
    return {
        "base_directory": str(BASE_DIR),
        "log_level": LOG_LEVEL,
        "api_config": {
            "host": API_HOST,
            "port": API_PORT,
            "workers": API_WORKERS
        },
        "memory_layer": {
            "enabled": FEATURE_MEMORY_LAYER,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_dim": EMBEDDING_DIM,
            "retention_days": MEMORY_RETENTION_DAYS
        },
        "mathematical_intelligence": {
            "enabled": FEATURE_MATHEMATICAL_INTELLIGENCE,
            "risk_free_rate": DEFAULT_RISK_FREE_RATE,
            "confidence_level": DEFAULT_CONFIDENCE_LEVEL
        },
        "anomaly_detection": {
            "enabled": FEATURE_ANOMALY_DETECTION,
            "threshold": ANOMALY_DETECTION_THRESHOLD,
            "contamination": ANOMALY_CONTAMINATION
        },
        "goal_planning": {
            "enabled": FEATURE_GOAL_PLANNING,
            "default_timeline_months": DEFAULT_GOAL_TIMELINE_MONTHS
        },
        "portfolio_optimization": {
            "enabled": FEATURE_PORTFOLIO_OPTIMIZATION,
            "method": OPTIMIZATION_METHOD,
            "target_volatility": DEFAULT_TARGET_VOLATILITY
        },
        "performance_monitoring": {
            "enabled": PERFORMANCE_MONITORING_ENABLED,
            "health_check_interval": HEALTH_CHECK_INTERVAL_SECONDS
        },
        "development": {
            "debug_mode": DEBUG_MODE,
            "testing_mode": TESTING_MODE,
            "mock_data_enabled": MOCK_DATA_ENABLED
        }
    }

# Initialize configuration validation
if __name__ == "__main__":
    errors = validate_configuration()
    if errors:
        print("Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
        exit(1)
    else:
        print("Configuration validation passed")
        print("\nConfiguration Summary:")
        import json
        print(json.dumps(get_config_summary(), indent=2))
