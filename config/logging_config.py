"""
Logging Configuration for Reflexionary

This module provides a centralized configuration for logging throughout the application.
It sets up different loggers, handlers, and formatters for various components.
"""

import os
import sys
import logging
import logging.config
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

# --- Logging Levels ---
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}

# --- Log Directories ---
LOG_DIR = os.getenv('LOG_DIR', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Current date for log file naming
CURRENT_DATE = datetime.now().strftime('%Y-%m-%d')

# --- Log File Paths ---
LOG_FILE_PATHS = {
    'app': os.path.join(LOG_DIR, f'app_{CURRENT_DATE}.log'),
    'error': os.path.join(LOG_DIR, f'error_{CURRENT_DATE}.log'),
    'debug': os.path.join(LOG_DIR, f'debug_{CURRENT_DATE}.log'),
    'performance': os.path.join(LOG_DIR, f'performance_{CURRENT_DATE}.log'),
    'security': os.path.join(LOG_DIR, f'security_{CURRENT_DATE}.log'),
}

# --- Log Formatters ---
class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[41m',  # White on Red
        'RESET': '\033[0m',      # Reset
    }
    
    def format(self, record):
        log_message = super().format(record)
        if record.levelname in self.COLORS:
            return f"{self.COLORS[record.levelname]}{log_message}{self.COLORS['RESET']}"
        return log_message

# Standard formatters
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
JSON_FORMAT = {
    'timestamp': '%(asctime)s',
    'name': '%(name)s',
    'level': '%(levelname)s',
    'message': '%(message)s',
    'pathname': '%(pathname)s',
    'lineno': '%(lineno)d',
    'funcName': '%(funcName)s',
}

# --- Log Handlers ---
HANDLERS = {
    'console': {
        'class': 'logging.StreamHandler',
        'level': LOG_LEVEL,
        'formatter': 'colored',
        'stream': 'ext://sys.stdout',
    },
    'file_app': {
        'class': 'logging.handlers.RotatingFileHandler',
        'level': 'INFO',
        'formatter': 'standard',
        'filename': LOG_FILE_PATHS['app'],
        'maxBytes': 10 * 1024 * 1024,  # 10MB
        'backupCount': 5,
        'encoding': 'utf8',
    },
    'file_error': {
        'class': 'logging.handlers.RotatingFileHandler',
        'level': 'ERROR',
        'formatter': 'standard',
        'filename': LOG_FILE_PATHS['error'],
        'maxBytes': 10 * 1024 * 1024,  # 10MB
        'backupCount': 5,
        'encoding': 'utf8',
    },
    'file_debug': {
        'class': 'logging.handlers.RotatingFileHandler',
        'level': 'DEBUG',
        'formatter': 'verbose',
        'filename': LOG_FILE_PATHS['debug'],
        'maxBytes': 10 * 1024 * 1024,  # 10MB
        'backupCount': 5,
        'encoding': 'utf8',
    },
    'file_performance': {
        'class': 'logging.handlers.RotatingFileHandler',
        'level': 'INFO',
        'formatter': 'json',
        'filename': LOG_FILE_PATHS['performance'],
        'maxBytes': 10 * 1024 * 1024,  # 10MB
        'backupCount': 5,
        'encoding': 'utf8',
    },
    'file_security': {
        'class': 'logging.handlers.RotatingFileHandler',
        'level': 'WARNING',
        'formatter': 'json',
        'filename': LOG_FILE_PATHS['security'],
        'maxBytes': 10 * 1024 * 1024,  # 10MB
        'backupCount': 5,
        'encoding': 'utf8',
    },
}

# --- Loggers ---
LOGGERS = {
    'root': {
        'level': LOG_LEVEL,
        'handlers': ['console', 'file_app', 'file_error'],
        'propagate': False,
    },
    'app': {
        'level': LOG_LEVEL,
        'handlers': ['console', 'file_app'],
        'propagate': False,
    },
    'debug': {
        'level': 'DEBUG',
        'handlers': ['file_debug'],
        'propagate': False,
    },
    'performance': {
        'level': 'INFO',
        'handlers': ['file_performance'],
        'propagate': False,
    },
    'security': {
        'level': 'WARNING',
        'handlers': ['file_security', 'console'],
        'propagate': False,
    },
    'uvicorn': {
        'level': 'WARNING',
        'handlers': ['console', 'file_app'],
        'propagate': False,
    },
    'fastapi': {
        'level': 'INFO',
        'handlers': ['console', 'file_app'],
        'propagate': False,
    },
    'sqlalchemy': {
        'level': 'WARNING',
        'handlers': ['console', 'file_app'],
        'propagate': False,
    },
}

# --- Logging Configuration Dictionary ---
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': DEFAULT_FORMAT,
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'colored': {
            '()': 'reflexionary.config.logging_config.ColoredFormatter',
            'format': DEFAULT_FORMAT,
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'verbose': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(funcName)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'json': {
            '()': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': json.dumps(JSON_FORMAT),
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },
    'handlers': HANDLERS,
    'loggers': LOGGERS,
    'root': {
        'level': LOG_LEVEL,
        'handlers': ['console', 'file_app', 'file_error'],
    },
}

# --- Logger Initialization ---
def setup_logging():
    """Configure logging using the LOGGING_CONFIG dictionary."""
    try:
        # Configure logging
        logging.config.dictConfig(LOGGING_CONFIG)
        
        # Set up root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(LOG_LEVELS.get(LOG_LEVEL, logging.INFO))
        
        # Set up specific loggers
        for logger_name, logger_config in LOGGERS.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(logger_config.get('level', LOG_LEVEL))
            logger.propagate = logger_config.get('propagate', False)
        
        # Capture warnings from the warnings module
        logging.captureWarnings(True)
        
        # Log successful logging setup
        logger = get_logger(__name__)
        logger.info("Logging configuration completed successfully")
        
    except Exception as e:
        print(f"Error setting up logging: {e}", file=sys.stderr)
        raise

def get_logger(name: str, log_level: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Name of the logger (usually __name__)
        log_level: Optional log level override
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if log_level is not None:
        logger.setLevel(log_level.upper())
    
    return logger

class PerformanceLogger:
    """Context manager for performance logging."""
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger('performance')
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        self.logger.info(
            "Performance - %s",
            {
                'event': 'performance',
                'name': self.name,
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': round(duration, 6),
                'status': 'error' if exc_type else 'success',
                'exception': str(exc_val) if exc_val else None,
            },
            extra={'metric': 'performance'}
        )
        
        # Don't suppress exceptions
        return False

# Initialize logging when module is imported
setup_logging()

# Example usage:
if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Performance logging example
    with PerformanceLogger("example_operation"):
        # Simulate some work
        import time
        time.sleep(0.5)
        
    # JSON logging example
    logger.info("User logged in", extra={
        'user_id': 12345,
        'ip_address': '192.168.1.1',
        'user_agent': 'Mozilla/5.0',
    })
