"""
Tacit - Core Application Settings

This module contains the core application settings and configurations.
It serves as the central place for all application-wide settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# --- Application Settings ---
APP_NAME = "Reflexionary: Your AI Financial Co-Pilot"
APP_VERSION = "0.1.0"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))

# --- Path Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"

# Ensure directories exist
for directory in [DATA_DIR, LOG_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# --- Security Settings ---
SECRET_KEY = os.getenv("SECRET_KEY", "django-insecure-your-secret-key-here")
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "*").split(",")
CORS_ORIGIN_WHITELIST = os.getenv("CORS_ORIGIN_WHITELIST", "").split(",")

# --- API Rate Limiting ---
RATE_LIMIT = os.getenv("RATE_LIMIT", "1000/day")
RATE_LIMIT_STORAGE_URL = os.getenv("RATE_LIMIT_STORAGE_URL", "memory://")

# --- Feature Flags ---
FEATURE_ANOMALY_DETECTION = os.getenv("FEATURE_ANOMALY_DETECTION", "true").lower() == "true"
FEATURE_GOAL_TRACKING = os.getenv("FEATURE_GOAL_TRACKING", "true").lower() == "true"
