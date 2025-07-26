import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv()

# --- Application Settings ---
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- Path Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Google Gemini API Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

# --- Firebase Configuration ---
FIREBASE_SERVICE_ACCOUNT_KEY_PATH = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY_PATH")

# --- Fi-MCP Simulator Configuration ---
FI_MCP_SIMULATOR_BASE_URL = os.getenv("FI_MCP_SIMULATOR_BASE_URL", "http://127.0.0.1:5000")
FI_MCP_SIMULATOR_API_KEY = os.getenv("FI_MCP_SIMULATOR_API_KEY")

# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# --- Annoy Index Configuration ---
ANNOY_INDEX_DIR = os.path.join(BASE_DIR, os.getenv("ANNOY_INDEX_DIR", "annoy_indexes"))
ANNOY_N_TREES = int(os.getenv("ANNOY_N_TREES", "10"))

# --- Anomaly Detection Configuration ---
ANOMALY_DETECTION_THRESHOLD_PERCENT = float(os.getenv("ANOMALY_DETECTION_THRESHOLD_PERCENT", "20.0"))
ANOMALY_BASELINE_WINDOW_TXNS = int(os.getenv("ANOMALY_BASELINE_WINDOW_TXNS", "30"))

# --- Streamlit UI Configuration ---
STREAMLIT_TITLE = os.getenv("STREAMLIT_TITLE", "Tacit: Your AI Financial Co-Pilot")

# Ensure directories exist
os.makedirs(ANNOY_INDEX_DIR, exist_ok=True)
