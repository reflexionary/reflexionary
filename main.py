"""
Tacit - Your AI Financial Co-Pilot
Main Application Entry Point
"""
import os
import logging
from pathlib import Path

# Configure logging before other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_application():
    """Initialize the Tacit application components."""
    from config.app_settings import (
        LOG_LEVEL, 
        ANNOY_INDEX_DIR,
        GEMINI_API_KEY,
        FIREBASE_SERVICE_ACCOUNT_KEY_PATH
    )
    
    # Set log level from config
    logging.getLogger().setLevel(LOG_LEVEL.upper())
    
    logger.info("Initializing Tacit AI Financial Co-Pilot...")
    
    # Verify required environment variables
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY is not set. Some features may not work.")
    
    if not FIREBASE_SERVICE_ACCOUNT_KEY_PATH or not os.path.exists(FIREBASE_SERVICE_ACCOUNT_KEY_PATH):
        logger.warning("FIREBASE_SERVICE_ACCOUNT_KEY_PATH is not set or invalid. Firebase features will be disabled.")
    
    # Ensure required directories exist
    os.makedirs(ANNOY_INDEX_DIR, exist_ok=True)
    
    logger.info("Application initialization complete.")

def run_cli():
    """Run the application in CLI mode."""
    print("\n=== Tacit AI Financial Co-Pilot ===")
    print("CLI Mode: This is a placeholder for future CLI functionality.")
    print("To launch the web UI, run: streamlit run ui/streamlit_app.py\n")

def main():
    """Main entry point for the application."""
    try:
        initialize_application()
        run_cli()
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
