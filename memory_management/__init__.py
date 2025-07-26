"""
Tethys - Memory Management Package

This package contains components for managing Tethys's memory system, including:
- GeminiConnector: Interface to Google's Gemini AI models
- MemoryManager: Core memory management functionality
- Vector-based semantic memory storage and retrieval
"""

# Version of the memory management package
__version__ = "0.1.0"

# Import key components to make them available at the package level
from .gemini_connector import GeminiConnector, GeminiConfig  # noqa: F401

__all__ = ["GeminiConnector", "GeminiConfig"]
