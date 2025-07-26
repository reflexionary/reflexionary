"""
Tacit - Retrieval Pipelines Package

This package contains modular retrieval pipelines for the Tacit memory system.
Each pipeline implements different strategies for retrieving relevant memories
based on various criteria (semantic similarity, temporal relevance, etc.).
"""

# Import core retrieval pipelines to make them available at package level
from .semantic_retriever import SemanticRetriever

__all__ = ['SemanticRetriever']
