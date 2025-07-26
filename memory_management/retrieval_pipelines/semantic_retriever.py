"""
Tacit - Semantic Retriever

This module implements a semantic retriever that uses vector embeddings to find
semantically similar memories. It provides a clean, modular interface for
performing semantic search operations on the memory store.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Import core components
from core_components.embedding_service import get_embedding
from core_components.vector_index import search_index_for_similar_memories
from core_components.persistent_storage import retrieve_raw_memory

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@dataclass
class MemoryResult:
    """Data class representing a retrieved memory with its metadata."""
    doc_id: str
    text: str
    memory_type: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None  # Similarity score (if applicable)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the memory result to a dictionary."""
        return {
            'doc_id': self.doc_id,
            'text': self.text,
            'memory_type': self.memory_type,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'score': self.score
        }

class SemanticRetriever:
    """
    A semantic retriever that finds relevant memories using vector similarity.
    
    This class provides methods to search for memories based on semantic
    similarity to a query string or embedding, with support for filtering
    and result formatting.
    """
    
    def __init__(self, user_id: str):
        """
        Initialize the semantic retriever for a specific user.
        
        Args:
            user_id: The unique identifier for the user.
        """
        self.user_id = user_id
        
    def search(
        self, 
        query: str, 
        num_results: int = 3, 
        min_score: Optional[float] = None,
        memory_types: Optional[List[str]] = None
    ) -> List[MemoryResult]:
        """
        Search for memories semantically similar to the query.
        
        Args:
            query: The search query string.
            num_results: Maximum number of results to return.
            min_score: Optional minimum similarity score (0.0 to 1.0).
            memory_types: Optional list of memory types to filter by.
            
        Returns:
            A list of MemoryResult objects containing the matched memories.
        """
        logger.info(f"Performing semantic search for user {self.user_id}: '{query[:50]}...'")
        
        # Get the embedding for the query
        query_embedding = get_embedding(query)
        
        # Search the vector index for similar memories
        similar_doc_ids = search_index_for_similar_memories(
            self.user_id, 
            query_embedding, 
            num_results=num_results
        )
        
        # Retrieve the full memory data for each result
        results = []
        for doc_id in similar_doc_ids:
            memory_data = retrieve_raw_memory(self.user_id, doc_id)
            if not memory_data:
                logger.warning(f"Memory not found for doc_id: {doc_id}")
                continue
                
            # Create a MemoryResult object
            memory = MemoryResult(
                doc_id=doc_id,
                text=memory_data.get('text', ''),
                memory_type=memory_data.get('memory_type', 'unknown'),
                timestamp=memory_data.get('timestamp', ''),
                metadata=memory_data.get('metadata', {})
            )
            
            # Apply filters
            if memory_types and memory.memory_type not in memory_types:
                continue
                
            results.append(memory)
        
        # Sort by score if available (higher is better)
        if all(r.score is not None for r in results):
            results.sort(key=lambda x: x.score or 0.0, reverse=True)
        
        # Apply min_score filter if specified
        if min_score is not None:
            results = [r for r in results if r.score is None or r.score >= min_score]
        
        logger.info(f"Found {len(results)} relevant memories for query: '{query[:30]}...'")
        return results
    
    def find_similar_to_memory(
        self, 
        memory_id: str, 
        num_results: int = 3, 
        exclude_self: bool = True
    ) -> List[MemoryResult]:
        """
        Find memories that are semantically similar to a specific memory.
        
        Args:
            memory_id: The ID of the memory to find similar memories for.
            num_results: Maximum number of results to return.
            exclude_self: Whether to exclude the original memory from results.
            
        Returns:
            A list of MemoryResult objects containing similar memories.
        """
        # First, retrieve the memory to use as a query
        memory_data = retrieve_raw_memory(self.user_id, memory_id)
        if not memory_data:
            logger.warning(f"Memory not found: {memory_id}")
            return []
            
        # Use the memory's text as the query
        query_text = memory_data.get('text', '')
        if not query_text:
            logger.warning(f"Memory {memory_id} has no text content")
            return []
            
        # Perform the search
        results = self.search(query_text, num_results=num_results + (1 if exclude_self else 0))
        
        # Filter out the original memory if needed
        if exclude_self:
            results = [r for r in results if r.doc_id != memory_id]
            
        return results[:num_results]
    
    def find_recent_memories(
        self, 
        days: int = 7, 
        memory_types: Optional[List[str]] = None
    ) -> List[MemoryResult]:
        """
        Find recent memories, optionally filtered by type.
        
        Note: This is a simplified implementation. For large datasets,
        consider using Firestore queries with date filters.
        
        Args:
            days: Number of days to look back.
            memory_types: Optional list of memory types to include.
            
        Returns:
            A list of recent MemoryResult objects.
        """
        # This is a placeholder implementation. In a real system, you would:
        # 1. Query Firestore for memories within the date range
        # 2. Optionally filter by memory_types
        # 3. Return the results
        
        # For now, we'll return an empty list as a placeholder
        logger.warning("Recent memories retrieval not fully implemented")
        return []

# Example usage:
if __name__ == "__main__":
    # Initialize the retriever for a user
    retriever = SemanticRetriever(user_id="test_user_123")
    
    # Example search
    print("Searching for memories about 'retirement planning'...")
    results = retriever.search("retirement planning", num_results=3)
    
    for i, memory in enumerate(results, 1):
        print(f"\nResult {i} (Score: {memory.score:.3f}):")
        print(f"Type: {memory.memory_type}")
        print(f"Date: {memory.timestamp}")
        print(f"Text: {memory.text[:150]}...")
    
    # Example of finding similar memories
    if results:
        memory_id = results[0].doc_id
        print(f"\nFinding memories similar to: {memory_id}")
        similar = retriever.find_similar_to_memory(memory_id, num_results=2)
        
        for i, memory in enumerate(similar, 1):
            print(f"\nSimilar {i} (Score: {memory.score:.3f}):")
            print(f"Type: {memory.memory_type}")
            print(f"Text: {memory.text[:150]}...")
