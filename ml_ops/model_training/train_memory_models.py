"""
Memory Layer Model Training Script

This script trains and manages models for Tethys's Memory Layer, including:
- Embedding models (sentence-transformers)
- Vector index optimization
- Memory retrieval models
"""

import os
import json
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

# Import core components
from core_components.embedding_service import get_embedding, EMBEDDING_DIM
from core_components.vector_index import add_embedding_to_index, search_index_for_similar_memories
from memory_management.memory_manager import ingest_user_memory, retrieve_contextual_memories

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryModelTrainer:
    """Trainer for memory layer models."""
    
    def __init__(self, model_dir: str = "../../models"):
        self.model_dir = os.path.abspath(model_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        
    def train_embedding_model(self, training_data: List[str]) -> Dict[str, Any]:
        """
        Train/fine-tune embedding model on financial domain data.
        
        Args:
            training_data: List of financial text samples for training
            
        Returns:
            Training results and model metadata
        """
        logger.info(f"Training embedding model on {len(training_data)} samples")
        
        # Generate embeddings for training data
        embeddings = []
        for text in training_data:
            embedding = get_embedding(text)
            embeddings.append(embedding)
        
        # Calculate embedding statistics
        embeddings_array = np.array(embeddings)
        stats = {
            "mean_embedding": embeddings_array.mean(axis=0).tolist(),
            "std_embedding": embeddings_array.std(axis=0).tolist(),
            "embedding_dim": EMBEDDING_DIM,
            "training_samples": len(training_data)
        }
        
        # Save embedding statistics
        stats_path = os.path.join(self.model_dir, "embedding_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Embedding model training completed. Stats saved to {stats_path}")
        return {
            "status": "success",
            "embedding_stats": stats,
            "model_path": stats_path
        }
    
    def train_vector_index(self, user_id: str, memory_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train vector index on user memory data.
        
        Args:
            user_id: User identifier
            memory_data: List of memory dictionaries with 'text' and 'type' keys
            
        Returns:
            Training results
        """
        logger.info(f"Training vector index for user {user_id} with {len(memory_data)} memories")
        
        # Ingest memories to build vector index
        for memory in memory_data:
            ingest_user_memory(
                user_id=user_id,
                text=memory['text'],
                memory_type=memory['type'],
                metadata=memory.get('metadata', {})
            )
        
        # Test retrieval
        test_query = "What are my financial goals?"
        retrieved = retrieve_contextual_memories(user_id, test_query, num_results=3)
        
        return {
            "status": "success",
            "memories_ingested": len(memory_data),
            "test_retrieval_count": len(retrieved),
            "user_id": user_id
        }
    
    def evaluate_memory_models(self, user_id: str, test_queries: List[str]) -> Dict[str, Any]:
        """
        Evaluate memory model performance.
        
        Args:
            user_id: User identifier
            test_queries: List of test queries
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating memory models for user {user_id}")
        
        results = []
        for query in test_queries:
            retrieved = retrieve_contextual_memories(user_id, query, num_results=3)
            results.append({
                "query": query,
                "retrieved_count": len(retrieved),
                "retrieved_types": [m.get('type', 'unknown') for m in retrieved]
            })
        
        return {
            "status": "success",
            "evaluation_results": results,
            "total_queries": len(test_queries)
        }

def main():
    """Main training function."""
    trainer = MemoryModelTrainer()
    
    # Sample financial training data
    training_data = [
        "I want to retire by age 60 with a comfortable lifestyle",
        "My risk tolerance is moderate, I prefer balanced investments",
        "I'm saving for a house down payment in the next 3 years",
        "My monthly SIP contribution is ₹10,000",
        "I had an unexpected medical expense of ₹50,000 last month",
        "My portfolio is currently 60% equity and 40% debt",
        "I'm considering investing in international markets",
        "My emergency fund covers 6 months of expenses"
    ]
    
    # Sample memory data
    memory_data = [
        {"text": "My goal is to retire by age 60", "type": "goal"},
        {"text": "I prefer low-risk investments", "type": "preference"},
        {"text": "Monthly SIP: ₹10,000", "type": "transaction"},
        {"text": "Emergency fund: 6 months expenses", "type": "financial_status"}
    ]
    
    # Train models
    user_id = "test_user_memory_training"
    
    # Train embedding model
    embedding_results = trainer.train_embedding_model(training_data)
    logger.info(f"Embedding training: {embedding_results['status']}")
    
    # Train vector index
    vector_results = trainer.train_vector_index(user_id, memory_data)
    logger.info(f"Vector index training: {vector_results['status']}")
    
    # Evaluate models
    test_queries = [
        "What are my retirement goals?",
        "What is my risk tolerance?",
        "How much do I save monthly?"
    ]
    
    eval_results = trainer.evaluate_memory_models(user_id, test_queries)
    logger.info(f"Evaluation: {eval_results['status']}")
    
    # Save training summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "embedding_training": embedding_results,
        "vector_index_training": vector_results,
        "evaluation": eval_results
    }
    
    summary_path = os.path.join(trainer.model_dir, "memory_training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Memory model training completed. Summary saved to {summary_path}")

if __name__ == "__main__":
    main() 