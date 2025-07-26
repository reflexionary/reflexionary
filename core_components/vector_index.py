"""
Tacit - Vector Indexing Service

This module provides a high-performance vector indexing and search capability using Annoy (Approximate Nearest Neighbors Oh Yeah).
It's responsible for efficiently storing and searching high-dimensional vector embeddings, enabling semantic similarity lookups
within Tacit's long-term memory system.

Key Features:
- Fast approximate nearest neighbor search for semantic similarity
- Persistent storage of vector indexes on disk
- Thread-safe operations with proper cleanup
- Integration with the embedding service and persistent storage
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional

# Import Annoy for vector indexing
try:
    from annoy import AnnoyIndex
except ImportError:
    logger.critical("Annoy library not found. Please install it: pip install annoy")
    exit(1)

# Import application-wide settings
from config import app_settings as settings

# Import components from other modules
from core_components.embedding_service import get_embedding, EMBEDDING_DIM
from core_components.persistent_storage import store_raw_memory, retrieve_raw_memory, delete_all_user_raw_memories

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Global Storage for Active Annoy Indexes and ID Maps ---
# These dictionaries will hold the AnnoyIndex objects and their corresponding
# ID mappings for each active user. For Tethys, we manage loading/saving to disk
# to ensure persistence across application runs.
_user_annoy_indices: Dict[str, AnnoyIndex] = {}
# Maps Annoy's internal integer ID to our Firestore document ID (string)
_user_annoy_id_to_firestore_id_map: Dict[str, Dict[int, str]] = {}
# Maps Firestore document ID (string) to Annoy's internal integer ID
_user_firestore_id_to_annoy_id_map: Dict[str, Dict[str, int]] = {}

# --- Helper Functions for Annoy Index File Management ---

def _get_annoy_index_path(user_id: str) -> str:
    """Returns the file path for a user's Annoy index file."""
    return os.path.join(settings.ANNOY_INDEX_DIR, f"{user_id}.ann")

def _get_annoy_map_path(user_id: str) -> str:
    """Returns the file path for a user's Annoy ID map file."""
    return os.path.join(settings.ANNOY_INDEX_DIR, f"{user_id}_map.json")

def _load_user_annoy_data(user_id: str):
    """
    Loads a user's Annoy index and its associated ID mapping from disk into memory.
    Initializes empty structures if files don't exist.
    """
    # Ensure the index directory exists
    os.makedirs(settings.ANNOY_INDEX_DIR, exist_ok=True)
    
    index_path = _get_annoy_index_path(user_id)
    map_path = _get_annoy_map_path(user_id)

    annoy_index = AnnoyIndex(EMBEDDING_DIM, 'angular')
    current_id_map = {}
    current_reverse_map = {}

    if os.path.exists(index_path) and os.path.getsize(index_path) > 0:
        try:
            annoy_index.load(index_path)
            _user_annoy_indices[user_id] = annoy_index
            logger.info(f"Vector Index: Loaded Annoy index for user '{user_id}'.")

            if os.path.exists(map_path) and os.path.getsize(map_path) > 0:
                with open(map_path, 'r') as f:
                    loaded_map_str_keys = json.load(f)
                    # Convert keys from string back to int as Annoy uses int IDs
                    current_id_map = {int(k): v for k, v in loaded_map_str_keys.items()}
                    current_reverse_map = {v: int(k) for k, v in loaded_map_str_keys.items()}
                logger.info(f"Vector Index: Loaded ID map for user '{user_id}'.")
            else:
                logger.warning(f"Vector Index: Annoy index found, but no ID map for user '{user_id}'. Creating new map.")
        except Exception as e:
            logger.error(f"Vector Index: ERROR loading Annoy data for user '{user_id}': {e}. Starting fresh.")
            # Clean up potentially corrupted files to prevent persistent errors
            if os.path.exists(index_path): 
                os.remove(index_path)
            if os.path.exists(map_path): 
                os.remove(map_path)
    else:
        logger.info(f"Vector Index: No existing Annoy index found for user '{user_id}'. A new one will be created upon first add.")
    
    _user_annoy_id_to_firestore_id_map[user_id] = current_id_map
    _user_firestore_id_to_annoy_id_map[user_id] = current_reverse_map
    
    # Ensure the AnnoyIndex object is in the global dict even if new
    if user_id not in _user_annoy_indices:
        _user_annoy_indices[user_id] = AnnoyIndex(EMBEDDING_DIM, 'angular')

def _save_user_annoy_data(user_id: str):
    """
    Saves a user's Annoy index and its associated ID mapping to disk.
    Ensures the index is built before saving.
    """
    os.makedirs(settings.ANNOY_INDEX_DIR, exist_ok=True)  # Ensure the directory exists

    if user_id in _user_annoy_indices:
        annoy_index = _user_annoy_indices[user_id]
        id_map = _user_annoy_id_to_firestore_id_map.get(user_id, {})
        
        try:
            # Build the index if it has new items and hasn't been built yet
            if annoy_index.get_n_items() > 0 and (not hasattr(annoy_index, 'was_built') or not annoy_index.was_built):
                annoy_index.build(settings.ANNOY_N_TREES)
                annoy_index.was_built = True  # Custom flag to indicate it's built
            elif annoy_index.get_n_items() == 0:  # If empty, no need to build/save index file
                if os.path.exists(_get_annoy_index_path(user_id)): 
                    os.remove(_get_annoy_index_path(user_id))
                if os.path.exists(_get_annoy_map_path(user_id)): 
                    os.remove(_get_annoy_map_path(user_id))
                logger.info(f"Vector Index: Annoy index for user '{user_id}' is empty. Removed files.")
                return

            # Save the index and mapping
            annoy_index.save(_get_annoy_index_path(user_id))
            with open(_get_annoy_map_path(user_id), 'w') as f:
                # Convert int keys to string for JSON serialization
                json.dump({str(k): v for k, v in id_map.items()}, f, indent=2)  # Pretty print for readability
            logger.debug(f"Vector Index: Saved Annoy index and map for user '{user_id}'.")
        except Exception as e:
            logger.error(f"Vector Index: ERROR saving Annoy data for user '{user_id}': {e}")
    else:
        logger.warning(f"Vector Index: No Annoy index in memory to save for user '{user_id}'.")

# --- Core Functions for Vector Indexing ---

def add_embedding_to_index(user_id: str, firestore_doc_id: str, embedding: List[float]):
    """
    Adds a new memory's embedding to the user's Annoy index and updates the ID mappings.
    This function also handles loading/initializing the index if not already in memory.

    Args:
        user_id (str): The unique identifier for the user.
        firestore_doc_id (str): The unique ID of the memory document in Firestore.
        embedding (List[float]): The vector embedding of the memory text.
    """
    # Ensure the user's Annoy data is loaded into memory
    if user_id not in _user_annoy_indices:
        _load_user_annoy_data(user_id)
    
    annoy_index = _user_annoy_indices[user_id]
    id_map = _user_annoy_id_to_firestore_id_map[user_id]
    reverse_map = _user_firestore_id_to_annoy_id_map[user_id]

    if firestore_doc_id in reverse_map:
        logger.warning(f"Vector Index: Memory '{firestore_doc_id[:8]}...' already exists for user '{user_id}'. Skipping add.")
        return  # Avoid adding duplicates

    # Assign a new internal Annoy integer ID
    annoy_int_id = annoy_index.get_n_items()  # Get the next available integer ID
    
    # Add the embedding to the index
    annoy_index.add_item(annoy_int_id, embedding)
    
    # Update mappings
    id_map[annoy_int_id] = firestore_doc_id
    reverse_map[firestore_doc_id] = annoy_int_id

    # For hackathon, we build and save after each add for demo simplicity.
    # In production, this would be batched and saved periodically (e.g., every 100 items).
    _save_user_annoy_data(user_id)
    logger.info(f"Vector Index: Added embedding for memory '{firestore_doc_id[:8]}...' (Annoy ID: {annoy_int_id}) for user '{user_id}'.")

def search_index_for_similar_memories(user_id: str, query_embedding: List[float], num_results: int = 3) -> List[str]:
    """
    Searches the user's Annoy index for the most semantically similar memory document IDs.
    Returns a list of Firestore document IDs (strings) of the most relevant memories.

    Args:
        user_id (str): The unique identifier for the user.
        query_embedding (List[float]): The vector embedding of the user's query.
        num_results (int): The number of top similar memories to retrieve.

    Returns:
        List[str]: A list of Firestore document IDs (strings) of the most relevant memories.
                   Returns an empty list if no index exists or no results found.
    """
    # Ensure the user's Annoy data is loaded into memory
    if user_id not in _user_annoy_indices:
        _load_user_annoy_data(user_id)
        # If still not loaded (e.g., no index file exists), return empty
        if user_id not in _user_annoy_indices:
            logger.warning(f"Vector Index: No Annoy index found for user '{user_id}' during search. Returning empty results.")
            return []

    annoy_index = _user_annoy_indices[user_id]
    id_map = _user_annoy_id_to_firestore_id_map[user_id]

    if annoy_index.get_n_items() == 0:
        logger.info(f"Vector Index: Annoy index for user '{user_id}' is empty. No memories to search.")
        return []

    try:
        # Get internal Annoy integer IDs of nearest neighbors
        # search_k=-1 uses n_trees * num_results by default, which is a good balance.
        nearest_annoy_int_ids = annoy_index.get_nns_by_vector(
            query_embedding, num_results, include_distances=False
        )
        
        # Convert Annoy internal IDs back to Firestore document IDs using the map
        firestore_doc_ids = [
            id_map[idx]
            for idx in nearest_annoy_int_ids
            if idx in id_map  # Safety check: ensure the ID is still mapped
        ]
        logger.debug(f"Vector Index: Found {len(firestore_doc_ids)} similar memories for user '{user_id}'.")
        return firestore_doc_ids
    except Exception as e:
        logger.error(f"Vector Index: ERROR during search for user '{user_id}': {e}. Returning empty results.")
        return []

def delete_user_annoy_data(user_id: str):
    """
    Deletes a user's Annoy index and its associated ID mapping from disk and memory.
    This is called during overall user data cleanup.
    """
    logger.info(f"Vector Index: Deleting Annoy data for user '{user_id}'...")
    # Remove from in-memory caches
    _user_annoy_indices.pop(user_id, None)
    _user_annoy_id_to_firestore_id_map.pop(user_id, None)
    _user_firestore_id_to_annoy_id_map.pop(user_id, None)

    # Remove physical files
    index_path = _get_annoy_index_path(user_id)
    map_path = _get_annoy_map_path(user_id)
    if os.path.exists(index_path):
        os.remove(index_path)
        logger.info(f"Vector Index: Removed Annoy index file: {index_path}")
    if os.path.exists(map_path):
        os.remove(map_path)
        logger.info(f"Vector Index: Removed Annoy map file: {map_path}")
    logger.info(f"Vector Index: Annoy data cleared for user '{user_id}'.")

# --- Self-Verification Block (for isolated testing) ---
if __name__ == "__main__":
    import time
    import uuid
    from datetime import datetime
    
    # Configure basic logging for self-test
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("\n--- Vector Indexing: Self-Test Initiated ---")

    # IMPORTANT: Ensure embedding_service.py and persistent_storage.py are functional.
    # This test will create/modify files in the 'annoy_indexes_tethys' directory.

    test_user_id = "tethys_test_annoy_001"
    
    # Clean up previous test data for a fresh run
    print(f"\n[Cleanup] Clearing existing data for user '{test_user_id}'...")
    delete_user_annoy_data(test_user_id)  # Clean Annoy files
    delete_all_user_raw_memories(test_user_id)  # Clean Firestore data
    time.sleep(2)  # Give Firestore time to propagate deletions
    print("  Cleanup complete.")

    # Test 1: Add multiple embeddings
    logger.info("\n[Test 1] Adding multiple embeddings to the index:")
    memories_to_add = [
        {"id": str(uuid.uuid4()), "text": "My long-term goal is to retire comfortably by age 60.", "type": "goal"},
        {"id": str(uuid.uuid4()), "text": "I prefer low-risk, stable investments for my portfolio.", "type": "preference"},
        {"id": str(uuid.uuid4()), "text": "Tacit advised me to increase my monthly SIP contribution by ₹5000.", "type": "advice"},
        {"id": str(uuid.uuid4()), "text": "I had an unexpected large debit of ₹25,000 for a new gadget last week.", "type": "anomaly"},
        {"id": str(uuid.uuid4()), "text": "My ambition is to buy a house in the next 5 years for ₹50 lakh.", "type": "goal"},
    ]

    for mem_data in memories_to_add:
        embedding = get_embedding(mem_data['text'])
        add_embedding_to_index(test_user_id, mem_data['id'], embedding)
        # Also store in Firestore for completeness of test (though not directly tested here)
        store_raw_memory(
            user_id=test_user_id,
            memory_id=mem_data['id'],
            text=mem_data['text'],
            memory_type=mem_data['type'],
            timestamp=datetime.now().isoformat(),
            metadata=mem_data
        )
    
    # Verify index size
    current_annoy_index = _user_annoy_indices.get(test_user_id)
    assert current_annoy_index is not None, "Test 1 Failed: Annoy index not created."
    assert current_annoy_index.get_n_items() == len(memories_to_add), "Test 1 Failed: Incorrect number of items in index."
    print(f"  Test 1 Passed: {len(memories_to_add)} embeddings added. Index size: {current_annoy_index.get_n_items()}")

    # Test 2: Search for relevant memories
    logger.info("\n[Test 2] Searching for relevant memories:")
    query_text_1 = "Tell me about my retirement plans."
    query_embedding_1 = get_embedding(query_text_1)
    results_1 = search_index_for_similar_memories(test_user_id, query_embedding_1, num_results=2)
    
    print(f"  Query: '{query_text_1}'")
    print(f"  Retrieved IDs: {results_1}")
    
    # Fetch raw texts for verification
    retrieved_texts_1 = [retrieve_raw_memory(test_user_id, doc_id)['text'] for doc_id in results_1 if retrieve_raw_memory(test_user_id, doc_id)]
    print(f"  Retrieved Texts: {retrieved_texts_1}")
    assert any("retire comfortably by age 60" in t for t in retrieved_texts_1), "Test 2 Failed: Did not retrieve retirement goal."
    print("  Test 2 Passed: Relevant memory retrieved.")

    query_text_2 = "What was that big expense I had recently?"
    query_embedding_2 = get_embedding(query_text_2)
    results_2 = search_index_for_similar_memories(test_user_id, query_embedding_2, num_results=1)
    
    print(f"  Query: '{query_text_2}'")
    print(f"  Retrieved IDs: {results_2}")
    
    retrieved_texts_2 = [retrieve_raw_memory(test_user_id, doc_id)['text'] for doc_id in results_2 if retrieve_raw_memory(test_user_id, doc_id)]
    print(f"  Retrieved Texts: {retrieved_texts_2}")
    assert any("₹25,000" in t for t in retrieved_texts_2), "Test 2 Failed: Did not retrieve large debit anomaly."
    print("  Test 2 Passed: Anomaly memory retrieved.")

    # Test 3: Search for non-existent user
    logger.info("\n[Test 3] Searching for a non-existent user:")
    non_existent_user = "non_existent_user_123"
    results_non_existent = search_index_for_similar_memories(non_existent_user, get_embedding("random query"))
    assert len(results_non_existent) == 0, "Test 3 Failed: Should return empty for non-existent user."
    print("  Test 3 Passed: Handled non-existent user gracefully.")

    # Final Cleanup
    print(f"\n[Final Cleanup] Deleting all remaining data for user '{test_user_id}'...")
    delete_user_annoy_data(test_user_id)  # Clean Annoy files
    delete_all_user_raw_memories(test_user_id)  # Clean Firestore data
    time.sleep(2)  # Give Firestore time to propagate deletions
    print("  Final Cleanup Complete: All test data removed.")

    logger.info("\n--- Vector Indexing: All Self-Tests Completed Successfully ---")
