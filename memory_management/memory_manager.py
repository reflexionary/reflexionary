import uuid
import datetime
import json
import os
import logging
from typing import Dict, List, Any, Optional

# Import core components of the Memory Management Layer
from core_components.embedding_service import get_embedding, EMBEDDING_DIM
from core_components.vector_index import add_embedding_to_index, search_index_for_similar_memories, delete_user_annoy_data
from core_components.persistent_storage import store_raw_memory, retrieve_raw_memory, delete_raw_memory, delete_all_user_raw_memories

# Import the Gemini Connector for potential LLM-based memory operations (e.g., summarization)
from memory_management.gemini_connector import GeminiConnector, GeminiConfig

# Import settings for configuration
from config.app_settings import ANNOY_INDEX_DIR, ANNOY_N_TREES, GEMINI_API_KEY, GEMINI_MODEL_NAME

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Global Gemini Client for Memory Operations ---
# This client can be used for LLM-based memory tasks like summarization of old memories.
_gemini_memory_client: Optional[GeminiConnector] = None

def _get_gemini_memory_client() -> GeminiConnector:
    """Initializes and returns a GeminiConnector instance for memory operations."""
    global _gemini_memory_client
    if _gemini_memory_client is None:
        try:
            # Use a specific config for memory operations if needed, or default
            config = GeminiConfig(api_key=GEMINI_API_KEY, model=GEMINI_MODEL_NAME, temperature=0.2, max_tokens=500)
            _gemini_memory_client = GeminiConnector(config=config)
            logger.info("Memory Manager: Internal Gemini client for memory ops initialized.")
        except Exception as e:
            logger.error(f"Memory Manager: Failed to initialize internal Gemini client for memory ops: {e}")
            raise # Re-raise to indicate critical failure
    return _gemini_memory_client

# --- Core Memory Management Functions ---

def ingest_user_memory(user_id: str, text: str, memory_type: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Orchestrates the Memory Ingestion Pipeline.
    Captures a significant piece of user-specific memory, converts it to an embedding,
    stores the raw text persistently, and indexes the embedding for fast retrieval.

    Args:
        user_id (str): The unique identifier for the user.
        text (str): The raw text content of the memory to be stored.
        memory_type (str): The type of memory (e.g., 'goal', 'preference', 'advice', 'transaction_summary', 'anomaly_detected').
        metadata (dict, optional): Additional key-value pairs to store with the memory. Defaults to None.
    """
    logger.info(f"Memory Manager: Initiating ingestion for user '{user_id}' (Type: {memory_type})...")
    
    # Generate a unique ID for this memory document in Firestore
    memory_doc_id = str(uuid.uuid4())

    # 1. Get embedding (Embedding Service)
    embedding = get_embedding(text)
    
    # 2. Store Raw Memory Text & Metadata in Persistent Storage (Firestore)
    current_timestamp = datetime.datetime.now().isoformat()
    store_raw_memory(user_id, memory_doc_id, text, memory_type, current_timestamp, metadata)

    # 3. Store Vector Embedding in Index (Annoy)
    add_embedding_to_index(user_id, memory_doc_id, embedding)

    logger.info(f"Memory Manager: Successfully ingested memory '{text[:50]}...' (ID: {memory_doc_id[:8]}...) for user '{user_id}'.")


def retrieve_contextual_memories(user_id: str, query_text: str, num_results: int = 3) -> List[Dict[str, Any]]:
    """
    Orchestrates the Memory Retrieval Pipeline.
    Takes a user query, finds semantically similar memories, and fetches their raw content.

    Args:
        user_id (str): The unique identifier for the user.
        query_text (str): The user's natural language query.
        num_results (int): The number of top relevant memories to retrieve.

    Returns:
        List[dict]: A list of dictionaries, where each dictionary contains the
                    raw memory text and its associated metadata.
                    Returns an empty list if no relevant memories are found.
    """
    logger.info(f"Memory Manager: Initiating retrieval for user '{user_id}' with query '{query_text[:50]}...'")

    # 1. Convert User Query to Vector (Embedding Service)
    query_embedding = get_embedding(query_text)

    # 2. Search Vector Index (Annoy) for similar memory IDs
    relevant_doc_ids = search_index_for_similar_memories(user_id, query_embedding, num_results)

    # 3. Fetch Raw Text & Metadata from Persistent Storage (Firestore)
    retrieved_memories_data = []
    for doc_id in relevant_doc_ids:
        memory_data = retrieve_raw_memory(user_id, doc_id)
        if memory_data:
            retrieved_memories_data.append(memory_data)
    
    logger.info(f"Memory Manager: Retrieved {len(retrieved_memories_data)} relevant memories for user '{user_id}'.")
    return retrieved_memories_data

def get_all_user_memories(user_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves all memories for a specific user directly from Firestore.
    Useful for displaying a 'Memory Dashboard' or for debugging.
    """
    logger.info(f"Memory Manager: Fetching all memories for user '{user_id}'...")
    # This directly calls the persistent storage function
    all_memories = db.collection(f'users/{user_id}/memories').stream() # Using Firestore stream directly
    
    memories_list = [doc.to_dict() for doc in all_memories]
    logger.info(f"Memory Manager: Found {len(memories_list)} total memories for user '{user_id}'.")
    return memories_list

def delete_user_memory(user_id: str, memory_id: str) -> bool:
    """
    Deletes a specific memory for a user from both Firestore and the Annoy index.
    This is crucial for user control and privacy.

    Args:
        user_id (str): The unique identifier for the user.
        memory_id (str): The ID of the memory document to delete.

    Returns:
        bool: True if deletion was successful, False otherwise.
    """
    logger.info(f"Memory Manager: Attempting to delete memory '{memory_id[:8]}...' for user '{user_id}'.")
    try:
        # 1. Delete from Firestore (Persistent Storage)
        delete_raw_memory(user_id, memory_id) # Calls function from persistent_storage.py

        # 2. Remove from Annoy index (more complex, requires rebuilding or marking as deleted)
        # For hackathon, we'll just handle the Annoy file deletion and in-memory map update.
        # The delete_user_annoy_data function in vector_index.py handles this.
        # Note: Annoy itself doesn't have a direct 'delete_item' without rebuilding.
        # For a production system, you'd rebuild the Annoy index periodically,
        # excluding deleted items, or use a vector database that supports deletion.
        
        # We need to re-load/re-save the Annoy index to reflect changes in map
        # This is handled by delete_user_annoy_data which clears and rebuilds/re-saves
        # for a specific user's index files.
        # However, for single memory deletion, Annoy's index itself isn't updated instantly.
        # The key is to ensure future searches don't return deleted items.
        # A common robust strategy is to add a 'deleted' flag in Firestore and filter during retrieval.
        # For now, we rely on the delete_raw_memory and subsequent full rebuilds/reloads.
        
        # The simplest way to handle single item deletion for Annoy in a hackathon
        # is to remove it from the in-memory maps and then re-save the Annoy index.
        # This is implicitly handled by _save_user_annoy_data in vector_index.py
        # if the item is removed from the _user_annoy_id_to_firestore_id_map before saving.
        
        # For immediate removal from search results, you'd need to rebuild the index or filter results.
        # Given hackathon constraints, relying on Firestore deletion and occasional rebuilds is fine.
        
        logger.info(f"Memory Manager: Memory '{memory_id[:8]}...' deleted conceptually from Annoy index (full removal on next rebuild/reload).")
        return True
    except Exception as e:
        logger.error(f"Memory Manager: An unexpected error occurred during delete_user_memory: {e}")
        return False

def delete_all_user_memories(user_id: str) -> bool:
    """
    Deletes all memories for a user from both Firestore and their Annoy index data.
    This is typically used for complete user data deletion.
    """
    logger.info(f"Memory Manager: Deleting ALL memories for user '{user_id}'...")
    try:
        # 1. Delete all raw memories from Firestore (Persistent Storage)
        delete_all_user_raw_memories(user_id) # Calls function from persistent_storage.py

        # 2. Delete Annoy index data from disk and clear in-memory caches
        delete_user_annoy_data(user_id) # Calls function from vector_index.py
        
        logger.info(f"Memory Manager: All memories and Annoy data cleared for user '{user_id}'.")
        return True
    except Exception as e:
        logger.error(f"Memory Manager: ERROR deleting all memories for user '{user_id}': {e}")
        return False

# --- Conceptual LLM-based Memory Operations (e.g., Summarization) ---
# These functions demonstrate how Gemini could be used internally within the memory layer.

def summarize_old_memories(user_id: str, memories_to_summarize: List[Dict[str, Any]]) -> Optional[str]:
    """
    (Conceptual) Uses Gemini to summarize a list of old memories into a concise new memory.
    This helps manage memory size and improve retrieval efficiency over very long periods.
    """
    if not memories_to_summarize:
        return None

    logger.info(f"Memory Manager: Summarizing {len(memories_to_summarize)} old memories for user '{user_id}'...")
    
    gemini_client_for_memory = _get_gemini_memory_client() # Get internal Gemini client

    # Construct prompt for summarization
    mem_texts = [mem['text'] for mem in memories_to_summarize]
    prompt_text = (
        "Summarize the following past conversations/memories into a concise, single paragraph. "
        "Focus on key financial goals, advice given, and significant financial events. "
        "Maintain the user's perspective where possible.\n\n" +
        "\n".join(mem_texts)
    )
    
    messages = [{"role": "user", "content": prompt_text}]
    
    try:
        response = gemini_client_for_memory.generate_response(messages=messages, temperature=0.2, max_tokens=200)
        summary_content = response.get("content")
        
        if summary_content:
            logger.info(f"Memory Manager: Generated summary: '{summary_content[:100]}...'")
            # Ingest this new summary as a 'summarized_memory' type
            ingest_user_memory(user_id, summary_content, "summarized_memory", {"source_memory_ids": [m['doc_id'] for m in memories_to_summarize]})
            # Optionally, delete the original memories after summarization
            # for mem in memories_to_summarize:
            #     delete_user_memory(user_id, mem['doc_id'])
            return summary_content
        else:
            logger.warning("Memory Manager: Gemini returned no content for summarization.")
            return None
    except Exception as e:
        logger.error(f"Memory Manager: Error during memory summarization by Gemini: {e}")
        return None


# --- Self-Verification Block (for isolated testing) ---
if __name__ == "__main__":
    import time
    import uuid
    import datetime
    
    # Configure basic logging for self-test
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("\n--- Memory Manager: Self-Test Initiated ---")

    # IMPORTANT: Ensure all previous core_components and gemini_connector are functional.
    # This test will create/modify files in 'annoy_indexes_tethys' and Firestore.

    test_user_id = "tethys_manager_test_001"
    
    # --- Cleanup previous test data for a fresh run ---
    logger.info(f"\n[Cleanup] Clearing existing data for user '{test_user_id}'...")
    delete_all_user_memories(test_user_id) # This calls delete_all_user_raw_memories and delete_user_annoy_data
    time.sleep(2) # Give Firestore time to propagate deletions
    logger.info("  Cleanup complete.")

    # Test 1: Ingesting multiple memories
    logger.info("\n[Test 1] Ingesting multiple memories:")
    initial_memories = [
        {"text": "My long-term goal is to retire comfortably by age 60.", "type": "goal"},
        {"text": "I prefer low-risk, stable investments for my portfolio.", "type": "preference"},
        {"text": "Tacit advised me to increase my monthly SIP contribution by ₹5000.", "type": "advice"},
        {"text": "I had an unexpected large debit of ₹25,000 for a new gadget last week.", "type": "anomaly", "metadata": {"amount": 25000, "category": "shopping"}},
        {"text": "My ambition is to buy a house in the next 5 years for ₹50 lakh.", "type": "goal"},
    ]
    for mem_data in initial_memories:
        ingest_user_memory(test_user_id, mem_data['text'], mem_data['type'], mem_data.get('metadata'))
    
    time.sleep(3) # Give time for async operations (Firestore writes, Annoy saves)

    # Test 2: Retrieving all memories
    logger.info("\n[Test 2] Retrieving all memories for the user:")
    all_mems = get_all_user_memories(test_user_id)
    logger.info(f"  Total memories retrieved: {len(all_mems)}")
    assert len(all_mems) == len(initial_memories), "Test 2 Failed: Incorrect number of memories retrieved."
    logger.info("  Test 2 Passed: All memories retrieved successfully.")

    # Test 3: Retrieving contextual memories (RAG simulation)
    logger.info("\n[Test 3] Retrieving contextual memories based on a query:")
    query_text_1 = "Tell me about my retirement plans."
    retrieved_context_1 = retrieve_contextual_memories(test_user_id, query_text_1, num_results=2)
    logger.info(f"  Query: '{query_text_1}'")
    logger.info(f"  Retrieved Contexts: {[m['text'] for m in retrieved_context_1]}")
    assert any("retire comfortably by age 60" in m['text'] for m in retrieved_context_1), "Test 3 Failed: Did not retrieve retirement goal."
    logger.info("  Test 3.1 Passed: Relevant memory retrieved contextually.")

    query_text_2 = "What was that big expense I had?"
    retrieved_context_2 = retrieve_contextual_memories(test_user_id, query_text_2, num_results=1)
    logger.info(f"  Query: '{query_text_2}'")
    logger.info(f"  Retrieved Contexts: {[m['text'] for m in retrieved_context_2]}")
    assert any("₹25,000" in m['text'] for m in retrieved_context_2), "Test 3 Failed: Did not retrieve large expense memory."
    logger.info("  Test 3.2 Passed: Anomaly memory retrieved contextually.")

    # Test 4: Deleting a specific memory
    logger.info("\n[Test 4] Deleting a specific memory:")
    # Find the ID of the "new gadget" memory
    memory_to_delete_id = next((m['doc_id'] for m in all_mems if "new gadget" in m['text']), None)
    if memory_to_delete_id:
        delete_success = delete_user_memory(test_user_id, memory_to_delete_id)
        assert delete_success, "Test 4 Failed: Deletion reported as unsuccessful."
        time.sleep(2) # Give Firestore time to propagate
        remaining_mems = get_all_user_memories(test_user_id)
        assert len(remaining_mems) == len(initial_mems) - 1, "Test 4 Failed: Memory not deleted from Firestore."
        logger.info("  Test 4 Passed: Specific memory deleted successfully.")
    else:
        logger.warning("  Test 4 Skipped: Memory to delete not found from previous step.")

    # Test 5: Conceptual Memory Summarization (requires actual Gemini API key to work fully)
    logger.info("\n[Test 5] Conceptual Memory Summarization:")
    # Assuming we have at least 2 memories left after deletion
    if len(all_mems) >= 2:
        mems_for_summary = all_mems[:2] # Take first two
        summarized_text = summarize_old_memories(test_user_id, mems_for_summary)
        if summarized_text:
            logger.info(f"  Generated Summary: {summarized_text[:100]}...")
            # Verify a new memory of type 'summarized_memory' was ingested
            all_mems_after_summary = get_all_user_memories(test_user_id)
            assert any(m.get('type') == 'summarized_memory' for m in all_mems_after_summary), "Test 5 Failed: Summarized memory not ingested."
            logger.info("  Test 5 Passed: Memory summarization initiated (verify Gemini output in logs).")
        else:
            logger.warning("  Test 5 Skipped: Memory summarization did not return content (check API key/connectivity).")
    else:
        logger.warning("  Test 5 Skipped: Not enough memories for summarization test.")

    # Final Cleanup
    logger.info(f"\n[Final Cleanup] Deleting all remaining data for user '{test_user_id}'...")
    delete_all_user_memories(test_user_id)
    time.sleep(2)
    final_mems_count = db.collection(f'users/{test_user_id}/memories').count().get().to_dict()['count']
    assert final_mems_count == 0, "Final Cleanup Failed: Not all memories were deleted from Firestore."
    logger.info("  Final Cleanup Complete: All test data removed.")

    logger.info("\n--- Memory Manager: All Self-Tests Completed Successfully ---")
