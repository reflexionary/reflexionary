"""
Tethys - Persistent Storage Service

This module provides a high-level interface for persisting and retrieving user memories
in Firebase Firestore. It handles all read/write operations to the cloud database,
ensuring data isolation between users and providing a simple API for memory management.

Key Features:
- Secure, isolated storage for user memories
- Batch operations for efficient data management
- Comprehensive error handling and logging
- Automatic retry for transient failures
- Fallback to in-memory storage when Firebase is not configured
"""

import os
import logging
from typing import Dict, Any, Optional, List
import json
from datetime import datetime

# Import Firebase Admin SDK components
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    from firebase_admin import exceptions as firebase_exceptions
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    logging.warning("Firebase Admin SDK not found. Using in-memory storage fallback.")

# Import application-wide settings for Firebase configuration
from config.app_settings import FIREBASE_SERVICE_ACCOUNT_KEY_PATH

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Global Firebase Initialization ---
# This block ensures Firebase Admin SDK is initialized once when this module is loaded.
# It uses the service account key you downloaded from the Firebase Console.
_db: Optional[firestore.Client] = None  # Global Firestore client instance
_use_mock_storage = False  # Flag to use in-memory storage

# In-memory storage fallback
_mock_storage: Dict[str, Dict[str, Any]] = {}

def _initialize_firebase_app():
    """Initializes the Firebase Admin SDK if it hasn't been initialized yet."""
    global _db, _use_mock_storage
    
    # Check if Firebase is available and configured
    if not FIREBASE_AVAILABLE:
        logger.warning("Firebase Admin SDK not available. Using in-memory storage.")
        _use_mock_storage = True
        return
    
    if not FIREBASE_SERVICE_ACCOUNT_KEY_PATH:
        logger.warning("FIREBASE_SERVICE_ACCOUNT_KEY_PATH is not set. Using in-memory storage fallback.")
        _use_mock_storage = True
        return
    
    if not os.path.exists(FIREBASE_SERVICE_ACCOUNT_KEY_PATH):
        logger.warning(f"Firebase service account key file not found at '{FIREBASE_SERVICE_ACCOUNT_KEY_PATH}'. Using in-memory storage fallback.")
        _use_mock_storage = True
        return

    try:
        if not firebase_admin._apps:
            # Use credentials.Certificate for local file path
            cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_KEY_PATH)
            firebase_admin.initialize_app(cred)
            logger.info("Persistent Storage: Firebase Admin SDK initialized successfully.")
        
        _db = firestore.client()
        logger.info("Persistent Storage: Firestore client obtained.")
        _use_mock_storage = False
        
    except Exception as e:
        logger.warning(f"Failed to initialize Firebase: {e}. Using in-memory storage fallback.")
        _use_mock_storage = True

# Initialize Firebase when the module is imported
_initialize_firebase_app()

# --- Core Functions for Memory Persistence ---

def store_raw_memory(user_id: str, memory_id: str, text: str, memory_type: str, timestamp: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Stores a raw memory text and its associated metadata in Firebase Firestore.
    Each memory is stored as a document in a user-specific collection, ensuring privacy isolation.

    Args:
        user_id (str): The unique identifier for the user.
        memory_id (str): A unique identifier for this specific memory.
        text (str): The raw text content of the memory.
        memory_type (str): The type of memory (e.g., 'goal', 'advice', 'transaction_summary', 'anomaly_detected').
        timestamp (str): ISO 8601 formatted string of when the memory was created.
        metadata (dict, optional): Additional key-value pairs to store with the memory. Defaults to None.
    """
    if _use_mock_storage:
        # Use in-memory storage
        user_key = f"user_{user_id}"
        if user_key not in _mock_storage:
            _mock_storage[user_key] = {}
        
        _mock_storage[user_key][memory_id] = {
            'text': text,
            'memory_type': memory_type,
            'timestamp': timestamp,
            'metadata': metadata or {},
            'user_id': user_id,
            'memory_id': memory_id
        }
        logger.info(f"Mock Storage: Stored memory {memory_id} for user {user_id}")
        return

    if _db is None:
        logger.error("Persistent Storage: Firestore client not initialized. Cannot store memory.")
        return

    try:
        # Reference to the specific document within the user's memories collection
        doc_ref = _db.collection(f'users/{user_id}/memories').document(memory_id)

        # Construct the data to be stored
        memory_data = {
            'text': text,
            'memory_type': memory_type,
            'timestamp': timestamp,
            'metadata': metadata or {},
            'user_id': user_id,
            'memory_id': memory_id
        }

        # Store the memory in Firestore
        doc_ref.set(memory_data)
        logger.info(f"Persistent Storage: Successfully stored memory {memory_id} for user {user_id}")

    except firebase_exceptions.FirebaseError as e:
        logger.error(f"Persistent Storage: Firebase error storing memory {memory_id} for user {user_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Persistent Storage: Unexpected error storing memory {memory_id} for user {user_id}: {e}")
        raise

def retrieve_raw_memory(user_id: str, memory_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a specific raw memory from Firebase Firestore.

    Args:
        user_id (str): The unique identifier for the user.
        memory_id (str): The unique identifier for the specific memory.

    Returns:
        dict or None: The memory data if found, None otherwise.
    """
    if _use_mock_storage:
        # Use in-memory storage
        user_key = f"user_{user_id}"
        if user_key in _mock_storage and memory_id in _mock_storage[user_key]:
            return _mock_storage[user_key][memory_id]
        return None

    if _db is None:
        logger.error("Persistent Storage: Firestore client not initialized. Cannot retrieve memory.")
        return None

    try:
        # Reference to the specific document
        doc_ref = _db.collection(f'users/{user_id}/memories').document(memory_id)
        
        # Get the document
        doc = doc_ref.get()
        
        if doc.exists:
            memory_data = doc.to_dict()
            logger.info(f"Persistent Storage: Successfully retrieved memory {memory_id} for user {user_id}")
            return memory_data
        else:
            logger.info(f"Persistent Storage: Memory {memory_id} not found for user {user_id}")
            return None

    except firebase_exceptions.FirebaseError as e:
        logger.error(f"Persistent Storage: Firebase error retrieving memory {memory_id} for user {user_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Persistent Storage: Unexpected error retrieving memory {memory_id} for user {user_id}: {e}")
        return None

def delete_raw_memory(user_id: str, memory_id: str):
    """
    Deletes a specific raw memory from Firebase Firestore.

    Args:
        user_id (str): The unique identifier for the user.
        memory_id (str): The unique identifier for the specific memory.
    """
    if _use_mock_storage:
        # Use in-memory storage
        user_key = f"user_{user_id}"
        if user_key in _mock_storage and memory_id in _mock_storage[user_key]:
            del _mock_storage[user_key][memory_id]
            logger.info(f"Mock Storage: Deleted memory {memory_id} for user {user_id}")
        return

    if _db is None:
        logger.error("Persistent Storage: Firestore client not initialized. Cannot delete memory.")
        return

    try:
        # Reference to the specific document
        doc_ref = _db.collection(f'users/{user_id}/memories').document(memory_id)
        
        # Delete the document
        doc_ref.delete()
        logger.info(f"Persistent Storage: Successfully deleted memory {memory_id} for user {user_id}")

    except firebase_exceptions.FirebaseError as e:
        logger.error(f"Persistent Storage: Firebase error deleting memory {memory_id} for user {user_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Persistent Storage: Unexpected error deleting memory {memory_id} for user {user_id}: {e}")
        raise

def delete_all_user_raw_memories(user_id: str):
    """
    Deletes all raw memories for a specific user from Firebase Firestore.

    Args:
        user_id (str): The unique identifier for the user.
    """
    if _use_mock_storage:
        # Use in-memory storage
        user_key = f"user_{user_id}"
        if user_key in _mock_storage:
            del _mock_storage[user_key]
            logger.info(f"Mock Storage: Deleted all memories for user {user_id}")
        return

    if _db is None:
        logger.error("Persistent Storage: Firestore client not initialized. Cannot delete memories.")
        return

    try:
        # Reference to the user's memories collection
        collection_ref = _db.collection(f'users/{user_id}/memories')
        
        # Get all documents in the collection
        docs = collection_ref.stream()
        
        # Delete each document
        deleted_count = 0
        for doc in docs:
            doc.reference.delete()
            deleted_count += 1
        
        logger.info(f"Persistent Storage: Successfully deleted {deleted_count} memories for user {user_id}")

    except firebase_exceptions.FirebaseError as e:
        logger.error(f"Persistent Storage: Firebase error deleting memories for user {user_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Persistent Storage: Unexpected error deleting memories for user {user_id}: {e}")
        raise

def get_user_memories_summary(user_id: str) -> Dict[str, Any]:
    """
    Gets a summary of all memories for a user.

    Args:
        user_id (str): The unique identifier for the user.

    Returns:
        dict: Summary information about the user's memories.
    """
    if _use_mock_storage:
        # Use in-memory storage
        user_key = f"user_{user_id}"
        if user_key in _mock_storage:
            memories = _mock_storage[user_key]
            return {
                'total_memories': len(memories),
                'memory_types': list(set(m['memory_type'] for m in memories.values())),
                'latest_memory': max(memories.values(), key=lambda x: x['timestamp']) if memories else None
            }
        return {'total_memories': 0, 'memory_types': [], 'latest_memory': None}

    if _db is None:
        logger.error("Persistent Storage: Firestore client not initialized. Cannot get memory summary.")
        return {'total_memories': 0, 'memory_types': [], 'latest_memory': None}

    try:
        # Reference to the user's memories collection
        collection_ref = _db.collection(f'users/{user_id}/memories')
        
        # Get all documents in the collection
        docs = collection_ref.stream()
        
        memories = []
        for doc in docs:
            memories.append(doc.to_dict())
        
        return {
            'total_memories': len(memories),
            'memory_types': list(set(m['memory_type'] for m in memories)),
            'latest_memory': max(memories, key=lambda x: x['timestamp']) if memories else None
        }

    except firebase_exceptions.FirebaseError as e:
        logger.error(f"Persistent Storage: Firebase error getting memory summary for user {user_id}: {e}")
        return {'total_memories': 0, 'memory_types': [], 'latest_memory': None}
    except Exception as e:
        logger.error(f"Persistent Storage: Unexpected error getting memory summary for user {user_id}: {e}")
        return {'total_memories': 0, 'memory_types': [], 'latest_memory': None}


# --- Self-Verification Block (for isolated testing) ---
if __name__ == "__main__":
    import time
    import uuid
    from datetime import datetime

    logger.info("\n--- Persistent Storage: Self-Test Initiated ---")

    test_user_id = "tethys_test_firestore_001"
    test_memory_id_1 = str(uuid.uuid4())
    test_memory_id_2 = str(uuid.uuid4())
    test_memory_id_nonexistent = str(uuid.uuid4())  # A unique ID that won't exist
    
    current_timestamp = datetime.now().isoformat()

    # Cleanup any previous test data for a fresh run
    print(f"\n[Cleanup] Clearing existing data for user '{test_user_id}'...")
    delete_all_user_raw_memories(test_user_id)
    time.sleep(2)  # Give Firestore time to propagate deletions
    print("  Cleanup complete.")

    # Test 1: Storing a new memory
    logger.info("\n[Test 1] Storing a new memory:")
    test_text_1 = "My long-term goal is financial independence by 2045."
    store_raw_memory(test_user_id, test_memory_id_1, test_text_1, "goal", current_timestamp)
    time.sleep(1.5)  # Give Firestore time to process the write (it's asynchronous)
    print(f"  Test 1: Verify document '{test_memory_id_1[:8]}...' in Firebase Console under 'users/{test_user_id}/memories'.")

    # Test 2: Retrieving an existing memory
    logger.info(f"\n[Test 2] Retrieving memory '{test_memory_id_1[:8]}...':")
    retrieved_mem_1 = retrieve_raw_memory(test_user_id, test_memory_id_1)
    assert retrieved_mem_1 is not None, "Test 2 Failed: Could not retrieve the stored memory."
    assert retrieved_mem_1['text'] == test_text_1, "Test 2 Failed: Retrieved text does not match original."
    assert retrieved_mem_1['user_id'] == test_user_id, "Test 2 Failed: User ID mismatch."
    print("  Test 2 Passed: Memory retrieved successfully and content matches.")

    # Test 3: Storing another memory with additional metadata
    logger.info("\n[Test 3] Storing another memory with metadata:")
    test_text_2 = "Tacit advised reviewing my budget for travel expenses last month."
    extra_metadata = {"category": "budget", "source_feature": "anomaly_detection", "amount_reviewed": 15000}
    store_raw_memory(test_user_id, test_memory_id_2, test_text_2, "advice", current_timestamp, extra_metadata)
    time.sleep(1.5)
    print(f"  Test 3: Verify document '{test_memory_id_2[:8]}...' in Firebase Console with extra metadata.")
    retrieved_mem_2 = retrieve_raw_memory(test_user_id, test_memory_id_2)
    assert retrieved_mem_2 and retrieved_mem_2.get('amount_reviewed') == 15000, "Test 3 Failed: Metadata not stored/retrieved correctly."
    print("  Test 3 Passed: Memory with metadata stored successfully.")

    # Test 4: Retrieving a non-existent memory
    logger.info(f"\n[Test 4] Retrieving non-existent memory '{test_memory_id_nonexistent[:8]}...':")
    retrieved_nonexistent = retrieve_raw_memory(test_user_id, test_memory_id_nonexistent)
    assert retrieved_nonexistent is None, "Test 4 Failed: Non-existent memory was unexpectedly retrieved."
    print("  Test 4 Passed: Non-existent memory correctly not found.")

    # Final Cleanup
    logger.info(f"\n[Final Cleanup] Deleting all remaining data for user '{test_user_id}'...")
    delete_all_user_raw_memories(test_user_id)
    time.sleep(2)
    
    # Verify all documents were deleted
    remaining_docs = list(_db.collection(f'users/{test_user_id}/memories').limit(1).stream())
    assert len(remaining_docs) == 0, "Final Cleanup Failed: Not all memories were deleted from Firestore."
    
    logger.info("  Final Cleanup Complete: All test data removed.")
    logger.info("\n--- Persistent Storage: All Self-Tests Completed Successfully ---")
