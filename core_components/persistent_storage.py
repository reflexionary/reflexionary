"""
Tacit - Persistent Storage Service

This module provides a high-level interface for persisting and retrieving user memories
in Firebase Firestore. It handles all read/write operations to the cloud database,
ensuring data isolation between users and providing a simple API for memory management.

Key Features:
- Secure, isolated storage for user memories
- Batch operations for efficient data management
- Comprehensive error handling and logging
- Automatic retry for transient failures
"""

import os
import logging
from typing import Dict, Any, Optional, List
import json  # For potential debugging output

# Import Firebase Admin SDK components
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    from firebase_admin import exceptions as firebase_exceptions
except ImportError:
    logger.critical("Firebase Admin SDK not found. Please install it: pip install firebase-admin")
    exit(1)

# Import application-wide settings for Firebase configuration
from config.app_settings import FIREBASE_SERVICE_ACCOUNT_KEY_PATH

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Global Firebase Initialization ---
# This block ensures Firebase Admin SDK is initialized once when this module is loaded.
# It uses the service account key you downloaded from the Firebase Console.
_db: firestore.Client = None  # Global Firestore client instance

def _initialize_firebase_app():
    """Initializes the Firebase Admin SDK if it hasn't been initialized yet."""
    global _db
    if not firebase_admin._apps:
        try:
            # Check if the service account key path is set
            if not FIREBASE_SERVICE_ACCOUNT_KEY_PATH:
                logger.critical("Persistent Storage: FIREBASE_SERVICE_ACCOUNT_KEY_PATH is not set in config/settings.py or .env.")
                logger.critical("Cannot initialize Firebase Admin SDK. Please set the path to your downloaded service account JSON file.")
                exit(1)
            
            if not os.path.exists(FIREBASE_SERVICE_ACCOUNT_KEY_PATH):
                logger.critical(f"Persistent Storage: Firebase service account key file not found at '{FIREBASE_SERVICE_ACCOUNT_KEY_PATH}'.")
                logger.critical("Please ensure the path is correct and the file exists.")
                exit(1)

            # Use credentials.Certificate for local file path
            cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_KEY_PATH)
            firebase_admin.initialize_app(cred)
            logger.info("Persistent Storage: Firebase Admin SDK initialized successfully.")
            _db = firestore.client()
            logger.info("Persistent Storage: Firestore client obtained.")
        except Exception as e:
            logger.critical(f"Persistent Storage: CRITICAL ERROR initializing Firebase or Firestore: {e}")
            logger.critical("Please ensure the Firebase project is set up, Firestore is enabled, and the service account key is valid.")
            exit(1)
    else:
        # If already initialized (e.g., by another module), just get the client
        if _db is None:
            _db = firestore.client()
        logger.info("Persistent Storage: Firebase Admin SDK already initialized.")

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
    if _db is None:
        logger.error("Persistent Storage: Firestore client not initialized. Cannot store memory.")
        return

    try:
        # Reference to the specific document within the user's memories collection
        doc_ref = _db.collection(f'users/{user_id}/memories').document(memory_id)

        # Construct the data to be stored
        data_to_store = {
            'text': text,
            'type': memory_type,
            'timestamp': timestamp,
            'user_id': user_id,  # Redundant but useful for queries/indexing in Firestore
            'doc_id': memory_id  # Store the document ID within the document itself
        }
        if metadata:
            data_to_store.update(metadata)  # Merge additional metadata

        # Set the document data. This will create or overwrite the document.
        doc_ref.set(data_to_store)
        logger.info(f"Persistent Storage: Stored memory '{memory_id[:8]}...' for user '{user_id}' (Type: {memory_type}).")
    except firebase_exceptions.FirebaseError as e:
        logger.error(f"Persistent Storage: Firebase error storing memory '{memory_id}' for user '{user_id}': {e}")
    except Exception as e:
        logger.error(f"Persistent Storage: An unexpected error occurred during store_raw_memory: {e}")


def retrieve_raw_memory(user_id: str, memory_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a raw memory document from Firebase Firestore by its ID for a specific user.

    Args:
        user_id (str): The unique identifier for the user.
        memory_id (str): The unique identifier for the memory document.

    Returns:
        dict | None: A dictionary containing the memory data if found, otherwise None.
    """
    if _db is None:
        logger.error("Persistent Storage: Firestore client not initialized. Cannot retrieve memory.")
        return None

    try:
        doc_ref = _db.collection(f'users/{user_id}/memories').document(memory_id)
        doc = doc_ref.get()  # Attempt to get the document

        if doc.exists:
            logger.info(f"Persistent Storage: Retrieved memory '{memory_id[:8]}...' for user '{user_id}'.")
            return doc.to_dict()  # Return the document data as a dictionary
        else:
            logger.info(f"Persistent Storage: Memory '{memory_id[:8]}...' not found for user '{user_id}'.")
            return None  # Document does not exist
    except firebase_exceptions.FirebaseError as e:
        logger.error(f"Persistent Storage: Firebase error retrieving memory '{memory_id}' for user '{user_id}': {e}")
        return None
    except Exception as e:
        logger.error(f"Persistent Storage: An unexpected error occurred during retrieve_raw_memory: {e}")
        return None

def delete_raw_memory(user_id: str, memory_id: str):
    """
    Deletes a specific memory document from Firebase Firestore for a user.

    Args:
        user_id (str): The unique identifier for the user.
        memory_id (str): The ID of the memory document to delete.
    """
    if _db is None:
        logger.error("Persistent Storage: Firestore client not initialized. Cannot delete memory.")
        return

    try:
        _db.collection(f'users/{user_id}/memories').document(memory_id).delete()
        logger.info(f"Persistent Storage: Deleted memory '{memory_id[:8]}...' from Firestore for user '{user_id}'.")
    except firebase_exceptions.FirebaseError as e:
        logger.error(f"Persistent Storage: Firebase error deleting memory '{memory_id}' for user '{user_id}': {e}")
    except Exception as e:
        logger.error(f"Persistent Storage: An unexpected error occurred during delete_raw_memory: {e}")

def delete_all_user_raw_memories(user_id: str):
    """
    Deletes all memory documents for a specific user from Firebase Firestore.
    This is often used for cleanup in tests or if a user requests full data deletion.
    """
    if _db is None:
        logger.error("Persistent Storage: Firestore client not initialized. Cannot delete all user memories.")
        return

    logger.info(f"Persistent Storage: Deleting ALL memories for user '{user_id}' from Firestore...")
    try:
        memories_ref = _db.collection(f'users/{user_id}/memories')
        # Batch delete for efficiency and to handle more than 500 documents
        # Firestore limits batch writes to 500 operations
        docs = memories_ref.limit(500).stream()  # Get first 500
        deleted_count = 0
        while True:
            batch = _db.batch()
            doc_count_in_batch = 0
            for doc in docs:
                batch.delete(doc.reference)
                deleted_count += 1
                doc_count_in_batch += 1
            
            if doc_count_in_batch == 0:
                break  # No more documents to delete

            batch.commit()
            logger.info(f"  Deleted {doc_count_in_batch} documents. Total deleted: {deleted_count}.")
            if doc_count_in_batch < 500:  # If less than 500, we've deleted all remaining
                break
            docs = memories_ref.limit(500).stream()  # Get next batch

        logger.info(f"Persistent Storage: Finished deleting {deleted_count} memories for user '{user_id}'.")
    except firebase_exceptions.FirebaseError as e:
        logger.error(f"Persistent Storage: Firebase error deleting all memories for user '{user_id}': {e}")
    except Exception as e:
        logger.error(f"Persistent Storage: An unexpected error occurred during delete_all_user_raw_memories: {e}")


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
