"""
Database Configuration for Reflexionary

This module contains all database-related configurations including:
- Firebase/Firestore connection settings
- Collection/table names
- Security rules templates
- Connection pools and timeouts
"""
import os
from typing import Dict, Any, Optional

# --- Firebase Configuration ---
# Project and authentication settings
FIREBASE_PROJECT_ID: str = os.getenv('FIREBASE_PROJECT_ID', '')
FIREBASE_SERVICE_ACCOUNT_PATH: str = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH', 'config/firebase-service-account.json')
FIREBASE_DATABASE_URL: str = f"https://{FIREBASE_PROJECT_ID}.firebaseio.com" if FIREBASE_PROJECT_ID else ""

# Collection/Table Names
COLLECTION_NAMES: Dict[str, str] = {
    'memories': 'memories',
    'users': 'users',
    'sessions': 'user_sessions',
    'portfolios': 'user_portfolios',
    'transactions': 'transactions',
    'market_data': 'market_data_cache',
    'analytics': 'usage_analytics',
}

# Security Rules Template
SECURITY_RULES_TEMPLATE: str = """
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // User-specific data access
    match /users/{userId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
      
      // Subcollections under user
      match /{document=**} {
        allow read, write: if request.auth != null && request.auth.uid == userId;
      }
    }
    
    // Public read-only data
    match /market_data/{document=**} {
      allow read: if true;
      allow write: if false;  // Only admin can write
    }
  }
}
"""

# Firestore Settings
FIRESTORE_SETTINGS: Dict[str, Any] = {
    'project': FIREBASE_PROJECT_ID,
    'database': FIREBASE_DATABASE_URL,
    'timeout': 30,  # seconds
    'max_retries': 3,
    'batch_size': 500,  # Max documents per batch operation
}

# Connection Pool Settings
CONNECTION_POOL: Dict[str, Any] = {
    'max_connections': 100,
    'max_keepalive_connections': 20,
    'keepalive_expiry': 300,  # seconds
    'retry_delay': 1.0,  # seconds
}

# Caching Configuration
CACHE_SETTINGS: Dict[str, Any] = {
    'enabled': True,
    'ttl': 300,  # seconds (5 minutes)
    'max_size': 1000,  # items
    'strategy': 'lru',  # Least Recently Used eviction policy
}

# Index Configuration
INDEX_CONFIG: Dict[str, Any] = {
    'auto_create': True,  # Automatically create indexes if they don't exist
    'composite_indexes': [
        {
            'collection': 'memories',
            'fields': [
                {'fieldPath': 'userId', 'order': 'ascending'},
                {'fieldPath': 'timestamp', 'order': 'descending'}
            ]
        },
        {
            'collection': 'transactions',
            'fields': [
                {'fieldPath': 'userId', 'order': 'ascending'},
                {'fieldPath': 'date', 'order': 'descending'}
            ]
        }
    ]
}

# Backup Configuration
BACKUP_CONFIG: Dict[str, Any] = {
    'enabled': True,
    'frequency': 'daily',  # 'daily', 'weekly', or 'monthly'
    'time': '02:00',  # 2 AM
    'retention_days': 30,
    'storage_bucket': f"{FIREBASE_PROJECT_ID}.appspot.com" if FIREBASE_PROJECT_ID else ""
}

# Migration Settings
MIGRATION_CONFIG: Dict[str, Any] = {
    'enabled': True,
    'auto_apply': False,  # Require manual confirmation for migrations
    'migrations_dir': 'migrations',
    'version_table': 'schema_migrations',
}

def get_database_config() -> Dict[str, Any]:
    """
    Returns the complete database configuration as a dictionary.
    Can be used to initialize database connections.
    """
    return {
        'firebase': {
            'project_id': FIREBASE_PROJECT_ID,
            'database_url': FIREBASE_DATABASE_URL,
            'service_account_path': FIREBASE_SERVICE_ACCOUNT_PATH,
            'collections': COLLECTION_NAMES,
        },
        'settings': FIRESTORE_SETTINGS,
        'connection_pool': CONNECTION_POOL,
        'cache': CACHE_SETTINGS,
        'indexes': INDEX_CONFIG,
        'backup': BACKUP_CONFIG,
        'migrations': MIGRATION_CONFIG,
    }
