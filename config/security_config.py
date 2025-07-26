"""
Security Configuration for Reflexionary

This module contains security-related configurations including:
- Authentication settings
- Authorization rules
- Password policies
- Session management
- CORS and security headers
"""

import os
from typing import Dict, List, Optional, Union
from datetime import timedelta

# --- Authentication ---
AUTH_METHODS = ['jwt', 'session', 'api_key']
DEFAULT_AUTH_METHOD = 'jwt'

# JWT Configuration
JWT_CONFIG = {
    'SECRET_KEY': os.getenv('JWT_SECRET_KEY', 'your-secret-key-please-change-in-production'),
    'ALGORITHM': 'HS256',  # Or 'RS256' for asymmetric keys
    'ACCESS_TOKEN_EXPIRE_MINUTES': 60 * 24,  # 24 hours
    'REFRESH_TOKEN_EXPIRE_DAYS': 30,  # 30 days
    'ISSUER': 'reflexionary',
    'AUDIENCE': ['reflexionary-web', 'reflexionary-mobile'],
    'LEEWAY': 30,  # seconds
}

# API Key Configuration
API_KEY_HEADER = 'X-API-Key'
API_KEY_LENGTH = 64
API_KEY_PREFIX = 'refx_'  # Prefix for API keys

# Session Configuration
SESSION_CONFIG = {
    'SECRET_KEY': os.getenv('SESSION_SECRET_KEY', 'your-session-secret-key-change-in-production'),
    'SESSION_COOKIE_NAME': 'reflexionary_session',
    'SESSION_COOKIE_SECURE': True,  # Only send over HTTPS
    'SESSION_COOKIE_HTTPONLY': True,  # Not accessible via JavaScript
    'SESSION_COOKIE_SAMESITE': 'Lax',  # CSRF protection
    'PERMANENT_SESSION_LIFETIME': timedelta(days=7),  # 7 days
    'SESSION_REFRESH_EACH_REQUEST': True,
}

# Password Policy
PASSWORD_POLICY = {
    'MIN_LENGTH': 12,
    'REQUIRE_UPPERCASE': True,
    'REQUIRE_LOWERCASE': True,
    'REQUIRE_NUMBERS': True,
    'REQUIRE_SPECIAL_CHARS': True,
    'SPECIAL_CHARS': '!@#$%^&*()_+-=[]{}|;:,.<>?',
    'MAX_PASSWORD_AGE_DAYS': 90,  # Require password change after 90 days
    'PASSWORD_HISTORY': 5,  # Remember last 5 passwords
    'MAX_FAILED_ATTEMPTS': 5,  # Lock account after 5 failed attempts
    'LOCKOUT_DURATION_MINUTES': 30,  # 30 minutes lockout
}

# CORS Configuration
CORS_CONFIG = {
    'ALLOW_ORIGINS': [
        'http://localhost:3000',  # Default React dev server
        'http://localhost:8000',  # Default FastAPI dev server
        'https://reflexionary.example.com',  # Production domain
    ],
    'ALLOW_METHODS': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    'ALLOW_HEADERS': [
        'Authorization',
        'Content-Type',
        'X-API-Key',
        'X-Requested-With',
    ],
    'ALLOW_CREDENTIALS': True,
    'MAX_AGE': 600,  # seconds
}

# Security Headers
SECURITY_HEADERS = {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block',
    'Content-Security-Policy': "default-src 'self'",
    'Referrer-Policy': 'strict-origin-when-cross-origin',
    'Permissions-Policy': 'geolocation=(), microphone=()',
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
}

# Rate Limiting
RATE_LIMITS = {
    'AUTH_ENDPOINTS': '100/hour',
    'API_ENDPOINTS': '1000/hour',
    'PUBLIC_ENDPOINTS': '10000/day',
    'STORAGE': 'redis://localhost:6379/0',  # For distributed rate limiting
}

# Security Middleware Configuration
MIDDLEWARE_CONFIG = {
    'TRUSTED_HOSTS': ['reflexionary.example.com', 'localhost'],
    'ALLOWED_HOSTS': ['*'],  # Restrict in production
    'PROXY_COUNT': 1,  # Number of proxy servers in front of the app
    'FORCE_HTTPS': True,
    'ENABLE_HSTS': True,
    'HSTS_PRELOAD': False,  # Only enable if you're sure
    'HSTS_INCLUDE_SUBDOMAINS': True,
    'HSTS_MAX_AGE': 31536000,  # 1 year
}

# OAuth2 Providers (if applicable)
OAUTH_PROVIDERS = {
    'google': {
        'client_id': os.getenv('GOOGLE_CLIENT_ID', ''),
        'client_secret': os.getenv('GOOGLE_CLIENT_SECRET', ''),
        'authorize_url': 'https://accounts.google.com/o/oauth2/auth',
        'token_url': 'https://oauth2.googleapis.com/token',
        'userinfo_url': 'https://www.googleapis.com/oauth2/v3/userinfo',
        'scopes': ['openid', 'email', 'profile'],
    },
    'github': {
        'client_id': os.getenv('GITHUB_CLIENT_ID', ''),
        'client_secret': os.getenv('GITHUB_CLIENT_SECRET', ''),
        'authorize_url': 'https://github.com/login/oauth/authorize',
        'token_url': 'https://github.com/login/oauth/access_token',
        'userinfo_url': 'https://api.github.com/user',
        'scopes': ['user:email'],
    },
}

# API Security
API_SECURITY = {
    'VERSION_HEADER': 'X-API-Version',
    'API_PREFIX': '/api/v1',
    'RATE_LIMIT': '1000/hour',
    'ENABLE_DOCS': True,  # Disable in production
    'DOCS_URL': '/docs',
    'REDOC_URL': '/redoc',
    'OPENAPI_URL': '/openapi.json',
}

# Data Privacy
DATA_PRIVACY = {
    'ANONYMIZE_IP': True,
    'LOG_PII': False,  # Log Personally Identifiable Information
    'ENCRYPT_FIELDS': [
        'email',
        'phone',
        'address',
        'ssn',
        'credit_card',
    ],
    'DATA_RETENTION_DAYS': {
        'user_data': 30,  # Days to keep user data after account deletion
        'logs': 90,
        'audit_logs': 365,
    },
}

# Security Monitoring
SECURITY_MONITORING = {
    'LOG_LOGIN_ATTEMPTS': True,
    'LOG_PASSWORD_CHANGES': True,
    'LOG_PERMISSION_CHANGES': True,
    'ALERT_ON_BRUTE_FORCE': True,
    'ALERT_ON_UNUSUAL_ACTIVITY': True,
    'ALERT_EMAILS': ['security@reflexionary.example.com'],
}

def is_valid_password(password: str) -> tuple[bool, Optional[str]]:
    """
    Validate a password against the password policy.
    
    Args:
        password: The password to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(password) < PASSWORD_POLICY['MIN_LENGTH']:
        return False, f"Password must be at least {PASSWORD_POLICY['MIN_LENGTH']} characters long"
        
    if PASSWORD_POLICY['REQUIRE_UPPERCASE'] and not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
        
    if PASSWORD_POLICY['REQUIRE_LOWERCASE'] and not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"
        
    if PASSWORD_POLICY['REQUIRE_NUMBERS'] and not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
        
    if PASSWORD_POLICY['REQUIRE_SPECIAL_CHARS'] and not any(c in PASSWORD_POLICY['SPECIAL_CHARS'] for c in password):
        return False, f"Password must contain at least one special character: {PASSWORD_POLICY['SPECIAL_CHARS']}"
        
    return True, None

def generate_api_key() -> str:
    """
    Generate a new API key.
    
    Returns:
        A new API key with the configured prefix and length
    """
    import secrets
    import string
    
    # Generate a random string of the specified length
    alphabet = string.ascii_letters + string.digits
    random_part = ''.join(secrets.choice(alphabet) for _ in range(API_KEY_LENGTH - len(API_KEY_PREFIX)))
    
    return f"{API_KEY_PREFIX}{random_part}"

def get_security_headers() -> Dict[str, str]:
    """
    Get the security headers to add to responses.
    
    Returns:
        Dictionary of security headers
    """
    return SECURITY_HEADERS

# Example usage
if __name__ == "__main__":
    # Test password validation
    test_passwords = [
        'weak',
        'BetterButStillWeak1',
        'StrongPassword123!@#',
        'NoSpecialChars123',
    ]
    
    for pwd in test_passwords:
        is_valid, message = is_valid_password(pwd)
        print(f"Password: {pwd}")
        print(f"Valid: {is_valid}")
        if not is_valid:
            print(f"Reason: {message}")
        print()
    
    # Generate an API key
    print(f"Generated API Key: {generate_api_key()}")
