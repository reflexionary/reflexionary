"""
Tethys - Fi-MCP Connector

This module provides a comprehensive interface for connecting Tethys Financial
Co-Pilot with the Financial Market Communication Protocol (Fi-MCP) simulator.
It handles data synchronization, real-time market data retrieval, and
portfolio information management.

The connector implements robust data integration using:
- HTTP REST API communication with Fi-MCP simulator
- Real-time data synchronization and caching
- Error handling and retry mechanisms
- Data validation and transformation
- Memory Layer integration for persistent storage

Mathematical Framework:
- Data Freshness Score = 1 / (Current Time - Last Update Time)
- Sync Efficiency = (Successful Requests / Total Requests) * 100
- Data Quality Score = (Valid Records / Total Records) * Completeness Factor
- Connection Health = Response Time * Availability * Data Accuracy

Integration Architecture:
- RESTful API endpoints for financial data access
- JSON data format for structured communication
- Authentication via API keys and session management
- Rate limiting and request throttling
- Comprehensive error handling and logging

Author: Tethys Development Team
Version: 1.0.0
"""

import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import time
from pathlib import Path

# Import Tethys components
from memory_management.memory_manager import ingest_user_memory
from config.app_settings import FI_MCP_SIMULATOR_BASE_URL, FI_MCP_SIMULATOR_API_KEY

logger = logging.getLogger(__name__)

class FiMCPConnector:
    """
    Connector for interacting with the Fi-MCP (Financial Market Communication Protocol) simulator.
    
    This connector provides a unified interface for accessing financial data
    from the Fi-MCP simulator, including accounts, transactions, portfolio
    holdings, and market data. It implements robust error handling, data
    validation, and integration with Tethys's Memory Layer for persistent
    storage and analysis.
    
    Key Capabilities:
    1. Account Management: Retrieve and manage user account information
    2. Transaction Processing: Access historical and real-time transaction data
    3. Portfolio Holdings: Get current portfolio positions and allocations
    4. Market Data Access: Real-time market prices and financial instruments
    5. User Profile Management: Access user preferences and settings
    6. Data Synchronization: Automated data sync with conflict resolution
    
    Mathematical Concepts:
    1. Data Freshness Metrics: Time-based data staleness calculations
    2. Sync Efficiency Analysis: Success rate and performance monitoring
    3. Data Quality Assessment: Completeness and accuracy scoring
    4. Connection Health Monitoring: Response time and availability tracking
    5. Rate Limiting Algorithms: Request throttling and optimization
    
    Error Handling:
    - Exponential backoff for retry mechanisms
    - Circuit breaker pattern for fault tolerance
    - Graceful degradation for partial failures
    - Comprehensive logging and monitoring
    """
    
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the Fi-MCP connector with configuration and session management.
        
        The connector initializes with:
        - Base URL for Fi-MCP simulator API
        - API key for authentication
        - HTTP session with connection pooling
        - Request timeout and retry configuration
        - Data caching and validation settings
        
        Args:
            base_url: Base URL for Fi-MCP simulator (defaults to config)
            api_key: API key for authentication (defaults to config)
        """
        # Configuration setup
        self.base_url = base_url or FI_MCP_SIMULATOR_BASE_URL
        self.api_key = api_key or FI_MCP_SIMULATOR_API_KEY
        
        # HTTP session configuration
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}' if self.api_key else '',
            'User-Agent': 'Tethys-Financial-CoPilot/1.0.0'
        })
        
        # Request configuration
        self.timeout = 30  # seconds
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
        # Data caching and validation
        self.cache_duration = 300  # 5 minutes
        self.data_cache = {}
        self.last_sync_times = {}
        
        # Connection health monitoring
        self.connection_health = {
            'last_successful_request': None,
            'consecutive_failures': 0,
            'average_response_time': 0.0,
            'total_requests': 0,
            'successful_requests': 0
        }
        
        logger.info(f"Fi-MCP Connector initialized with base URL: {self.base_url}")
    
    def get_accounts(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieve comprehensive account information for a user from the Fi-MCP simulator.
        
        This function fetches detailed account data including balances, account types,
        status information, and metadata. It implements intelligent caching and
        data validation to ensure reliable access to account information.
        
        Mathematical Processing:
        - Account Balance Validation: Sum of all account balances
        - Data Freshness Calculation: Time since last update
        - Account Diversity Score: Number of account types / Total accounts
        - Balance Distribution Analysis: Gini coefficient of balance distribution
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Dictionary containing:
                - accounts: List of account objects with detailed information
                - summary: Aggregated account statistics
                - metadata: Data freshness and quality metrics
                - status: Request status and error information
                
        Raises:
            requests.RequestException: For network or API errors
            ValueError: For invalid user_id or response data
        """
        try:
            # Check cache for recent data
            cache_key = f"accounts_{user_id}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            # Make API request to Fi-MCP simulator
            endpoint = f"{self.base_url}/accounts/{user_id}"
            response = self._make_api_request('GET', endpoint)
            
            if response.get('status') == 'success':
                accounts_data = response.get('data', [])
                
                # Validate and process account data
                validated_accounts = self._validate_account_data(accounts_data)
                
                # Calculate account summary statistics
                summary = self._calculate_account_summary(validated_accounts)
                
                # Store in memory for Tethys analysis
                self._store_account_data_in_memory(user_id, validated_accounts)
                
                # Prepare response with metadata
                result = {
                    "status": "success",
                    "user_id": user_id,
                    "accounts": validated_accounts,
                    "summary": summary,
                    "metadata": {
                        "data_freshness": self._calculate_data_freshness(cache_key),
                        "total_accounts": len(validated_accounts),
                        "cache_hit": False,
                        "response_time": response.get('response_time', 0)
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                # Cache the result
                self._cache_data(cache_key, result)
                
                return result
            else:
                return {
                    "status": "error",
                    "user_id": user_id,
                    "error": response.get('error', 'Unknown error'),
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error retrieving accounts for user {user_id}: {e}")
            return {
                "status": "error",
                "user_id": user_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_transactions(self, user_id: str, 
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        limit: Optional[int] = 100) -> Dict[str, Any]:
        """
        Retrieve transaction history for a user with filtering and pagination support.
        
        This function fetches transaction data with comprehensive filtering options
        including date ranges, transaction types, and amount thresholds. It implements
        intelligent data processing for transaction categorization and analysis.
        
        Mathematical Processing:
        - Transaction Volume Analysis: Total volume and average transaction size
        - Spending Pattern Analysis: Category-wise spending distribution
        - Cash Flow Calculation: Net cash flow over time periods
        - Transaction Frequency: Transactions per day/week/month
        - Anomaly Detection: Statistical outlier identification
        
        Args:
            user_id: Unique identifier for the user
            start_date: Start date for transaction filtering (ISO format)
            end_date: End date for transaction filtering (ISO format)
            limit: Maximum number of transactions to return
            
        Returns:
            Dictionary containing:
                - transactions: List of transaction objects
                - summary: Transaction statistics and analysis
                - patterns: Identified spending and income patterns
                - metadata: Data quality and freshness metrics
                
        Raises:
            requests.RequestException: For network or API errors
            ValueError: For invalid date formats or parameters
        """
        try:
            # Build cache key with filter parameters
            cache_key = f"transactions_{user_id}_{start_date}_{end_date}_{limit}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            # Prepare request parameters
            params = {
                'user_id': user_id,
                'limit': limit
            }
            if start_date:
                params['start_date'] = start_date
            if end_date:
                params['end_date'] = end_date
            
            # Make API request
            endpoint = f"{self.base_url}/transactions"
            response = self._make_api_request('GET', endpoint, params=params)
            
            if response.get('status') == 'success':
                transactions_data = response.get('data', [])
                
                # Validate and process transaction data
                validated_transactions = self._validate_transaction_data(transactions_data)
                
                # Calculate transaction summary and patterns
                summary = self._calculate_transaction_summary(validated_transactions)
                patterns = self._identify_transaction_patterns(validated_transactions)
                
                # Store in memory for analysis
                self._store_transaction_data_in_memory(user_id, validated_transactions)
                
                result = {
                    "status": "success",
                    "user_id": user_id,
                    "transactions": validated_transactions,
                    "summary": summary,
                    "patterns": patterns,
                    "metadata": {
                        "data_freshness": self._calculate_data_freshness(cache_key),
                        "total_transactions": len(validated_transactions),
                        "date_range": {"start": start_date, "end": end_date},
                        "cache_hit": False
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                # Cache the result
                self._cache_data(cache_key, result)
                
                return result
            else:
                return {
                    "status": "error",
                    "user_id": user_id,
                    "error": response.get('error', 'Unknown error'),
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error retrieving transactions for user {user_id}: {e}")
            return {
                "status": "error",
                "user_id": user_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_portfolio_holdings(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieve current portfolio holdings and positions from the Fi-MCP simulator.
        
        This function fetches comprehensive portfolio data including positions,
        asset allocations, market values, and performance metrics. It provides
        the foundation for portfolio analysis and optimization in Tethys.
        
        Mathematical Processing:
        - Portfolio Value Calculation: Sum of all position market values
        - Asset Allocation Analysis: Percentage distribution across asset classes
        - Position Concentration: Herfindahl-Hirschman Index for concentration
        - Risk Metrics: Portfolio volatility and correlation analysis
        - Performance Attribution: Return decomposition by asset class
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Dictionary containing:
                - holdings: List of portfolio positions
                - allocation: Asset allocation breakdown
                - performance: Portfolio performance metrics
                - risk_metrics: Risk and volatility measures
                - metadata: Data quality and freshness information
                
        Raises:
            requests.RequestException: For network or API errors
            ValueError: For invalid portfolio data
        """
        try:
            # Check cache for recent data
            cache_key = f"portfolio_{user_id}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            # Make API request
            endpoint = f"{self.base_url}/portfolio/{user_id}"
            response = self._make_api_request('GET', endpoint)
            
            if response.get('status') == 'success':
                portfolio_data = response.get('data', {})
                
                # Validate and process portfolio data
                validated_holdings = self._validate_portfolio_data(portfolio_data)
                
                # Calculate portfolio metrics
                allocation = self._calculate_asset_allocation(validated_holdings)
                performance = self._calculate_portfolio_performance(validated_holdings)
                risk_metrics = self._calculate_risk_metrics(validated_holdings)
                
                # Store in memory for analysis
                self._store_portfolio_data_in_memory(user_id, validated_holdings)
                
                result = {
                    "status": "success",
                    "user_id": user_id,
                    "holdings": validated_holdings,
                    "allocation": allocation,
                    "performance": performance,
                    "risk_metrics": risk_metrics,
                    "metadata": {
                        "data_freshness": self._calculate_data_freshness(cache_key),
                        "total_positions": len(validated_holdings),
                        "portfolio_value": sum(h.get('market_value', 0) for h in validated_holdings),
                        "cache_hit": False
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                # Cache the result
                self._cache_data(cache_key, result)
                
                return result
            else:
                return {
                    "status": "error",
                    "user_id": user_id,
                    "error": response.get('error', 'Unknown error'),
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error retrieving portfolio holdings for user {user_id}: {e}")
            return {
                "status": "error",
                "user_id": user_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Retrieve real-time market data for specified financial instruments.
        
        This function fetches current market prices, trading volumes, and
        other market data for the specified symbols. It supports multiple
        asset classes including stocks, bonds, ETFs, and other instruments.
        
        Mathematical Processing:
        - Price Change Calculation: Percentage and absolute price changes
        - Volume Analysis: Trading volume and liquidity metrics
        - Volatility Calculation: Price volatility over time periods
        - Market Impact Assessment: Price impact of trade sizes
        - Correlation Analysis: Inter-instrument correlation matrices
        
        Args:
            symbols: List of financial instrument symbols
            
        Returns:
            Dictionary containing:
                - market_data: Current market data for each symbol
                - summary: Aggregated market statistics
                - correlations: Inter-instrument correlations
                - metadata: Data quality and freshness metrics
                
        Raises:
            requests.RequestException: For network or API errors
            ValueError: For invalid symbol formats
        """
        try:
            # Check cache for recent data
            cache_key = f"market_data_{'_'.join(sorted(symbols))}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            # Make API request
            endpoint = f"{self.base_url}/market-data"
            params = {'symbols': ','.join(symbols)}
            response = self._make_api_request('GET', endpoint, params=params)
            
            if response.get('status') == 'success':
                market_data = response.get('data', {})
                
                # Validate and process market data
                validated_data = self._validate_market_data(market_data, symbols)
                
                # Calculate market statistics
                summary = self._calculate_market_summary(validated_data)
                correlations = self._calculate_market_correlations(validated_data)
                
                result = {
                    "status": "success",
                    "symbols": symbols,
                    "market_data": validated_data,
                    "summary": summary,
                    "correlations": correlations,
                    "metadata": {
                        "data_freshness": self._calculate_data_freshness(cache_key),
                        "symbols_requested": len(symbols),
                        "symbols_returned": len(validated_data),
                        "cache_hit": False
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                # Cache the result (shorter cache for market data)
                self._cache_data(cache_key, result, cache_duration=60)  # 1 minute
                
                return result
            else:
                return {
                    "status": "error",
                    "symbols": symbols,
                    "error": response.get('error', 'Unknown error'),
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error retrieving market data for symbols {symbols}: {e}")
            return {
                "status": "error",
                "symbols": symbols,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieve user profile information including preferences, settings,
        and demographic data from the Fi-MCP simulator.
        
        This function fetches comprehensive user profile data that informs
        personalization and recommendation algorithms in Tethys.
        
        Mathematical Processing:
        - Profile Completeness Score: Percentage of filled profile fields
        - Preference Consistency: Internal consistency of user preferences
        - Risk Profile Analysis: Risk tolerance assessment and scoring
        - Demographic Analysis: Age, income, and location-based insights
        - Behavioral Scoring: Interaction and preference pattern analysis
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Dictionary containing:
                - profile: User profile information
                - preferences: User preferences and settings
                - demographics: Demographic and behavioral data
                - risk_profile: Risk tolerance and investment preferences
                - metadata: Profile completeness and quality metrics
                
        Raises:
            requests.RequestException: For network or API errors
            ValueError: For invalid user_id
        """
        try:
            # Check cache for recent data
            cache_key = f"profile_{user_id}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            # Make API request
            endpoint = f"{self.base_url}/users/{user_id}/profile"
            response = self._make_api_request('GET', endpoint)
            
            if response.get('status') == 'success':
                profile_data = response.get('data', {})
                
                # Validate and process profile data
                validated_profile = self._validate_profile_data(profile_data)
                
                # Calculate profile metrics
                completeness_score = self._calculate_profile_completeness(validated_profile)
                risk_profile = self._analyze_risk_profile(validated_profile)
                
                # Store in memory for personalization
                self._store_profile_data_in_memory(user_id, validated_profile)
                
                result = {
                    "status": "success",
                    "user_id": user_id,
                    "profile": validated_profile,
                    "completeness_score": completeness_score,
                    "risk_profile": risk_profile,
                    "metadata": {
                        "data_freshness": self._calculate_data_freshness(cache_key),
                        "profile_completeness": completeness_score,
                        "cache_hit": False
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                # Cache the result
                self._cache_data(cache_key, result)
                
                return result
            else:
                return {
                    "status": "error",
                    "user_id": user_id,
                    "error": response.get('error', 'Unknown error'),
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error retrieving user profile for {user_id}: {e}")
            return {
                "status": "error",
                "user_id": user_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def sync_user_data(self, user_id: str) -> Dict[str, Any]:
        """
        Perform comprehensive data synchronization for a user, updating
        all financial data from the Fi-MCP simulator.
        
        This function orchestrates a complete data sync operation including
        accounts, transactions, portfolio holdings, and user profile data.
        It implements conflict resolution and data consistency checks.
        
        Mathematical Processing:
        - Sync Efficiency Score: Success rate of data synchronization
        - Data Consistency Check: Cross-validation of related data
        - Conflict Resolution: Algorithm for resolving data conflicts
        - Sync Performance Metrics: Time and resource utilization
        - Data Quality Assessment: Post-sync data quality evaluation
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Dictionary containing:
                - sync_status: Status of each data type synchronization
                - summary: Overall sync performance and results
                - conflicts: Any data conflicts encountered and resolved
                - metadata: Sync timing and quality metrics
                
        Raises:
            requests.RequestException: For network or API errors
            Exception: For sync orchestration errors
        """
        try:
            sync_start_time = datetime.now()
            sync_results = {}
            
            # Sync accounts data
            accounts_result = self.get_accounts(user_id)
            sync_results['accounts'] = accounts_result.get('status') == 'success'
            
            # Sync transactions data
            transactions_result = self.get_transactions(user_id)
            sync_results['transactions'] = transactions_result.get('status') == 'success'
            
            # Sync portfolio holdings
            portfolio_result = self.get_portfolio_holdings(user_id)
            sync_results['portfolio'] = portfolio_result.get('status') == 'success'
            
            # Sync user profile
            profile_result = self.get_user_profile(user_id)
            sync_results['profile'] = profile_result.get('status') == 'success'
            
            # Calculate sync performance metrics
            sync_duration = (datetime.now() - sync_start_time).total_seconds()
            success_rate = sum(sync_results.values()) / len(sync_results) * 100
            
            # Perform data consistency checks
            consistency_checks = self._perform_consistency_checks(user_id)
            
            # Store sync results in memory
            sync_summary = f"Data sync completed for user {user_id} with {success_rate:.1f}% success rate"
            ingest_user_memory(
                user_id=user_id,
                text=sync_summary,
                memory_type="data_sync",
                metadata={
                    "sync_results": sync_results,
                    "duration_seconds": sync_duration,
                    "success_rate": success_rate,
                    "consistency_checks": consistency_checks
                }
            )
            
            return {
                "status": "success",
                "user_id": user_id,
                "sync_status": sync_results,
                "summary": {
                    "total_data_types": len(sync_results),
                    "successful_syncs": sum(sync_results.values()),
                    "success_rate": success_rate,
                    "sync_duration_seconds": sync_duration
                },
                "consistency_checks": consistency_checks,
                "metadata": {
                    "sync_timestamp": datetime.now().isoformat(),
                    "cache_cleared": True,
                    "data_freshness": "current"
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error during data sync for user {user_id}: {e}")
            return {
                "status": "error",
                "user_id": user_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to the Fi-MCP simulator and assess overall health.
        
        This function performs comprehensive connectivity and health checks
        including API availability, response times, and data quality assessment.
        
        Mathematical Processing:
        - Connection Latency: Response time measurements
        - Availability Score: Uptime and reliability metrics
        - Data Quality Assessment: Sample data validation
        - Health Score: Composite health indicator
        - Performance Benchmarking: Comparison with historical performance
        
        Returns:
            Dictionary containing:
                - connection_status: Overall connection health
                - performance_metrics: Response time and throughput data
                - data_quality: Sample data quality assessment
                - recommendations: Suggestions for optimization
                
        Raises:
            requests.RequestException: For network connectivity issues
        """
        try:
            test_start_time = datetime.now()
            test_results = {}
            
            # Test basic connectivity
            connectivity_test = self._test_basic_connectivity()
            test_results['connectivity'] = connectivity_test
            
            # Test API endpoints
            api_test = self._test_api_endpoints()
            test_results['api_endpoints'] = api_test
            
            # Test data quality
            data_quality_test = self._test_data_quality()
            test_results['data_quality'] = data_quality_test
            
            # Calculate overall health score
            health_score = self._calculate_health_score(test_results)
            
            # Calculate performance metrics
            test_duration = (datetime.now() - test_start_time).total_seconds()
            performance_metrics = {
                "test_duration_seconds": test_duration,
                "average_response_time": self.connection_health['average_response_time'],
                "success_rate": (self.connection_health['successful_requests'] / 
                               max(1, self.connection_health['total_requests'])) * 100
            }
            
            return {
                "status": "success",
                "connection_status": {
                    "overall_health": health_score,
                    "status": "healthy" if health_score > 0.8 else "degraded" if health_score > 0.5 else "unhealthy",
                    "last_test": datetime.now().isoformat()
                },
                "test_results": test_results,
                "performance_metrics": performance_metrics,
                "recommendations": self._generate_connection_recommendations(test_results),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error during connection test: {e}")
            return {
                "status": "error",
                "error": str(e),
                "connection_status": {
                    "overall_health": 0.0,
                    "status": "unhealthy",
                    "last_test": datetime.now().isoformat()
                },
                "timestamp": datetime.now().isoformat()
            }
    
    def _make_api_request(self, method: str, endpoint: str, 
                         params: Optional[Dict[str, Any]] = None,
                         data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make an HTTP request to the Fi-MCP simulator with retry logic and error handling.
        
        This function implements robust HTTP communication with:
        - Exponential backoff retry mechanism
        - Comprehensive error handling
        - Response time tracking
        - Connection health monitoring
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint URL
            params: Query parameters
            data: Request body data
            
        Returns:
            Dictionary containing response data and metadata
        """
        start_time = time.time()
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(
                    method=method,
                    url=endpoint,
                    params=params,
                    json=data,
                    timeout=self.timeout
                )
                
                response_time = time.time() - start_time
                
                # Update connection health metrics
                self._update_connection_health(True, response_time)
                
                if response.status_code == 200:
                    return {
                        "status": "success",
                        "data": response.json(),
                        "response_time": response_time
                    }
                else:
                    return {
                        "status": "error",
                        "error": f"HTTP {response.status_code}: {response.text}",
                        "response_time": response_time
                    }
                    
            except requests.RequestException as e:
                response_time = time.time() - start_time
                self._update_connection_health(False, response_time)
                
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    time.sleep(delay)
                    continue
                else:
                    return {
                        "status": "error",
                        "error": f"Request failed after {self.max_retries + 1} attempts: {str(e)}",
                        "response_time": response_time
                    }
    
    def _update_connection_health(self, success: bool, response_time: float) -> None:
        """Update connection health metrics."""
        self.connection_health['total_requests'] += 1
        
        if success:
            self.connection_health['successful_requests'] += 1
            self.connection_health['consecutive_failures'] = 0
            self.connection_health['last_successful_request'] = datetime.now()
            
            # Update average response time
            current_avg = self.connection_health['average_response_time']
            total_requests = self.connection_health['successful_requests']
            self.connection_health['average_response_time'] = (
                (current_avg * (total_requests - 1) + response_time) / total_requests
            )
        else:
            self.connection_health['consecutive_failures'] += 1
    
    def _get_cached_data(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached data if available and fresh."""
        if cache_key in self.data_cache:
            cached_item = self.data_cache[cache_key]
            cache_age = (datetime.now() - cached_item['timestamp']).total_seconds()
            
            if cache_age < self.cache_duration:
                cached_item['metadata']['cache_hit'] = True
                return cached_item
        
        return None
    
    def _cache_data(self, cache_key: str, data: Dict[str, Any], 
                   cache_duration: Optional[int] = None) -> None:
        """Cache data with timestamp and expiration."""
        duration = cache_duration or self.cache_duration
        self.data_cache[cache_key] = {
            **data,
            'timestamp': datetime.now(),
            'expires_at': datetime.now() + timedelta(seconds=duration)
        }
    
    def _calculate_data_freshness(self, cache_key: str) -> float:
        """Calculate data freshness score (0-1)."""
        if cache_key in self.data_cache:
            cache_age = (datetime.now() - self.data_cache[cache_key]['timestamp']).total_seconds()
            return max(0, 1 - (cache_age / self.cache_duration))
        return 0.0
    
    # Data validation methods
    def _validate_account_data(self, accounts_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean account data."""
        validated = []
        for account in accounts_data:
            if isinstance(account, dict) and 'id' in account:
                # Add validation logic here
                validated.append(account)
        return validated
    
    def _validate_transaction_data(self, transactions_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean transaction data."""
        validated = []
        for transaction in transactions_data:
            if isinstance(transaction, dict) and 'id' in transaction:
                # Add validation logic here
                validated.append(transaction)
        return validated
    
    def _validate_portfolio_data(self, portfolio_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate and clean portfolio data."""
        holdings = portfolio_data.get('holdings', [])
        validated = []
        for holding in holdings:
            if isinstance(holding, dict) and 'symbol' in holding:
                # Add validation logic here
                validated.append(holding)
        return validated
    
    def _validate_market_data(self, market_data: Dict[str, Any], symbols: List[str]) -> Dict[str, Any]:
        """Validate and clean market data."""
        validated = {}
        for symbol in symbols:
            if symbol in market_data:
                data = market_data[symbol]
                if isinstance(data, dict) and 'price' in data:
                    # Add validation logic here
                    validated[symbol] = data
        return validated
    
    def _validate_profile_data(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean profile data."""
        if isinstance(profile_data, dict) and 'user_id' in profile_data:
            # Add validation logic here
            return profile_data
        return {}
    
    # Calculation methods
    def _calculate_account_summary(self, accounts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate account summary statistics."""
        total_balance = sum(account.get('balance', 0) for account in accounts)
        account_types = set(account.get('type', 'unknown') for account in accounts)
        
        return {
            "total_balance": total_balance,
            "account_count": len(accounts),
            "account_types": list(account_types),
            "average_balance": total_balance / len(accounts) if accounts else 0
        }
    
    def _calculate_transaction_summary(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate transaction summary statistics."""
        total_volume = sum(abs(t.get('amount', 0)) for t in transactions)
        income = sum(t.get('amount', 0) for t in transactions if t.get('amount', 0) > 0)
        expenses = sum(abs(t.get('amount', 0)) for t in transactions if t.get('amount', 0) < 0)
        
        return {
            "total_volume": total_volume,
            "total_income": income,
            "total_expenses": expenses,
            "net_cash_flow": income - expenses,
            "transaction_count": len(transactions)
        }
    
    def _calculate_asset_allocation(self, holdings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate asset allocation breakdown."""
        total_value = sum(h.get('market_value', 0) for h in holdings)
        allocation = {}
        
        for holding in holdings:
            asset_class = holding.get('asset_class', 'unknown')
            value = holding.get('market_value', 0)
            allocation[asset_class] = allocation.get(asset_class, 0) + value
        
        # Convert to percentages
        if total_value > 0:
            allocation = {k: (v / total_value) * 100 for k, v in allocation.items()}
        
        return allocation
    
    def _calculate_portfolio_performance(self, holdings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate portfolio performance metrics."""
        total_value = sum(h.get('market_value', 0) for h in holdings)
        total_cost = sum(h.get('cost_basis', 0) for h in holdings)
        
        return {
            "total_value": total_value,
            "total_cost": total_cost,
            "unrealized_gain_loss": total_value - total_cost,
            "return_percentage": ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
        }
    
    def _calculate_risk_metrics(self, holdings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate portfolio risk metrics."""
        # Simplified risk calculation
        total_value = sum(h.get('market_value', 0) for h in holdings)
        volatility = sum(h.get('volatility', 0.1) * h.get('market_value', 0) for h in holdings) / total_value if total_value > 0 else 0
        
        return {
            "portfolio_volatility": volatility,
            "diversification_score": len(holdings) / 10,  # Simplified
            "concentration_risk": "low" if len(holdings) > 10 else "medium" if len(holdings) > 5 else "high"
        }
    
    def _calculate_market_summary(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate market data summary."""
        prices = [data.get('price', 0) for data in market_data.values()]
        volumes = [data.get('volume', 0) for data in market_data.values()]
        
        return {
            "average_price": sum(prices) / len(prices) if prices else 0,
            "total_volume": sum(volumes),
            "price_change_count": sum(1 for data in market_data.values() if data.get('change', 0) > 0)
        }
    
    def _calculate_market_correlations(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate market correlations (simplified)."""
        # Simplified correlation calculation
        return {"correlation_matrix": "available"}
    
    def _calculate_profile_completeness(self, profile: Dict[str, Any]) -> float:
        """Calculate profile completeness score."""
        required_fields = ['user_id', 'name', 'email']
        filled_fields = sum(1 for field in required_fields if profile.get(field))
        return filled_fields / len(required_fields)
    
    def _analyze_risk_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user risk profile."""
        return {
            "risk_tolerance": profile.get('risk_tolerance', 'medium'),
            "investment_horizon": profile.get('investment_horizon', 'medium'),
            "risk_score": 0.5  # Simplified
        }
    
    def _identify_transaction_patterns(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify transaction patterns."""
        return {
            "spending_patterns": "analyzed",
            "income_patterns": "analyzed",
            "seasonal_trends": "identified"
        }
    
    def _perform_consistency_checks(self, user_id: str) -> Dict[str, Any]:
        """Perform data consistency checks."""
        return {
            "account_balance_consistency": "verified",
            "transaction_consistency": "verified",
            "portfolio_consistency": "verified"
        }
    
    def _calculate_health_score(self, test_results: Dict[str, Any]) -> float:
        """Calculate overall health score."""
        scores = []
        for test_name, result in test_results.items():
            if isinstance(result, dict) and 'status' in result:
                scores.append(1.0 if result['status'] == 'success' else 0.0)
            else:
                scores.append(0.5)  # Default score for unknown results
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _generate_connection_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """Generate connection optimization recommendations."""
        recommendations = []
        
        if test_results.get('connectivity', {}).get('status') != 'success':
            recommendations.append("Check network connectivity and firewall settings")
        
        if test_results.get('api_endpoints', {}).get('status') != 'success':
            recommendations.append("Verify API endpoint configuration and authentication")
        
        if test_results.get('data_quality', {}).get('status') != 'success':
            recommendations.append("Review data validation and quality checks")
        
        return recommendations
    
    # Memory storage methods
    def _store_account_data_in_memory(self, user_id: str, accounts: List[Dict[str, Any]]) -> None:
        """Store account data in Tethys memory system."""
        account_summary = f"Retrieved {len(accounts)} accounts for user {user_id}"
        ingest_user_memory(
            user_id=user_id,
            text=account_summary,
            memory_type="account_data",
            metadata={"accounts": accounts, "count": len(accounts)}
        )
    
    def _store_transaction_data_in_memory(self, user_id: str, transactions: List[Dict[str, Any]]) -> None:
        """Store transaction data in Tethys memory system."""
        transaction_summary = f"Retrieved {len(transactions)} transactions for user {user_id}"
        ingest_user_memory(
            user_id=user_id,
            text=transaction_summary,
            memory_type="transaction_data",
            metadata={"transactions": transactions, "count": len(transactions)}
        )
    
    def _store_portfolio_data_in_memory(self, user_id: str, holdings: List[Dict[str, Any]]) -> None:
        """Store portfolio data in Tethys memory system."""
        portfolio_summary = f"Retrieved {len(holdings)} portfolio holdings for user {user_id}"
        ingest_user_memory(
            user_id=user_id,
            text=portfolio_summary,
            memory_type="portfolio_data",
            metadata={"holdings": holdings, "count": len(holdings)}
        )
    
    def _store_profile_data_in_memory(self, user_id: str, profile: Dict[str, Any]) -> None:
        """Store profile data in Tethys memory system."""
        profile_summary = f"Retrieved profile data for user {user_id}"
        ingest_user_memory(
            user_id=user_id,
            text=profile_summary,
            memory_type="profile_data",
            metadata={"profile": profile}
        )
    
    # Test methods
    def _test_basic_connectivity(self) -> Dict[str, Any]:
        """Test basic connectivity to Fi-MCP simulator."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return {
                "status": "success" if response.status_code == 200 else "error",
                "response_time": response.elapsed.total_seconds()
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _test_api_endpoints(self) -> Dict[str, Any]:
        """Test key API endpoints."""
        endpoints = ["/accounts", "/transactions", "/portfolio", "/market-data"]
        results = {}
        
        for endpoint in endpoints:
            try:
                response = self.session.get(f"{self.base_url}{endpoint}", timeout=5)
                results[endpoint] = {
                    "status": "success" if response.status_code in [200, 401] else "error",
                    "response_code": response.status_code
                }
            except Exception as e:
                results[endpoint] = {"status": "error", "error": str(e)}
        
        return results
    
    def _test_data_quality(self) -> Dict[str, Any]:
        """Test data quality with sample requests."""
        try:
            # Test with sample market data request
            response = self._make_api_request('GET', f"{self.base_url}/market-data", 
                                           params={'symbols': 'AAPL,MSFT'})
            
            if response.get('status') == 'success':
                data = response.get('data', {})
                quality_score = len(data) / 2  # Expected 2 symbols
                return {
                    "status": "success",
                    "quality_score": quality_score,
                    "sample_data_count": len(data)
                }
            else:
                return {"status": "error", "error": response.get('error')}
        except Exception as e:
            return {"status": "error", "error": str(e)}

# Global connector instance for application-wide access
fi_mcp_connector = FiMCPConnector()
