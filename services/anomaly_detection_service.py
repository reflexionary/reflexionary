"""
Tethys - Anomaly Detection Service

This service orchestrates anomaly detection logic, alert generation, and user preferences
for the Tethys Financial Co-Pilot. It integrates with the Memory Layer and Mathematical
Intelligence Layer to provide intelligent anomaly detection.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

# Import Tethys components
from memory_management.memory_manager import ingest_user_memory, retrieve_contextual_memories
from financial_intelligence.risk_analysis.anomaly_detector import detect_financial_anomaly
from financial_intelligence.risk_analysis.risk_calculator import analyze_return_distribution
from core_components.embedding_service import get_embedding

logger = logging.getLogger(__name__)

class AnomalyDetectionService:
    """
    Service for detecting and managing financial anomalies in Tethys.
    
    This service combines:
    - Transaction pattern analysis
    - Portfolio performance monitoring
    - User behavior learning
    - Alert generation and management
    """
    
    def __init__(self):
        """Initialize the anomaly detection service."""
        self.alert_history = {}
        self.user_preferences = {}
    
    def detect_transaction_anomalies(self, user_id: str, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect anomalies in transaction data.
        
        Args:
            user_id: User identifier
            transaction_data: Transaction information including amount, category, timestamp
            
        Returns:
            Dictionary containing anomaly detection results
        """
        try:
            # Analyze transaction patterns
            anomaly_result = detect_financial_anomaly(
                user_id=user_id,
                data_type="transaction",
                data=transaction_data
            )
            
            # Store anomaly in memory if detected
            if anomaly_result.get("is_anomaly", False):
                memory_text = f"Anomalous transaction detected: {transaction_data.get('amount', 0)} in {transaction_data.get('category', 'unknown')} category"
                ingest_user_memory(
                    user_id=user_id,
                    text=memory_text,
                    memory_type="anomaly",
                    metadata={
                        "anomaly_score": anomaly_result.get("anomaly_score", 0),
                        "transaction_data": transaction_data,
                        "detection_timestamp": datetime.now().isoformat()
                    }
                )
                
                # Generate alert
                alert = self._generate_alert(user_id, "transaction_anomaly", anomaly_result)
                self._store_alert(user_id, alert)
            
            return {
                "status": "success",
                "anomaly_detected": anomaly_result.get("is_anomaly", False),
                "anomaly_score": anomaly_result.get("anomaly_score", 0),
                "confidence": anomaly_result.get("confidence", 0),
                "recommendations": anomaly_result.get("recommendations", []),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error detecting transaction anomalies for user {user_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def detect_portfolio_anomalies(self, user_id: str) -> Dict[str, Any]:
        """
        Detect anomalies in portfolio performance.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary containing portfolio anomaly detection results
        """
        try:
            # Get user's portfolio context from memory
            portfolio_memories = retrieve_contextual_memories(
                user_id=user_id,
                query_text="portfolio performance risk tolerance investment strategy",
                num_results=5
            )
            
            # Analyze portfolio returns distribution
            distribution_analysis = analyze_return_distribution(user_id)
            
            # Detect portfolio anomalies
            anomaly_result = detect_financial_anomaly(
                user_id=user_id,
                data_type="portfolio",
                data=distribution_analysis
            )
            
            if anomaly_result.get("is_anomaly", False):
                memory_text = f"Portfolio anomaly detected: {anomaly_result.get('description', 'Unusual portfolio behavior')}"
                ingest_user_memory(
                    user_id=user_id,
                    text=memory_text,
                    memory_type="anomaly",
                    metadata={
                        "anomaly_type": "portfolio",
                        "anomaly_score": anomaly_result.get("anomaly_score", 0),
                        "portfolio_metrics": distribution_analysis,
                        "detection_timestamp": datetime.now().isoformat()
                    }
                )
                
                # Generate alert
                alert = self._generate_alert(user_id, "portfolio_anomaly", anomaly_result)
                self._store_alert(user_id, alert)
            
            return {
                "status": "success",
                "anomaly_detected": anomaly_result.get("is_anomaly", False),
                "anomaly_score": anomaly_result.get("anomaly_score", 0),
                "portfolio_context": {
                    "memories_count": len(portfolio_memories),
                    "distribution_analysis": distribution_analysis
                },
                "recommendations": anomaly_result.get("recommendations", []),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error detecting portfolio anomalies for user {user_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_user_anomaly_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get user's anomaly detection preferences.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary containing user preferences
        """
        # Retrieve user preferences from memory
        preference_memories = retrieve_contextual_memories(
            user_id=user_id,
            query_text="anomaly detection preferences alert settings notification",
            num_results=3
        )
        
        # Default preferences
        default_preferences = {
            "transaction_threshold": 1000,  # Amount threshold for transaction alerts
            "portfolio_volatility_threshold": 0.15,  # Volatility threshold
            "alert_frequency": "immediate",  # immediate, daily, weekly
            "notification_channels": ["memory"],  # memory, email, push
            "risk_tolerance": "medium"  # low, medium, high
        }
        
        # Override with user-specific preferences from memory
        user_preferences = default_preferences.copy()
        for memory in preference_memories:
            if memory.get("type") == "preference":
                metadata = memory.get("metadata", {})
                user_preferences.update(metadata)
        
        return {
            "status": "success",
            "user_id": user_id,
            "preferences": user_preferences,
            "timestamp": datetime.now().isoformat()
        }
    
    def update_user_anomaly_preferences(self, user_id: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update user's anomaly detection preferences.
        
        Args:
            user_id: User identifier
            preferences: New preferences to set
            
        Returns:
            Dictionary containing update status
        """
        try:
            # Store preferences in memory
            preference_text = f"Updated anomaly detection preferences: {json.dumps(preferences, indent=2)}"
            ingest_user_memory(
                user_id=user_id,
                text=preference_text,
                memory_type="preference",
                metadata=preferences
            )
            
            return {
                "status": "success",
                "message": "Anomaly detection preferences updated successfully",
                "user_id": user_id,
                "preferences": preferences,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating anomaly preferences for user {user_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_alert_history(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Get user's anomaly alert history.
        
        Args:
            user_id: User identifier
            days: Number of days to look back
            
        Returns:
            Dictionary containing alert history
        """
        # Retrieve anomaly memories from the specified time period
        cutoff_date = datetime.now() - timedelta(days=days)
        
        anomaly_memories = retrieve_contextual_memories(
            user_id=user_id,
            query_text="anomaly alert detection unusual pattern",
            num_results=50  # Get more results to filter by date
        )
        
        # Filter by date and type
        recent_anomalies = []
        for memory in anomaly_memories:
            if memory.get("type") == "anomaly":
                metadata = memory.get("metadata", {})
                detection_time = metadata.get("detection_timestamp")
                if detection_time:
                    try:
                        detection_date = datetime.fromisoformat(detection_time.replace('Z', '+00:00'))
                        if detection_date >= cutoff_date:
                            recent_anomalies.append({
                                "id": memory.get("id"),
                                "text": memory.get("text"),
                                "anomaly_score": metadata.get("anomaly_score", 0),
                                "anomaly_type": metadata.get("anomaly_type", "unknown"),
                                "detection_timestamp": detection_time,
                                "metadata": metadata
                            })
                    except ValueError:
                        continue
        
        return {
            "status": "success",
            "user_id": user_id,
            "alert_history": recent_anomalies,
            "total_alerts": len(recent_anomalies),
            "time_period_days": days,
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_alert(self, user_id: str, alert_type: str, anomaly_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an alert based on anomaly detection results."""
        alert = {
            "id": f"alert_{user_id}_{datetime.now().timestamp()}",
            "user_id": user_id,
            "type": alert_type,
            "severity": self._calculate_severity(anomaly_result.get("anomaly_score", 0)),
            "title": self._generate_alert_title(alert_type, anomaly_result),
            "description": anomaly_result.get("description", "Anomaly detected"),
            "anomaly_score": anomaly_result.get("anomaly_score", 0),
            "recommendations": anomaly_result.get("recommendations", []),
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        return alert
    
    def _store_alert(self, user_id: str, alert: Dict[str, Any]):
        """Store alert in the alert history."""
        if user_id not in self.alert_history:
            self.alert_history[user_id] = []
        self.alert_history[user_id].append(alert)
    
    def _calculate_severity(self, anomaly_score: float) -> str:
        """Calculate alert severity based on anomaly score."""
        if anomaly_score >= 0.8:
            return "high"
        elif anomaly_score >= 0.5:
            return "medium"
        else:
            return "low"
    
    def _generate_alert_title(self, alert_type: str, anomaly_result: Dict[str, Any]) -> str:
        """Generate alert title based on type and result."""
        if alert_type == "transaction_anomaly":
            return "Unusual Transaction Detected"
        elif alert_type == "portfolio_anomaly":
            return "Portfolio Anomaly Alert"
        else:
            return "Financial Anomaly Detected"

# Global service instance
anomaly_detection_service = AnomalyDetectionService()
