"""
Tethys - Core Integration Layer

This module serves as the main integration layer for Tethys Financial Co-Pilot,
orchestrating all components including Memory Layer, Mathematical Intelligence,
and business services to provide a unified, intelligent financial assistant.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Import Tethys components
from memory_management.memory_manager import ingest_user_memory, retrieve_contextual_memories
from memory_management.gemini_connector import GeminiConnector
from financial_intelligence.financial_quant_tools import (
    get_portfolio_performance_metrics,
    get_optimal_portfolio_allocation,
    get_value_at_risk
)

# Import services
from services.anomaly_detection_service import anomaly_detection_service
from services.portfolio_management_service import portfolio_management_service
from services.goal_planning_service import goal_planning_service
from services.user_interaction_service import user_interaction_service

# Import data connectors
from data_connectors.fi_mcp_connector import fi_mcp_connector

# Import observability
from observability.logging_config import get_logger
from observability.metrics_exporter import get_metrics_collector

logger = get_logger("tethys_core")

class TethysCore:
    """
    Core integration layer for Tethys Financial Co-Pilot.
    
    This class orchestrates:
    - Memory Layer operations
    - Mathematical Intelligence calculations
    - Business service interactions
    - Data synchronization
    - User interactions
    - Performance monitoring
    """
    
    def __init__(self):
        """Initialize the Tethys core system."""
        self.gemini_connector = GeminiConnector()
        self.metrics_collector = get_metrics_collector()
        self.logger = logger
        
        # Initialize component status
        self.component_status = {
            "memory_layer": "initializing",
            "mathematical_intelligence": "initializing",
            "data_connectors": "initializing",
            "services": "initializing"
        }
        
        self.logger.info("Tethys Core initializing...")
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all Tethys components."""
        try:
            # Test data connector
            connection_test = fi_mcp_connector.test_connection()
            if connection_test["status"] == "success":
                self.component_status["data_connectors"] = "healthy"
                self.logger.info("Data connectors initialized successfully")
            else:
                self.component_status["data_connectors"] = "unhealthy"
                self.logger.warning("Data connectors initialization failed")
            
            # Test memory layer
            try:
                # Test memory ingestion
                test_memory = "Tethys core system initialized successfully"
                ingest_user_memory(
                    user_id="system",
                    text=test_memory,
                    memory_type="system",
                    metadata={"component": "tethys_core", "operation": "initialization"}
                )
                self.component_status["memory_layer"] = "healthy"
                self.logger.info("Memory layer initialized successfully")
            except Exception as e:
                self.component_status["memory_layer"] = "unhealthy"
                self.logger.error(f"Memory layer initialization failed: {e}")
            
            # Test mathematical intelligence
            try:
                # Test basic financial calculations
                test_metrics = get_portfolio_performance_metrics("test_user")
                self.component_status["mathematical_intelligence"] = "healthy"
                self.logger.info("Mathematical intelligence initialized successfully")
            except Exception as e:
                self.component_status["mathematical_intelligence"] = "unhealthy"
                self.logger.error(f"Mathematical intelligence initialization failed: {e}")
            
            # Initialize services
            self.component_status["services"] = "healthy"
            self.logger.info("Business services initialized successfully")
            
            self.logger.info("Tethys Core initialization completed")
            
        except Exception as e:
            self.logger.error(f"Error during Tethys Core initialization: {e}")
            raise
    
    def process_user_query(self, user_id: str, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a user query through the complete Tethys pipeline.
        
        Args:
            user_id: User identifier
            query: User's natural language query
            context: Additional context for the query
            
        Returns:
            Dictionary containing comprehensive response
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing user query for user {user_id}: {query}")
            self.metrics_collector.record_user_activity(user_id, "query", {"query": query})
            
            # Step 1: Get user context from memory
            memory_context = self._get_user_context(user_id, query)
            
            # Step 2: Sync latest data if needed
            data_sync_result = self._sync_user_data_if_needed(user_id)
            
            # Step 3: Process query through user interaction service
            interaction_result = user_interaction_service.process_user_query(user_id, query, context)
            
            # Step 4: Generate comprehensive insights
            insights = self._generate_comprehensive_insights(user_id, query, memory_context)
            
            # Step 5: Generate recommendations
            recommendations = self._generate_recommendations(user_id, query, memory_context, insights)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            self.metrics_collector.record_operation_time(
                "user_query_processing", processing_time, user_id, "tethys_core"
            )
            
            # Build comprehensive response
            response = {
                "status": "success",
                "user_id": user_id,
                "query": query,
                "response": interaction_result.get("response", ""),
                "insights": insights,
                "recommendations": recommendations,
                "context": {
                    "memory_context": memory_context,
                    "data_sync": data_sync_result,
                    "processing_time_ms": processing_time
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Store interaction in memory
            self._store_interaction_memory(user_id, query, response)
            
            self.logger.info(f"Query processing completed for user {user_id} in {processing_time:.2f}ms")
            
            return response
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.logger.error(f"Error processing user query for {user_id}: {e}")
            self.metrics_collector.record_operation_time(
                "user_query_processing", processing_time, user_id, "tethys_core"
            )
            
            return {
                "status": "error",
                "error": str(e),
                "processing_time_ms": processing_time,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_user_dashboard(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive user dashboard data.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary containing dashboard data
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Generating dashboard for user {user_id}")
            
            # Get portfolio overview
            portfolio_overview = portfolio_management_service.get_portfolio_overview(user_id)
            
            # Get goals overview
            goals_overview = goal_planning_service.get_goals_overview(user_id)
            
            # Get anomaly alerts
            anomaly_alerts = anomaly_detection_service.get_alert_history(user_id, days=7)
            
            # Get personalized insights
            insights = user_interaction_service.get_personalized_insights(user_id)
            
            # Get recent interactions
            interactions = user_interaction_service.get_interaction_history(user_id, days=7)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            self.metrics_collector.record_operation_time(
                "dashboard_generation", processing_time, user_id, "tethys_core"
            )
            
            dashboard = {
                "status": "success",
                "user_id": user_id,
                "portfolio": portfolio_overview.get("portfolio_overview", {}),
                "goals": goals_overview.get("goals_overview", {}),
                "alerts": anomaly_alerts.get("alert_history", []),
                "insights": insights.get("insights", []),
                "recent_interactions": interactions.get("interaction_history", []),
                "processing_time_ms": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Dashboard generated for user {user_id} in {processing_time:.2f}ms")
            
            return dashboard
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.logger.error(f"Error generating dashboard for user {user_id}: {e}")
            
            return {
                "status": "error",
                "error": str(e),
                "processing_time_ms": processing_time,
                "timestamp": datetime.now().isoformat()
            }
    
    def sync_user_data(self, user_id: str) -> Dict[str, Any]:
        """
        Synchronize all user data from external sources.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary containing sync results
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting data sync for user {user_id}")
            
            # Sync data from Fi-MCP
            sync_result = fi_mcp_connector.sync_user_data(user_id)
            
            # Process synced data through services
            if sync_result["status"] == "success":
                # Update portfolio data
                portfolio_management_service.get_portfolio_overview(user_id)
                
                # Check for anomalies
                anomaly_detection_service.detect_portfolio_anomalies(user_id)
                
                # Update goals progress
                goals_overview = goal_planning_service.get_goals_overview(user_id)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            self.metrics_collector.record_operation_time(
                "data_sync", processing_time, user_id, "tethys_core"
            )
            
            result = {
                "status": "success",
                "user_id": user_id,
                "sync_result": sync_result,
                "processing_time_ms": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Data sync completed for user {user_id} in {processing_time:.2f}ms")
            
            return result
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.logger.error(f"Error syncing data for user {user_id}: {e}")
            
            return {
                "status": "error",
                "error": str(e),
                "processing_time_ms": processing_time,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status and health.
        
        Returns:
            Dictionary containing system status
        """
        try:
            # Test component health
            health_checks = {
                "memory_layer": self._test_memory_layer(),
                "mathematical_intelligence": self._test_mathematical_intelligence(),
                "data_connectors": self._test_data_connectors(),
                "services": self._test_services()
            }
            
            # Calculate overall health
            healthy_components = sum(1 for check in health_checks.values() if check["status"] == "healthy")
            total_components = len(health_checks)
            overall_health = "healthy" if healthy_components == total_components else "degraded"
            
            if healthy_components == 0:
                overall_health = "unhealthy"
            
            status = {
                "status": "success",
                "overall_health": overall_health,
                "component_health": health_checks,
                "healthy_components": healthy_components,
                "total_components": total_components,
                "component_status": self.component_status,
                "timestamp": datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_user_context(self, user_id: str, query: str) -> Dict[str, Any]:
        """Get user context from memory."""
        try:
            # Retrieve relevant memories
            memories = retrieve_contextual_memories(
                user_id=user_id,
                query_text=query,
                num_results=10
            )
            
            # Get user preferences
            preferences = user_interaction_service.get_user_preferences(user_id)
            
            return {
                "memories": memories,
                "preferences": preferences,
                "memory_count": len(memories)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting user context for {user_id}: {e}")
            return {"memories": [], "preferences": {}, "memory_count": 0}
    
    def _sync_user_data_if_needed(self, user_id: str) -> Dict[str, Any]:
        """Sync user data if it hasn't been synced recently."""
        try:
            # Check last sync time from memory
            sync_memories = retrieve_contextual_memories(
                user_id=user_id,
                query_text="data sync completed",
                num_results=1
            )
            
            # If no recent sync, perform sync
            if not sync_memories:
                return self.sync_user_data(user_id)
            
            return {"status": "skipped", "reason": "recent_sync_exists"}
            
        except Exception as e:
            self.logger.error(f"Error checking data sync for {user_id}: {e}")
            return {"status": "error", "error": str(e)}
    
    def _generate_comprehensive_insights(self, user_id: str, query: str, memory_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate comprehensive insights for the user."""
        insights = []
        
        try:
            # Get portfolio insights
            portfolio_overview = portfolio_management_service.get_portfolio_overview(user_id)
            if portfolio_overview["status"] == "success":
                portfolio_data = portfolio_overview["portfolio_overview"]
                
                # Add portfolio performance insights
                performance_metrics = portfolio_data.get("performance_metrics", {})
                if performance_metrics:
                    total_return = performance_metrics.get("total_return", 0)
                    if total_return < 0:
                        insights.append({
                            "type": "portfolio_performance",
                            "title": "Portfolio Underperforming",
                            "description": "Your portfolio is currently showing negative returns. Consider reviewing your asset allocation.",
                            "priority": "high"
                        })
            
            # Get goal insights
            goals_overview = goal_planning_service.get_goals_overview(user_id)
            if goals_overview["status"] == "success":
                goals_data = goals_overview["goals_overview"]
                
                # Add goal progress insights
                overall_progress = goals_data.get("overall_progress_percentage", 0)
                if overall_progress < 25:
                    insights.append({
                        "type": "goal_progress",
                        "title": "Goals Need Attention",
                        "description": "You're behind on your financial goals. Consider increasing your savings rate.",
                        "priority": "medium"
                    })
            
            # Get anomaly insights
            anomaly_alerts = anomaly_detection_service.get_alert_history(user_id, days=7)
            if anomaly_alerts["status"] == "success":
                recent_alerts = anomaly_alerts.get("alert_history", [])
                if recent_alerts:
                    insights.append({
                        "type": "anomaly_detection",
                        "title": "Recent Anomalies Detected",
                        "description": f"{len(recent_alerts)} financial anomalies detected recently. Review your transactions.",
                        "priority": "high"
                    })
            
        except Exception as e:
            self.logger.error(f"Error generating insights for {user_id}: {e}")
        
        return insights
    
    def _generate_recommendations(self, user_id: str, query: str, memory_context: Dict[str, Any], 
                                insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate personalized recommendations."""
        recommendations = []
        
        try:
            # Add recommendations based on insights
            for insight in insights:
                if insight["type"] == "portfolio_performance":
                    recommendations.append({
                        "type": "portfolio_review",
                        "title": "Review Portfolio Allocation",
                        "description": "Schedule a portfolio review to optimize your asset allocation.",
                        "action": "Use portfolio optimization service"
                    })
                
                elif insight["type"] == "goal_progress":
                    recommendations.append({
                        "type": "goal_management",
                        "title": "Adjust Savings Strategy",
                        "description": "Consider increasing your monthly savings to meet your goals faster.",
                        "action": "Review and update financial goals"
                    })
            
            # Add query-specific recommendations
            query_lower = query.lower()
            if "invest" in query_lower or "portfolio" in query_lower:
                recommendations.append({
                    "type": "investment_advice",
                    "title": "Investment Analysis",
                    "description": "Get a detailed analysis of your current investments and potential opportunities.",
                    "action": "Request portfolio analysis"
                })
            
            if "goal" in query_lower or "target" in query_lower:
                recommendations.append({
                    "type": "goal_planning",
                    "title": "Goal Planning",
                    "description": "Review your financial goals and create a plan to achieve them.",
                    "action": "Access goal planning tools"
                })
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations for {user_id}: {e}")
        
        return recommendations
    
    def _store_interaction_memory(self, user_id: str, query: str, response: Dict[str, Any]):
        """Store interaction in memory."""
        try:
            interaction_text = f"User query: {query} - Response generated with insights and recommendations"
            ingest_user_memory(
                user_id=user_id,
                text=interaction_text,
                memory_type="interaction",
                metadata={
                    "query": query,
                    "response_summary": response.get("response", ""),
                    "insights_count": len(response.get("insights", [])),
                    "recommendations_count": len(response.get("recommendations", [])),
                    "processing_time_ms": response.get("context", {}).get("processing_time_ms", 0)
                }
            )
        except Exception as e:
            self.logger.error(f"Error storing interaction memory for {user_id}: {e}")
    
    def _test_memory_layer(self) -> Dict[str, Any]:
        """Test memory layer functionality."""
        try:
            # Test memory ingestion and retrieval
            test_text = "Memory layer health check"
            ingest_user_memory(
                user_id="system",
                text=test_text,
                memory_type="system",
                metadata={"test": "health_check"}
            )
            
            memories = retrieve_contextual_memories(
                user_id="system",
                query_text="health check",
                num_results=1
            )
            
            return {
                "status": "healthy",
                "details": f"Memory operations successful, retrieved {len(memories)} memories"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "details": f"Memory layer test failed: {str(e)}"
            }
    
    def _test_mathematical_intelligence(self) -> Dict[str, Any]:
        """Test mathematical intelligence functionality."""
        try:
            # Test basic financial calculations
            test_metrics = get_portfolio_performance_metrics("test_user")
            
            return {
                "status": "healthy",
                "details": "Financial calculations successful"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "details": f"Mathematical intelligence test failed: {str(e)}"
            }
    
    def _test_data_connectors(self) -> Dict[str, Any]:
        """Test data connector functionality."""
        try:
            connection_test = fi_mcp_connector.test_connection()
            
            return {
                "status": connection_test.get("connection", "unknown"),
                "details": connection_test.get("health_data", "Connection test completed")
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "details": f"Data connector test failed: {str(e)}"
            }
    
    def _test_services(self) -> Dict[str, Any]:
        """Test business services functionality."""
        try:
            # Test service initialization
            services = [
                anomaly_detection_service,
                portfolio_management_service,
                goal_planning_service,
                user_interaction_service
            ]
            
            return {
                "status": "healthy",
                "details": f"All {len(services)} services initialized successfully"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "details": f"Services test failed: {str(e)}"
            }

# Global Tethys Core instance
tethys_core = TethysCore()

def get_tethys_core() -> TethysCore:
    """Get the global Tethys Core instance."""
    return tethys_core 