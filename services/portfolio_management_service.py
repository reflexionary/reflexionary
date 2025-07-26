"""
Tethys - Portfolio Management Service

This service manages portfolio optimization, metrics, and asset allocation workflows
for the Tethys Financial Co-Pilot. It integrates the Memory Layer with Mathematical
Intelligence to provide personalized portfolio management.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

# Import Tethys components
from memory_management.memory_manager import ingest_user_memory, retrieve_contextual_memories
from financial_intelligence.financial_quant_tools import (
    get_portfolio_performance_metrics,
    get_optimal_portfolio_allocation,
    get_value_at_risk
)
from financial_intelligence.portfolio_opt.optimizer import PortfolioOptimizer
from financial_intelligence.core_metrics.performance_analyzer import PerformanceAnalyzer

logger = logging.getLogger(__name__)

class PortfolioManagementService:
    """
    Service for managing investment portfolios in Tethys.
    
    This service provides:
    - Portfolio performance analysis
    - Asset allocation optimization
    - Risk management
    - Goal-based portfolio planning
    - Rebalancing recommendations
    """
    
    def __init__(self):
        """Initialize the portfolio management service."""
        self.optimizer = PortfolioOptimizer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.rebalancing_history = {}
    
    def get_portfolio_overview(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive portfolio overview for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary containing portfolio overview
        """
        try:
            # Get user's financial context from memory
            financial_memories = retrieve_contextual_memories(
                user_id=user_id,
                query_text="investment goals risk tolerance portfolio strategy financial objectives",
                num_results=10
            )
            
            # Get portfolio performance metrics
            performance_metrics = get_portfolio_performance_metrics(user_id)
            
            # Get risk metrics
            risk_metrics = get_value_at_risk(user_id)
            
            # Extract user preferences and goals from memory
            user_context = self._extract_user_context(financial_memories)
            
            return {
                "status": "success",
                "user_id": user_id,
                "portfolio_overview": {
                    "performance_metrics": performance_metrics,
                    "risk_metrics": risk_metrics,
                    "user_context": user_context,
                    "last_updated": datetime.now().isoformat()
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio overview for user {user_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def optimize_portfolio(self, user_id: str, optimization_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize portfolio allocation based on user preferences and goals.
        
        Args:
            user_id: User identifier
            optimization_params: Optimization parameters including risk tolerance, goals, etc.
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            # Get user's financial context
            financial_memories = retrieve_contextual_memories(
                user_id=user_id,
                query_text="investment goals risk tolerance time horizon financial objectives",
                num_results=10
            )
            
            # Extract optimization parameters
            risk_tolerance = optimization_params.get("risk_tolerance", "medium")
            investment_amount = optimization_params.get("investment_amount")
            time_horizon = optimization_params.get("time_horizon", "medium")
            optimization_method = optimization_params.get("method", "max_sharpe")
            
            # Run portfolio optimization
            optimization_result = get_optimal_portfolio_allocation(
                user_id=user_id,
                risk_tolerance=risk_tolerance,
                total_investment_value=investment_amount
            )
            
            # Store optimization result in memory
            optimization_text = f"Portfolio optimization completed with {risk_tolerance} risk tolerance using {optimization_method} method"
            ingest_user_memory(
                user_id=user_id,
                text=optimization_text,
                memory_type="portfolio_optimization",
                metadata={
                    "optimization_params": optimization_params,
                    "optimization_result": optimization_result,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            return {
                "status": "success",
                "user_id": user_id,
                "optimization": {
                    "parameters": optimization_params,
                    "result": optimization_result,
                    "user_context": self._extract_user_context(financial_memories)
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio for user {user_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_rebalancing_recommendations(self, user_id: str) -> Dict[str, Any]:
        """
        Get portfolio rebalancing recommendations.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary containing rebalancing recommendations
        """
        try:
            # Get current portfolio allocation
            current_allocation = self._get_current_allocation(user_id)
            
            # Get target allocation from memory
            target_allocation = self._get_target_allocation(user_id)
            
            # Calculate rebalancing needs
            rebalancing_needs = self._calculate_rebalancing_needs(
                current_allocation, target_allocation
            )
            
            # Generate recommendations
            recommendations = self._generate_rebalancing_recommendations(
                user_id, rebalancing_needs
            )
            
            # Store rebalancing analysis in memory
            rebalancing_text = f"Portfolio rebalancing analysis completed with {len(recommendations)} recommendations"
            ingest_user_memory(
                user_id=user_id,
                text=rebalancing_text,
                memory_type="portfolio_rebalancing",
                metadata={
                    "current_allocation": current_allocation,
                    "target_allocation": target_allocation,
                    "rebalancing_needs": rebalancing_needs,
                    "recommendations": recommendations,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            return {
                "status": "success",
                "user_id": user_id,
                "rebalancing": {
                    "current_allocation": current_allocation,
                    "target_allocation": target_allocation,
                    "rebalancing_needs": rebalancing_needs,
                    "recommendations": recommendations
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting rebalancing recommendations for user {user_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def set_investment_goals(self, user_id: str, goals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set or update investment goals for a user.
        
        Args:
            user_id: User identifier
            goals: Investment goals and parameters
            
        Returns:
            Dictionary containing goal setting status
        """
        try:
            # Store goals in memory
            goal_text = f"Investment goals set: {json.dumps(goals, indent=2)}"
            ingest_user_memory(
                user_id=user_id,
                text=goal_text,
                memory_type="goal",
                metadata=goals
            )
            
            # Generate goal-based portfolio recommendations
            recommendations = self._generate_goal_based_recommendations(user_id, goals)
            
            return {
                "status": "success",
                "user_id": user_id,
                "goals": goals,
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error setting investment goals for user {user_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_portfolio_history(self, user_id: str, days: int = 90) -> Dict[str, Any]:
        """
        Get portfolio performance history.
        
        Args:
            user_id: User identifier
            days: Number of days to look back
            
        Returns:
            Dictionary containing portfolio history
        """
        try:
            # Get portfolio memories from the specified time period
            cutoff_date = datetime.now() - timedelta(days=days)
            
            portfolio_memories = retrieve_contextual_memories(
                user_id=user_id,
                query_text="portfolio performance allocation optimization rebalancing",
                num_results=100
            )
            
            # Filter by date and type
            recent_portfolio_events = []
            for memory in portfolio_memories:
                if memory.get("type") in ["portfolio_optimization", "portfolio_rebalancing", "goal"]:
                    metadata = memory.get("metadata", {})
                    event_time = metadata.get("timestamp")
                    if event_time:
                        try:
                            event_date = datetime.fromisoformat(event_time.replace('Z', '+00:00'))
                            if event_date >= cutoff_date:
                                recent_portfolio_events.append({
                                    "id": memory.get("id"),
                                    "type": memory.get("type"),
                                    "text": memory.get("text"),
                                    "timestamp": event_time,
                                    "metadata": metadata
                                })
                        except ValueError:
                            continue
            
            return {
                "status": "success",
                "user_id": user_id,
                "portfolio_history": recent_portfolio_events,
                "total_events": len(recent_portfolio_events),
                "time_period_days": days,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio history for user {user_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _extract_user_context(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract user context from memories."""
        context = {
            "risk_tolerance": "medium",
            "time_horizon": "medium",
            "investment_goals": [],
            "preferences": {}
        }
        
        for memory in memories:
            if memory.get("type") == "goal":
                metadata = memory.get("metadata", {})
                if "risk_tolerance" in metadata:
                    context["risk_tolerance"] = metadata["risk_tolerance"]
                if "time_horizon" in metadata:
                    context["time_horizon"] = metadata["time_horizon"]
                if "goals" in metadata:
                    context["investment_goals"].extend(metadata["goals"])
            
            elif memory.get("type") == "preference":
                metadata = memory.get("metadata", {})
                context["preferences"].update(metadata)
        
        return context
    
    def _get_current_allocation(self, user_id: str) -> Dict[str, float]:
        """Get current portfolio allocation."""
        # This would typically come from the Fi-MCP data
        # For now, return a mock allocation
        return {
            "equity": 0.60,
            "bonds": 0.25,
            "cash": 0.10,
            "alternatives": 0.05
        }
    
    def _get_target_allocation(self, user_id: str) -> Dict[str, float]:
        """Get target portfolio allocation from memory."""
        # Retrieve target allocation from memory
        allocation_memories = retrieve_contextual_memories(
            user_id=user_id,
            query_text="target allocation portfolio weights asset allocation",
            num_results=5
        )
        
        # Default target allocation
        default_target = {
            "equity": 0.70,
            "bonds": 0.20,
            "cash": 0.05,
            "alternatives": 0.05
        }
        
        # Override with user-specific target from memory
        for memory in allocation_memories:
            if memory.get("type") == "portfolio_optimization":
                metadata = memory.get("metadata", {})
                if "optimization_result" in metadata:
                    result = metadata["optimization_result"]
                    if "target_allocation" in result:
                        return result["target_allocation"]
        
        return default_target
    
    def _calculate_rebalancing_needs(self, current: Dict[str, float], target: Dict[str, float]) -> Dict[str, float]:
        """Calculate rebalancing needs."""
        rebalancing_needs = {}
        for asset_class in target:
            if asset_class in current:
                rebalancing_needs[asset_class] = target[asset_class] - current[asset_class]
            else:
                rebalancing_needs[asset_class] = target[asset_class]
        
        return rebalancing_needs
    
    def _generate_rebalancing_recommendations(self, user_id: str, rebalancing_needs: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate rebalancing recommendations."""
        recommendations = []
        
        for asset_class, adjustment in rebalancing_needs.items():
            if abs(adjustment) > 0.05:  # 5% threshold
                action = "buy" if adjustment > 0 else "sell"
                recommendations.append({
                    "asset_class": asset_class,
                    "action": action,
                    "adjustment": abs(adjustment),
                    "reason": f"Rebalance to target allocation of {asset_class}",
                    "priority": "high" if abs(adjustment) > 0.10 else "medium"
                })
        
        return recommendations
    
    def _generate_goal_based_recommendations(self, user_id: str, goals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate goal-based portfolio recommendations."""
        recommendations = []
        
        # Analyze goals and generate recommendations
        for goal in goals.get("investment_goals", []):
            goal_type = goal.get("type", "unknown")
            target_amount = goal.get("target_amount", 0)
            time_horizon = goal.get("time_horizon", "medium")
            
            if goal_type == "retirement":
                recommendations.append({
                    "type": "goal_based",
                    "goal": goal_type,
                    "recommendation": f"Consider increasing equity allocation for long-term retirement goal",
                    "priority": "high"
                })
            elif goal_type == "emergency_fund":
                recommendations.append({
                    "type": "goal_based",
                    "goal": goal_type,
                    "recommendation": f"Maintain high cash allocation for emergency fund goal",
                    "priority": "high"
                })
        
        return recommendations

# Global service instance
portfolio_management_service = PortfolioManagementService()
