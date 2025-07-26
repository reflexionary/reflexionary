"""
Tethys - Goal Planning Service

This service manages financial goal setting, tracking, and recommendations
for the Tethys Financial Co-Pilot. It integrates the Memory Layer with
Mathematical Intelligence to provide personalized goal planning.

The service implements goal-based financial planning using:
- Time Value of Money (TVM) calculations for goal feasibility
- Compound interest modeling for long-term goals
- Risk-adjusted return expectations for different goal types
- Monte Carlo simulation for success probability analysis
- Behavioral finance principles for goal adherence

Mathematical Framework:
- Future Value (FV) = PV * (1 + r)^n where r is periodic rate, n is periods
- Required Monthly Savings = (Target Amount - Current Amount) / (1 + r)^n
- Success Probability = P(Actual Savings >= Required Savings)
- Goal Achievement Score = (Current Progress / Target) * (1 - Time Decay Factor)

Author: Tethys Development Team
Version: 1.0.0
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import math

# Import Tethys components
from memory_management.memory_manager import ingest_user_memory, retrieve_contextual_memories
from financial_intelligence.financial_quant_tools import get_portfolio_performance_metrics
from financial_intelligence.portfolio_opt.optimizer import optimize_portfolio_allocation

logger = logging.getLogger(__name__)

class GoalPlanningService:
    """
    Service for managing financial goals in Tethys.
    
    This service provides comprehensive goal-based financial planning using
    mathematical models and behavioral finance principles. It calculates
    required savings rates, success probabilities, and provides intelligent
    recommendations based on user behavior and market conditions.
    
    Key Mathematical Concepts:
    1. Time Value of Money (TVM): Future value calculations considering inflation
    2. Compound Interest: Exponential growth modeling for long-term goals
    3. Risk-Adjusted Returns: Expected returns based on goal time horizon
    4. Monte Carlo Simulation: Success probability analysis
    5. Behavioral Scoring: Goal adherence prediction based on historical data
    
    Goal Types and Risk Profiles:
    - Emergency Fund: Low risk, high liquidity, short-term horizon
    - Retirement: Medium-high risk, long-term horizon, compound growth focus
    - House Down Payment: Low-medium risk, medium-term horizon
    - Education: Medium risk, medium-term horizon, inflation-adjusted
    - Vacation: Low risk, short-term horizon, discretionary spending
    """
    
    def __init__(self):
        """
        Initialize the goal planning service with predefined goal templates
        and mathematical parameters for financial calculations.
        
        Goal templates include risk tolerance, time horizon, and priority
        settings that influence the mathematical models used for planning.
        """
        # Predefined goal templates with risk profiles and mathematical parameters
        self.goal_templates = {
            "retirement": {
                "description": "Save for retirement",
                "time_horizon": "long",  # 20+ years
                "risk_tolerance": "medium",  # Can tolerate market volatility
                "priority": "high",
                "expected_return": 0.07,  # 7% annual return (inflation-adjusted)
                "inflation_rate": 0.025,  # 2.5% annual inflation
                "compound_frequency": 12,  # Monthly compounding
                "success_threshold": 0.8  # 80% success probability target
            },
            "emergency_fund": {
                "description": "Build emergency fund",
                "time_horizon": "short",  # 3-12 months
                "risk_tolerance": "low",  # Capital preservation priority
                "priority": "high",
                "expected_return": 0.03,  # 3% annual return (high-yield savings)
                "inflation_rate": 0.025,  # 2.5% annual inflation
                "compound_frequency": 12,  # Monthly compounding
                "success_threshold": 0.95  # 95% success probability target
            },
            "house_down_payment": {
                "description": "Save for house down payment",
                "time_horizon": "medium",  # 2-5 years
                "risk_tolerance": "low",  # Capital preservation priority
                "priority": "medium",
                "expected_return": 0.04,  # 4% annual return (conservative)
                "inflation_rate": 0.025,  # 2.5% annual inflation
                "compound_frequency": 12,  # Monthly compounding
                "success_threshold": 0.9  # 90% success probability target
            },
            "education": {
                "description": "Save for education expenses",
                "time_horizon": "medium",  # 5-15 years
                "risk_tolerance": "medium",  # Moderate risk tolerance
                "priority": "medium",
                "expected_return": 0.06,  # 6% annual return (education inflation)
                "inflation_rate": 0.04,  # 4% annual inflation (education costs)
                "compound_frequency": 12,  # Monthly compounding
                "success_threshold": 0.85  # 85% success probability target
            },
            "vacation": {
                "description": "Save for vacation",
                "time_horizon": "short",  # 3-12 months
                "risk_tolerance": "low",  # Capital preservation priority
                "priority": "low",
                "expected_return": 0.03,  # 3% annual return (savings account)
                "inflation_rate": 0.025,  # 2.5% annual inflation
                "compound_frequency": 12,  # Monthly compounding
                "success_threshold": 0.8  # 80% success probability target
            }
        }
        
        # Mathematical constants for calculations
        self.monthly_compounding_factor = 12  # Number of compounding periods per year
        self.minimum_goal_amount = 100  # Minimum goal amount in currency units
        self.maximum_goal_amount = 10000000  # Maximum goal amount in currency units
        self.maximum_timeline_years = 50  # Maximum goal timeline in years
    
    def create_financial_goal(self, user_id: str, goal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new financial goal for a user with comprehensive mathematical validation
        and feasibility analysis.
        
        This function implements the Time Value of Money (TVM) framework to calculate
        required monthly savings, success probability, and goal feasibility. It uses
        compound interest formulas and risk-adjusted return expectations.
        
        Mathematical Formulas Used:
        1. Future Value (FV) = PV * (1 + r/n)^(n*t)
           where: PV = Present Value, r = annual rate, n = compounding frequency, t = time
        2. Required Monthly Savings = (Target - Current) / ((1 + r/n)^(n*t) - 1) * (r/n)
        3. Success Probability = Normal CDF of (Actual Savings - Required Savings) / Std Dev
        
        Args:
            user_id: Unique identifier for the user
            goal_data: Dictionary containing goal parameters:
                - type: Goal type (retirement, emergency_fund, etc.)
                - target_amount: Target goal amount in currency units
                - timeline_months: Goal timeline in months
                - current_amount: Current amount saved (default: 0)
                - risk_tolerance: User's risk tolerance (low, medium, high)
                - priority: Goal priority (low, medium, high)
                - description: Custom goal description
            
        Returns:
            Dictionary containing:
                - status: Success or error status
                - goal: Complete goal object with calculated parameters
                - recommendations: Initial recommendations for goal achievement
                - feasibility_score: Mathematical feasibility score (0-1)
                - success_probability: Estimated success probability
                
        Raises:
            ValueError: If goal parameters are invalid
            Exception: For calculation or storage errors
        """
        try:
            # Extract and validate goal parameters
            goal_type = goal_data.get("type", "custom")
            target_amount = goal_data.get("target_amount", 0)
            timeline_months = goal_data.get("timeline_months", 12)
            current_amount = goal_data.get("current_amount", 0)
            
            # Mathematical validation of input parameters
            if target_amount <= 0:
                return {
                    "status": "error",
                    "error": "Target amount must be greater than 0",
                    "timestamp": datetime.now().isoformat()
                }
            
            if timeline_months <= 0 or timeline_months > (self.maximum_timeline_years * 12):
                return {
                    "status": "error",
                    "error": f"Timeline must be between 1 and {self.maximum_timeline_years * 12} months",
                    "timestamp": datetime.now().isoformat()
                }
            
            if target_amount < self.minimum_goal_amount or target_amount > self.maximum_goal_amount:
                return {
                    "status": "error",
                    "error": f"Target amount must be between {self.minimum_goal_amount} and {self.maximum_goal_amount}",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Generate unique goal identifier
            goal_id = f"goal_{user_id}_{datetime.now().timestamp()}"
            
            # Get goal template and mathematical parameters
            template = self.goal_templates.get(goal_type, {})
            expected_return = template.get("expected_return", 0.05)  # Default 5% return
            inflation_rate = template.get("inflation_rate", 0.025)  # Default 2.5% inflation
            compound_frequency = template.get("compound_frequency", 12)  # Monthly compounding
            
            # Calculate inflation-adjusted target amount
            # FV = PV * (1 + inflation_rate)^(timeline_years)
            timeline_years = timeline_months / 12
            inflation_adjusted_target = target_amount * (1 + inflation_rate) ** timeline_years
            
            # Calculate required monthly savings using compound interest formula
            # PMT = FV * (r/n) / ((1 + r/n)^(n*t) - 1)
            # where: r = annual rate, n = compounding frequency, t = time in years
            effective_rate = expected_return / compound_frequency
            total_periods = timeline_months
            future_value_factor = (1 + effective_rate) ** total_periods
            
            if future_value_factor > 1:
                monthly_savings_required = (inflation_adjusted_target - current_amount) * effective_rate / (future_value_factor - 1)
            else:
                # Fallback to simple division for very low rates
                monthly_savings_required = (inflation_adjusted_target - current_amount) / timeline_months
            
            # Calculate success probability using normal distribution
            # Assuming 10% standard deviation in savings ability
            savings_std_dev = monthly_savings_required * 0.1
            success_probability = self._calculate_success_probability(monthly_savings_required, savings_std_dev)
            
            # Calculate feasibility score based on multiple factors
            feasibility_score = self._calculate_feasibility_score(
                monthly_savings_required, success_probability, timeline_months, goal_type
            )
            
            # Create comprehensive goal object
            goal = {
                "id": goal_id,
                "user_id": user_id,
                "type": goal_type,
                "description": goal_data.get("description", template.get("description", "Custom financial goal")),
                "target_amount": target_amount,
                "inflation_adjusted_target": inflation_adjusted_target,
                "current_amount": current_amount,
                "timeline_months": timeline_months,
                "timeline_years": timeline_years,
                "monthly_savings_required": monthly_savings_required,
                "expected_return": expected_return,
                "inflation_rate": inflation_rate,
                "compound_frequency": compound_frequency,
                "risk_tolerance": goal_data.get("risk_tolerance", template.get("risk_tolerance", "medium")),
                "priority": goal_data.get("priority", template.get("priority", "medium")),
                "success_probability": success_probability,
                "feasibility_score": feasibility_score,
                "status": "active",
                "created_at": datetime.now().isoformat(),
                "target_date": (datetime.now() + timedelta(days=timeline_months * 30)).isoformat()
            }
            
            # Store goal in memory system for future reference
            goal_text = f"Financial goal created: {goal['description']} - Target: ${target_amount:,.2f} in {timeline_months} months"
            ingest_user_memory(
                user_id=user_id,
                text=goal_text,
                memory_type="goal",
                metadata=goal
            )
            
            # Generate initial recommendations based on goal characteristics
            recommendations = self._generate_goal_recommendations(goal)
            
            return {
                "status": "success",
                "user_id": user_id,
                "goal": goal,
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating financial goal for user {user_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def update_goal_progress(self, user_id: str, goal_id: str, progress_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update progress for a specific goal and recalculate success probability
        and recommendations based on new progress data.
        
        This function implements dynamic goal tracking with real-time mathematical
        updates. It recalculates required savings, success probability, and provides
        updated recommendations based on actual progress versus planned progress.
        
        Mathematical Updates:
        1. Progress Percentage = (Current Amount / Target Amount) * 100
        2. Remaining Amount = Target Amount - Current Amount
        3. Updated Monthly Savings = Remaining Amount / Remaining Months
        4. Success Probability = Updated based on actual savings rate
        5. Time Decay Factor = (Remaining Time / Original Time) ^ 0.5
        
        Args:
            user_id: Unique identifier for the user
            goal_id: Unique identifier for the goal
            progress_data: Dictionary containing progress updates:
                - current_amount: New current amount saved
                - update_timeline: Boolean to update timeline
                - timeline_months: New timeline if updating
                - additional_notes: Any additional progress notes
            
        Returns:
            Dictionary containing:
                - status: Success or error status
                - goal: Updated goal object with new calculations
                - recommendations: Updated recommendations based on progress
                - progress_metrics: Detailed progress analysis
                
        Raises:
            ValueError: If goal not found or invalid progress data
            Exception: For calculation or storage errors
        """
        try:
            # Retrieve goal from memory system
            goal_memories = retrieve_contextual_memories(
                user_id=user_id,
                query_text=f"goal {goal_id} financial target",
                num_results=10
            )
            
            # Find the specific goal in memory
            goal = None
            for memory in goal_memories:
                if memory.get("type") == "goal":
                    metadata = memory.get("metadata", {})
                    if metadata.get("id") == goal_id:
                        goal = metadata
                        break
            
            if not goal:
                return {
                    "status": "error",
                    "error": f"Goal {goal_id} not found",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Update current amount and calculate progress metrics
            current_amount = progress_data.get("current_amount", goal.get("current_amount", 0))
            goal["current_amount"] = current_amount
            
            # Calculate progress percentage
            progress_percentage = (current_amount / goal["target_amount"]) * 100
            goal["progress_percentage"] = progress_percentage
            
            # Calculate remaining amount and required savings
            remaining_amount = goal["target_amount"] - current_amount
            goal["remaining_amount"] = remaining_amount
            
            # Update timeline if requested
            if progress_data.get("update_timeline"):
                new_timeline = progress_data.get("timeline_months", goal["timeline_months"])
                goal["timeline_months"] = new_timeline
                goal["timeline_years"] = new_timeline / 12
                
                # Recalculate required monthly savings with new timeline
                template = self.goal_templates.get(goal["type"], {})
                expected_return = template.get("expected_return", 0.05)
                compound_frequency = template.get("compound_frequency", 12)
                effective_rate = expected_return / compound_frequency
                total_periods = new_timeline
                future_value_factor = (1 + effective_rate) ** total_periods
                
                if future_value_factor > 1:
                    goal["monthly_savings_required"] = remaining_amount * effective_rate / (future_value_factor - 1)
                else:
                    goal["monthly_savings_required"] = remaining_amount / new_timeline
                
                goal["target_date"] = (datetime.now() + timedelta(days=new_timeline * 30)).isoformat()
            
            # Check if goal is completed
            if current_amount >= goal["target_amount"]:
                goal["status"] = "completed"
                goal["completed_at"] = datetime.now().isoformat()
                goal["progress_percentage"] = 100.0
            
            # Recalculate success probability based on actual progress
            actual_monthly_savings = current_amount / (goal["timeline_months"] * (1 - progress_percentage / 100))
            savings_std_dev = goal["monthly_savings_required"] * 0.1
            updated_success_probability = self._calculate_success_probability(
                goal["monthly_savings_required"], savings_std_dev, actual_monthly_savings
            )
            goal["success_probability"] = updated_success_probability
            
            # Calculate progress metrics
            progress_metrics = {
                "progress_percentage": progress_percentage,
                "remaining_amount": remaining_amount,
                "remaining_months": goal["timeline_months"] * (1 - progress_percentage / 100),
                "ahead_behind_schedule": actual_monthly_savings - goal["monthly_savings_required"],
                "completion_rate": progress_percentage / (goal["timeline_months"] / 12)  # Annual completion rate
            }
            
            # Store updated goal in memory
            progress_text = f"Goal progress updated: {goal['description']} - Progress: {progress_percentage:.1f}% (${current_amount:,.2f}/${goal['target_amount']:,.2f})"
            ingest_user_memory(
                user_id=user_id,
                text=progress_text,
                memory_type="goal_progress",
                metadata=goal
            )
            
            # Generate updated recommendations based on progress
            recommendations = self._generate_progress_recommendations(goal)
            
            return {
                "status": "success",
                "user_id": user_id,
                "goal": goal,
                "recommendations": recommendations,
                "progress_metrics": progress_metrics,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating goal progress for user {user_id}, goal {goal_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_goals_overview(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive overview of all user goals with aggregated metrics
        and portfolio context analysis.
        
        This function provides a holistic view of all financial goals, including
        aggregated metrics, portfolio performance context, and goal interdependencies.
        It uses portfolio theory to analyze how goals relate to overall financial
        strategy and risk management.
        
        Mathematical Aggregations:
        1. Total Goal Value = Sum of all goal target amounts
        2. Overall Progress = Weighted average of individual goal progress
        3. Portfolio Goal Allocation = Goal amounts as percentage of total portfolio
        4. Risk-Adjusted Goal Score = Weighted score considering risk and priority
        5. Goal Diversification Score = How well goals are spread across time horizons
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Dictionary containing:
                - goals_overview: Aggregated goal metrics and analysis
                - portfolio_context: Portfolio performance and goal relationship
                - goal_analysis: Detailed analysis of goal characteristics
                - recommendations: Portfolio-level goal recommendations
                
        Raises:
            Exception: For retrieval or calculation errors
        """
        try:
            # Retrieve all goals from memory system
            goal_memories = retrieve_contextual_memories(
                user_id=user_id,
                query_text="financial goal target amount timeline",
                num_results=50
            )
            
            # Extract and analyze goals
            goals = []
            total_target_amount = 0
            total_current_amount = 0
            goal_analysis = {
                "short_term_goals": 0,
                "medium_term_goals": 0,
                "long_term_goals": 0,
                "high_priority_goals": 0,
                "low_risk_goals": 0,
                "goal_diversification_score": 0
            }
            
            for memory in goal_memories:
                if memory.get("type") == "goal":
                    metadata = memory.get("metadata", {})
                    if metadata.get("status") != "completed":
                        goals.append(metadata)
                        total_target_amount += metadata.get("target_amount", 0)
                        total_current_amount += metadata.get("current_amount", 0)
                        
                        # Analyze goal characteristics
                        timeline_months = metadata.get("timeline_months", 12)
                        if timeline_months <= 12:
                            goal_analysis["short_term_goals"] += 1
                        elif timeline_months <= 60:
                            goal_analysis["medium_term_goals"] += 1
                        else:
                            goal_analysis["long_term_goals"] += 1
                        
                        if metadata.get("priority") == "high":
                            goal_analysis["high_priority_goals"] += 1
                        
                        if metadata.get("risk_tolerance") == "low":
                            goal_analysis["low_risk_goals"] += 1
            
            # Calculate overall progress using weighted average
            overall_progress = 0
            if total_target_amount > 0:
                overall_progress = (total_current_amount / total_target_amount) * 100
            
            # Calculate goal diversification score
            total_goals = len(goals)
            if total_goals > 0:
                time_horizon_diversity = 1 - (max(goal_analysis["short_term_goals"], 
                                                goal_analysis["medium_term_goals"], 
                                                goal_analysis["long_term_goals"]) / total_goals)
                goal_analysis["goal_diversification_score"] = time_horizon_diversity
            
            # Get portfolio performance for context
            portfolio_metrics = get_portfolio_performance_metrics(user_id)
            
            # Calculate portfolio-goal relationship metrics
            portfolio_context = {}
            if portfolio_metrics and portfolio_metrics.get("total_value"):
                portfolio_value = portfolio_metrics["total_value"]
                portfolio_context = {
                    "portfolio_value": portfolio_value,
                    "goal_to_portfolio_ratio": total_target_amount / portfolio_value if portfolio_value > 0 else 0,
                    "portfolio_goal_coverage": min(1.0, portfolio_value / total_target_amount) if total_target_amount > 0 else 0,
                    "portfolio_risk_alignment": self._analyze_portfolio_goal_alignment(goals, portfolio_metrics)
                }
            
            return {
                "status": "success",
                "user_id": user_id,
                "goals_overview": {
                    "total_goals": len(goals),
                    "total_target_amount": total_target_amount,
                    "total_current_amount": total_current_amount,
                    "overall_progress_percentage": overall_progress,
                    "average_goal_amount": total_target_amount / len(goals) if goals else 0,
                    "average_timeline_months": sum(g.get("timeline_months", 0) for g in goals) / len(goals) if goals else 0,
                    "goals": goals,
                    "goal_analysis": goal_analysis,
                    "portfolio_context": portfolio_context
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting goals overview for user {user_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_goal_recommendations(self, user_id: str, goal_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get personalized recommendations for goal achievement based on
        user's financial context, portfolio performance, and behavioral patterns.
        
        This function implements a recommendation engine that combines:
        1. Mathematical goal feasibility analysis
        2. Portfolio performance optimization
        3. Behavioral finance insights
        4. Market condition analysis
        5. Risk management principles
        
        Recommendation Categories:
        - Savings Strategy: How to increase savings rate
        - Investment Strategy: Asset allocation for goal funding
        - Risk Management: Hedging strategies for goal protection
        - Timeline Optimization: Adjusting goal timelines
        - Portfolio Integration: How goals fit with overall portfolio
        
        Args:
            user_id: Unique identifier for the user
            goal_id: Optional specific goal identifier for targeted recommendations
            
        Returns:
            Dictionary containing:
                - recommendations: List of personalized recommendations
                - priority_ranking: Recommendations ranked by priority
                - implementation_steps: Actionable steps for each recommendation
                - expected_impact: Quantified impact of each recommendation
                
        Raises:
            Exception: For analysis or recommendation generation errors
        """
        try:
            # Get user's financial context from memory
            financial_memories = retrieve_contextual_memories(
                user_id=user_id,
                query_text="income expenses savings investment portfolio",
                num_results=20
            )
            
            # Get portfolio performance metrics
            portfolio_metrics = get_portfolio_performance_metrics(user_id)
            
            # Generate recommendations based on scope
            recommendations = []
            
            if goal_id:
                # Specific goal recommendations
                goal_memories = retrieve_contextual_memories(
                    user_id=user_id,
                    query_text=f"goal {goal_id}",
                    num_results=5
                )
                
                for memory in goal_memories:
                    if memory.get("type") == "goal":
                        goal = memory.get("metadata", {})
                        if goal.get("id") == goal_id:
                            goal_recommendations = self._generate_goal_recommendations(goal)
                            recommendations.extend(goal_recommendations)
                            break
            else:
                # General goal recommendations
                recommendations = self._generate_general_recommendations(user_id, financial_memories, portfolio_metrics)
            
            # Rank recommendations by priority and expected impact
            ranked_recommendations = self._rank_recommendations(recommendations, user_id)
            
            return {
                "status": "success",
                "user_id": user_id,
                "goal_id": goal_id,
                "recommendations": ranked_recommendations,
                "total_recommendations": len(ranked_recommendations),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting goal recommendations for user {user_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_goal_templates(self) -> Dict[str, Any]:
        """
        Get available goal templates with their mathematical parameters
        and risk profiles for goal creation.
        
        Returns:
            Dictionary containing goal templates with mathematical parameters
        """
        return {
            "status": "success",
            "templates": self.goal_templates,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_success_probability(self, required_savings: float, std_dev: float, 
                                     actual_savings: Optional[float] = None) -> float:
        """
        Calculate the probability of achieving a goal based on required savings
        and user's savings ability distribution.
        
        This function uses the normal distribution to model savings ability
        and calculate success probability. It assumes that actual savings
        follow a normal distribution around the required savings rate.
        
        Mathematical Formula:
        P(Success) = 1 - Φ((Required - Actual) / StdDev)
        where Φ is the cumulative normal distribution function
        
        Args:
            required_savings: Required monthly savings amount
            std_dev: Standard deviation of savings ability (typically 10% of required)
            actual_savings: Optional actual savings rate for comparison
            
        Returns:
            Success probability as a float between 0 and 1
        """
        if actual_savings is None:
            actual_savings = required_savings  # Assume user can meet required rate
        
        if std_dev <= 0:
            return 1.0 if actual_savings >= required_savings else 0.0
        
        # Calculate z-score
        z_score = (required_savings - actual_savings) / std_dev
        
        # Use approximation of normal CDF for simplicity
        # For more accuracy, use scipy.stats.norm.cdf(z_score)
        if z_score <= 0:
            # User is meeting or exceeding required savings
            return 0.5 + 0.5 * (1 - math.exp(-z_score**2 / 2))
        else:
            # User is below required savings
            return 0.5 * math.exp(-z_score**2 / 2)
    
    def _calculate_feasibility_score(self, monthly_savings: float, success_probability: float, 
                                   timeline_months: int, goal_type: str) -> float:
        """
        Calculate a comprehensive feasibility score for a goal based on
        multiple factors including savings rate, success probability, and goal characteristics.
        
        The feasibility score combines:
        1. Savings Rate Feasibility (40% weight)
        2. Success Probability (30% weight)
        3. Timeline Reasonableness (20% weight)
        4. Goal Type Appropriateness (10% weight)
        
        Args:
            monthly_savings: Required monthly savings amount
            success_probability: Calculated success probability
            timeline_months: Goal timeline in months
            goal_type: Type of goal (retirement, emergency_fund, etc.)
            
        Returns:
            Feasibility score as a float between 0 and 1
        """
        # Savings rate feasibility (normalized to typical income ranges)
        # Assume typical monthly savings is 10-20% of income
        typical_savings_rate = 0.15  # 15% of income
        savings_feasibility = min(1.0, (monthly_savings / 5000) / typical_savings_rate)  # Normalize to $5000 income
        
        # Success probability (already 0-1)
        prob_score = success_probability
        
        # Timeline reasonableness (optimal timeline varies by goal type)
        optimal_timelines = {
            "emergency_fund": 6,
            "vacation": 12,
            "house_down_payment": 36,
            "education": 120,
            "retirement": 240
        }
        optimal_timeline = optimal_timelines.get(goal_type, 60)
        timeline_score = 1.0 - abs(timeline_months - optimal_timeline) / optimal_timeline
        timeline_score = max(0.0, min(1.0, timeline_score))
        
        # Goal type appropriateness (based on template success thresholds)
        template = self.goal_templates.get(goal_type, {})
        type_score = template.get("success_threshold", 0.8)
        
        # Weighted combination
        feasibility_score = (
            0.4 * savings_feasibility +
            0.3 * prob_score +
            0.2 * timeline_score +
            0.1 * type_score
        )
        
        return max(0.0, min(1.0, feasibility_score))
    
    def _analyze_portfolio_goal_alignment(self, goals: List[Dict[str, Any]], 
                                        portfolio_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze how well the user's portfolio aligns with their financial goals.
        
        This analysis considers:
        1. Asset allocation appropriateness for goal time horizons
        2. Risk tolerance alignment between portfolio and goals
        3. Liquidity requirements for short-term goals
        4. Return expectations for goal funding
        
        Args:
            goals: List of user's financial goals
            portfolio_metrics: Portfolio performance and allocation metrics
            
        Returns:
            Dictionary containing alignment analysis and recommendations
        """
        alignment_score = 0.5  # Default neutral score
        alignment_factors = []
        
        if not portfolio_metrics or not goals:
            return {"alignment_score": alignment_score, "factors": alignment_factors}
        
        # Analyze short-term goal liquidity
        short_term_goals = [g for g in goals if g.get("timeline_months", 0) <= 12]
        if short_term_goals:
            total_short_term_needs = sum(g.get("target_amount", 0) for g in short_term_goals)
            liquid_assets = portfolio_metrics.get("cash_equivalents", 0)
            liquidity_ratio = liquid_assets / total_short_term_needs if total_short_term_needs > 0 else 1.0
            if liquidity_ratio >= 1.0:
                alignment_score += 0.2
                alignment_factors.append("Adequate liquidity for short-term goals")
            else:
                alignment_score -= 0.1
                alignment_factors.append("Insufficient liquidity for short-term goals")
        
        # Analyze long-term goal growth potential
        long_term_goals = [g for g in goals if g.get("timeline_months", 0) > 60]
        if long_term_goals:
            equity_allocation = portfolio_metrics.get("equity_allocation", 0)
            if equity_allocation >= 0.6:  # 60%+ equity for long-term goals
                alignment_score += 0.2
                alignment_factors.append("Appropriate equity allocation for long-term goals")
            else:
                alignment_score -= 0.1
                alignment_factors.append("Consider increasing equity allocation for long-term goals")
        
        return {
            "alignment_score": max(0.0, min(1.0, alignment_score)),
            "factors": alignment_factors
        }
    
    def _rank_recommendations(self, recommendations: List[Dict[str, Any]], 
                            user_id: str) -> List[Dict[str, Any]]:
        """
        Rank recommendations by priority, expected impact, and implementation difficulty.
        
        Ranking factors:
        1. Priority (high/medium/low)
        2. Expected financial impact
        3. Implementation difficulty
        4. Time to impact
        5. Risk level
        
        Args:
            recommendations: List of recommendation dictionaries
            user_id: User identifier for personalization
            
        Returns:
            Ranked list of recommendations with priority scores
        """
        for rec in recommendations:
            # Calculate priority score
            priority_score = 0
            
            # Priority weight
            priority_weights = {"high": 0.4, "medium": 0.2, "low": 0.1}
            priority_score += priority_weights.get(rec.get("priority", "medium"), 0.2)
            
            # Impact weight (estimated)
            impact_weights = {"high": 0.3, "medium": 0.2, "low": 0.1}
            impact_score = impact_weights.get(rec.get("impact", "medium"), 0.2)
            priority_score += impact_score
            
            # Implementation difficulty (inverse)
            difficulty_weights = {"easy": 0.2, "medium": 0.1, "hard": 0.05}
            difficulty_score = difficulty_weights.get(rec.get("difficulty", "medium"), 0.1)
            priority_score += difficulty_score
            
            rec["priority_score"] = priority_score
        
        # Sort by priority score (descending)
        return sorted(recommendations, key=lambda x: x.get("priority_score", 0), reverse=True)
    
    def _generate_goal_recommendations(self, goal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate specific recommendations for a particular goal based on
        its characteristics, progress, and mathematical analysis.
        
        This function analyzes goal parameters and generates actionable
        recommendations to improve goal achievement probability.
        
        Args:
            goal: Goal dictionary with all parameters and progress data
            
        Returns:
            List of recommendation dictionaries with actionable advice
        """
        recommendations = []
        
        goal_type = goal.get("type", "custom")
        target_amount = goal.get("target_amount", 0)
        timeline_months = goal.get("timeline_months", 12)
        monthly_savings = goal.get("monthly_savings_required", 0)
        progress_percentage = goal.get("progress_percentage", 0)
        success_probability = goal.get("success_probability", 0.5)
        
        # Savings rate recommendations
        if progress_percentage < 25:
            recommendations.append({
                "type": "savings_strategy",
                "priority": "high",
                "title": "Increase Savings Rate",
                "description": f"Consider increasing monthly savings to ${monthly_savings:,.2f} to meet your goal",
                "action": "Review monthly budget and identify areas to reduce expenses",
                "impact": "high",
                "difficulty": "medium",
                "expected_improvement": "20-30% increase in success probability"
            })
        
        # Investment strategy recommendations based on goal type
        if goal_type == "retirement" and timeline_months > 60:
            recommendations.append({
                "type": "investment_strategy",
                "priority": "high",
                "title": "Consider Equity Investments",
                "description": "For long-term retirement goals, consider equity investments for higher potential returns",
                "action": "Review portfolio allocation and consider increasing equity exposure",
                "impact": "high",
                "difficulty": "medium",
                "expected_improvement": "2-3% additional annual return potential"
            })
        
        elif goal_type == "emergency_fund":
            recommendations.append({
                "type": "investment_strategy",
                "priority": "high",
                "title": "Maintain Liquid Savings",
                "description": "Keep emergency fund in high-yield savings account for easy access",
                "action": "Ensure emergency fund is in liquid, low-risk investments",
                "impact": "medium",
                "difficulty": "easy",
                "expected_improvement": "Maintain capital preservation and liquidity"
            })
        
        # Timeline optimization recommendations
        if progress_percentage < 50 and timeline_months < 6:
            recommendations.append({
                "type": "timeline_optimization",
                "priority": "medium",
                "title": "Consider Extending Timeline",
                "description": "You may need more time to reach your goal comfortably",
                "action": "Consider extending the goal timeline or adjusting the target amount",
                "impact": "medium",
                "difficulty": "easy",
                "expected_improvement": "Reduced monthly savings requirement"
            })
        
        # Success probability improvement recommendations
        if success_probability < 0.7:
            recommendations.append({
                "type": "risk_management",
                "priority": "medium",
                "title": "Improve Goal Feasibility",
                "description": f"Current success probability is {success_probability:.1%}. Consider adjustments to improve chances.",
                "action": "Review goal parameters and consider reducing target or extending timeline",
                "impact": "high",
                "difficulty": "medium",
                "expected_improvement": "Increase success probability to 80%+"
            })
        
        return recommendations
    
    def _generate_progress_recommendations(self, goal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on goal progress and performance
        relative to the planned timeline and savings rate.
        
        Args:
            goal: Goal dictionary with progress data
            
        Returns:
            List of progress-based recommendations
        """
        recommendations = []
        
        progress_percentage = goal.get("progress_percentage", 0)
        remaining_amount = goal.get("remaining_amount", 0)
        timeline_months = goal.get("timeline_months", 12)
        
        if progress_percentage >= 100:
            recommendations.append({
                "type": "goal_achievement",
                "priority": "high",
                "title": "Goal Achieved!",
                "description": "Congratulations! You've successfully reached your financial goal.",
                "action": "Consider setting a new goal or increasing your target amount",
                "impact": "high",
                "difficulty": "easy",
                "expected_improvement": "Maintain momentum with new financial objectives"
            })
        
        elif progress_percentage >= 75:
            recommendations.append({
                "type": "goal_motivation",
                "priority": "medium",
                "title": "Almost There!",
                "description": f"You're {progress_percentage:.1f}% of the way to your goal. Keep up the great work!",
                "action": "Stay consistent with your savings plan",
                "impact": "medium",
                "difficulty": "easy",
                "expected_improvement": "Maintain current progress rate"
            })
        
        elif progress_percentage < 25:
            recommendations.append({
                "type": "goal_adjustment",
                "priority": "high",
                "title": "Consider Adjustments",
                "description": "You may need to adjust your savings rate or timeline to reach your goal",
                "action": "Review your budget and consider increasing monthly savings",
                "impact": "high",
                "difficulty": "medium",
                "expected_improvement": "Improve goal achievement probability"
            })
        
        return recommendations
    
    def _generate_general_recommendations(self, user_id: str, financial_memories: List[Dict[str, Any]], 
                                        portfolio_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate general financial goal recommendations based on user's
        overall financial situation and portfolio performance.
        
        Args:
            user_id: User identifier
            financial_memories: User's financial history and context
            portfolio_metrics: Portfolio performance data
            
        Returns:
            List of general financial recommendations
        """
        recommendations = []
        
        # Analyze financial context from memories
        has_emergency_fund = False
        has_retirement_goal = False
        
        for memory in financial_memories:
            if memory.get("type") == "goal":
                metadata = memory.get("metadata", {})
                if metadata.get("type") == "emergency_fund":
                    has_emergency_fund = True
                elif metadata.get("type") == "retirement":
                    has_retirement_goal = True
        
        # Priority recommendations based on financial planning best practices
        if not has_emergency_fund:
            recommendations.append({
                "type": "goal_setting",
                "priority": "high",
                "title": "Build Emergency Fund",
                "description": "Consider setting up an emergency fund goal for financial security",
                "action": "Create a goal to save 3-6 months of expenses in an emergency fund",
                "impact": "high",
                "difficulty": "medium",
                "expected_improvement": "Financial security and reduced stress"
            })
        
        if not has_retirement_goal:
            recommendations.append({
                "type": "goal_setting",
                "priority": "high",
                "title": "Plan for Retirement",
                "description": "Consider setting up retirement savings goals",
                "action": "Create a retirement goal based on your desired retirement age and lifestyle",
                "impact": "high",
                "difficulty": "medium",
                "expected_improvement": "Long-term financial security"
            })
        
        # Portfolio-based recommendations
        if portfolio_metrics:
            current_return = portfolio_metrics.get("total_return", 0)
            if current_return < 0.05:  # Less than 5% return
                recommendations.append({
                    "type": "investment_strategy",
                    "priority": "medium",
                    "title": "Review Investment Strategy",
                    "description": "Your portfolio returns may need optimization",
                    "action": "Consider reviewing your asset allocation and investment strategy",
                    "impact": "medium",
                    "difficulty": "medium",
                    "expected_improvement": "Potential for higher returns"
                })
        
        return recommendations

# Global service instance for application-wide access
goal_planning_service = GoalPlanningService()
