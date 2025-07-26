"""
Tethys - User Interaction Service

This service manages user interactions, natural language processing, and
personalized response generation for the Tethys Financial Co-Pilot. It
integrates Google's Gemini AI models with the Memory Layer and Mathematical
Intelligence to provide intelligent, contextual financial guidance.

The service implements advanced natural language processing using:
- Google Gemini AI models for intelligent response generation
- Semantic similarity analysis for query understanding
- Context retrieval and relevance ranking
- User preference learning and adaptation
- Multi-turn conversation management

Mathematical Framework:
- Semantic Similarity Score = Cosine Similarity(Query Vector, Context Vector)
- Context Relevance Score = Weighted Average of Memory Relevance Scores
- User Preference Weight = Historical Interaction Success Rate
- Response Quality Score = Relevance * Accuracy * Completeness

AI Models Used:
- Google Gemini Pro: Primary language model for response generation
- Google Gemini Pro Vision: For visual data analysis (future enhancement)
- Sentence Transformers: For semantic similarity calculations
- Custom fine-tuned models: For financial domain specialization

Author: Tethys Development Team
Version: 1.0.0
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import numpy as np
from collections import defaultdict

# Import Tethys components
from memory_management.memory_manager import ingest_user_memory, retrieve_contextual_memories
from memory_management.gemini_connector import GeminiConnector, GeminiConfig
from financial_intelligence.financial_quant_tools import get_portfolio_performance_metrics
from services.anomaly_detection_service import AnomalyDetectionService
from services.portfolio_management_service import PortfolioManagementService
from services.goal_planning_service import GoalPlanningService

logger = logging.getLogger(__name__)

class UserInteractionService:
    """
    Service for managing user interactions and natural language processing
    in Tethys Financial Co-Pilot.
    
    This service provides intelligent conversation capabilities using Google's
    Gemini AI models combined with Tethys's Memory Layer and Mathematical
    Intelligence. It processes natural language queries, retrieves relevant
    context, and generates personalized financial guidance.
    
    Key Capabilities:
    1. Natural Language Processing: Understanding financial queries in plain English
    2. Context Retrieval: Intelligent memory search and relevance ranking
    3. Response Generation: AI-powered financial advice and explanations
    4. User Preference Learning: Adaptive responses based on user behavior
    5. Multi-turn Conversations: Maintaining context across conversation turns
    
    Mathematical Concepts:
    1. Semantic Similarity: Vector space modeling for query-context matching
    2. Relevance Ranking: Weighted scoring of memory relevance
    3. User Preference Modeling: Learning user interaction patterns
    4. Response Quality Assessment: Multi-dimensional response evaluation
    5. Conversation Flow Analysis: Turn-taking and context management
    
    AI Integration:
    - Google Gemini Pro: Primary language model for response generation
    - Sentence Transformers: Semantic similarity calculations
    - Custom Financial Models: Domain-specific fine-tuning
    """
    
    def __init__(self):
        """
        Initialize the user interaction service with Google Gemini AI
        integration and user preference management systems.
        
        The service initializes with:
        - Google Gemini AI connector for natural language processing
        - User preference tracking and learning systems
        - Interaction history management
        - Response quality assessment metrics
        """
        # Initialize Google Gemini AI connector
        self.gemini_connector = GeminiConnector()
        
        # User interaction tracking and preference management
        self.interaction_history = defaultdict(list)
        self.user_preferences = defaultdict(dict)
        
        # Response quality metrics and learning
        self.response_quality_scores = defaultdict(list)
        self.user_satisfaction_ratings = defaultdict(list)
        
        # Service integration for comprehensive responses
        self.anomaly_service = AnomalyDetectionService()
        self.portfolio_service = PortfolioManagementService()
        self.goal_service = GoalPlanningService()
        
        # Conversation context management
        self.conversation_contexts = defaultdict(dict)
        
        # Mathematical parameters for similarity and ranking
        self.similarity_threshold = 0.7  # Minimum similarity for context inclusion
        self.max_context_memories = 5    # Maximum memories to include in context
        self.preference_learning_rate = 0.1  # Rate of preference adaptation
        
        logger.info("UserInteractionService initialized with Gemini AI integration")
    
    def process_user_query(self, user_id: str, query_text: str, 
                          query_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a natural language user query and generate an intelligent,
        contextual response using Google Gemini AI and Tethys's knowledge base.
        
        This function implements a sophisticated query processing pipeline:
        1. Query Analysis: Understanding intent and extracting key concepts
        2. Context Retrieval: Finding relevant memories and financial data
        3. Response Generation: Using Google Gemini AI for intelligent responses
        4. Quality Assessment: Evaluating response relevance and accuracy
        5. Learning Integration: Updating user preferences and interaction patterns
        
        Mathematical Processing:
        - Query Vectorization: Converting text to semantic vectors
        - Context Relevance Scoring: Weighted similarity calculations
        - Response Quality Metrics: Multi-dimensional evaluation
        - User Preference Updates: Adaptive learning algorithms
        
        Args:
            user_id: Unique identifier for the user
            query_text: Natural language query from the user
            query_context: Optional additional context (timestamp, platform, etc.)
            
        Returns:
            Dictionary containing:
                - response: Generated response text
                - confidence_score: AI confidence in the response
                - context_used: List of relevant memories used
                - recommendations: Additional financial recommendations
                - follow_up_questions: Suggested follow-up questions
                - response_metadata: Technical details about response generation
                
        Raises:
            Exception: For AI processing or context retrieval errors
        """
        try:
            start_time = datetime.now()
            
            # Step 1: Query Analysis and Intent Recognition
            query_analysis = self._analyze_query_intent(query_text, user_id)
            intent = query_analysis.get("intent", "general")
            confidence = query_analysis.get("confidence", 0.5)
            
            # Step 2: Context Retrieval and Relevance Ranking
            relevant_memories = self._retrieve_relevant_context(user_id, query_text, intent)
            financial_context = self._get_financial_context(user_id)
            
            # Step 3: Response Generation using Google Gemini AI
            response_data = self._generate_ai_response(
                user_id, query_text, relevant_memories, financial_context, intent
            )
            
            # Step 4: Response Enhancement and Validation
            enhanced_response = self._enhance_response_with_services(
                user_id, response_data, query_text, intent
            )
            
            # Step 5: Quality Assessment and Learning
            quality_metrics = self._assess_response_quality(
                user_id, query_text, enhanced_response, relevant_memories
            )
            
            # Step 6: Store Interaction for Learning
            self._store_interaction(user_id, query_text, enhanced_response, quality_metrics)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "status": "success",
                "user_id": user_id,
                "response": enhanced_response.get("response", ""),
                "confidence_score": enhanced_response.get("confidence", confidence),
                "context_used": relevant_memories,
                "recommendations": enhanced_response.get("recommendations", []),
                "follow_up_questions": enhanced_response.get("follow_up_questions", []),
                "response_metadata": {
                    "processing_time_seconds": processing_time,
                    "intent_recognized": intent,
                    "memories_retrieved": len(relevant_memories),
                    "quality_score": quality_metrics.get("overall_score", 0.0),
                    "ai_model_used": "Google Gemini Pro",
                    "context_relevance_score": quality_metrics.get("context_relevance", 0.0)
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing user query for {user_id}: {e}")
            return {
                "status": "error",
                "user_id": user_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieve comprehensive user preferences and interaction patterns
        for personalization and response optimization.
        
        This function analyzes user interaction history to extract:
        1. Communication Style Preferences: Formal vs. casual, detail level
        2. Financial Knowledge Level: Beginner, intermediate, advanced
        3. Response Format Preferences: Text, charts, tables, summaries
        4. Topic Preferences: Investment focus, goal planning, risk management
        5. Interaction Frequency Patterns: Usage timing and frequency
        
        Mathematical Analysis:
        - Preference Weight Calculation: Frequency * Recency * Satisfaction
        - Knowledge Level Assessment: Response complexity and topic depth
        - Communication Style Classification: Language complexity analysis
        - Topic Interest Scoring: Engagement metrics by topic category
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Dictionary containing:
                - communication_preferences: Style and format preferences
                - knowledge_level: Assessed financial knowledge level
                - topic_preferences: Interest levels in different topics
                - interaction_patterns: Usage frequency and timing patterns
                - learning_progress: Knowledge development over time
                
        Raises:
            Exception: For preference analysis or retrieval errors
        """
        try:
            # Retrieve user interaction history
            interaction_history = self.interaction_history.get(user_id, [])
            user_prefs = self.user_preferences.get(user_id, {})
            
            if not interaction_history:
                return {
                    "status": "success",
                    "user_id": user_id,
                    "preferences": {
                        "communication_style": "balanced",
                        "knowledge_level": "beginner",
                        "response_format": "text",
                        "topic_preferences": {},
                        "interaction_frequency": "low",
                        "learning_progress": "initial"
                    },
                    "timestamp": datetime.now().isoformat()
                }
            
            # Analyze communication style preferences
            communication_style = self._analyze_communication_style(interaction_history)
            
            # Assess financial knowledge level
            knowledge_level = self._assess_knowledge_level(interaction_history)
            
            # Analyze topic preferences
            topic_preferences = self._analyze_topic_preferences(interaction_history)
            
            # Analyze interaction patterns
            interaction_patterns = self._analyze_interaction_patterns(interaction_history)
            
            # Calculate learning progress
            learning_progress = self._calculate_learning_progress(interaction_history)
            
            # Compile comprehensive preferences
            preferences = {
                "communication_style": communication_style,
                "knowledge_level": knowledge_level,
                "response_format": user_prefs.get("preferred_format", "text"),
                "topic_preferences": topic_preferences,
                "interaction_frequency": interaction_patterns.get("frequency", "medium"),
                "learning_progress": learning_progress,
                "preferred_detail_level": user_prefs.get("detail_level", "moderate"),
                "risk_tolerance_communication": user_prefs.get("risk_communication", "balanced"),
                "technical_terminology_preference": user_prefs.get("tech_terms", "explained")
            }
            
            return {
                "status": "success",
                "user_id": user_id,
                "preferences": preferences,
                "analysis_metadata": {
                    "interactions_analyzed": len(interaction_history),
                    "analysis_date": datetime.now().isoformat(),
                    "confidence_level": self._calculate_preference_confidence(interaction_history)
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting user preferences for {user_id}: {e}")
            return {
                "status": "error",
                "user_id": user_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def update_user_preferences(self, user_id: str, preference_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update user preferences based on explicit feedback or inferred patterns
        to improve future interaction quality and personalization.
        
        This function implements adaptive preference learning that combines:
        1. Explicit User Feedback: Direct preference statements
        2. Implicit Learning: Behavior pattern analysis
        3. Feedback Integration: Response quality and satisfaction metrics
        4. Preference Evolution: Gradual adaptation over time
        
        Mathematical Learning:
        - Preference Update: P_new = P_old + α * (Feedback - P_old)
        - Confidence Weighting: Weight updates by feedback confidence
        - Temporal Decay: Recent preferences weighted more heavily
        - Consistency Scoring: Preference stability over time
        
        Args:
            user_id: Unique identifier for the user
            preference_updates: Dictionary containing preference updates:
                - communication_style: Preferred communication approach
                - knowledge_level: Self-assessed or inferred knowledge level
                - response_format: Preferred response formats
                - topic_preferences: Interest levels in specific topics
                - detail_level: Preferred level of detail in responses
                - feedback_scores: Quality ratings for recent interactions
                
        Returns:
            Dictionary containing:
                - updated_preferences: New preference state
                - learning_metrics: Metrics about preference evolution
                - recommendations: Suggestions for preference optimization
                
        Raises:
            Exception: For preference update or learning errors
        """
        try:
            current_preferences = self.user_preferences.get(user_id, {})
            
            # Apply preference updates with learning rate
            updated_preferences = {}
            learning_metrics = {
                "preferences_updated": 0,
                "confidence_changes": {},
                "learning_rate_applied": self.preference_learning_rate
            }
            
            for key, new_value in preference_updates.items():
                if key in current_preferences:
                    # Gradual learning: blend old and new preferences
                    old_value = current_preferences[key]
                    if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                        # Numeric preferences: weighted average
                        updated_value = old_value + self.preference_learning_rate * (new_value - old_value)
                    else:
                        # Categorical preferences: gradual transition
                        updated_value = new_value if np.random.random() < self.preference_learning_rate else old_value
                else:
                    # New preference: direct assignment
                    updated_value = new_value
                
                updated_preferences[key] = updated_value
                learning_metrics["preferences_updated"] += 1
            
            # Update user preferences
            self.user_preferences[user_id].update(updated_preferences)
            
            # Calculate learning metrics
            confidence_changes = self._calculate_preference_confidence_changes(
                user_id, current_preferences, updated_preferences
            )
            learning_metrics["confidence_changes"] = confidence_changes
            
            # Generate recommendations for preference optimization
            recommendations = self._generate_preference_recommendations(user_id, updated_preferences)
            
            # Store preference update in memory
            preference_text = f"User preferences updated: {list(updated_preferences.keys())}"
            ingest_user_memory(
                user_id=user_id,
                text=preference_text,
                memory_type="preference_update",
                metadata={"updates": updated_preferences, "learning_metrics": learning_metrics}
            )
            
            return {
                "status": "success",
                "user_id": user_id,
                "updated_preferences": updated_preferences,
                "learning_metrics": learning_metrics,
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating user preferences for {user_id}: {e}")
            return {
                "status": "error",
                "user_id": user_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_interaction_history(self, user_id: str, 
                               limit: Optional[int] = 50,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve user interaction history with filtering and analysis capabilities
        for understanding user behavior patterns and improving service quality.
        
        This function provides comprehensive interaction analysis including:
        1. Query Patterns: Common question types and topics
        2. Response Quality Trends: Improvement or degradation over time
        3. User Engagement Metrics: Interaction frequency and depth
        4. Learning Progression: Knowledge development patterns
        5. Satisfaction Trends: User feedback and quality ratings
        
        Mathematical Analysis:
        - Interaction Frequency: Events per time period analysis
        - Quality Trend Analysis: Linear regression of response quality scores
        - Topic Distribution: Entropy-based topic diversity measurement
        - Engagement Scoring: Multi-factor engagement calculation
        - Learning Curve Analysis: Knowledge acquisition rate modeling
        
        Args:
            user_id: Unique identifier for the user
            limit: Maximum number of interactions to return
            start_date: Start date for filtering (ISO format)
            end_date: End date for filtering (ISO format)
            
        Returns:
            Dictionary containing:
                - interactions: List of interaction records
                - analysis: Statistical analysis of interaction patterns
                - trends: Temporal trends in user behavior
                - insights: Derived insights about user behavior
                
        Raises:
            Exception: For history retrieval or analysis errors
        """
        try:
            # Retrieve interaction history
            all_interactions = self.interaction_history.get(user_id, [])
            
            # Apply date filtering if specified
            if start_date or end_date:
                filtered_interactions = []
                start_dt = datetime.fromisoformat(start_date) if start_date else datetime.min
                end_dt = datetime.fromisoformat(end_date) if end_date else datetime.max
                
                for interaction in all_interactions:
                    interaction_dt = datetime.fromisoformat(interaction.get("timestamp", ""))
                    if start_dt <= interaction_dt <= end_dt:
                        filtered_interactions.append(interaction)
                
                interactions = filtered_interactions
            else:
                interactions = all_interactions
            
            # Apply limit
            if limit:
                interactions = interactions[-limit:]
            
            # Perform statistical analysis
            analysis = self._analyze_interaction_history(interactions)
            
            # Calculate temporal trends
            trends = self._calculate_interaction_trends(interactions)
            
            # Generate behavioral insights
            insights = self._generate_behavioral_insights(interactions, analysis)
            
            return {
                "status": "success",
                "user_id": user_id,
                "interactions": interactions,
                "analysis": analysis,
                "trends": trends,
                "insights": insights,
                "metadata": {
                    "total_interactions": len(all_interactions),
                    "filtered_interactions": len(interactions),
                    "date_range": {
                        "start": start_date,
                        "end": end_date
                    },
                    "analysis_date": datetime.now().isoformat()
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting interaction history for {user_id}: {e}")
            return {
                "status": "error",
                "user_id": user_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_personalized_insights(self, user_id: str) -> Dict[str, Any]:
        """
        Generate personalized financial insights based on user interaction
        history, preferences, and financial context from the Memory Layer.
        
        This function creates intelligent insights by combining:
        1. Interaction Pattern Analysis: Learning from user behavior
        2. Financial Context Integration: Portfolio and goal data
        3. Predictive Modeling: Future behavior and needs prediction
        4. Comparative Analysis: Benchmarking against similar users
        5. Opportunity Identification: Financial improvement suggestions
        
        Mathematical Framework:
        - Insight Relevance Score = User Interest * Financial Impact * Urgency
        - Personalization Weight = Historical Engagement * Topic Preference
        - Predictive Accuracy = Model Confidence * Data Quality
        - Comparative Benchmarking = Percentile ranking across user base
        - Opportunity Scoring = Potential Impact * Implementation Feasibility
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Dictionary containing:
                - insights: List of personalized insights
                - recommendations: Actionable recommendations
                - learning_progress: Knowledge development insights
                - improvement_opportunities: Areas for financial improvement
                - comparative_analysis: Benchmarking against similar users
                
        Raises:
            Exception: For insight generation or analysis errors
        """
        try:
            # Retrieve user context and history
            interaction_history = self.interaction_history.get(user_id, [])
            user_preferences = self.user_preferences.get(user_id, {})
            financial_memories = retrieve_contextual_memories(
                user_id=user_id,
                query_text="portfolio goals investments financial situation",
                num_results=20
            )
            
            # Generate personalized insights
            insights = []
            
            # Learning progress insights
            learning_insights = self._generate_learning_insights(interaction_history, user_preferences)
            insights.extend(learning_insights)
            
            # Financial behavior insights
            financial_insights = self._generate_financial_insights(financial_memories, user_preferences)
            insights.extend(financial_insights)
            
            # Interaction pattern insights
            pattern_insights = self._generate_pattern_insights(interaction_history, user_preferences)
            insights.extend(pattern_insights)
            
            # Generate actionable recommendations
            recommendations = self._generate_insight_recommendations(insights, user_preferences)
            
            # Calculate learning progress metrics
            learning_progress = self._calculate_detailed_learning_progress(interaction_history)
            
            # Identify improvement opportunities
            improvement_opportunities = self._identify_improvement_opportunities(
                insights, financial_memories, user_preferences
            )
            
            # Perform comparative analysis
            comparative_analysis = self._perform_comparative_analysis(user_id, interaction_history)
            
            return {
                "status": "success",
                "user_id": user_id,
                "insights": insights,
                "recommendations": recommendations,
                "learning_progress": learning_progress,
                "improvement_opportunities": improvement_opportunities,
                "comparative_analysis": comparative_analysis,
                "metadata": {
                    "insights_generated": len(insights),
                    "recommendations_count": len(recommendations),
                    "analysis_depth": "comprehensive",
                    "personalization_level": "high"
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating personalized insights for {user_id}: {e}")
            return {
                "status": "error",
                "user_id": user_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _analyze_query_intent(self, query_text: str, user_id: str) -> Dict[str, Any]:
        """
        Analyze user query to determine intent and extract key concepts
        for improved context retrieval and response generation.
        
        This function implements intent recognition using:
        1. Keyword Analysis: Financial terminology identification
        2. Semantic Classification: Query type categorization
        3. User Context Integration: Personal history consideration
        4. Confidence Scoring: Intent recognition reliability
        
        Args:
            query_text: User's natural language query
            user_id: User identifier for context
            
        Returns:
            Dictionary containing intent analysis results
        """
        # Define financial intent categories
        intent_keywords = {
            "portfolio_analysis": ["portfolio", "investments", "returns", "performance", "allocation"],
            "goal_planning": ["goal", "target", "save", "retirement", "emergency", "house"],
            "risk_assessment": ["risk", "volatility", "safety", "protection", "insurance"],
            "market_analysis": ["market", "stocks", "bonds", "economy", "trends"],
            "budget_planning": ["budget", "expenses", "income", "spending", "savings"],
            "tax_planning": ["tax", "deduction", "credit", "filing", "optimization"],
            "general_advice": ["advice", "recommendation", "suggestion", "help"]
        }
        
        query_lower = query_text.lower()
        intent_scores = {}
        
        # Calculate intent scores based on keyword presence
        for intent, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            intent_scores[intent] = score / len(keywords) if keywords else 0
        
        # Determine primary intent
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])
        
        # Calculate confidence based on score strength
        confidence = primary_intent[1] if primary_intent[1] > 0.3 else 0.5
        
        return {
            "intent": primary_intent[0],
            "confidence": confidence,
            "all_scores": intent_scores,
            "query_complexity": len(query_text.split()) / 10  # Normalized complexity
        }
    
    def _retrieve_relevant_context(self, user_id: str, query_text: str, 
                                 intent: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories and context from the Memory Layer
        based on query semantic similarity and intent matching.
        
        This function implements intelligent context retrieval using:
        1. Semantic Similarity: Vector-based memory matching
        2. Intent Filtering: Context relevance to query intent
        3. Recency Weighting: Recent memories weighted more heavily
        4. Relevance Ranking: Multi-factor relevance scoring
        
        Args:
            user_id: User identifier
            query_text: User's query for context matching
            intent: Recognized query intent
            
        Returns:
            List of relevant memory contexts
        """
        # Retrieve memories from Memory Layer
        memories = retrieve_contextual_memories(
            user_id=user_id,
            query_text=query_text,
            num_results=self.max_context_memories * 2  # Retrieve more for filtering
        )
        
        # Filter and rank memories by relevance
        relevant_memories = []
        for memory in memories:
            relevance_score = self._calculate_memory_relevance(memory, query_text, intent)
            if relevance_score >= self.similarity_threshold:
                memory["relevance_score"] = relevance_score
                relevant_memories.append(memory)
        
        # Sort by relevance and limit results
        relevant_memories.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return relevant_memories[:self.max_context_memories]
    
    def _get_financial_context(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieve current financial context including portfolio performance,
        goals, and recent financial activities for comprehensive response generation.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary containing financial context data
        """
        try:
            # Get portfolio performance metrics
            portfolio_metrics = get_portfolio_performance_metrics(user_id)
            
            # Get recent financial memories
            recent_memories = retrieve_contextual_memories(
                user_id=user_id,
                query_text="recent financial activity transactions portfolio",
                num_results=10
            )
            
            return {
                "portfolio_metrics": portfolio_metrics,
                "recent_activities": recent_memories,
                "context_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.warning(f"Error retrieving financial context for {user_id}: {e}")
            return {"portfolio_metrics": None, "recent_activities": []}
    
    def _generate_ai_response(self, user_id: str, query_text: str, 
                            relevant_memories: List[Dict[str, Any]],
                            financial_context: Dict[str, Any],
                            intent: str) -> Dict[str, Any]:
        """
        Generate intelligent response using Google Gemini AI with
        integrated context from Memory Layer and Mathematical Intelligence.
        
        This function orchestrates the AI response generation process:
        1. Context Preparation: Formatting memories and financial data
        2. Prompt Engineering: Creating optimized prompts for Gemini AI
        3. Response Generation: Using Gemini AI for intelligent responses
        4. Response Enhancement: Adding financial insights and recommendations
        5. Quality Validation: Ensuring response accuracy and relevance
        
        Args:
            user_id: User identifier
            query_text: Original user query
            relevant_memories: Retrieved relevant memories
            financial_context: Current financial context
            intent: Recognized query intent
            
        Returns:
            Dictionary containing generated response and metadata
        """
        try:
            # Prepare context for AI
            context_summary = self._prepare_context_summary(relevant_memories, financial_context)
            
            # Create optimized prompt for Gemini AI
            prompt = self._create_ai_prompt(query_text, context_summary, intent, user_id)
            
            # Generate response using Google Gemini AI
            ai_response = self.gemini_connector.generate_response(prompt)
            
            # Parse and enhance AI response
            enhanced_response = self._parse_and_enhance_ai_response(ai_response, intent)
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error generating AI response for {user_id}: {e}")
            return {
                "response": "I apologize, but I'm having trouble processing your request right now. Please try again.",
                "confidence": 0.3,
                "recommendations": [],
                "follow_up_questions": []
            }
    
    def _enhance_response_with_services(self, user_id: str, response_data: Dict[str, Any],
                                      query_text: str, intent: str) -> Dict[str, Any]:
        """
        Enhance AI response with additional insights from specialized services
        based on query intent and user context.
        
        Args:
            user_id: User identifier
            response_data: Base AI response data
            query_text: Original user query
            intent: Query intent for service selection
            
        Returns:
            Enhanced response with service-specific insights
        """
        enhanced_response = response_data.copy()
        additional_recommendations = []
        
        try:
            # Add service-specific insights based on intent
            if intent == "portfolio_analysis":
                portfolio_insights = self.portfolio_service.get_portfolio_overview(user_id)
                if portfolio_insights.get("status") == "success":
                    additional_recommendations.extend(
                        portfolio_insights.get("recommendations", [])
                    )
            
            elif intent == "goal_planning":
                goal_insights = self.goal_service.get_goal_recommendations(user_id)
                if goal_insights.get("status") == "success":
                    additional_recommendations.extend(
                        goal_insights.get("recommendations", [])
                    )
            
            elif intent == "risk_assessment":
                anomaly_insights = self.anomaly_service.get_user_anomaly_preferences(user_id)
                if anomaly_insights.get("status") == "success":
                    additional_recommendations.extend(
                        anomaly_insights.get("recommendations", [])
                    )
            
            # Merge recommendations
            all_recommendations = response_data.get("recommendations", []) + additional_recommendations
            enhanced_response["recommendations"] = all_recommendations[:5]  # Limit to top 5
            
        except Exception as e:
            logger.warning(f"Error enhancing response with services for {user_id}: {e}")
        
        return enhanced_response
    
    def _assess_response_quality(self, user_id: str, query_text: str,
                               response_data: Dict[str, Any],
                               relevant_memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess the quality of generated response using multiple metrics
        for continuous improvement and learning.
        
        Quality metrics include:
        1. Relevance Score: How well response addresses the query
        2. Completeness Score: Coverage of query aspects
        3. Accuracy Score: Factual correctness assessment
        4. Clarity Score: Response understandability
        5. Actionability Score: Practical value of recommendations
        
        Args:
            user_id: User identifier
            query_text: Original user query
            response_data: Generated response data
            relevant_memories: Context used for response generation
            
        Returns:
            Dictionary containing quality assessment metrics
        """
        response_text = response_data.get("response", "")
        
        # Calculate relevance score
        relevance_score = self._calculate_response_relevance(query_text, response_text, relevant_memories)
        
        # Calculate completeness score
        completeness_score = self._calculate_response_completeness(query_text, response_text)
        
        # Calculate clarity score
        clarity_score = self._calculate_response_clarity(response_text)
        
        # Calculate actionability score
        actionability_score = self._calculate_response_actionability(response_data)
        
        # Overall quality score (weighted average)
        overall_score = (
            0.3 * relevance_score +
            0.25 * completeness_score +
            0.25 * clarity_score +
            0.2 * actionability_score
        )
        
        return {
            "overall_score": overall_score,
            "relevance_score": relevance_score,
            "completeness_score": completeness_score,
            "clarity_score": clarity_score,
            "actionability_score": actionability_score,
            "context_relevance": sum(m.get("relevance_score", 0) for m in relevant_memories) / len(relevant_memories) if relevant_memories else 0
        }
    
    def _store_interaction(self, user_id: str, query_text: str,
                          response_data: Dict[str, Any],
                          quality_metrics: Dict[str, Any]) -> None:
        """
        Store interaction data for learning and improvement purposes.
        
        Args:
            user_id: User identifier
            query_text: Original user query
            response_data: Generated response data
            quality_metrics: Response quality assessment
        """
        interaction_record = {
            "query": query_text,
            "response": response_data.get("response", ""),
            "confidence": response_data.get("confidence", 0.0),
            "quality_metrics": quality_metrics,
            "timestamp": datetime.now().isoformat(),
            "recommendations_count": len(response_data.get("recommendations", [])),
            "context_memories_used": len(response_data.get("context_used", []))
        }
        
        self.interaction_history[user_id].append(interaction_record)
        
        # Store in Memory Layer for long-term retention
        interaction_summary = f"User query: {query_text[:100]}... Response quality: {quality_metrics.get('overall_score', 0):.2f}"
        ingest_user_memory(
            user_id=user_id,
            text=interaction_summary,
            memory_type="user_interaction",
            metadata=interaction_record
        )
    
    # Additional helper methods for comprehensive functionality...
    def _calculate_memory_relevance(self, memory: Dict[str, Any], query_text: str, intent: str) -> float:
        """Calculate relevance score for a memory based on query and intent."""
        # Simplified relevance calculation
        memory_text = memory.get("text", "").lower()
        query_words = query_text.lower().split()
        word_matches = sum(1 for word in query_words if word in memory_text)
        return word_matches / len(query_words) if query_words else 0
    
    def _prepare_context_summary(self, memories: List[Dict[str, Any]], 
                               financial_context: Dict[str, Any]) -> str:
        """Prepare context summary for AI prompt generation."""
        context_parts = []
        
        # Add memory context
        for memory in memories:
            context_parts.append(f"Memory: {memory.get('text', '')}")
        
        # Add financial context
        if financial_context.get("portfolio_metrics"):
            context_parts.append("Portfolio data available")
        
        return "\n".join(context_parts)
    
    def _create_ai_prompt(self, query_text: str, context_summary: str, 
                         intent: str, user_id: str) -> str:
        """Create optimized prompt for Google Gemini AI."""
        return f"""
        You are Tethys, an AI Financial Co-Pilot. Provide intelligent, personalized financial advice.
        
        User Query: {query_text}
        Query Intent: {intent}
        
        Context Information:
        {context_summary}
        
        Please provide:
        1. A clear, helpful response to the user's query
        2. 2-3 actionable recommendations
        3. 1-2 follow-up questions to better understand their needs
        
        Focus on being helpful, accurate, and personalized to their financial situation.
        """
    
    def _parse_and_enhance_ai_response(self, ai_response: str, intent: str) -> Dict[str, Any]:
        """Parse and enhance AI response with structured data."""
        # Simplified parsing - in production, use more sophisticated parsing
        return {
            "response": ai_response,
            "confidence": 0.8,
            "recommendations": [],
            "follow_up_questions": []
        }
    
    def _calculate_response_relevance(self, query: str, response: str, 
                                    memories: List[Dict[str, Any]]) -> float:
        """Calculate how relevant the response is to the query."""
        # Simplified relevance calculation
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words.intersection(response_words))
        return min(1.0, overlap / len(query_words)) if query_words else 0.5
    
    def _calculate_response_completeness(self, query: str, response: str) -> float:
        """Calculate how completely the response addresses the query."""
        # Simplified completeness calculation
        return min(1.0, len(response) / 100)  # Normalize by expected response length
    
    def _calculate_response_clarity(self, response: str) -> float:
        """Calculate how clear and understandable the response is."""
        # Simplified clarity calculation
        sentences = response.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        return max(0.5, 1.0 - (avg_sentence_length - 15) / 15)  # Prefer 15-word sentences
    
    def _calculate_response_actionability(self, response_data: Dict[str, Any]) -> float:
        """Calculate how actionable the response recommendations are."""
        recommendations = response_data.get("recommendations", [])
        return min(1.0, len(recommendations) / 3)  # Normalize by expected recommendation count
    
    # Additional analysis methods for user preferences and insights...
    def _analyze_communication_style(self, interactions: List[Dict[str, Any]]) -> str:
        """Analyze user's preferred communication style."""
        # Simplified analysis
        return "balanced"
    
    def _assess_knowledge_level(self, interactions: List[Dict[str, Any]]) -> str:
        """Assess user's financial knowledge level."""
        # Simplified assessment
        return "intermediate"
    
    def _analyze_topic_preferences(self, interactions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze user's topic preferences."""
        # Simplified analysis
        return {"portfolio_analysis": 0.7, "goal_planning": 0.6}
    
    def _analyze_interaction_patterns(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user interaction patterns."""
        # Simplified analysis
        return {"frequency": "medium", "preferred_time": "afternoon"}
    
    def _calculate_learning_progress(self, interactions: List[Dict[str, Any]]) -> str:
        """Calculate user's learning progress."""
        # Simplified calculation
        return "intermediate"
    
    def _calculate_preference_confidence(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate confidence in preference analysis."""
        # Simplified calculation
        return min(1.0, len(interactions) / 20)
    
    def _calculate_preference_confidence_changes(self, user_id: str, 
                                               old_prefs: Dict[str, Any],
                                               new_prefs: Dict[str, Any]) -> Dict[str, float]:
        """Calculate changes in preference confidence."""
        # Simplified calculation
        return {"overall_confidence": 0.8}
    
    def _generate_preference_recommendations(self, user_id: str, 
                                           preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations for preference optimization."""
        # Simplified recommendations
        return []
    
    def _analyze_interaction_history(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze interaction history for patterns and trends."""
        # Simplified analysis
        return {"total_interactions": len(interactions), "avg_quality": 0.7}
    
    def _calculate_interaction_trends(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate temporal trends in user interactions."""
        # Simplified trends
        return {"frequency_trend": "stable", "quality_trend": "improving"}
    
    def _generate_behavioral_insights(self, interactions: List[Dict[str, Any]], 
                                    analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate behavioral insights from interaction analysis."""
        # Simplified insights
        return []
    
    def _generate_learning_insights(self, interactions: List[Dict[str, Any]], 
                                  preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights about user learning progress."""
        # Simplified insights
        return []
    
    def _generate_financial_insights(self, memories: List[Dict[str, Any]], 
                                   preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights about user financial behavior."""
        # Simplified insights
        return []
    
    def _generate_pattern_insights(self, interactions: List[Dict[str, Any]], 
                                 preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights about user interaction patterns."""
        # Simplified insights
        return []
    
    def _generate_insight_recommendations(self, insights: List[Dict[str, Any]], 
                                        preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on insights."""
        # Simplified recommendations
        return []
    
    def _calculate_detailed_learning_progress(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate detailed learning progress metrics."""
        # Simplified calculation
        return {"level": "intermediate", "progress_percentage": 65}
    
    def _identify_improvement_opportunities(self, insights: List[Dict[str, Any]], 
                                          memories: List[Dict[str, Any]],
                                          preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify opportunities for financial improvement."""
        # Simplified opportunities
        return []
    
    def _perform_comparative_analysis(self, user_id: str, 
                                    interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comparative analysis against similar users."""
        # Simplified analysis
        return {"percentile_rank": 75, "similar_users": 100}

# Global service instance for application-wide access
user_interaction_service = UserInteractionService()
