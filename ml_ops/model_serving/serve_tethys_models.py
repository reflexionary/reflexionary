"""
Tethys Model Serving API

This FastAPI application serves all Tethys models including:
- Memory Layer models (embedding, vector search, memory retrieval)
- Mathematical Intelligence models (time series, portfolio optimization, risk)
- Integrated models (combining memory and mathematical intelligence)
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
import joblib
import json
import os
import logging
from datetime import datetime

# Import Tethys components
from core_components.embedding_service import get_embedding
from memory_management.memory_manager import ingest_user_memory, retrieve_contextual_memories
from financial_intelligence.financial_quant_tools import (
    get_portfolio_performance_metrics,
    get_optimal_portfolio_allocation,
    get_value_at_risk,
    get_asset_price_forecast
)
from financial_intelligence.ml_quant_models.time_series_forecasting import TimeSeriesForecaster

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Tethys Financial Co-Pilot API",
    description="API for Tethys's Memory Layer and Mathematical Intelligence Layer",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class MemoryRequest(BaseModel):
    user_id: str
    text: str
    memory_type: str = Field(..., description="Type of memory: goal, preference, transaction, advice, anomaly")
    metadata: Optional[Dict[str, Any]] = None

class MemoryQueryRequest(BaseModel):
    user_id: str
    query: str
    num_results: int = Field(default=3, ge=1, le=10)

class TimeSeriesRequest(BaseModel):
    ticker: str
    lookback: int = Field(default=30, ge=10, le=100)
    horizon: int = Field(default=5, ge=1, le=30)
    model_type: str = Field(default="tft", description="Model type: tft, nbeats, lstm")

class PortfolioRequest(BaseModel):
    user_id: str
    risk_tolerance: str = Field(default="medium", description="Risk tolerance: low, medium, high")
    investment_amount: Optional[float] = None

class RiskRequest(BaseModel):
    user_id: str
    confidence_level: float = Field(default=0.95, ge=0.8, le=0.99)
    time_horizon_days: int = Field(default=1, ge=1, le=30)

class IntegratedRequest(BaseModel):
    user_id: str
    query: str
    include_memory: bool = True
    include_quant: bool = True
    num_memory_results: int = Field(default=3, ge=1, le=10)

# Model loading utilities
def load_model(model_path: str):
    """Load a trained model from disk."""
    if not os.path.exists(model_path):
        return None
    try:
        return joblib.load(model_path)
    except Exception as e:
        logger.error(f"Error loading model {model_path}: {e}")
        return None

# Memory Layer Endpoints
@app.post("/memory/ingest")
async def ingest_memory(request: MemoryRequest):
    """Ingest a new memory into the user's memory store."""
    try:
        ingest_user_memory(
            user_id=request.user_id,
            text=request.text,
            memory_type=request.memory_type,
            metadata=request.metadata
        )
        return {
            "status": "success",
            "message": f"Memory ingested successfully for user {request.user_id}",
            "memory_type": request.memory_type,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error ingesting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/query")
async def query_memories(request: MemoryQueryRequest):
    """Query user memories using semantic search."""
    try:
        memories = retrieve_contextual_memories(
            user_id=request.user_id,
            query_text=request.query,
            num_results=request.num_results
        )
        return {
            "status": "success",
            "user_id": request.user_id,
            "query": request.query,
            "memories": memories,
            "count": len(memories),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error querying memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/embedding")
async def get_text_embedding(text: str):
    """Get embedding for a text string."""
    try:
        embedding = get_embedding(text)
        return {
            "status": "success",
            "text": text,
            "embedding_dim": len(embedding),
            "embedding": embedding[:10] + ["..."] if len(embedding) > 10 else embedding,  # Truncate for response
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mathematical Intelligence Endpoints
@app.post("/quant/portfolio/performance")
async def get_portfolio_performance(request: PortfolioRequest):
    """Get comprehensive portfolio performance metrics."""
    try:
        metrics = get_portfolio_performance_metrics(request.user_id)
        return {
            "status": "success",
            "user_id": request.user_id,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting portfolio performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quant/portfolio/optimize")
async def optimize_portfolio(request: PortfolioRequest):
    """Optimize portfolio allocation."""
    try:
        result = get_optimal_portfolio_allocation(
            user_id=request.user_id,
            risk_tolerance=request.risk_tolerance,
            total_investment_value=request.investment_amount
        )
        return {
            "status": "success",
            "user_id": request.user_id,
            "optimization": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error optimizing portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quant/risk/var")
async def calculate_var(request: RiskRequest):
    """Calculate Value at Risk."""
    try:
        var_result = get_value_at_risk(
            user_id=request.user_id,
            confidence_level=request.confidence_level,
            time_horizon_days=request.time_horizon_days
        )
        return {
            "status": "success",
            "user_id": request.user_id,
            "var_result": var_result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error calculating VaR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quant/forecast/timeseries")
async def forecast_timeseries(request: TimeSeriesRequest):
    """Generate time series forecasts."""
    try:
        # Load trained model if available
        model_path = f"../../models/timeseries_{request.ticker}_{request.model_type}.pkl"
        model = load_model(model_path)
        
        if model is None:
            # Use conceptual model if trained model not available
            model = TimeSeriesForecaster(
                model_type=request.model_type,
                lookback=request.lookback,
                horizon=request.horizon
            )
        
        # Generate forecast (this would use real data in production)
        forecast_result = model.predict(n_steps=request.horizon)
        
        return {
            "status": "success",
            "ticker": request.ticker,
            "model_type": request.model_type,
            "forecast": forecast_result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Integrated Endpoints (Combining Memory and Mathematical Intelligence)
@app.post("/integrated/query")
async def integrated_query(request: IntegratedRequest):
    """Integrated query combining memory and mathematical intelligence."""
    try:
        result = {
            "user_id": request.user_id,
            "query": request.query,
            "memory_context": None,
            "quantitative_insights": None,
            "integrated_response": None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Get memory context if requested
        if request.include_memory:
            memories = retrieve_contextual_memories(
                user_id=request.user_id,
                query_text=request.query,
                num_results=request.num_memory_results
            )
            result["memory_context"] = {
                "memories": memories,
                "count": len(memories)
            }
        
        # Get quantitative insights if requested
        if request.include_quant:
            try:
                # Get portfolio performance
                portfolio_metrics = get_portfolio_performance_metrics(request.user_id)
                
                # Get risk metrics
                var_result = get_value_at_risk(request.user_id)
                
                result["quantitative_insights"] = {
                    "portfolio_performance": portfolio_metrics,
                    "risk_metrics": var_result
                }
            except Exception as e:
                logger.warning(f"Error getting quantitative insights: {e}")
                result["quantitative_insights"] = {"error": str(e)}
        
        # Generate integrated response
        result["integrated_response"] = {
            "summary": f"Query processed for user {request.user_id}",
            "memory_available": result["memory_context"] is not None,
            "quant_available": result["quantitative_insights"] is not None
        }
        
        return {
            "status": "success",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error in integrated query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/integrated/financial_advice")
async def get_financial_advice(request: IntegratedRequest):
    """Get personalized financial advice combining memory and quantitative analysis."""
    try:
        # Get user's financial context from memory
        memories = retrieve_contextual_memories(
            user_id=request.user_id,
            query_text=request.query,
            num_results=5
        )
        
        # Get quantitative analysis
        portfolio_metrics = get_portfolio_performance_metrics(request.user_id)
        var_result = get_value_at_risk(request.user_id)
        
        # Generate personalized advice
        advice = {
            "user_context": {
                "memories": memories,
                "portfolio_metrics": portfolio_metrics,
                "risk_profile": var_result
            },
            "personalized_advice": {
                "summary": f"Personalized advice for {request.user_id} based on {len(memories)} memories and quantitative analysis",
                "risk_level": "moderate",  # Would be calculated based on user profile
                "recommendations": [
                    "Consider rebalancing portfolio based on current market conditions",
                    "Review emergency fund adequacy",
                    "Monitor risk exposure regularly"
                ]
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "status": "success",
            "advice": advice
        }
        
    except Exception as e:
        logger.error(f"Error generating financial advice: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Tethys Financial Co-Pilot API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

# Model status endpoint
@app.get("/models/status")
async def model_status():
    """Check status of trained models."""
    model_dir = "../../models"
    models = {}
    
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            if file.endswith(('.pkl', '.json')):
                file_path = os.path.join(model_dir, file)
                models[file] = {
                    "exists": True,
                    "size_bytes": os.path.getsize(file_path),
                    "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                }
    
    return {
        "status": "success",
        "models": models,
        "total_models": len(models),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 