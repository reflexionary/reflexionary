# Model Serving

This folder contains scripts for serving trained machine learning models via APIs for Tethys Financial Co-Pilot.

## Available Serving Scripts

### 1. Basic Anomaly Detector API (`serve_anomaly_detector.py`)
Simple FastAPI server for the basic anomaly detection model.

**Usage:**
```bash
uvicorn serve_anomaly_detector:app --reload
```

**Endpoint:** `POST /predict`
```json
{
  "feature1": 0.5,
  "feature2": 4.2
}
```

---

### 2. Comprehensive Tethys API (`serve_tethys_models.py`)
Full-featured FastAPI server serving all Tethys models including Memory Layer and Mathematical Intelligence Layer.

**Usage:**
```bash
uvicorn serve_tethys_models:app --reload --host 0.0.0.0 --port 8000
```

**Features:**
- Memory Layer endpoints (embedding, vector search, memory retrieval)
- Mathematical Intelligence endpoints (portfolio optimization, risk assessment, time series forecasting)
- Integrated endpoints (combining memory and quantitative analysis)
- Health checks and model status monitoring

---

## API Endpoints

### Memory Layer Endpoints

#### 1. Ingest Memory
**POST** `/memory/ingest`
```json
{
  "user_id": "user_123",
  "text": "My goal is to retire by age 60",
  "memory_type": "goal",
  "metadata": {"priority": "high"}
}
```

#### 2. Query Memories
**POST** `/memory/query`
```json
{
  "user_id": "user_123",
  "query": "What are my retirement goals?",
  "num_results": 3
}
```

#### 3. Get Text Embedding
**POST** `/memory/embedding`
```json
{
  "text": "I want to invest in low-risk options"
}
```

### Mathematical Intelligence Endpoints

#### 1. Portfolio Performance
**POST** `/quant/portfolio/performance`
```json
{
  "user_id": "user_123",
  "risk_tolerance": "medium",
  "investment_amount": 100000
}
```

#### 2. Portfolio Optimization
**POST** `/quant/portfolio/optimize`
```json
{
  "user_id": "user_123",
  "risk_tolerance": "medium",
  "investment_amount": 100000
}
```

#### 3. Value at Risk
**POST** `/quant/risk/var`
```json
{
  "user_id": "user_123",
  "confidence_level": 0.95,
  "time_horizon_days": 1
}
```

#### 4. Time Series Forecasting
**POST** `/quant/forecast/timeseries`
```json
{
  "ticker": "RELIANCE.NS",
  "lookback": 30,
  "horizon": 5,
  "model_type": "tft"
}
```

### Integrated Endpoints

#### 1. Integrated Query
**POST** `/integrated/query`
```json
{
  "user_id": "user_123",
  "query": "How should I adjust my portfolio?",
  "include_memory": true,
  "include_quant": true,
  "num_memory_results": 3
}
```

#### 2. Financial Advice
**POST** `/integrated/financial_advice`
```json
{
  "user_id": "user_123",
  "query": "What should I do about my investments?",
  "include_memory": true,
  "include_quant": true,
  "num_memory_results": 5
}
```

### Utility Endpoints

#### 1. Health Check
**GET** `/health`
```json
{
  "status": "healthy",
  "service": "Tethys Financial Co-Pilot API",
  "version": "1.0.0",
  "timestamp": "2024-01-01T12:00:00"
}
```

#### 2. Model Status
**GET** `/models/status`
```json
{
  "status": "success",
  "models": {
    "timeseries_RELIANCE.NS_tft.pkl": {
      "exists": true,
      "size_bytes": 1024000,
      "modified": "2024-01-01T12:00:00"
    }
  },
  "total_models": 1,
  "timestamp": "2024-01-01T12:00:00"
}
```

---

## API Documentation

### Interactive Documentation
Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### OpenAPI Specification
The API follows OpenAPI 3.0 specification and is automatically generated from the FastAPI code.

---

## Configuration

### Environment Variables
Set these environment variables for production deployment:
```bash
export FIREBASE_SERVICE_ACCOUNT_KEY_PATH="/path/to/firebase-key.json"
export GEMINI_API_KEY="your-gemini-api-key"
export EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2"
```

### Model Paths
Models are loaded from `../../models/` directory. Ensure trained models exist before starting the server.

---

## Deployment

### Development
```bash
uvicorn serve_tethys_models:app --reload --host 0.0.0.0 --port 8000
```

### Production
```bash
# Using Gunicorn
gunicorn serve_tethys_models:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Using Docker
docker build -t tethys-api .
docker run -p 8000:8000 tethys-api
```

### Docker Example
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "ml_ops.model_serving.serve_tethys_models:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Testing

### Health Check
```bash
curl http://localhost:8000/health
```

### Memory Ingestion
```bash
curl -X POST "http://localhost:8000/memory/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "text": "I want to save for retirement",
    "memory_type": "goal"
  }'
```

### Portfolio Analysis
```bash
curl -X POST "http://localhost:8000/quant/portfolio/performance" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "risk_tolerance": "medium"
  }'
```

### Integrated Query
```bash
curl -X POST "http://localhost:8000/integrated/financial_advice" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "query": "How should I invest my money?",
    "include_memory": true,
    "include_quant": true
  }'
```

---

## Monitoring

### Logs
The API logs all requests and errors. Check logs for:
- Request/response times
- Error rates
- Model loading issues
- Memory usage

### Metrics
Consider adding Prometheus metrics for:
- Request count by endpoint
- Response times
- Error rates
- Model inference times

---

## Security

### Authentication
For production, add authentication middleware:
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your token verification logic
    pass
```

### Rate Limiting
Add rate limiting to prevent abuse:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```

---

## Troubleshooting

### Common Issues
1. **Model not found**: Ensure models are trained and saved to `../../models/`
2. **Firebase connection**: Check Firebase credentials and network connectivity
3. **Memory errors**: Increase server memory or reduce batch sizes
4. **Port conflicts**: Change port with `--port 8001`

### Debug Mode
Run with debug logging:
```bash
uvicorn serve_tethys_models:app --reload --log-level debug
``` 