# 🌊 Tethys - Your AI Financial Co-Pilot

> **Powered by Memory Layer + Mathematical Intelligence**

Tethys is an intelligent financial co-pilot that combines the power of AI memory with advanced mathematical intelligence to provide personalized financial guidance, portfolio management, and goal planning.

## 🚀 Features

### 🧠 Memory Layer - Your Financial Elephant
- **Personalized Memory**: Remembers your financial goals, preferences, and history
- **Semantic Search**: Lightning-fast retrieval of relevant financial context
- **Continuous Learning**: Builds understanding of your financial personality over time
- **Privacy-First**: Your sensitive data stays private while enabling personalization

### 🧮 Mathematical Intelligence - Your Personal Quant Guru
- **Portfolio Optimization**: Advanced asset allocation using modern portfolio theory
- **Risk Assessment**: Comprehensive risk metrics including VaR, CVaR, and stress testing
- **Time Series Forecasting**: Multi-model predictions using TFT, N-BEATS, and LSTM
- **Anomaly Detection**: Intelligent detection of unusual financial patterns
- **Performance Analytics**: Deep portfolio performance analysis with 50+ metrics

### 🎯 Goal Planning & Tracking
- **Smart Goal Setting**: AI-powered goal recommendations based on your profile
- **Progress Monitoring**: Real-time tracking with intelligent insights
- **Timeline Optimization**: Dynamic adjustment of savings strategies
- **Success Probability**: Predictive analysis of goal achievement likelihood

### 🔍 Anomaly Detection & Alerts
- **Transaction Monitoring**: Real-time detection of unusual spending patterns
- **Portfolio Alerts**: Intelligent monitoring of portfolio performance anomalies
- **Risk Alerts**: Proactive identification of potential financial risks
- **Customizable Preferences**: Personalized alert thresholds and frequencies

### 📊 Comprehensive Analytics
- **Portfolio Performance**: 50+ financial metrics and ratios
- **Risk Analysis**: Multi-dimensional risk assessment
- **Goal Progress**: Visual tracking and predictive insights
- **Behavioral Analysis**: Understanding of financial patterns and preferences

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Tethys Core Layer                        │
├─────────────────────────────────────────────────────────────┤
│  🌊 Tethys Core Integration                                │
│  ├── User Interaction Service                              │
│  ├── Portfolio Management Service                          │
│  ├── Goal Planning Service                                 │
│  └── Anomaly Detection Service                             │
├─────────────────────────────────────────────────────────────┤
│                    Memory Layer                             │
│  🧠 Embedding Service (sentence-transformers)              │
│  🧠 Vector Index (Annoy)                                   │
│  🧠 Persistent Storage (Firebase Firestore)                │
│  🧠 Memory Manager                                         │
├─────────────────────────────────────────────────────────────┤
│                Mathematical Intelligence                    │
│  🧮 Portfolio Optimization (PyPortfolioOpt)               │
│  🧮 Risk Analysis (VaR, CVaR, Stress Testing)             │
│  🧮 Time Series Forecasting (TFT, N-BEATS, LSTM)          │
│  🧮 Performance Analytics (quantstats)                     │
│  🧮 Anomaly Detection (Isolation Forest)                   │
├─────────────────────────────────────────────────────────────┤
│                   Data Connectors                          │
│  📡 Fi-MCP Connector (Financial Data)                     │
│  📡 Market Data Connector                                  │
│  📡 External API Connectors                                │
├─────────────────────────────────────────────────────────────┤
│                   Observability                            │
│  📊 Structured Logging (JSON + Console)                   │
│  📊 Metrics Collection (Prometheus Format)                 │
│  📊 Performance Monitoring                                 │
│  📊 Health Checks                                          │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- Firebase project with Firestore
- Google Gemini API key
- Fi-MCP simulator (for financial data)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd tethys
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   export FIREBASE_SERVICE_ACCOUNT_KEY_PATH="/path/to/firebase-key.json"
   export FI_MCP_SIMULATOR_BASE_URL="http://localhost:3000"
   ```

4. **Initialize Tethys**
   ```bash
   python main.py
   ```

## 🚀 Usage

### CLI Mode (Interactive)
```bash
python main.py --mode cli
```

**Available Commands:**
- `query <user_id> <query>` - Process a user query
- `dashboard <user_id>` - Get user dashboard
- `sync <user_id>` - Sync user data
- `status` - Show system status
- `help` - Show help
- `exit` - Exit Tethys

### API Mode
```bash
# Start the API server
uvicorn ml_ops.model_serving.serve_tethys_models:app --reload --host 0.0.0.0 --port 8000
```

**API Endpoints:**
- `POST /memory/ingest` - Ingest user memory
- `POST /memory/query` - Query user memories
- `POST /quant/portfolio/optimize` - Optimize portfolio
- `POST /quant/risk/var` - Calculate Value at Risk
- `POST /integrated/financial_advice` - Get integrated financial advice

### Testing
```bash
# Test with specific user and query
python main.py --mode cli --user-id test_user --query "How should I invest my money?"
```

## 📁 Project Structure

```
tethys/
├── core_components/           # Core infrastructure
│   ├── embedding_service.py   # Text embedding generation
│   ├── vector_index.py        # Vector search and indexing
│   └── persistent_storage.py  # Firebase Firestore integration
├── memory_management/         # Memory Layer
│   ├── memory_manager.py      # Memory orchestration
│   └── gemini_connector.py    # AI model integration
├── financial_intelligence/    # Mathematical Intelligence
│   ├── ml_quant_models/       # ML models (TFT, N-BEATS, LSTM)
│   ├── portfolio_opt/         # Portfolio optimization
│   ├── risk_analysis/         # Risk assessment
│   └── core_metrics/          # Performance analytics
├── services/                  # Business services
│   ├── anomaly_detection_service.py
│   ├── portfolio_management_service.py
│   ├── goal_planning_service.py
│   └── user_interaction_service.py
├── data_connectors/           # Data integration
│   └── fi_mcp_connector.py    # Financial data connector
├── observability/             # Monitoring and logging
│   ├── logging_config.py      # Structured logging
│   └── metrics_exporter.py    # Metrics collection
├── ml_ops/                    # MLOps pipeline
│   ├── model_training/        # Model training scripts
│   └── model_serving/         # Model serving APIs
├── config/                    # Configuration
│   └── app_settings.py        # Application settings
├── tethys_core.py            # Main integration layer
└── main.py                   # Application entry point
```

## 🔧 Configuration

### Environment Variables
```bash
# Required
GEMINI_API_KEY=your-gemini-api-key
FIREBASE_SERVICE_ACCOUNT_KEY_PATH=/path/to/firebase-key.json

# Optional
FI_MCP_SIMULATOR_BASE_URL=http://localhost:3000
LOG_LEVEL=INFO
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
ANOMALY_DETECTION_THRESHOLD=0.8
DEFAULT_RISK_FREE_RATE=0.05
```

### Feature Flags
```bash
FEATURE_MEMORY_LAYER=true
FEATURE_MATHEMATICAL_INTELLIGENCE=true
FEATURE_ANOMALY_DETECTION=true
FEATURE_GOAL_PLANNING=true
FEATURE_PORTFOLIO_OPTIMIZATION=true
```

## 📊 MLOps Pipeline

### Model Training
```bash
# Train memory models
python ml_ops/model_training/train_memory_models.py

# Train quantitative models
python ml_ops/model_training/train_quant_models.py

# Train anomaly detection
python ml_ops/model_training/train_anomaly_detector.py
```

### Model Serving
```bash
# Serve all models
uvicorn ml_ops.model_serving.serve_tethys_models:app --reload

# Serve specific model
uvicorn ml_ops.model_serving.serve_anomaly_detector:app --reload
```

## 🔍 Monitoring & Observability

### Logging
- **Structured Logging**: JSON format for machine readability
- **Component-Specific**: Separate log files for each component
- **Performance Tracking**: Operation timing and metrics
- **Error Tracking**: Comprehensive error logging with context

### Metrics
- **Performance Metrics**: Response times, throughput, error rates
- **Business Metrics**: User activity, memory operations, financial calculations
- **System Metrics**: Health checks, component status, resource usage
- **Export Formats**: JSON and Prometheus formats

### Health Checks
```bash
# Check system status
curl http://localhost:8000/health

# Check model status
curl http://localhost:8000/models/status
```

## 🛡️ Security & Privacy

### Data Privacy
- **PII Masking**: Automatic masking of sensitive information
- **Data Encryption**: End-to-end encryption of financial data
- **Audit Logging**: Comprehensive audit trails
- **Access Control**: Role-based access control

### Security Features
- **API Rate Limiting**: Protection against abuse
- **Input Validation**: Comprehensive input sanitization
- **Error Handling**: Secure error messages
- **Circuit Breakers**: Protection against cascading failures

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/unit/
```

### Integration Tests
```bash
python -m pytest tests/integration/
```

### Performance Tests
```bash
python -m pytest tests/performance/
```

## 📈 Performance

### Benchmarks
- **Memory Retrieval**: < 100ms for semantic search
- **Portfolio Optimization**: < 5s for complex portfolios
- **Anomaly Detection**: < 1s for real-time detection
- **API Response**: < 2s for complex queries

### Scalability
- **Memory Layer**: Supports 1M+ memories per user
- **Vector Search**: Sub-second retrieval from 10M+ vectors
- **API**: Handles 1000+ concurrent requests
- **Storage**: Efficient compression and indexing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini**: For AI model capabilities
- **Firebase**: For scalable data storage
- **PyPortfolioOpt**: For portfolio optimization
- **quantstats**: For financial analytics
- **sentence-transformers**: For semantic embeddings
- **Annoy**: For vector search

## 📞 Support

- **Documentation**: [Wiki](link-to-wiki)
- **Issues**: [GitHub Issues](link-to-issues)
- **Discussions**: [GitHub Discussions](link-to-discussions)
- **Email**: support@tethys.ai

---

**🌊 Tethys - Where Memory Meets Mathematics for Smarter Finance**
