# Tacit AI Financial Co-Pilot

![Tacit Logo](docs/images/logo.png)

> Your intelligent financial assistant powered by AI

## 🚀 Features

- **AI-Powered Financial Insights**: Get personalized financial advice using Gemini AI
- **Portfolio Analysis**: Advanced analytics for your investment portfolio
- **Goal Tracking**: Set and track financial goals with smart recommendations
- **Anomaly Detection**: Identify unusual spending patterns and potential fraud
- **Multi-Platform**: Access via web UI or CLI

## 🛠️ Prerequisites

- Python 3.9+
- Google Gemini API Key
- Firebase account (for data persistence)
- Node.js 16+ (for development)

## ⚙️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/tacit.git
   cd tacit
   ```

2. **Set up a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Create a `.env` file in the project root with the following variables:

   ```env
   # Google Gemini API
   GEMINI_API_KEY=your_gemini_api_key_here
   
   # Firebase
   FIREBASE_SERVICE_ACCOUNT_KEY_PATH=path/to/your/firebase-key.json
   
   # Application Settings
   LOG_LEVEL=INFO
   ```

## 🚦 Quick Start

### Web UI

```bash
streamlit run ui/streamlit_app.py
```

### CLI Mode

```bash
python main.py
```

## 🏗️ Project Structure

```text
tacit/
├── config/               # Application configuration
│   ├── __init__.py
│   ├── app_settings.py   # Core application settings
│   ├── db_config.py      # Database configuration
│   ├── fi_mcp_config.py  # Fi-MCP simulator settings
│   ├── llm_config.py     # LLM model configuration
│   └── ml_config.py      # ML model configuration
├── core_components/      # Core application components
│   ├── __init__.py
│   ├── embedding_service.py  # Text embedding service
│   └── vector_index.py       # Vector similarity search
├── data/                 # Data storage
│   ├── raw/              # Raw data files
│   └── processed/        # Processed data
├── data_connectors/      # External data integrations
│   ├── __init__.py
│   ├── fi_mcp_connector.py  # Fi-MCP simulator client
│   └── market_data.py    # Market data APIs
├── financial_intelligence/  # Financial analysis
│   ├── __init__.py
│   ├── quant_tools/     # Quantitative analysis tools
│   ├── models/          # ML models
│   ├── evaluation/      # Model evaluation
│   └── serving/         # Model serving
├── memory_management/   # Memory and retrieval
│   ├── __init__.py
│   ├── connectors/      # Data connectors
│   ├── ingestion/       # Data ingestion pipelines
│   ├── retrieval/       # Retrieval pipelines
│   └── llm_agents/      # LLM agent implementations
├── services/            # Business logic services
│   ├── __init__.py
│   ├── portfolio_service.py
│   └── recommendation_service.py
├── ui/                  # User interface
│   ├── __init__.py
│   ├── streamlit_app.py # Streamlit web interface
│   └── components/      # Reusable UI components
├── tests/               # Test suite
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   └── e2e/             # End-to-end tests
├── deploy/              # Deployment configurations
│   ├── docker/          # Dockerfiles
│   └── kubernetes/      # K8s manifests
├── infra/               # Infrastructure as Code
│   └── terraform/       # Terraform configurations
├── ml_ops/              # ML Operations
│   ├── training/        # Model training pipelines
│   ├── serving/         # Model serving
│   └── data_versioning/ # Data version control
├── observability/       # Monitoring and logging
│   ├── logging_config.py
│   ├── metrics.py
│   └── tracing.py
├── scripts/             # Utility scripts
├── main.py              # Application entry point
├── requirements.txt     # Python dependencies
└── README.md            # This file

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Gemini AI](https://ai.google.dev/) for the powerful language model
- [Streamlit](https://streamlit.io/) for the awesome web framework
- [Firebase](https://firebase.google.com/) for data persistence
- [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) for portfolio optimization
