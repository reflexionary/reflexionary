# Model Training

This folder contains scripts for training machine learning models for Tethys Financial Co-Pilot.

## Available Training Scripts

### 1. Anomaly Detector (`train_anomaly_detector.py`)
Basic anomaly detection using Isolation Forest.

**Usage:**
```bash
python train_anomaly_detector.py
```

**Output:** `../../models/anomaly_detector.pkl`

---

### 2. Memory Layer Models (`train_memory_models.py`)
Trains models for Tethys's Memory Layer including:
- Embedding model statistics and optimization
- Vector index training for semantic search
- Memory retrieval model evaluation

**Usage:**
```bash
python train_memory_models.py
```

**Outputs:**
- `../../models/embedding_stats.json` - Embedding model statistics
- `../../models/memory_training_summary.json` - Training summary

**Features:**
- Trains embedding models on financial domain data
- Builds vector indexes for semantic memory retrieval
- Evaluates memory retrieval performance
- Supports user-specific memory training

---

### 3. Quantitative Finance Models (`train_quant_models.py`)
Trains models for Tethys's Mathematical Intelligence Layer including:
- Time series forecasting models (TFT, N-BEATS, LSTM)
- Portfolio optimization models
- Risk assessment models (VaR, anomaly detection)
- Factor exposure models

**Usage:**
```bash
python train_quant_models.py
```

**Outputs:**
- `../../models/timeseries_*.pkl` - Time series forecasting models
- `../../models/portfolio_opt_*.json` - Portfolio optimization results
- `../../models/risk_anomaly_*.pkl` - Risk assessment models
- `../../models/factor_exposure_*.json` - Factor exposure analysis
- `../../models/quant_training_summary.json` - Training summary

**Features:**
- Multi-model time series forecasting
- Portfolio optimization with different risk tolerances
- Risk assessment including VaR and anomaly detection
- Factor exposure analysis for portfolio attribution

---

## Training Pipeline

### Using DVC (Recommended)
Run the complete training pipeline:
```bash
dvc repro train_memory_models
dvc repro train_quant_models
```

### Manual Training
Train models individually:
```bash
# Memory Layer
python train_memory_models.py

# Mathematical Intelligence
python train_quant_models.py

# Basic anomaly detection
python train_anomaly_detector.py
```

---

## Model Dependencies

### Memory Layer Models
- `sentence-transformers` - For embedding generation
- `annoy` - For vector indexing
- `firebase-admin` - For persistent storage

### Quantitative Models
- `scikit-learn` - For anomaly detection and preprocessing
- `pandas`, `numpy` - For data manipulation
- `quantstats` - For financial metrics
- `pypfopt` - For portfolio optimization

---

## Configuration

All models use configuration from:
- `config/app_settings.py` - Application settings
- `config/ml_config.py` - ML-specific configuration

---

## Model Artifacts

Trained models are saved to `../../models/` with the following structure:
```
models/
├── anomaly_detector.pkl
├── embedding_stats.json
├── memory_training_summary.json
├── quant_training_summary.json
├── timeseries_*.pkl
├── portfolio_opt_*.json
├── risk_anomaly_*.pkl
└── factor_exposure_*.json
```

---

## Evaluation

Each training script includes built-in evaluation:
- Memory models: Semantic search accuracy and retrieval performance
- Quantitative models: Forecasting accuracy, portfolio performance metrics
- Anomaly detection: Detection rate and false positive analysis

---

## Troubleshooting

### Common Issues
1. **Missing dependencies**: Install required packages from `requirements.txt`
2. **Data not found**: Ensure data files exist in `../../data/`
3. **Memory errors**: Reduce batch sizes or model complexity
4. **Firebase connection**: Check Firebase credentials in configuration

### Logs
Training logs are written to console and can be redirected to files:
```bash
python train_memory_models.py > memory_training.log 2>&1
python train_quant_models.py > quant_training.log 2>&1
``` 