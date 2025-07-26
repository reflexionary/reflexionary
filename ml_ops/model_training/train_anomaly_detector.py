import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

# Load processed data (for demo, generate mock data if not found)
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed_data.csv'))
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    # Generate mock data
    np.random.seed(42)
    df = pd.DataFrame({'feature1': np.random.normal(0, 1, 1000), 'feature2': np.random.normal(5, 2, 1000)})

# Train Isolation Forest for anomaly detection
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(df)

# Save the trained model
os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models')), exist_ok=True)
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/anomaly_detector.pkl'))
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
