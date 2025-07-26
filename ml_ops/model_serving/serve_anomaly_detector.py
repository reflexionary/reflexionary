from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

class Features(BaseModel):
    feature1: float
    feature2: float

# Load the trained model
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/anomaly_detector.pkl'))
if not os.path.exists(model_path):
    raise RuntimeError(f"Model file not found at {model_path}. Please train the model first.")
model = joblib.load(model_path)

@app.post("/predict")
def predict(features: Features):
    X = np.array([[features.feature1, features.feature2]])
    try:
        pred = model.predict(X)
        is_anomaly = int(pred[0] == -1)
        return {"anomaly": bool(is_anomaly)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 