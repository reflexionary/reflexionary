# Model Serving

This folder contains scripts for serving trained machine learning models via an API.

## Anomaly Detector API

To start the FastAPI server:

1. Ensure the model is trained and available at `../../models/anomaly_detector.pkl`.
2. Install FastAPI and Uvicorn:
   ```
   pip install fastapi uvicorn
   ```
3. Start the server:
   ```
   uvicorn serve_anomaly_detector:app --reload
   ```
4. Send POST requests to `/predict` with JSON body:
   ```json
   {
     "feature1": 0.5,
     "feature2": 4.2
   }
   ``` 