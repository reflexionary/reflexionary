# Model Training

This folder contains scripts for training machine learning models.

## Anomaly Detector

To train the anomaly detector model:

1. Ensure you have prepared the data (see ../data_versioning/prepare_data.py).
2. Run the training script:
   ```
   python train_anomaly_detector.py
   ```
3. The trained model will be saved to `../../models/anomaly_detector.pkl`. 