# Data Versioning with DVC

This folder contains scripts and configuration for data versioning using DVC.

## Usage

1. Install DVC: `pip install dvc`
2. Initialize DVC in your repo (if not already): `dvc init`
3. Run the data preparation stage:
   ```
dvc repro prepare_data
```
4. Run the model training stage:
   ```
dvc repro train
```
5. Track data and model artifacts with DVC:
   ```
dvc add ../../data/processed_data.csv
   dvc add ../../models/anomaly_detector.pkl
```
6. Push to remote storage (optional):
   ```
dvc push
```

See [DVC documentation](https://dvc.org/doc) for more details. 