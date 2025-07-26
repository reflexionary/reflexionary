import pandas as pd
import numpy as np
import os

# Generate mock data for demonstration
np.random.seed(42)
df = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.normal(5, 2, 1000)
})

# Save to data/processed_data.csv
os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data')), exist_ok=True)
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed_data.csv'))
df.to_csv(data_path, index=False)
print(f"Processed data saved to {data_path}") 