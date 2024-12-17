# src/load_data.py

import pandas as pd

def load_data(filepath):
    """Loads the dataset from a CSV file."""
    return pd.read_csv(filepath)
