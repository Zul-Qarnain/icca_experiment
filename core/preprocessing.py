import pandas as pd
import numpy as np

def serialize_row(row, target_col):
    """Converts a pandas row into a natural language string."""
    items = [f"{str(col).strip()}: {val}" for col, val in row.items() if col != target_col]
    return ", ".join(items)

def inject_missing_data(df, column_name, missing_fraction=0.30, seed=42):
    """Randomly masks a specific column with NaN to simulate data degradation."""
    if column_name not in df.columns:
        return df # Safely return if column doesn't exist
    
    perturbed_df = df.copy()
    
    # FIX: Convert the column to 'object' dtype so it can hold both numbers and text
    perturbed_df[column_name] = perturbed_df[column_name].astype(object)
    
    np.random.seed(seed)
    mask = np.random.rand(len(perturbed_df)) < missing_fraction
    perturbed_df.loc[mask, column_name] = "NaN (Missing Value)"
    
    return perturbed_df
