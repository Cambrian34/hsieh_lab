import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("your_file.csv")

# --- 1. Create time bins ---
bin_size = 0.01  # 10 ms bins (adjust as needed)
df["time_bin"] = (df["spiketime"] // bin_size).astype(int)

# --- 2. Convert to spike count matrix ---
features = df.pivot_table(
    index="time_bin",
    columns="electrodes",
    aggfunc="size",
    fill_value=0
)

# Optional: sort columns
features = features.sort_index(axis=1)

print(features.head())