import os
import pandas as pd
import numpy as np

# Root directory
root_dir = r"C:\\Users\\alich\\Downloads\\hsieh_lab\\real-time predictor of network bursts\\Output"

# Parameters
bin_size = 0.01  # 10 ms bins

all_features = {}

# Loop through each experiment folder
for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)

    if os.path.isdir(folder_path):
        print(f"\nProcessing folder: {folder}")

        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                file_path = os.path.join(folder_path, file)

                print(f"  -> Loading {file}")

                # Load data
                df = pd.read_csv(file_path)

                # --- 1. Create time bins ---
                df["time_bin"] = (df["spiketime"] // bin_size).astype(int)

                # --- 2. Convert to spike count matrix ---
                features = df.pivot_table(
                    index="time_bin",
                    columns="electrodes",
                    aggfunc="size",
                    fill_value=0
                )

                features = features.sort_index(axis=1)

                # Store using unique key
                key = f"{folder}_{file}"
                all_features[key] = features

                print(features.head())

print("\nFinished processing all folders.")