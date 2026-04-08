import json
import re

with open("Transformer_Monitoring_System.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
        source = "".join(cell['source'])
        
        if "NEW: MASSIVE SYNTHETIC DGA EXPANSION" in source:
            print("Found block, rebuilding it perfectly...")
            
            pattern = re.compile(r'# ---------------------------------------------------------\n# NEW: MASSIVE SYNTHETIC DGA EXPANSION.*?y_dga_class_massive = df_dga_massive\[\'health_category\'\]', re.DOTALL | re.IGNORECASE)
            
            new_block = """# ---------------------------------------------------------
# NEW: MASSIVE SYNTHETIC DGA EXPANSION (GAUSSIAN SAMPLING)
# Generate a robust, 4,000-row synthetic dataset to solve
# extreme class imbalances and incredibly small test sizes.
# ---------------------------------------------------------
print("\\n\\u2697\\ufe0f Synthetically expanding dataset using Gaussian Distribution profiles...")
import numpy as np
import pandas as pd

synthetic_dfs = []
target_size_per_class = 1000

y_series = pd.Series(y_dga_class.flatten() if isinstance(y_dga_class, np.ndarray) else y_dga_class)
X_df = pd.DataFrame(X_dga) if isinstance(X_dga, np.ndarray) else X_dga

for cat in np.unique(y_series):
    class_mask = (y_series == cat)
    class_features = X_df.loc[class_mask]
    
    # Calculate Mean & Standard Deviation for each feature
    means = class_features.mean().values
    stds = class_features.std().values
    
    # Add small epsilon to stds to prevent 0 variance failure
    stds = np.where(stds == 0, 1e-5, stds)
    
    # Randomly sample from normal distribution
    np.random.seed(42)
    synthetic_samples = np.random.normal(loc=means, scale=stds, size=(target_size_per_class, len(means)))
    
    # Clip absolute physical constraints (no negative gas readings)
    synthetic_samples = np.clip(synthetic_samples, 0.0, None)
    
    # Form a new DataFrame
    df_synthetic = pd.DataFrame(synthetic_samples, columns=X_df.columns)
    df_synthetic['health_category'] = cat
    synthetic_dfs.append(df_synthetic)

# Combine Real & Synthetic!
df_dga_massive = pd.concat([pd.concat([X_df.reset_index(drop=True), y_series.rename('health_category').reset_index(drop=True)], axis=1)] + synthetic_dfs, ignore_index=True)

X_dga_massive = df_dga_massive[X_df.columns]
y_dga_class_massive = df_dga_massive['health_category']"""
            
            match = pattern.search(source)
            if match:
                new_source = source.replace(match.group(0), new_block)
                lines = [line + '\n' for line in new_source.split('\n')]
                lines[-1] = lines[-1].rstrip('\n')
                if not lines[-1]:
                    lines.pop()
                cell['source'] = lines

with open("Transformer_Monitoring_System.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")

print("Notebook cleansed and cleanly patched.")
