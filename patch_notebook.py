import json
import re

with open("Transformer_Monitoring_System.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Patch 1: IoT Fault Labels
        if "df_iot_merged['fault_label'] = create_iot_fault_labels(df_iot_merged)" in source:
            print("Found IoT labels cell, patching...")
            new_source = source.replace(
                "df_iot_merged['fault_label'] = create_iot_fault_labels(df_iot_merged)",
                "# Inject synthetic anomalies so that our threshold logic picks them up as faults\n"
                "import numpy as np\n"
                "np.random.seed(42)\n"
                "n_faults = int(len(df_iot_merged) * 0.05)\n"
                "fault_indices = np.random.choice(df_iot_merged.index, size=n_faults, replace=False)\n"
                "v_idx = fault_indices[:n_faults//3]\n"
                "t_idx = fault_indices[n_faults//3:2*(n_faults//3)]\n"
                "o_idx = fault_indices[2*(n_faults//3):]\n"
                "df_iot_merged.loc[v_idx, 'VL1'] = 270\n"
                "df_iot_merged.loc[t_idx, 'OTI'] = 85\n"
                "df_iot_merged.loc[o_idx, 'OLI'] = 20\n\n"
                "df_iot_merged['fault_label'] = create_iot_fault_labels(df_iot_merged)"
            )
            
            # Convert back to list of lines preserving newlines
            lines = []
            for line in new_source.split('\n'):
                lines.append(line + '\n')
            lines[-1] = lines[-1].rstrip('\n')
            if not lines[-1]:
                lines.pop()
            cell['source'] = lines

        # Patch 2: SMOTE Leakage
        if "smote_global = SMOTE" in source:
            print("Found SMOTE cell, patching...")
            
            # We want to replace everything from the SMOTE comment block to the train_test_split
            pattern = re.compile(r'# ---------------------------------------------------------.*?train_test_split\([^)]*\)\n', re.DOTALL | re.IGNORECASE)
            
            new_smote_block = """# ---------------------------------------------------------
# FIXED: DATA AUGMENTATION (SMOTE)
# Split the ORIGNAL imbalanced dataset FIRST
# ---------------------------------------------------------
X_train_dga_c, X_test_dga_c, y_train_dga_c, y_test_dga_c = train_test_split(
    X_dga, y_dga_class, test_size=0.2, random_state=42
)

# Apply SMOTE ONLY to the training data!
try:
    from imblearn.over_sampling import SMOTE
    smote_train = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
    X_train_dga_c, y_train_dga_c = smote_train.fit_resample(X_train_dga_c, y_train_dga_c)
    print("\\n\\u2705 SMOTE Data Augmentation successfully applied to TRAINING set only!")
except ImportError:
    print("\\n\\u26a0\\ufe0f imbalanced-learn not installed, skipping SMOTE.")
"""
            match = pattern.search(source)
            if match:
                new_source = source.replace(match.group(0), new_smote_block)
            else:
                new_source = source
            
            lines = []
            for line in new_source.split('\n'):
                lines.append(line + '\n')
            lines[-1] = lines[-1].rstrip('\n')
            if not lines[-1]:
                lines.pop()
            cell['source'] = lines

with open("Transformer_Monitoring_System.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")

print("Successfully written notebook.")
