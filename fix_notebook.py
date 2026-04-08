import json

with open("Transformer_Monitoring_System.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        if "NEW: MASSIVE SYNTHETIC DGA EXPANSION" in source:
            print("Found block, patching numpy bug...")
            new_source = source.replace(
                "for cat in y_dga_class.unique():",
                "y_series = pd.Series(y_dga_class.flatten() if isinstance(y_dga_class, np.ndarray) else y_dga_class)\n    "
                "X_df = pd.DataFrame(X_dga) if isinstance(X_dga, np.ndarray) else X_dga\n    "
                "for cat in np.unique(y_series):"
            )
            new_source = new_source.replace("class_mask = (y_dga_class == cat)", "class_mask = (y_series == cat)")
            new_source = new_source.replace("class_features = X_dga[class_mask]", "class_features = X_df.loc[class_mask]")
            new_source = new_source.replace("columns=X_dga.columns", "columns=X_df.columns")
            new_source = new_source.replace("pd.concat([X_dga, y_dga_class]", "pd.concat([X_df.reset_index(drop=True), y_series.rename('health_category').reset_index(drop=True)]")
            new_source = new_source.replace("len(X_dga)", "len(X_df)")
            new_source = new_source.replace("df_dga_massive[X_dga.columns]", "df_dga_massive[X_df.columns]")
            
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
