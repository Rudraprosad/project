import json

with open("Transformer_Monitoring_System.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

errors = []
for cell in nb['cells']:
    if cell.get('outputs'):
        for out in cell['outputs']:
            if out.get('output_type') == 'error':
                errors.append(out.get('ename', '') + ': ' + out.get('evalue', ''))

print(f"Total notebook errors found: {len(errors)}")
if errors:
    print(errors[0])
