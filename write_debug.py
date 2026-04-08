import json
with open('Transformer_Monitoring_System.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('debug_output.txt', 'w', encoding='utf-8') as out:
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code':
            source = ''.join(cell['source'])
            if 'NEW: MASSIVE SYNTHETIC DGA EXPANSION' in source:
                out.write(source + '\n\nOUTPUTS:\n')
                for err in cell.get('outputs', []):
                    if err.get('output_type') == 'error':
                        out.write(f"ERR: {err.get('ename')} {err.get('evalue')}\n")
