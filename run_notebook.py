import nbformat
from nbclient import NotebookClient

print("Loading notebook...")
with open('Transformer_Monitoring_System.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

print("Executing notebook...")
client = NotebookClient(nb, timeout=600, kernel_name='python3')
client.execute()

print("Saving notebook...")
with open('Transformer_Monitoring_System.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook executed successfully!")
