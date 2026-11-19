"""Run every tutorial cell with the local Python kernel and save a copy."""
import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("output", type=Path, help="Destination for the executed notebook")
a = p.parse_args()
base = Path(__file__).resolve().parent
notebook = nbformat.read(base / "quickstart.ipynb", as_version=4)
NotebookClient(
    notebook,
    kernel_name="python3",
    timeout=180,
    resources={"metadata": {"path": str(base)}},
).execute()
for cell in notebook.cells:
    if cell.cell_type == "code":
        cell.metadata.pop("execution", None)
a.output.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(notebook, a.output)
print(a.output)
