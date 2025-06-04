import sys
import json
import re
from pathlib import Path

def md_to_notebook(md_path, notebook_path):

	md_path = Path(md_path)
	with md_path.open("r", encoding="utf-8") as f:
		lines = f.readlines()

	cells = []

	cell_lines = []
	in_code_block = False

	for line in lines:

		if '```python' in line:
			cells.append({
				"cell_type": "markdown",
				"metadata": {},
				"source": cell_lines
			})
			cell_lines = []
			in_code_block = True

		elif '```' in line and in_code_block:
			cells.append({
				"cell_type": "code",
				"execution_count": None,
				"metadata": {},
				"outputs": [],
				"source": cell_lines
			})
			cell_lines = []
			in_code_block = False

		else:
			cell_lines.append(line)

	cells.append({
		"cell_type": "markdown",
		"metadata": {},
		"source": cell_lines
	})

	notebook = {
		"cells": cells,
		"metadata": {
			"kernelspec": {
				"display_name": "Python 3",
				"language": "python",
				"name": "python3"
			},
			"language_info": {
				"name": "python",
				"version": "3.x"
			}
		},
		"nbformat": 4,
		"nbformat_minor": 5
	}
	with open(notebook_path, "w", encoding="utf-8") as f:
		json.dump(notebook, f, indent=2)
	print(f"Notebook saved to {notebook_path}")

if __name__ == "__main__":
	md_to_notebook(sys.argv[1], sys.argv[2])