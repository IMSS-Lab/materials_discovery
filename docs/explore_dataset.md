
---



```markdown
# explore_dataset.ipynb Documentation

## Overview
This Jupyter notebook is designed for:
- Verifying the loading of the GNOME dataset.
- Visualizing basic structure-label mappings.
- Preparing for further machine learning experiments.

---

## Notebook Outline

### 1. Setup
- Appends the `scripts/` folder to Python's path to allow importing the `GNOMEDataset` class.
- Imports necessary libraries (`pandas`, `pymatgen`, etc.).

### 2. Dataset Initialization
- Creates a `GNOMEDataset` instance with:
  - `structure_dir`: Path to the CIF files organized by reduced formula.
  - `label_path`: Path to the stable materials summary CSV.

- Prints the number of structures loaded.

### 3. First Sample Inspection
- Loads the first material sample.
- Prints:
  - Formula
  - Associated property information (label dictionary)

---

## Future Extensions
- Visualize structures in 3D (requires additional libraries like `nglview`).
- Extract features from structures for ML models.
- Prepare PyTorch Datasets and DataLoaders for model training.

---
