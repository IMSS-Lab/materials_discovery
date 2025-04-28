# gnome_dataset.py Documentation

## Overview
The `GNOMEDataset` class is responsible for:
- Loading crystal structure files (.CIF) from the GNOME dataset.
- Loading tabular label information from the stable materials summary CSV.
- Linking structures to their material properties based on reduced formulas.
- Preparing each material as a structured sample for further machine learning usage.

It acts as a light-weight PyTorch-style dataset (but without requiring graph conversion yet).

---

## Class: `GNOMEDataset`

### `__init__(self, structure_dir, label_path)`
- **Purpose**: Initialize the dataset by loading structures and corresponding labels.
- **Arguments**:
  - `structure_dir (str)`: Path to the directory containing CIF structure files.
  - `label_path (str)`: Path to the CSV file containing stable materials summary data.
- **Process**:
  - Reads the label CSV using pandas.
  - Builds a lookup dictionary mapping each formula to its property information.
  - Prepares a list of CIF files for dataset indexing.

---

### `__len__(self)`
- **Purpose**: Returns the number of structures in the dataset.

---

### `__getitem__(self, idx)`
- **Purpose**: Given an index, loads and returns:
  - The structure (as a pymatgen `Structure` object).
  - Its associated label information (dictionary of properties).

- **Returns**:
  ```python
  {
    "formula": str,
    "structure": Structure,
    "label_info": dict
  }
