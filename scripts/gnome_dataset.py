import os
import pandas as pd
from pymatgen.core import Structure

class GNOMEDataset:
    def __init__(self, structure_dir, label_path):
        """
        Args:
            structure_dir (str): Directory containing .CIF structure files
            label_path (str): Path to stable_materials_summary.csv
        """
        self.structure_dir = structure_dir
        self.structure_files = sorted([f for f in os.listdir(structure_dir) if f.endswith('.CIF')])

        # Load label CSV
        self.labels_df = pd.read_csv(label_path)

        # Create a lookup dictionary {formula_string: label_row}
        self.label_lookup = self._build_label_lookup()

    def _build_label_lookup(self):
        """Build a lookup from formula string to label data."""
        lookup = {}
        for _, row in self.labels_df.iterrows():
            formula = row['Reduced Formula']  # ✅ (Important fix: capital R, space, capital F)
            lookup[formula] = row
        return lookup

    def __len__(self):
        return len(self.structure_files)

    def __getitem__(self, idx):
        file_name = self.structure_files[idx]
        file_path = os.path.join(self.structure_dir, file_name)

        # Load structure
        structure = Structure.from_file(file_path)

        # Extract formula (remove .CIF ending)
        formula = file_name.replace('.CIF', '')

        # Find labels
        label_info = self.label_lookup.get(formula, None)

        data = {
            'structure': structure,
            'formula': formula,
            'label_info': label_info.to_dict() if label_info is not None else None
        }
        return data

