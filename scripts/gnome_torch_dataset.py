import torch
from torch.utils.data import Dataset
from pymatgen.core import Structure
import os
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor

class GNOMEPyTorchDataset(Dataset):
    def __init__(self, structure_dir, label_path, target_property="Formation Energy Per Atom"):
        """
        Args:
            structure_dir (str): Directory containing .CIF structure files
            label_path (str): Path to stable_materials_summary.csv
            target_property (str): Property to predict
        """
        self.structure_dir = structure_dir
        self.structure_files = sorted([f for f in os.listdir(structure_dir) if f.endswith('.CIF')])
        self.labels_df = pd.read_csv(label_path)
        self.target_property = target_property

        # Build lookup: formula --> label row
        self.label_lookup = self._build_label_lookup()

    def _build_label_lookup(self):
        lookup = {}
        for _, row in self.labels_df.iterrows():
            formula = row['Reduced Formula']
            lookup[formula] = row
        return lookup

    def __len__(self):
        return len(self.structure_files)

    def __getitem__(self, idx):
        file_name = self.structure_files[idx]
        file_path = os.path.join(self.structure_dir, file_name)

        # Load structure
        structure = Structure.from_file(file_path)
        formula = file_name.replace('.CIF', '')

        # Get label info
        label_row = self.label_lookup.get(formula, None)
        
        if label_row is None:
            raise ValueError(f"Label not found for formula {formula}")

        target_value = label_row[self.target_property]

        # Convert to tensor
        target_tensor = torch.tensor(target_value, dtype=torch.float32)

        # Also return ASE object if needed later
        ase_structure = AseAtomsAdaptor.get_atoms(structure)

        return {
            "structure": structure,  # pymatgen Structure
            "ase_structure": ase_structure,  # ASE Atoms
            "formula": formula,
            "target": target_tensor  # regression target
        }
