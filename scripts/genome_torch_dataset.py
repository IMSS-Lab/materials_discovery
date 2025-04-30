import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import logging

# Optional imports
try:
    from pymatgen.core import Structure
    from pymatgen.io.cif import CifParser
    from pymatgen.io.ase import AseAtomsAdaptor
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    logging.warning("Pymatgen/ASE not found. GNOMEPyTorchDataset structure loading/conversion will be limited.")


class GNOMEPyTorchDataset(Dataset):
    """
    PyTorch Dataset for loading GNoME structures and labels.

    Loads structures from CIF files and converts them to a format suitable
    for PyTorch geometric or other graph-based models (e.g., ASE Atoms objects).
    """
    def __init__(self, structure_dir: str, label_path: str,
                 target_property: str = "Formation Energy Per Atom",
                 max_data_size: Optional[int] = None,
                 use_ase: bool = True):
        """
        Args:
            structure_dir (str): Directory containing .CIF structure files.
            label_path (str): Path to 'stable_materials_summary.csv'.
            target_property (str): Property column name in label_path to use as the target.
            max_data_size (Optional[int]): Maximum number of samples to load.
            use_ase (bool): Whether to convert structures to ASE Atoms objects. Requires ASE and Pymatgen.
        """
        if not os.path.isdir(structure_dir):
             raise FileNotFoundError(f"Structure directory not found: {structure_dir}")
        if not os.path.isfile(label_path):
             raise FileNotFoundError(f"Label file not found: {label_path}")
        if use_ase and not PYMATGEN_AVAILABLE: # ASE conversion relies on pymatgen adapter here
             raise ImportError("use_ase=True requires Pymatgen and ASE.")

        self.structure_dir = structure_dir
        self.target_property = target_property
        self.use_ase = use_ase

        # Load label CSV
        try:
            self.labels_df = pd.read_csv(label_path)
            logging.info(f"Loaded labels for {len(self.labels_df)} materials from {label_path}")
            if 'Reduced Formula' not in self.labels_df.columns:
                 raise ValueError("Label file must contain 'Reduced Formula' column.")
            if self.target_property not in self.labels_df.columns:
                 raise ValueError(f"Target property '{self.target_property}' not found in label file columns.")
        except Exception as e:
            logging.error(f"Failed to load or parse label file {label_path}: {e}")
            raise

        # Build lookup: reduced_formula --> label row dict
        self.label_lookup = self._build_label_lookup()

        # List structure files and map them to labels
        self.data_index = self._build_data_index()

        # Limit dataset size if requested
        if max_data_size is not None and max_data_size < len(self.data_index):
            logging.info(f"Limiting dataset size from {len(self.data_index)} to {max_data_size}")
            self.data_index = self.data_index[:max_data_size]

        logging.info(f"Initialized GNOMEPyTorchDataset with {len(self.data_index)} valid entries.")

    def _build_label_lookup(self):
        """Builds a lookup from reduced formula string to label data."""
        lookup = {}
        for _, row in self.labels_df.iterrows():
            formula = row['Reduced Formula']
            lookup[formula] = row.to_dict()
        logging.info(f"Built label lookup for {len(lookup)} unique reduced formulas.")
        return lookup

    def _build_data_index(self):
        """ Creates a list of valid data points (file exists and has label). """
        data_index = []
        structure_files = sorted([
            f for f in os.listdir(self.structure_dir)
            if f.endswith('.CIF') and not f.startswith('.')
        ])
        logging.info(f"Found {len(structure_files)} CIF files in {self.structure_dir}.")

        found_labels_count = 0
        for file_name in structure_files:
             reduced_formula = file_name.replace('.CIF', '')
             if reduced_formula in self.label_lookup:
                  label_info = self.label_lookup[reduced_formula]
                  target_value = label_info.get(self.target_property)
                  # Check if target value is valid (not NaN)
                  if target_value is not None and pd.notna(target_value):
                       file_path = os.path.join(self.structure_dir, file_name)
                       data_index.append({
                           "file_path": file_path,
                           "formula": reduced_formula,
                           "target_value": float(target_value) # Ensure float
                       })
                       found_labels_count += 1
                  # else:
                  #      logging.debug(f"Skipping {file_name}: Target property '{self.target_property}' is missing or NaN.")
             # else:
             #      logging.debug(f"Skipping {file_name}: Reduced formula '{reduced_formula}' not found in labels.")

        logging.info(f"Matched {found_labels_count} structure files with valid labels.")
        return data_index


    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        if idx >= len(self.data_index):
             raise IndexError("Index out of range.")

        index_entry = self.data_index[idx]
        file_path = index_entry["file_path"]
        formula = index_entry["formula"]
        target_value = index_entry["target_value"]

        # Load structure and optionally convert to ASE
        structure_pmg = None
        structure_ase = None
        if PYMATGEN_AVAILABLE:
             try:
                  parser = CifParser(file_path)
                  structure_pmg = parser.get_structures(primitive=False)[0]
                  if self.use_ase:
                       structure_ase = AseAtomsAdaptor.get_atoms(structure_pmg)
             except Exception as e:
                  logging.warning(f"Could not parse/convert structure file {file_path}: {e}")
                  # Depending on the downstream model, this might be an error or handled later
                  # For now, return None structures but keep target value.

        # Convert target to tensor
        target_tensor = torch.tensor(target_value, dtype=torch.float32)

        # Return data dictionary
        # Models downstream will need to handle None structures if parsing fails
        # Or filter out problematic samples in a preprocessing step / collate_fn
        item = {
            "structure_pmg": structure_pmg,  # Pymatgen Structure object or None
            "structure_ase": structure_ase,    # ASE Atoms object or None
            "formula": formula,
            "target": target_tensor,         # Regression target tensor
            "file_path": file_path
        }

        # Add graph representation if needed (e.g., for PyTorch Geometric)
        # item['graph'] = self._create_graph_representation(structure_ase or structure_pmg)

        return item

    # def _create_graph_representation(self, structure):
    #     """ Placeholder: Convert structure (ASE/Pymatgen) to graph format (e.g., Data object). """
    #     if structure is None: return None
    #     # Implementation depends on the GNN library (PyG, DGL)
    #     # Example using ASE:
    #     # from torch_geometric.data import Data
    #     # edge_index = ... # Build neighbor list
    #     # pos = torch.tensor(structure.get_positions(), dtype=torch.float32)
    #     # x = ... # Node features (e.g., atomic numbers)
    #     # data = Data(x=x, edge_index=edge_index, pos=pos, ...)
    #     # return data
    #     return None # Return None for now