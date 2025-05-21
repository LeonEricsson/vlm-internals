import numpy as np
from pathlib import Path
from typing import Dict, List


def load_extractions(input_dir: str, keys: List[str]) -> Dict[str, np.ndarray]:
    """Load specified keys from NPZ files in the input directory and concatenate them."""
    all_data = {key: [] for key in keys}

    # Find all NPZ files in the directory
    npz_files = sorted(Path(input_dir).glob("all_tensors_*.npz"))
    if not npz_files:
        raise ValueError(f"No NPZ files found in {input_dir}")

    # Load and concatenate all files
    for npz_file in npz_files:
        data = np.load(npz_file)
        for key in keys:
            if key not in data:
                raise ValueError(f"Key {key} not found in {npz_file}")
            if key == "attns":
                all_data[key].append(np.swapaxes(data[key], 0, 1))
            else:
                all_data[key].append(data[key])

    # Concatenate along the sample dimension
    return {k: np.concatenate(v, axis=0) for k, v in all_data.items()}
