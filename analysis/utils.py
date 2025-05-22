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


def image_patch_bb(
    patch_size: int,
    merge_size: int,
    grid_thw: list[int],
    orig_resolution: list[int],
):
    H_orig, W_orig = orig_resolution

    # --- grab geometry ----------------------------
    _, H_grid, W_grid = grid_thw

    # grid after merging
    H_tok, W_tok = H_grid // merge_size, W_grid // merge_size

    H_real = H_grid * patch_size
    W_real = W_grid * patch_size

    sx = W_orig / W_real
    sy = H_orig / H_real

    # --- bounding-box for token k ------------------
    def token_bbox(k: int):
        r, c = divmod(k, W_tok)
        r0_p = r * merge_size
        c0_p = c * merge_size

        x0 = int(round(c0_p * patch_size * sx))
        y0 = int(round(r0_p * patch_size * sy))
        x1 = int(round((c0_p + merge_size) * patch_size * sx))
        y1 = int(round((r0_p + merge_size) * patch_size * sy))
        return x0, y0, x1, y1

    num_img_tokens = H_tok * W_tok
    return [token_bbox(k) for k in range(num_img_tokens)]
