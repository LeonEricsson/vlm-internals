"""Helper utilities for loading and preprocessing extracted data."""

import numpy as np
from pathlib import Path
from typing import Dict, List
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import os


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
            all_data[key].append(data[key])

    # Concatenate along the sample dimension
    return {k: np.concatenate(v, axis=0) for k, v in all_data.items()}


def image_patch_bb(
    patch_size: int,
    merge_size: int,
    grid_thw: list[int],
    orig_resolution: list[int],
):
    """
    Compute per-token bounding-boxes for a ViT-style vision encoder
    (e.g. Qwen-VL, CLIP-ViT) after optional patch-merging. A processed
    dataset will have the same bounding boxes unless the original /
    unprocessed dataset contains images of varying resolutions.

    Parameters
    ----------
    patch_size        – raw ViT patch edge in px (e.g. 14)
    merge_size        – patch-merger factor (`1` = no merge, `2` = 2×2→1)
    grid_thw          – [T, H_grid, W_grid] from `image_grid_thw`
    orig_resolution   – [H_orig, W_orig] of the *original* image

    Returns
    -------
    List[(x0, y0, x1, y1)] in original-image pixels, length
    `(H_grid//merge_size) * (W_grid//merge_size)`, ordered row-major.
    """
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


def calculate_patch_information(image, input_dir):
    """Calculates patch bounding boxes based on the image and processor config."""
    orig_h, orig_w = image.size[::-1]

    processor_path = os.path.join(input_dir, "processor")
    try:
        processor = AutoProcessor.from_pretrained(processor_path)
    except Exception as e:
        print(f"Error loading processor from {processor_path}: {e}")
        raise

    patch_size = processor.image_processor.patch_size
    merge_size = processor.image_processor.merge_size

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": "Describe this image.",
                },
            ],
        }
    ]
    image_inputs, _ = process_vision_info(messages)

    # The processor call might vary based on model type
    inputs = processor(
        text=["Describe this image."],
        images=image_inputs if image_inputs is not None else image,
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    # Assuming inputs has 'image_grid_thw' for patch grid dimensions [T, H, W]
    # For a single image, T=1. grid_thw[0] would be [1, H_patches, W_patches]
    if not hasattr(inputs, "image_grid_thw") or inputs.image_grid_thw is None:
        # Fallback or error if image_grid_thw is not available
        # For ViT-like models, grid size can be (img_size // patch_size) x (img_size // patch_size)
        # This is a simplified fallback, actual calculation might be more complex
        print(
            "Warning: 'image_grid_thw' not found in processor output. Attempting fallback for grid calculation."
        )
        num_patch_h = orig_h // patch_size
        num_patch_w = orig_w // patch_size
        grid_thw = [1, num_patch_h, num_patch_w]  # [T, H, W]
    else:
        grid_thw = inputs.image_grid_thw[0].tolist()

    patch_boxes = image_patch_bb(
        patch_size=patch_size,
        merge_size=merge_size,
        grid_thw=grid_thw,
        orig_resolution=[orig_h, orig_w],
    )
    return patch_boxes
