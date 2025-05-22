"""Analysis utilities for extracted model tensors.

This script is intended to be run after :mod:`extract.py`. It loads the
``.npz`` files produced during extraction and provides a few common analyses,
such as visualizing attention maps or summarizing how much attention is paid to
image tokens versus text tokens.
"""

import os
import argparse
from analysis.utils import load_extractions, calculate_patch_information
from analysis.attention import compute_attention_ratios, precompute_attention_map
from analysis.visualization import plot_attention_map, plot_attention_ratios
from data import get_dataset


def visual_attention_map_analysis(
    input_dir: str,
    sample_idx: int,
    dataset_name: str,
) -> None:
    """Interactively browse attention maps for a given sample.

    Parameters
    ----------
    input_dir : str
        Directory containing ``.npz`` files produced by :mod:`extract.py`.
    sample_idx : int
        Which sample from the dataset to load.
    dataset_name : str
        Dataset identifier understood by :func:`data.get_dataset`.
    """
    dataset = get_dataset(dataset_name)  # Uses dummy if not overridden

    print(f"Loading extractions from {input_dir}...")
    data = load_extractions(input_dir, keys=["attn_weights", "image_token_mask"])

    if sample_idx >= len(data["attn_weights"]) or sample_idx >= len(
        data["image_token_mask"]
    ):
        print(
            f"Error: sample_idx {sample_idx} is out of bounds for loaded extraction data arrays."
        )
        return

    # Get attention weights for this sample. Assuming [Layers, Heads, Seq_relevant_for_source]
    # The .mean(axis=1) in _precompute_attention_map would average over heads.
    # If data["attn_weights"][sample_idx] is [Layers, Heads, Seq], this is fine.
    attn_weights_sample = data["attn_weights"][sample_idx]
    image_token_mask_sample = data["image_token_mask"][sample_idx]

    sample_data = dataset[sample_idx]
    image = sample_data["image_options"][0]

    num_actual_layers = (
        attn_weights_sample.shape[0] if attn_weights_sample.ndim > 0 else 0
    )

    patch_boxes = calculate_patch_information(image, input_dir)

    attention_maps = precompute_attention_map(
        attn_weights_sample, image_token_mask_sample, patch_boxes, image
    )

    plot_attention_map(attention_maps, num_actual_layers, image, sample_data)


def analyze_attention_split(
    input_dir: str,
) -> None:
    """Compute average attention on image versus text tokens.

    Parameters
    ----------
    input_dir : str
        Directory containing ``.npz`` files with ``attn_weights`` and
        ``image_token_mask`` arrays.
    """

    print(f"Loading extractions from {input_dir}...")
    data = load_extractions(input_dir, keys=["attn_weights", "image_token_mask"])

    print("Computing attention ratios...")
    image_attention, text_attention = compute_attention_ratios(
        data["attn_weights"], data["image_token_mask"]
    )

    print("Plotting attention distribution...")
    plot_attention_ratios(
        image_attention,
        text_attention,
        title=f"Attention Distribution Across Layers\n{os.path.basename(input_dir)}",
    )


def main():
    """Entry point for running analyses from the command line."""
    parser = argparse.ArgumentParser(
        description="Analyze attention patterns in VLM extractions"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the NPZ extraction files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Directory to save the analysis plots",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Index of sample to visualize attention maps for",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="VSR",
        help="Dataset name as understood by get_dataset().",
    )

    args = parser.parse_args()
    visual_attention_map_analysis(args.input_dir, args.sample_idx, args.dataset)

    print("Analysis complete!")


if __name__ == "__main__":
    main()
