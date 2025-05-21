import os
import argparse
from analysis.utils import load_extractions
from analysis.attention import compute_attention_ratios, plot_attention_ratios


def analyze_attention_split(
    input_dir: str,
) -> None:
    """
    Analyze how attention is distributed between image and text tokens across model layers.
    This analysis looks at the amount of attention placed on image tokens versus text tokens
    throughout the model's layers.
    """

    print(f"Loading extractions from {input_dir}...")
    data = load_extractions(input_dir, keys=["attns", "image_positions"])

    print("Computing attention ratios...")
    image_attention, text_attention = compute_attention_ratios(
        data["attns"], data["image_positions"]
    )

    print("Plotting attention distribution...")
    plot_attention_ratios(
        image_attention,
        text_attention,
        title=f"Attention Distribution Across Layers\n{os.path.basename(input_dir)}",
    )


def main():
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
    args = parser.parse_args()

    # Call whatever analysis function you want
    analyze_attention_split(args.input_dir)

    print("Analysis complete!")


if __name__ == "__main__":
    main()
