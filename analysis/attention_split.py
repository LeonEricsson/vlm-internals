import argparse
import numpy as np
import yaml
import matplotlib.pyplot as plt


def load_config(config_path):
    """
    Expects a YAML like:
    token_index_map:
      image: [0,1,2,3]      # positions of all image tokens in sequence
      text:  [4,5,6,...,N]  # positions of all text tokens
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["token_index_map"]


def compute_image_attention_fraction(attns, image_idxs, text_idxs):
    """
    attns: [num_layers, total_samples, heads, seq, seq]
    Returns: image_frac: [num_layers], text_frac: [num_layers]
    """
    num_layers, S, H, L, _ = attns.shape
    image_frac = np.zeros(num_layers)
    text_frac = np.zeros(num_layers)

    # collapse samples & heads for simplicity
    attn_all = attns.reshape(num_layers, S * H, L, L)

    # for each layer, sum attention weights going *to* image vs text tokens
    for layer in range(num_layers):
        A = attn_all[layer]  # [S*H, L, L]
        # sum over query positions and over all (sample×head) axes
        total_mass = A.sum()
        img_mass = A[..., image_idxs].sum()
        txt_mass = A[..., text_idxs].sum()
        # numerical check: img_mass+txt_mass ≈ total_mass
        image_frac[layer] = img_mass / total_mass
        text_frac[layer] = txt_mass / total_mass

    return image_frac, text_frac


def plot_attention_split(image_frac, text_frac, output_path):
    layers = np.arange(len(image_frac)) + 1
    plt.figure(figsize=(8, 4))
    plt.plot(layers, image_frac * 100, marker="o", label="Image tokens")
    plt.plot(layers, text_frac * 100, marker="s", label="Text tokens")
    plt.xlabel("Transformer Layer")
    plt.ylabel("Average Attention (%)")
    plt.title("Cross-Modal Attention Split per Layer")
    plt.xticks(layers)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce §3.1: ratio of attention to image vs text tokens per layer."
    )
    parser.add_argument(
        "--input_npz", type=str, required=True, help="Path to all_tensors.npz"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="YAML with token_index_map"
    )
    parser.add_argument("--output_plot", type=str, default="attention_split.png")
    args = parser.parse_args()

    # 1. load data
    data = np.load(args.input_npz)
    attns = data["attns"]  # [layers, samples, heads, seq, seq]

    # 2. load which positions are image/text
    token_map = load_config(args.config)
    image_idxs = token_map["image"]
    text_idxs = token_map["text"]

    # 3. compute fractions
    image_frac, text_frac = compute_image_attention_fraction(
        attns, image_idxs, text_idxs
    )

    # 4. plot
    plot_attention_split(image_frac, text_frac, args.output_plot)


if __name__ == "__main__":
    main()
