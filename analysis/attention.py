import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def compute_attention_ratios(
    attn_weights: np.ndarray,  # [Sample, Layer, Head, Seq]
    image_positions: np.ndarray,  # [Sample, Seq]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    - image_attention: [Layer, Sample] per-layer attention mass on image tokens
    - text_attention: [Layer, Sample] per-layer attention mass on text tokens
    """
    H = attn_weights.shape[2]  # number of heads

    # slhq,sq -> ls  (contract heads & seq, keep layer + sample)
    image_attention = np.einsum("slhq,sq->ls", attn_weights, image_positions) / H
    text_attention = np.einsum("slhq,sq->ls", attn_weights, ~image_positions) / H
    return image_attention, text_attention


def plot_attention_ratios(
    image_attention: np.ndarray,
    text_attention: np.ndarray,
    title: str = "Attention Distribution Across Layers",
) -> None:
    """
    Plot the average attention ratios across layers.
    Shows the figure instead of saving it.
    """
    mean_img_attn = np.mean(image_attention, axis=1)
    mean_txt_attn = np.mean(text_attention, axis=1)

    plt.figure(figsize=(10, 6))
    layers = np.arange(len(mean_img_attn))

    plt.plot(layers, mean_img_attn, "b-", label="Image Attention")
    plt.plot(layers, mean_txt_attn, "r-", label="Text Attention")

    plt.xlabel("Layer")
    plt.ylabel("Average Attention Ratio")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig("result.png")
    # plt.show()
