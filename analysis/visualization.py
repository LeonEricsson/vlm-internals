"""Plotting helpers for attention visualisation."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image


def plot_attention_ratios(
    image_attention: np.ndarray,
    text_attention: np.ndarray,
    title: str = "Attention Distribution Across Layers",
) -> None:
    """Plot average attention on image and text tokens per layer.

    Parameters
    ----------
    image_attention : np.ndarray
        Array ``[L, S]`` of attention mass on image tokens.
    text_attention : np.ndarray
        Array ``[L, S]`` of attention mass on text tokens.
    title : str, optional
        Title for the resulting plot.
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

    # plt.savefig("result.png")
    plt.show()


def plot_attention_map(
    attention_maps: dict, num_layers: int, image: Image.Image, sample: dict
):
    """Display an interactive heatmap of attention over the input image.

    Parameters
    ----------
    attention_maps : dict[int, np.ndarray]
        Precomputed per-layer attention maps from
        :func:`analysis.attention.precompute_attention_map`.
    num_layers : int
        Total number of layers in the model.
    image : PIL.Image
        Image that was fed into the model.
    sample_idx : int, optional
        Dataset index being visualized (for the window title).
    dataset_name : str, optional
        Name of the dataset for display purposes.
    """
    orig_h, orig_w = image.size[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7))

    fig.subplots_adjust(
        bottom=0.25,  # Space for slider
        top=0.88,  # Space for suptitle
        left=0.05,  # Left margin
        right=0.95,  # Right margin
        wspace=0.1,  # Reduced wspace
    )

    ax1.imshow(image)
    ax1.set_title("Original Image", fontsize=12)
    ax1.axis("off")

    all_map_values = [
        m for m in attention_maps.values() if m is not None and m.size > 0
    ]
    global_max_attn = 0.0
    if all_map_values:
        max_vals = [m.max() for m in all_map_values if m.ndim > 0 and m.size > 0]
        if max_vals:
            global_max_attn = max(max_vals)
    if global_max_attn <= 1e-9:
        global_max_attn = 1.0

    initial_layer_idx = 0
    if initial_layer_idx not in attention_maps:
        initial_layer_idx = sorted(attention_maps.keys())[0] if attention_maps else 0

    initial_attn_map = attention_maps.get(
        initial_layer_idx, np.zeros((orig_h, orig_w), dtype=np.float32)
    )

    ax2.imshow(image)
    attention_overlay_plot = ax2.imshow(
        initial_attn_map, cmap="viridis", alpha=0.65, vmin=0, vmax=global_max_attn
    )
    ax2.set_title(
        f"Attention Overlay - Layer {initial_layer_idx}/{num_layers - 1}", fontsize=12
    )
    ax2.axis("off")

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(
        attention_overlay_plot, cax=cax, label="Normalized Attention Weight"
    )

    slider_ax_position = [0.25, 0.1, 0.5, 0.03]
    ax_slider = fig.add_axes(slider_ax_position)
    layer_slider = None

    if num_layers > 1:
        layer_slider = Slider(
            ax=ax_slider,
            label="Layer",
            valmin=0,
            valmax=num_layers - 1,
            valinit=initial_layer_idx,
            valstep=1,
            valfmt="%d",
        )
        fig.text(
            slider_ax_position[0] - 0.03,
            slider_ax_position[1] + slider_ax_position[3] / 2,
            "0",
            ha="right",
            va="center",
            fontsize=10,
        )
        fig.text(
            slider_ax_position[0] + slider_ax_position[2] + 0.03,
            slider_ax_position[1] + slider_ax_position[3] / 2,
            f"{num_layers - 1}",
            ha="left",
            va="center",
            fontsize=10,
        )
    else:
        ax_slider.text(
            0.5,
            0.5,
            "Single Layer",
            ha="center",
            va="center",
            transform=ax_slider.transAxes,
        )
        ax_slider.set_axis_off()

    def update_plot_data(layer_idx_val):
        layer_idx = int(layer_idx_val)
        current_attn_map = attention_maps.get(
            layer_idx, np.zeros((orig_h, orig_w), dtype=np.float32)
        )
        attention_overlay_plot.set_data(current_attn_map)
        ax2.set_title(
            f"Attention Overlay - Layer {layer_idx}/{num_layers - 1}", fontsize=12
        )
        fig.canvas.draw_idle()

    if layer_slider:
        layer_slider.on_changed(update_plot_data)

    def on_key_press(event):
        if not layer_slider or not hasattr(event, "key") or event.key is None:
            return
        current_val, new_val = layer_slider.val, layer_slider.val
        if event.key == "right":
            new_val = min(current_val + 1, layer_slider.valmax)
        elif event.key == "left":
            new_val = max(current_val - 1, layer_slider.valmin)
        elif event.key == "home":
            new_val = layer_slider.valmin
        elif event.key == "end":
            new_val = layer_slider.valmax
        if new_val != current_val:
            layer_slider.set_val(new_val)

    fig.canvas.mpl_connect("key_press_event", on_key_press)

    suptitle_text = "Attention Visualization"
    suptitle_text += f": Sample {sample_idx}"
    fig.suptitle(suptitle_text, fontsize=16, y=0.97)

    plt.show()
