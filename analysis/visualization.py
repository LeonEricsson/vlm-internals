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
        Dictionary mapping from layer index to ``[H, W]`` heatmaps containing
        unnormalized attention weights.
    num_layers : int
        Total number of layers in the model.
    image : PIL.Image
        Image that was fed into the model.
    sample : dict
        Sample containing caption options and other metadata.
    """
    orig_h, orig_w = image.size[::-1]

    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    fig.subplots_adjust(
        bottom=0.25,  # Space for slider
        top=0.88,  # Space for suptitle
        left=0.05,  # Left margin
        right=0.95,  # Right margin
        wspace=0.2,  # Space between subplots
        hspace=0.3,  # Space between rows
    )

    # Original image
    ax1.imshow(image)
    ax1.set_title("Original Image", fontsize=12)
    ax1.axis("off")

    # Get global max attention for both views
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

    # Normalized attention overlay
    ax2.imshow(image)
    attention_overlay_plot = ax2.imshow(
        initial_attn_map, cmap="viridis", alpha=0.65, vmin=0, vmax=1.0
    )
    ax2.set_title(
        f"Normalized Attention Overlay - Layer {initial_layer_idx}/{num_layers - 1}",
        fontsize=12,
    )
    ax2.axis("off")

    # Unnormalized attention overlay
    ax3.imshow(image)
    unnorm_attention_overlay_plot = ax3.imshow(
        initial_attn_map, cmap="viridis", alpha=0.65, vmin=0, vmax=global_max_attn
    )
    ax3.set_title(
        f"Unnormalized Attention Overlay - Layer {initial_layer_idx}/{num_layers - 1}",
        fontsize=12,
    )
    ax3.axis("off")

    # Add colorbars
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cbar2 = fig.colorbar(
        attention_overlay_plot, cax=cax2, label="Normalized Attention Weight"
    )

    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.1)
    cbar3 = fig.colorbar(
        unnorm_attention_overlay_plot, cax=cax3, label="Unnormalized Attention Weight"
    )

    # Add attention statistics
    ax4.axis("off")
    stats_text = ax4.text(
        0.5,
        0.5,
        "Attention Statistics:\n\nLoading...",
        ha="center",
        va="center",
        fontsize=12,
        transform=ax4.transAxes,
    )
    ax4.set_title("Attention Distribution Statistics", fontsize=12)

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

        # Normalize the attention map for the normalized view
        attn_sum = current_attn_map.sum()
        if attn_sum > 1e-9:
            normalized_map = current_attn_map / attn_sum
        else:
            normalized_map = np.zeros_like(current_attn_map)

        # Update both attention overlays
        attention_overlay_plot.set_data(normalized_map)
        unnorm_attention_overlay_plot.set_data(current_attn_map)

        # Update titles
        ax2.set_title(
            f"Normalized Attention Overlay - Layer {layer_idx}/{num_layers - 1}",
            fontsize=12,
        )
        ax3.set_title(
            f"Unnormalized Attention Overlay - Layer {layer_idx}/{num_layers - 1}",
            fontsize=12,
        )

        # Calculate and update statistics
        if current_attn_map is not None and current_attn_map.size > 0:
            total_attention = np.sum(current_attn_map)
            max_attention = np.max(current_attn_map)
            mean_attention = np.mean(current_attn_map)
            std_attention = np.std(current_attn_map)

            # Calculate percentage of attention going to image tokens
            # Assuming the total attention across all tokens (image + text) is 1.0
            image_attention_percentage = total_attention * 100

            stats_text.set_text(
                f"Attention Statistics (Layer {layer_idx}):\n\n"
                f"Total Image Attention: {total_attention:.4f}\n"
                f"Max Attention: {max_attention:.4f}\n"
                f"Mean Attention: {mean_attention:.4f}\n"
                f"Std Attention: {std_attention:.4f}\n"
                f"Image Token Attention %: {image_attention_percentage:.2f}%"
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

    question = sample["caption_options"][0]
    suptitle_text = f"Caption: {question}"
    fig.suptitle(suptitle_text, fontsize=16, y=0.97)

    plt.show()
