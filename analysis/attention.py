import numpy as np
from typing import Tuple


def compute_attention_ratios(
    attn_weights: np.ndarray,  # [Sample, Layer, Head, Seq]
    image_token_mask: np.ndarray,  # [Sample, Seq]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    - image_attention: [Layer, Sample] per-layer attention mass on image tokens
    - text_attention: [Layer, Sample] per-layer attention mass on text tokens
    """
    H = attn_weights.shape[2]  # number of heads

    # slhq,sq -> ls  (contract heads & seq, keep layer + sample)
    image_attention = np.einsum("slhq,sq->ls", attn_weights, image_token_mask) / H
    text_attention = np.einsum("slhq,sq->ls", attn_weights, ~image_token_mask) / H
    return image_attention, text_attention


def precompute_attention_map(attn_weights, image_token_mask, patch_boxes, image):
    """Precomputes 2D attention maps for each layer."""
    orig_h, orig_w = image.size[::-1]
    precomputed_maps = {}
    num_layers_in_weights = attn_weights.shape[0]

    for layer_idx in range(num_layers_in_weights):
        # Assuming attn_weights[layer_idx] is [Heads, Seq_Tokens_Attended_By_Source]
        layer_attn_scores_for_tokens = attn_weights[layer_idx].mean(axis=0)

        img_attn_scores = layer_attn_scores_for_tokens[image_token_mask]

        img_attn_sum = img_attn_scores.sum()
        if img_attn_sum > 1e-9:
            img_attn_scores_normalized = img_attn_scores / img_attn_sum
        else:
            img_attn_scores_normalized = np.zeros_like(img_attn_scores)
            # Print warning only for the first problematic layer to avoid spam
            if not np.any(
                [
                    precomputed_maps[k].sum() > 1e-9
                    for k in precomputed_maps
                    if precomputed_maps[k].ndim > 0
                ]
            ):  # Crude check if other maps were also blank
                print(
                    f"Warning: Sum of attention for image tokens is zero/negligible for layer {layer_idx}. Map may appear blank."
                )

        attn_map_2d = np.zeros((orig_h, orig_w), dtype=np.float32)

        if len(img_attn_scores_normalized) != len(patch_boxes):
            print(
                f"Warning: Layer {layer_idx}: Mismatch between number of attention scores ({len(img_attn_scores_normalized)}) "
                f"and patch boxes ({len(patch_boxes)}). This layer's map will be empty."
            )
            precomputed_maps[layer_idx] = attn_map_2d  # Store empty map
            continue

        for patch_idx, (x0, y0, x1, y1) in enumerate(patch_boxes):
            attn_map_2d[y0:y1, x0:x1] = img_attn_scores_normalized[patch_idx]

        precomputed_maps[layer_idx] = attn_map_2d
    return precomputed_maps
