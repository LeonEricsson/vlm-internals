"""Numerical utilities for analysing attention tensors."""

import numpy as np
from typing import Tuple


def compute_attention_ratios(
    attn_weights: np.ndarray,  # [Sample, Layer, Head, Seq]
    image_token_mask: np.ndarray,  # [Sample, Seq]
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate attention mass over image and text tokens.

    Parameters
    ----------
    attn_weights : np.ndarray
        Attention tensor of shape ``[S, L, H, Q]`` where ``S`` is sample
        count and ``Q`` is the sequence length attended over.
    image_token_mask : np.ndarray
        Boolean mask ``[S, Q]`` indicating which tokens correspond to the image.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two arrays of shape ``[L, S]`` representing, for each layer and
        sample, the fraction of attention directed at image tokens and at text
        tokens respectively.
    """
    H = attn_weights.shape[2]  # number of heads

    # slhq,sq -> ls  (contract heads & seq, keep layer + sample)
    image_attention = np.einsum("slhq,sq->ls", attn_weights, image_token_mask) / H
    text_attention = np.einsum("slhq,sq->ls", attn_weights, ~image_token_mask) / H
    return image_attention, text_attention


def precompute_attention_map(attn_weights, image_token_mask, patch_boxes, image):
    """Project per-token attention scores back onto the image grid.

    Parameters
    ----------
    attn_weights : np.ndarray
        Attention weights for a single sample of shape ``[L, H, Q]``.
    image_token_mask : np.ndarray
        Boolean mask ``[Q]`` indicating the positions of image tokens.
    patch_boxes : List[Tuple[int, int, int, int]]
        Bounding boxes for each image patch in original pixel coordinates.
    image : PIL.Image
        Source image used for computing output resolution.

    Returns
    -------
    Dict[int, np.ndarray]
        Dictionary mapping from layer index to ``[H, W]`` heatmaps containing attention weights.
    """
    orig_h, orig_w = image.size[::-1]
    precomputed_maps = {}
    num_layers_in_weights = attn_weights.shape[0]

    for layer_idx in range(num_layers_in_weights):
        # Assuming attn_weights[layer_idx] is [Heads, Seq_Tokens_Attended_By_Source]
        layer_attn_scores_for_tokens = attn_weights[layer_idx].mean(axis=0)
        img_attn_scores = layer_attn_scores_for_tokens[image_token_mask]

        attn_map_2d = np.zeros((orig_h, orig_w), dtype=np.float32)

        if len(img_attn_scores) != len(patch_boxes):
            print(
                f"Warning: Layer {layer_idx}: Mismatch between number of attention scores ({len(img_attn_scores)}) "
                f"and patch boxes ({len(patch_boxes)}). This layer's map will be empty."
            )
            precomputed_maps[layer_idx] = attn_map_2d
            continue

        for patch_idx, (x0, y0, x1, y1) in enumerate(patch_boxes):
            attn_map_2d[y0:y1, x0:x1] = img_attn_scores[patch_idx]

        precomputed_maps[layer_idx] = attn_map_2d

    return precomputed_maps
