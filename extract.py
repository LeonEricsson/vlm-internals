"""Extraction pipeline for attention weights and activations.

This script is the entry point for running a vision-language model over a
dataset and saving intermediate tensors to disk. The saved ``.npz`` files can
later be analysed using :mod:`analyze.py`.
"""

import os
import argparse
import time

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from data import get_dataset
from model_utils import get_wrapper_collator


def debug_batch(loader, wrapper, device):
    """Run one batch and print the model's output for inspection.

    This helper is useful when wiring up a new dataset or model. It grabs the
    first batch from ``loader`` and runs a short generation step so you can
    verify that the preprocessing pipeline, tokenization and model weights are
    working as expected.

    Parameters
    ----------
    loader : DataLoader
        Loader yielding batches of preprocessed samples.
    wrapper : Qwen2_5_VLWrapper
        Model wrapper used to run inference.
    device : torch.device or str
        Device on which to run the model.
    """
    print("[DEBUG] Inspecting first batch via DataLoader + collator…\n")
    batch = next(iter(loader))

    # Move all inputs to device
    batch = {k: v.to(device) for k, v in batch.items()}
    wrapper.model.to(device)

    gen_ids = wrapper.model.generate(
        **batch,
        max_new_tokens=20,
    )
    generated = wrapper.processor.tokenizer.batch_decode(
        gen_ids, skip_special_tokens=True
    )[0]
    print(">>> [DEBUG] Generation:")
    print(generated, "\n")


def label_prediction(logits, labels, tokenizer):
    """Convert model logits into boolean correctness predictions.

    The function looks at the prediction for the final generated token of each
    sample and interprets it as either ``True`` or ``False``. Any other token is
    treated as ``None`` and subsequently marked incorrect.

    Parameters
    ----------
    logits : torch.Tensor
        Model output logits of shape ``[B, seq, vocab]``.
    labels : torch.Tensor
        Ground-truth labels (0 or 1) for each sample.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer used to decode the predictions.

    Returns
    -------
    List[bool]
        Whether each prediction matches the true label.
    """
    last_logits = logits[:, -1, :]
    pred_ids = last_logits.argmax(dim=-1)
    preds, trues = [], labels.tolist()
    for pid in pred_ids.cpu().tolist():
        tok = tokenizer.decode([pid], skip_special_tokens=True).strip().lower()
        if tok.startswith("true"):
            preds.append(1)
        elif tok.startswith("false"):
            preds.append(0)
        else:
            preds.append(None)
    correct = [bool(p == t) for p, t in zip(preds, trues)]
    return correct


def flush_storage(storage, output_dir, chunk_idx):
    """Write accumulated tensors to disk.

    Parameters
    ----------
    storage : dict
        Lists of per-batch arrays and masks collected during extraction.
    output_dir : str
        Directory in which the ``.npz`` file will be created.
    chunk_idx : int
        Index used to differentiate chunk files when ``save_n_samples`` is
        enabled.
    """
    attn_weights = np.concatenate(storage["attn_weights"], axis=0)
    acts = np.concatenate(storage["acts"], axis=0)
    mask = np.concatenate(storage["mask"], axis=0)
    correct = np.array(storage["correct"], dtype=bool)
    image_token_mask = np.concatenate(storage["image_token_mask"], axis=0)

    out_path = os.path.join(output_dir, f"all_tensors_{chunk_idx}.npz")
    np.savez_compressed(
        out_path,
        attn_weights=attn_weights,
        acts=acts,
        correct=correct,
        image_token_mask=image_token_mask,
        mask=mask,
    )
    print(f"Saved chunk {chunk_idx} ({correct.size} samples) to {out_path}")


def parse_args():
    """Parse command line arguments for extraction."""

    parser = argparse.ArgumentParser(
        description="Extract attention maps, activations, and logits from VLMs in chunks."
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument(
        "--dataset",
        type=str,
        default="VSR",
        help="Dataset name as understood by get_dataset().",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./extractions",
        help="Directory to save the extracted tensors (.npz).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for extraction."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of batches to process.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run a single debug generation and exit.",
    )
    parser.add_argument(
        "--save_n_samples",
        type=int,
        default=100,
        help="Save intermediate .npz files after this many samples.",
    )
    return parser.parse_args()


def main():
    """Run the extraction loop and save results to disk."""
    args = parse_args()

    model_safe = args.model.replace("/", "_")
    out_base = os.path.join(args.output_dir, f"{model_safe}_{args.dataset}")
    os.makedirs(out_base, exist_ok=True)

    dataset = get_dataset(args.dataset)
    wrapper, collator = get_wrapper_collator(
        model_name=args.model, device=args.device, eval=False
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    print(f"Wrapper: {wrapper.__class__}, Collator: {collator.__class__}")
    if args.debug:
        debug_batch(loader, wrapper, args.device)
        return

    storage = {
        "attn_weights": [],  # [Sample, Layer, Head, Seq]
        "acts": [],  # [Sample, Layer + 1, Hidden Dim]
        "mask": [],  # [Sample, Seq]
        "correct": [],  # [Sample]
        "image_token_mask": [],  # [Sample, Seq]
    }
    sample_count = 0
    chunk_idx = 0
    start_time = time.time()

    for idx, batch in enumerate(tqdm(loader, desc="Extracting", unit="batch")):
        # Extract diagnostic masks and labels
        img_pos = batch.pop("image_token_mask")
        labels = batch.pop("labels")
        # Move inputs to device
        batch = {k: v.to(args.device) for k, v in batch.items()}

        out = wrapper.forward_and_capture(batch)
        correct = label_prediction(out["logits"], labels, wrapper.processor.tokenizer)

        # Detach and convert to numpy (float16 for space savings)
        attn_np = np.stack(
            [a.detach().cpu().to(torch.float16).numpy() for a in out["attns"]],
            axis=0,
        )
        act_np = np.stack(
            [h.detach().cpu().to(torch.float16).numpy() for h in out["acts"]],
            axis=0,
        )
        act_np = act_np[:, :, -1, :]  # activations at last token
        attn_mask = batch["attention_mask"].detach().cpu().numpy()

        storage["attn_weights"].append(np.swapaxes(attn_np, 0, 1))
        storage["acts"].append(np.swapaxes(act_np, 0, 1))
        storage["mask"].append(attn_mask)
        storage["correct"].extend(correct)
        storage["image_token_mask"].append(img_pos.cpu().numpy())

        sample_count += len(correct)

        if args.save_n_samples and sample_count >= args.save_n_samples:
            flush_storage(storage, out_base, chunk_idx)
            chunk_idx += 1
            # Reset storage and counter for next chunk
            storage = {k: [] for k in storage}
            sample_count = 0

        if args.limit and idx + 1 >= args.limit:
            break

    # Final flush for any remaining samples
    if args.save_n_samples:
        if sample_count > 0:
            flush_storage(storage, out_base, chunk_idx)
    else:
        # No chunking: save all in one file
        flush_storage(storage, out_base, 0)

    wrapper.save_model_processor(out_base)

    elapsed = time.time() - start_time
    print(f"Extraction complete in {elapsed:.2f}s.")


if __name__ == "__main__":
    main()
