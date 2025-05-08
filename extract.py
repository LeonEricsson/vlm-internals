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
    """
    Run a single batch through the model in generate mode for debugging:
    - Fetches the first batch from the loader
    - Moves tensors to the device and generates a sample output
    - Prints the generated text for manual inspection
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
    """
    Compute True/False predictions from model logits:
    - Takes the last-token logits for each sample
    - Picks the top-1 token ID
    - Decodes to string and maps 'true'* to 1, 'false'* to 0, else None
    - Returns a list of booleans indicating correctness vs. ground truth
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
    """
    Concatenate stored batches and save them to a compressed NPZ file.

    Args:
        storage (dict): Lists of per-batch arrays and masks.
        output_dir (str): Directory to write the NPZ.
        chunk_idx (int): Index for naming the chunk file.
    """
    attns = np.concatenate(storage["attns"], axis=1)
    acts = np.concatenate(storage["acts"], axis=1)
    mask = np.concatenate(storage["mask"], axis=0)
    correct = np.array(storage["correct"], dtype=bool)
    image_positions = np.concatenate(storage["image_positions"], axis=0)
    text_positions = np.concatenate(storage["text_positions"], axis=0)

    out_path = os.path.join(output_dir, f"all_tensors_{chunk_idx}.npz")
    np.savez_compressed(
        out_path,
        attns=attns,
        acts=acts,
        correct=correct,
        image_positions=image_positions,
        text_positions=text_positions,
        mask=mask,
    )
    print(f"Saved chunk {chunk_idx} ({correct.size} samples) to {out_path}")


def parse_args():
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
        "attns": [],
        "acts": [],
        "mask": [],
        "correct": [],
        "image_positions": [],
        "text_positions": [],
    }
    sample_count = 0
    chunk_idx = 0
    start_time = time.time()

    for idx, batch in enumerate(tqdm(loader, desc="Extracting", unit="batch")):
        # Extract diagnostic masks and labels
        img_pos = batch.pop("image_positions")
        txt_pos = batch.pop("text_positions")
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
        attn_np = attn_np[:, :, :, -1, :]
        act_np = np.stack(
            [h.detach().cpu().to(torch.float16).numpy() for h in out["acts"]],
            axis=0,
        )
        act_np = act_np[:, :, -1, :]
        attn_mask = batch["attention_mask"].detach().cpu().numpy()

        storage["attns"].append(attn_np)
        storage["acts"].append(act_np)
        storage["mask"].append(attn_mask)
        storage["correct"].extend(correct)
        storage["image_positions"].append(img_pos.cpu().numpy())
        storage["text_positions"].append(txt_pos.cpu().numpy())

        sample_count += len(correct)
        # Flush if threshold reached
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

    elapsed = time.time() - start_time
    print(f"Extraction complete in {elapsed:.2f}s.")


if __name__ == "__main__":
    main()
